"""How much of a score difference is the change, and how much is the dice.

**Written because every claim made about an intervention in this project has been undermined by
this.** Three runs of one scenario produced overall 0.86, 0.89 and a third figure again, with
`tool_selection` at 1.00 then 0.50 and `draft_reliability` 0.83 then 0.56 — on unchanged code. A
FAIL became a PASS between two runs and was reported as an intervention working; the captured
bundle showed the intervention had not fired at all.

So this answers one question: **for each dimension, how large must a difference be before it means
something?** Two numbers do that.

- `sigma_att` — the spread *within* one scenario across repeated attempts. This is the dice: the
  same fixtures, the same code, a different answer.
- `sigma_sc` — the spread *between* scenario means. This is not noise; it is the scenarios being
  genuinely different problems. It matters because it says an unpaired comparison — run A on some
  scenarios, run B on others — is dominated by which scenarios were picked.

From `sigma_att` comes the minimum detectable effect for a **paired** comparison, which is the
only kind this harness supports honestly: the same scenarios, re-scored or re-run, before and
after. Pairing removes `sigma_sc` entirely, which is why it is worth insisting on.

The formula is the standard two-sided paired one at 95% confidence and 80% power,
`MDE = (z_{0.975} + z_{0.80}) * sigma_att * sqrt(2/n)`, and the coefficient is the same 2.80 that
`cortex.analysis.identifiability` uses — deliberately, because a project that computes a detection
floor two different ways will eventually disagree with itself about one.

Nothing here calls a provider. It reads bundles.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.analysis.identifiability import _z
from cortex.eval.fixtures import by_name
from cortex.eval.replay import load_bundle, replay_bundle
from cortex.eval.scorer import Scorer

__all__ = ["DimensionSpread", "Spread", "measure_spread", "render_spread"]

#: Confidence and power for the minimum detectable effect.
#:
#: 80% power rather than the 90% `minimum_separation` defaults to. That function decides whether
#: two candidate *causes* can be told apart, where a wrong answer is a false attribution; this one
#: sizes an eval comparison, where the cost of missing a small real improvement is another run.
ALPHA = 0.05
POWER = 0.80


@dataclass(frozen=True, slots=True)
class DimensionSpread:
    """One dimension's behaviour across repeated attempts."""

    name: str
    #: Mean over every attempt of every scenario.
    mean: float
    #: Pooled within-scenario standard deviation. The dice.
    sigma_attempt: float
    #: Standard deviation of the per-scenario means. Not noise -- scenario difficulty.
    sigma_scenario: float
    #: Scenarios that produced an identical score on every attempt.
    stable_scenarios: int
    scenarios: int
    attempts: int

    @property
    def paired_mde(self) -> float:
        """Smallest mean shift a paired before/after could detect, per scenario count."""
        if self.sigma_attempt == 0.0 or self.scenarios == 0:
            return 0.0
        coefficient = _z(1 - ALPHA / 2) + _z(POWER)
        return coefficient * self.sigma_attempt * (2 / self.scenarios) ** 0.5

    @property
    def deterministic(self) -> bool:
        """True when no scenario ever varied. A difference here is always real."""
        return self.sigma_attempt == 0.0


@dataclass(frozen=True, slots=True)
class Spread:
    dimensions: tuple[DimensionSpread, ...]
    scenarios: tuple[str, ...]
    attempts_per_scenario: dict[str, int]


async def measure_spread(session: AsyncSession, directory: Path) -> Spread:
    """Score every bundle in `directory` and describe each dimension's spread.

    **Scores are re-derived rather than stored, which is deliberate.** A bundle carries the
    report and the rows, not the scorecard, so every attempt here is scored by the *current*
    scorer. That is the comparable thing: three attempts scored by three different versions of a
    dimension would mix code changes into the noise estimate, and the whole point is to separate
    those.

    It costs nothing -- re-scoring is a local computation, and a bundle round-trips to a
    bit-identical scorecard, so the numbers are the ones those attempts would have got.
    """
    by_scenario: dict[str, list[dict[str, float]]] = defaultdict(list)
    scorer = Scorer()
    for path in sorted(directory.glob("*.json")):
        bundle = load_bundle(path)
        tenant, investigation_id, view, gate_result, verification = await replay_bundle(
            session, bundle
        )
        card = await scorer.score(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=by_name(bundle.scenario),
            investigation=view,
            gate_result=gate_result,
            verification=verification,
        )
        by_scenario[bundle.scenario].append({d.name: d.score for d in card.dimensions})

    names: list[str] = []
    for attempts in by_scenario.values():
        for scores in attempts:
            for name in scores:
                if name not in names:
                    names.append(name)

    dimensions: list[DimensionSpread] = []
    for name in names:
        per_scenario: list[list[float]] = [
            [scores[name] for scores in attempts if name in scores]
            for attempts in by_scenario.values()
        ]
        present = [values for values in per_scenario if values]
        if not present:
            continue
        flat = [value for values in present for value in values]
        # Pooled within-scenario variance: the average of each scenario's own variance, so a
        # scenario that never varies contributes zero rather than being dropped.
        variances = [statistics.pvariance(values) for values in present if len(values) > 1]
        sigma_attempt = (sum(variances) / len(variances)) ** 0.5 if variances else 0.0
        means = [statistics.fmean(values) for values in present]
        dimensions.append(
            DimensionSpread(
                name=name,
                mean=statistics.fmean(flat),
                sigma_attempt=sigma_attempt,
                sigma_scenario=statistics.pstdev(means) if len(means) > 1 else 0.0,
                stable_scenarios=sum(1 for values in present if len(set(values)) == 1),
                scenarios=len(present),
                attempts=len(flat),
            )
        )

    return Spread(
        dimensions=tuple(dimensions),
        scenarios=tuple(by_scenario),
        attempts_per_scenario={name: len(a) for name, a in by_scenario.items()},
    )


def render_spread(spread: Spread) -> str:
    """A table, and the sentence a reader needs from it."""
    if not spread.dimensions:
        return (
            "No bundles found. Capture a run with --repeat and --capture-to first; a single "
            "attempt per scenario has no spread to measure.\n"
        )

    counts = sorted(set(spread.attempts_per_scenario.values()))
    lines = [
        "Score spread across repeated attempts",
        "=" * 72,
        f"{len(spread.scenarios)} scenario(s), {'/'.join(str(c) for c in counts)} attempt(s) each",
        "",
        f"{'dimension':<18}{'mean':>7}{'sig_att':>9}{'sig_scen':>10}{'stable':>8}{'MDE':>8}",
        "-" * 72,
    ]
    for dimension in sorted(spread.dimensions, key=lambda d: -d.sigma_attempt):
        lines.append(
            f"{dimension.name:<18}{dimension.mean:>7.3f}{dimension.sigma_attempt:>9.3f}"
            f"{dimension.sigma_scenario:>10.3f}"
            f"{dimension.stable_scenarios:>5}/{dimension.scenarios:<2}"
            f"{dimension.paired_mde:>8.3f}"
        )

    noisy = [d for d in spread.dimensions if not d.deterministic]
    lines += ["", "-" * 72]
    if noisy:
        worst = max(noisy, key=lambda d: d.paired_mde)
        lines.append(
            f"Noisiest: {worst.name}. A paired before/after over "
            f"{worst.scenarios} scenario(s) cannot detect a mean shift smaller than "
            f"{worst.paired_mde:.3f} on it. Anything smaller is the dice."
        )
    deterministic = [d.name for d in spread.dimensions if d.deterministic]
    if deterministic:
        lines.append(
            f"Deterministic across every attempt: {', '.join(sorted(deterministic))}. "
            "A difference on these is always real."
        )
    lines.append(
        "sig_scen is scenario difficulty, not noise. It is why a comparison must be paired -- "
        "the same scenarios both sides -- and why an unpaired one mostly measures which "
        "scenarios were chosen."
    )
    return "\n".join(lines) + "\n"
