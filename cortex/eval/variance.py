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

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.analysis.identifiability import _z
from cortex.eval.fixtures import by_name
from cortex.eval.replay import load_bundle, replay_bundle
from cortex.eval.scorer import Scorer

__all__ = [
    "DimensionSpread",
    "Spread",
    "TrajectorySpread",
    "measure_spread",
    "render_spread",
]

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
    def attempts_per_scenario(self) -> float:
        """Mean attempts behind each scenario's mean. The `K` the detection floor divides by."""
        return self.attempts / self.scenarios if self.scenarios else 0.0

    @property
    def paired_mde(self) -> float:
        """Smallest mean shift a paired before/after could detect at this attempt count.

        **This was wrong, in a way that governed every claim made from it.** The formula read
        `2.80 * sigma_attempt * sqrt(2 / scenarios)`, which is the *unpaired two-arm* variance
        `2·sigma^2` with the attempt count silently fixed at one. Two consequences, both
        material:

          - It reported 0.262 for a suite measured at five attempts, when the correct figure
            for five attempts is 0.117. Improvements between 12 and 26 points were dismissed
            as noise on the strength of it.
          - It made "more attempts buy no power" look like a property of the design. That is
            an artifact of `K = 1`: the attempt-level variance component divides by `K`, so
            reaching a 10-point floor needs about **seven attempts on the eight scenarios that
            already exist**, not the fifty-five scenarios previously calculated.

        The reference decomposition is Evan Miller, *Adding Error Bars to Evals*
        (arXiv:2411.00640, 2024): `Var(mean) = (Var(x) + E[sigma_i^2]) / n`, where `Var(x)` is
        between-question difficulty -- irreducible by resampling -- and `E[sigma_i^2]` is
        within-question sampling noise, which `K` samples per question divide.

        **Pairing is what removes the difficulty term.** Run the same scenarios on both sides
        and `Var(x)` cancels, leaving `2 * sigma_attempt^2 / K` per scenario. That is the
        formula below. It assumes the two sides have equal attempt-level variance and are
        independent given the scenario, which is the right model for re-running one suite and
        the wrong one for comparing two different suites -- `sigma_scenario` is reported beside
        this precisely so an unpaired comparison cannot borrow this number.

        Miller also bounds what `K` can buy: for uniformly distributed binary difficulty the
        variance ratio is `(1 + 2/K) / 3`, so 33% reduction at `K = 2` against a 67%
        asymptotic ceiling, with returns thinning by `K` of four to six. Attempts are not free
        power, but they are not zero power either.
        """
        attempts = self.attempts_per_scenario
        if self.sigma_attempt == 0.0 or self.scenarios == 0 or attempts <= 0:
            return 0.0
        coefficient = _z(1 - ALPHA / 2) + _z(POWER)
        return coefficient * (2 * self.sigma_attempt**2 / attempts / self.scenarios) ** 0.5

    @property
    def attempts_for(self) -> Callable[[float], int]:
        """How many attempts per scenario a target detection floor needs, at this noise.

        Inverting the formula above. Reported because the question a reader has after seeing a
        floor is always "what would it take to halve it", and the answer -- attempts, not
        scenarios -- is the one the old formula could not give.
        """

        def _needed(target: float) -> int:
            if target <= 0 or self.sigma_attempt == 0.0 or self.scenarios == 0:
                return 0
            coefficient = _z(1 - ALPHA / 2) + _z(POWER)
            return math.ceil(
                2 * (coefficient * self.sigma_attempt) ** 2 / (self.scenarios * target**2)
            )

        return _needed

    @property
    def icc(self) -> float:
        """Share of variance that is scenario difficulty rather than run-to-run noise.

        `sigma_scenario^2 / (sigma_scenario^2 + sigma_attempt^2)`, the intraclass correlation
        the agent-eval literature reports (arXiv:2512.06710 measures 0.30-0.77 across GAIA and
        FRAMES). It says which lever binds: near 1, the scenarios differ and more attempts buy
        little; near 0, the same scenario answers differently each time and attempts are the
        cheaper fix than more scenarios.
        """
        total = self.sigma_scenario**2 + self.sigma_attempt**2
        return round(self.sigma_scenario**2 / total, 4) if total else 0.0

    @property
    def deterministic(self) -> bool:
        """True when no scenario ever varied. A difference here is always real."""
        return self.sigma_attempt == 0.0


#: The layer of consistency that predicts whether the answer is right.
#:
#: **Read from the literature rather than guessed at.** *How Consistent Are LLM Agents?
#: Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines* (arXiv 2605.28840)
#: measures three layers and finds only one of them matters:
#:
#:   - **Tool Sequence Similarity** -- did the agent call the same tools in the same order.
#:     Attempts in their high-TSS condition were **90.2%** correct against **61.2%** for
#:     low-TSS (Cohen's d = 0.81). Structural variance is where failures concentrate.
#:   - **Argument Consistency** -- did it pass the same parameters. *No* predictive power
#:     (r = 0.12, not significant). Parametric variance is benign.
#:   - **Output agreement** -- did it produce the same words. Under **5%** exact match even
#:     when tool sequences were identical, and uncorrelated with failure.
#:
#: That last figure is the one worth holding onto: an agent that words its answer differently
#: every time is behaving normally, and a project that treats prose variation as evidence of a
#: broken system will spend its effort in the wrong place. The question worth asking is whether
#: the *trajectory* was stable, and whether the *conclusion* was right.
#:
#: They also find divergence concentrates early -- 60% of it originates in the first two steps
#: -- which is why the survey prefix is excluded below and the first decision is reported
#: separately.
_SURVEY_CAPABILITIES = ("list_repositories", "list_events", "list_projects", "list_queries")


def _trajectory(bundle_calls: list[dict[str, Any]]) -> list[str]:
    """The tool calls this attempt *chose*, as `tool__capability` names.

    The survey prefix is dropped. Every investigation opens by calling each discovery
    capability, which is the loop's own behaviour rather than a decision the model made, so
    counting it inflates every similarity score by the same constant and hides the thing being
    measured.
    """
    names = [
        f"{call.get('tool_name')}__{call.get('capability')}"
        for call in bundle_calls
        if call.get("capability")
    ]
    index = 0
    while index < len(names) and names[index].split("__", 1)[-1] in _SURVEY_CAPABILITIES:
        index += 1
    return names[index:]


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if not left:
        return len(right)
    previous = list(range(len(right) + 1))
    for i, item in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item != other),
                )
            )
        previous = current
    return previous[-1]


def _sequence_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """1.0 for identical sequences, 0.0 for entirely different ones."""
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return 1.0 - _levenshtein(left, right) / longest


def _argument_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Jaccard over flattened key-value pairs, as the paper defines it."""

    def _pairs(params: dict[str, Any]) -> set[str]:
        return {f"{key}={value!r}" for key, value in sorted(params.items())}

    first, second = _pairs(left), _pairs(right)
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


@dataclass(frozen=True, slots=True)
class TrajectorySpread:
    """How much one scenario's *route* varied, beside whether its answer was right.

    Reported together on purpose. The two numbers answer different questions and the pair is
    what makes either actionable: a scenario that is always right by a different route needs
    nothing done about it, and one that is sometimes wrong by a different route has a cause to
    look for in its first two steps.
    """

    scenario: str
    attempts: int
    #: Mean pairwise Tool Sequence Similarity across attempts. 1.0 = the same route every time.
    tool_sequence_similarity: float
    #: Mean pairwise Argument Consistency over the calls the attempts had in common.
    argument_consistency: float
    #: How many *distinct* routes the attempts took. The paper's own headline unit -- it
    #: reports 2.0-4.2 distinct action sequences per 10 runs for ReAct agents on HotpotQA.
    distinct_routes: int
    #: Share of attempts that got the scenario's gating `accuracy` dimension right.
    accuracy_rate: float
    #: Whether every attempt agreed with every other about the answer, right or wrong.
    unanimous: bool


@dataclass(frozen=True, slots=True)
class Spread:
    dimensions: tuple[DimensionSpread, ...]
    scenarios: tuple[str, ...]
    attempts_per_scenario: dict[str, int]
    trajectories: tuple[TrajectorySpread, ...] = ()

    @property
    def route_accuracy_split(self) -> tuple[float, float] | None:
        """Mean accuracy of the scenarios that took one route, against those that took several.

        The replication of arXiv 2605.28840's central finding on this project's own data: they
        report 90.2% correct for high tool-sequence similarity against 61.2% for low, and if
        that holds here then the lever for accuracy is route stability rather than anything in
        the drafting. Returns None until both groups have a member.
        """
        single = [t.accuracy_rate for t in self.trajectories if t.distinct_routes == 1]
        several = [t.accuracy_rate for t in self.trajectories if t.distinct_routes > 1]
        if not single or not several:
            return None
        return statistics.fmean(single), statistics.fmean(several)


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
    routes: dict[str, list[list[str]]] = defaultdict(list)
    arguments: dict[str, list[list[tuple[str, dict[str, Any]]]]] = defaultdict(list)
    scorer = Scorer()
    for path in sorted(directory.glob("*.json")):
        bundle = load_bundle(path)
        # `sufficiency` is unpacked and passed on, and leaving it out is why this module went
        # unrun: `replay_bundle` grew a sixth return value when the sufficiency gate landed and
        # this call still expected five, so every invocation of `--variance` since then has
        # died on a tuple unpack. The instrument existed and produced no numbers.
        (
            tenant,
            investigation_id,
            view,
            gate_result,
            verification,
            applied,
        ) = await replay_bundle(session, bundle)
        card = await scorer.score(
            session,
            tenant,
            investigation_id=investigation_id,
            scenario=by_name(bundle.scenario),
            investigation=view,
            gate_result=gate_result,
            verification=verification,
            sufficiency=applied,
        )
        by_scenario[bundle.scenario].append({d.name: d.score for d in card.dimensions})
        routes[bundle.scenario].append(_trajectory(bundle.tool_calls))
        arguments[bundle.scenario].append(
            [
                (f"{c.get('tool_name')}__{c.get('capability')}", c.get("params") or {})
                for c in bundle.tool_calls
            ]
        )

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

    trajectories: list[TrajectorySpread] = []
    for scenario, attempts in routes.items():
        accuracies = [scores.get("accuracy") for scores in by_scenario[scenario]]
        scored = [value for value in accuracies if value is not None]
        pairs = [
            (attempts[i], attempts[j])
            for i in range(len(attempts))
            for j in range(i + 1, len(attempts))
        ]
        calls = arguments[scenario]
        argument_pairs = [
            _argument_similarity(dict(left_params), dict(right_params))
            for i in range(len(calls))
            for j in range(i + 1, len(calls))
            for (left_name, left_params) in calls[i]
            for (right_name, right_params) in calls[j]
            if left_name == right_name
        ]
        trajectories.append(
            TrajectorySpread(
                scenario=scenario,
                attempts=len(attempts),
                tool_sequence_similarity=(
                    round(statistics.fmean(_sequence_similarity(a, b) for a, b in pairs), 4)
                    if pairs
                    else 1.0
                ),
                argument_consistency=(
                    round(statistics.fmean(argument_pairs), 4) if argument_pairs else 1.0
                ),
                distinct_routes=len({tuple(route) for route in attempts}),
                accuracy_rate=round(statistics.fmean(scored), 4) if scored else 0.0,
                unanimous=len(set(scored)) <= 1,
            )
        )

    return Spread(
        dimensions=tuple(dimensions),
        scenarios=tuple(by_scenario),
        attempts_per_scenario={name: len(a) for name, a in by_scenario.items()},
        trajectories=tuple(trajectories),
    )


def _render_trajectories(spread: Spread) -> list[str]:
    """The route table, and the one sentence that decides where to spend effort.

    Separate from the score table because it answers a different question. The scores say how
    large a difference has to be before it means something; this says whether the *route* was
    stable and whether the answer was right -- and arXiv 2605.28840 finds that only the first
    of those predicts the second.
    """
    if not spread.trajectories:
        return []
    lines = [
        "",
        "Route stability across the same attempts",
        "=" * 72,
        f"{'scenario':<36}{'TSS':>7}{'AC':>7}{'routes':>8}{'right':>8}",
        "-" * 72,
    ]
    for trajectory in sorted(spread.trajectories, key=lambda t: t.tool_sequence_similarity):
        lines.append(
            f"{trajectory.scenario:<36}{trajectory.tool_sequence_similarity:>7.2f}"
            f"{trajectory.argument_consistency:>7.2f}{trajectory.distinct_routes:>8}"
            f"{trajectory.accuracy_rate:>8.2f}"
        )
    lines += ["-" * 72]

    divided = [t for t in spread.trajectories if not t.unanimous]
    if divided:
        lines.append(
            "Disagreed with itself about the answer: "
            + ", ".join(f"{t.scenario} ({t.accuracy_rate:.0%} right)" for t in divided)
            + ". This is the number that matters -- attempts wording one answer differently "
            "are behaving normally, attempts reaching different answers are not."
        )
    else:
        lines.append(
            "Every scenario reached the same answer on every attempt. Route and wording still "
            "vary, and neither is a defect on its own."
        )

    split = spread.route_accuracy_split
    if split is not None:
        single, several = split
        lines.append(
            f"One route: {single:.0%} right. Several routes: {several:.0%}. "
            "arXiv 2605.28840 reports 90% against 61% for this split and finds argument "
            "variance carries no signal at all, so route stability is where an intervention "
            "belongs -- and 60% of route divergence originates in the first two steps."
        )
    return lines


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
        f"{'dimension':<18}{'mean':>7}{'sig_att':>9}{'sig_scen':>10}{'ICC':>7}"
        f"{'stable':>8}{'MDE':>8}",
        "-" * 72,
    ]
    for dimension in sorted(spread.dimensions, key=lambda d: -d.sigma_attempt):
        lines.append(
            f"{dimension.name:<18}{dimension.mean:>7.3f}{dimension.sigma_attempt:>9.3f}"
            f"{dimension.sigma_scenario:>10.3f}{dimension.icc:>7.2f}"
            f"{dimension.stable_scenarios:>5}/{dimension.scenarios:<2}"
            f"{dimension.paired_mde:>8.3f}"
        )

    noisy = [d for d in spread.dimensions if not d.deterministic]
    lines += ["", "-" * 72]
    if noisy:
        worst = max(noisy, key=lambda d: d.paired_mde)
        lines.append(
            f"Noisiest: {worst.name}. A paired before/after over {worst.scenarios} "
            f"scenario(s) at {worst.attempts_per_scenario:.0f} attempt(s) each cannot detect a "
            f"mean shift smaller than {worst.paired_mde:.3f} on it. Anything smaller is the "
            "dice."
        )
        # The question a reader always has next, and the one the old formula could not answer
        # because it had fixed the attempt count at one.
        for target in (0.10, 0.05):
            if target < worst.paired_mde:
                lines.append(
                    f"  To reach {target:.2f} on it: {worst.attempts_for(target)} attempt(s) "
                    f"per scenario on these same {worst.scenarios}, no new scenarios needed."
                )
                break
        lines.append(
            "MDE divides by the attempt count, so attempts buy power -- Miller "
            "(arXiv:2411.00640) bounds the gain at 33% variance reduction by K=2 against a 67% "
            "ceiling, thinning by K of four to six. ICC says which lever binds: high is "
            "scenario difficulty, low is run-to-run noise."
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
    lines += _render_trajectories(spread)
    return "\n".join(lines) + "\n"
