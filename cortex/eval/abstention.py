"""Choose, once and offline, which reports are safe to deliver without a human.

**The decision this fits.** Every report carries a signal of how shaky it is -- the verifier
already judges each claim `supported`, `overstated` or `unsupported`, and records the ones it
could not judge at all. That is a number per report. The question is where to cut it: below
the cut a report goes out, above it a human looks first. Picking that cut by eye is how a
threshold ends up tuned to whichever examples were on screen.

**What is guaranteed, and what is not.** Split conformal gives a finite-sample bound on the
*selective risk* -- the error rate among reports that were delivered -- and the bound holds
for any scoring function, however bad. A poor score does not break the guarantee; it makes it
expensive, because holding the delivered-error rate down then means delivering less. So the
number to read alongside the threshold is how much still gets delivered. A threshold that
keeps 4% of reports is valid and useless.

**Which errors are controlled.** Only the ones that mislead a reader: going along with a
premise the data contradicts, answering a window the data never covered, refuting a movement
that did happen. An unnecessary escalation costs a human a minute and is reported separately
as the price, never folded into the risk -- a bound that counted both would buy one down by
selling the other up. `holds` against `none_asserted` on a question that asserts nothing is
not an error at all; see `premise_matrix.BENIGN`.

The bound is exact rather than asymptotic, and dependency-free: at the sample sizes a
calibration run produces, a normal approximation is doing arithmetic it has not earned.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import exp, lgamma, log, log1p

from cortex.eval.premise_matrix import VERDICTS, expected_of, is_benign

#: Confidence for the upper bound on delivered-error rate. The bound holds with probability
#: `1 - DELTA` over the draw of the calibration set.
DELTA = 0.05


@dataclass(frozen=True, slots=True)
class Point:
    """One calibration example: how shaky the report looked, and whether it was wrong."""

    scenario: str
    score: float
    expected: str
    stated: str | None
    #: Verdicts the evidence also supports here, which must not count against the bound.
    also_acceptable: tuple[str, ...] = ()

    @property
    def delivered(self) -> bool:
        """Whether a report reached the point of stating anything at all."""
        return self.stated is not None

    @property
    def misleading(self) -> bool:
        """Wrong in a way a reader would be misled by.

        An undelivered report is not misleading -- nobody read it. Over-abstention is not
        misleading either: `stated == "unverifiable"` declines to answer, which costs an
        answer and tells no untruth.
        """
        if self.stated is None or self.stated == "unverifiable":
            return False
        if self.stated in self.also_acceptable:
            # Labelled one way, answered another, and the evidence carries both. Counting it
            # as an error would spend the threshold's budget buying down a distinction the
            # data does not make.
            return False
        return self.stated != self.expected and not is_benign(self.expected, self.stated)


def score_of(bundle: dict) -> float:
    """How shaky one captured report looks, in [0, 1]. Higher is shakier.

    Built from what the verifier already produced, so it costs no extra call: the share of
    claims it would not certify, counting the ones it could not judge against the report
    rather than for it. A draft the gate refused outright scores 1.0 -- the system has
    already said it does not stand behind it.

    This is a first score, not a good one. The conformal bound holds whatever it is, so a
    better score can replace it later and be compared on the one number that matters: how
    much gets delivered at the same risk.
    """
    if bundle.get("rejected_because"):
        return 1.0
    verdicts = bundle.get("verdicts") or []
    unverified = bundle.get("unverified") or []
    total = len(verdicts) + len(unverified)
    if total == 0:
        # Nothing was checked, so nothing supports delivering it unreviewed.
        return 1.0
    doubted = sum(1 for v in verdicts if v.get("verdict") != "supported")
    return (doubted + len(unverified)) / total


def _log_choose(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(k + 1):
        total += exp(_log_choose(n, i) + i * log(p) + (n - i) * log1p(-p))
    return min(1.0, total)


def risk_upper_bound(errors: int, n: int, delta: float = DELTA) -> float:
    """Clopper-Pearson upper limit on the error rate, exact for this n.

    With no observations at all the honest bound is 1.0: zero errors out of zero is not
    evidence of safety, and returning 0.0 there would let a threshold that delivers nothing
    look perfect.
    """
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    low, high = errors / n, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        # The largest p whose lower tail still holds `delta` of its mass at or below `errors`.
        if binomial_cdf(errors, n, mid) > delta:
            low = mid
        else:
            high = mid
    return high


@dataclass(frozen=True, slots=True)
class Threshold:
    """A fitted cut, and what it costs."""

    #: Deliver a report when its score is at or below this.
    cut: float
    alpha: float
    delivered: int
    #: Reports the cut sends to a human that would have been fine.
    escalated_unnecessarily: int
    misleading_delivered: int
    risk_bound: float
    total: int

    @property
    def coverage(self) -> float:
        return self.delivered / self.total if self.total else 0.0

    @property
    def feasible(self) -> bool:
        """Whether any report can be delivered at this risk level."""
        return self.delivered > 0


def calibrate(points: Iterable[Point], alpha: float = 0.10, delta: float = DELTA) -> Threshold:
    """The most permissive cut whose delivered-error rate is provably at or below `alpha`.

    Searched over the observed scores rather than a grid: a cut between two observed values
    delivers exactly the same set as the lower of them, so the grid adds candidates that
    cannot differ and hides the ones that can.

    Ties matter and are handled by cutting at `<=`. Scores here are ratios over small
    denominators, so many reports share a value -- a strict cut would deliver a different
    set depending on floating-point equality.
    """
    examples = [p for p in points if p.delivered]
    if not examples:
        return Threshold(0.0, alpha, 0, 0, 0, 1.0, 0)

    best: Threshold | None = None
    for cut in sorted({p.score for p in examples}):
        kept = [p for p in examples if p.score <= cut]
        errors = sum(1 for p in kept if p.misleading)
        bound = risk_upper_bound(errors, len(kept), delta)
        if bound > alpha:
            continue
        candidate = Threshold(
            cut=cut,
            alpha=alpha,
            delivered=len(kept),
            escalated_unnecessarily=sum(1 for p in examples if p.score > cut and not p.misleading),
            misleading_delivered=errors,
            risk_bound=bound,
            total=len(examples),
        )
        # Most permissive wins: every feasible cut satisfies the bound, so the useful one is
        # whichever delivers most.
        if best is None or candidate.delivered > best.delivered:
            best = candidate
    if best is not None:
        return best
    # No cut is feasible. Reported rather than raised, with a cut that delivers nothing, so a
    # caller sees "this cannot be done at this alpha on this much data" instead of an error.
    return Threshold(
        cut=-1.0,
        alpha=alpha,
        delivered=0,
        escalated_unnecessarily=sum(1 for p in examples if not p.misleading),
        misleading_delivered=0,
        risk_bound=risk_upper_bound(0, 0, delta),
        total=len(examples),
    )


def points_from(
    bundles: Iterable[dict],
    labels: dict[str, str | dict],
    score: Callable[[dict], float] = score_of,
) -> tuple[Point, ...]:
    """Pair captured bundles with their computed labels."""
    out: list[Point] = []
    for bundle in bundles:
        name = bundle.get("scenario")
        if name not in labels:
            continue
        expected, acceptable = expected_of(labels[name])
        out.append(
            Point(
                scenario=name,
                score=score(bundle),
                expected=expected,
                also_acceptable=acceptable,
                stated=stated_premise(bundle),
            )
        )
    return tuple(out)


#: Files a capture holds beside its bundles. Skipped by name rather than by shape, so a
#: malformed bundle still raises instead of being quietly treated as a sidecar.
SIDECARS = frozenset({"premise-labels.json", "generated-run.json"})


def load_bundles(directory) -> tuple[dict, ...]:
    from pathlib import Path

    return tuple(
        json.loads(path.read_text())
        for path in sorted(Path(directory).glob("*.json"))
        if path.name not in SIDECARS
    )


def render(threshold: Threshold, points: tuple[Point, ...]) -> str:
    """The cut, the guarantee, and the price -- which must be read together."""
    lines = ["", "Abstention threshold (split conformal)", ""]
    if not threshold.feasible:
        lines += [
            f"  No cut delivers anything at alpha={threshold.alpha:.0%}.",
            f"  {threshold.total} delivered report(s) is too few to certify any error rate"
            f" this low, or the score does not separate the errors.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        f"  deliver when score <= {threshold.cut:.3f}",
        f"  delivered            {threshold.delivered}/{threshold.total}"
        f" ({threshold.coverage:.0%})",
        f"  misleading delivered {threshold.misleading_delivered}",
        f"  error rate bound     {threshold.risk_bound:.1%}"
        f" (<= alpha {threshold.alpha:.0%}, at {1 - DELTA:.0%} confidence)",
        f"  escalated for review {threshold.total - threshold.delivered}, of which"
        f" {threshold.escalated_unnecessarily} would have been fine",
        "",
    ]
    undelivered = [p for p in points if not p.delivered]
    if undelivered:
        lines.append(
            f"  {len(undelivered)} attempt(s) produced no report and are outside this bound."
        )
        lines.append("")
    return "\n".join(lines)


def stated_premise(bundle: dict) -> str | None:
    """The premise verdict a captured bundle's report carries, or None."""
    stated = (bundle.get("report") or {}).get("premise")
    return stated if stated in VERDICTS else None


def instability_of(bundles: Sequence[dict]) -> float:
    """Share of attempts that disagree with the most common premise verdict.

    **Why this score and not the verifier's.** The measured failure is not an unsupported
    claim, it is the same evidence yielding a different verdict on a re-run: one attempt at a
    case said `false` and the next said `unverifiable`, both describing the
    zero rows they saw in the same words. `score_of` cannot see that -- every claim in both
    reports is supported, and on the pilot it ranked both wrong reports as *safer* than both
    right ones. This measures the re-roll directly, because the re-roll is the defect.

    The price is real: it needs K attempts per case where `score_of` needs one. That is the
    trade to make deliberately, which is why both live here and the caller picks.

    0.0 is unanimity. With K attempts split evenly across two verdicts it approaches 0.5.
    """
    stated = [stated_premise(b) for b in bundles]
    answered = [s for s in stated if s is not None]
    if not answered:
        return 1.0
    agreed = Counter(answered).most_common(1)[0][1]
    # Undelivered attempts count against agreement rather than being dropped: a case that
    # only answered twice in seven is not as settled as one that answered twice in two.
    return 1.0 - agreed / len(stated)


def majority_premise(bundles: Sequence[dict]) -> str | None:
    """The verdict most attempts reached, or None if none delivered.

    Ties are broken toward the more cautious verdict. A case that says `false` twice and
    `unverifiable` twice has not established a contradiction, and picking the confident half
    of a coin flip is how an unstable verdict becomes a delivered falsehood.
    """
    answered = [s for s in (stated_premise(b) for b in bundles) if s]
    if not answered:
        return None
    counts = Counter(answered)
    best = max(counts.values())
    tied = {verdict for verdict, n in counts.items() if n == best}
    for cautious in ("unverifiable", "none_asserted", "false", "holds"):
        if cautious in tied:
            return cautious
    return next(iter(tied))


def points_from_repeats(bundles: Iterable[dict], labels: dict[str, str]) -> tuple[Point, ...]:
    """One point per case, scored by how much its attempts disagreed.

    Attempts are grouped by scenario, so a run made with `--repeat` yields a calibration set
    whose score is instability and whose answer is the majority verdict -- which is itself
    the intervention, not only the measurement: taking the modal answer of K attempts is
    self-consistency decoding, and it is what the threshold would be protecting.
    """
    grouped: dict[str, list[dict]] = {}
    for bundle in bundles:
        name = bundle.get("scenario")
        if name in labels:
            grouped.setdefault(name, []).append(bundle)
    points = []
    for name, attempts in sorted(grouped.items()):
        expected, acceptable = expected_of(labels[name])
        points.append(
            Point(
                scenario=name,
                score=instability_of(attempts),
                expected=expected,
                also_acceptable=acceptable,
                stated=majority_premise(attempts),
            )
        )
    return tuple(points)
