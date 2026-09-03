"""Whether a causal claim is available at all, decided before one is attempted.

Eight predicates, G0 to G7, derived from the assumptions a single-unit attribution actually
rests on. Two of them (G2a, G4) run on **shapes alone**, with no data values, which makes them the
cheapest and sharpest checks we have.

**The composition rule is worst-domain, following ROBINS-I: any single refusal refuses the whole
causal claim. Confidence does not average.** A gate that averaged would let six easy passes
outvote the one check that says the question is unanswerable.

**And the one-sidedness, which is the easiest thing to forget:** G0-G7 all passing does *not*
mean the change caused the shift. It means we found no reason it cannot be established. The
output for a fully-passing gate is "consistent with", never "caused by", and no computable
statistic can upgrade that.

Three of these require a human and must not be faked, because faking them is how a system
manufactures a causal claim out of arithmetic:

- **G1** needs a named intervention. A causal question with no specified intervention is not a
  question, and the system must never invent one by picking its own largest changepoint and then
  testing that same changepoint -- that is in-sample selection.
- **G3** needs someone to attest a control series was not exposed to the change. Correlation
  alone selects contaminated controls *preferentially*, because a series that moved because of
  our change correlates beautifully.
- **G7** needs a confounder ledger. An empty ledger is a claim -- "nothing else happened" -- and
  not a default. `None` (nobody looked) and `[]` (somebody looked and found nothing) must produce
  different output, so they are different values here.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from datetime import date

from cortex.analysis.series import Calendar

__all__ = [
    "Control",
    "GateCheck",
    "Identifiability",
    "Intervention",
    "assess_identifiability",
    "minimum_separation",
]

#: Default significance level.
DEFAULT_ALPHA = 0.05

#: Default power for the prospective separation calculation.
#:
#: **0.90, not the more usual 0.80, and the reason is a discrepancy found while implementing
#: this.** The research's separation table was validated by simulation (400 trials per cell,
#: 0.94-0.99 accuracy at `d_min`), and only power = 0.90 reproduces it. Solving each row for the
#: constant it implies gives intervals of (3.230, 3.287] and (3.217, 3.374]; `z(.975) + z(.90) =
#: 3.2415` lands inside every one, and `z(.975) + z(.80) = 2.8016` lands inside none of the tight
#: ones. At 0.80 this function returns 8 days where the simulation measured 11.
#:
#: The prose formula in section 5.5 reads `z_{1-power}`, which for power = 0.80 is `z_{0.20} =
#: -0.84` -- negative, and would *shrink* the required separation. That is a typo for `z_{power}`
#: (equivalently `z_{1-beta}`, the standard two-sample form), and the table is the authority.
DEFAULT_POWER = 0.90

#: Minimum pre-period correlation for a series to be *considered* as a control. Necessary and
#: nowhere near sufficient -- see `Control.admissible`.
DEFAULT_RHO_MIN = 0.5


class Gate(enum.StrEnum):
    """The eight checks, named as the research names them so the two can be read together."""

    COMPLETENESS = "G0"
    INTERVENTION = "G1"
    SEPARABILITY = "G2"
    CONTROLS = "G3"
    P_FLOOR = "G4"
    DETECTION_FLOOR = "G5"
    SENSITIVITY = "G6"
    CONFOUNDER_LEDGER = "G7"


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One predicate's verdict, and the thing that would lift it if it refused.

    `lifts_it` is not decoration. A refusal that does not say what is missing is the failure mode
    section 5.7 warns about -- not saying "I don't know" but saying it uninformatively -- and the
    alternative to an informative refusal is not silence, it is a human inventing a cause.
    """

    gate: Gate
    passed: bool
    detail: str
    lifts_it: str | None = None
    #: True when this check cannot fail, only report. G6 is the only one.
    advisory: bool = False


@dataclass(frozen=True, slots=True)
class Intervention:
    """A named change, at a known time, with a stated scope.

    Supplied by a caller. The system does not construct one for itself.
    """

    change_id: str
    at: date
    scope: str

    def __post_init__(self) -> None:
        if not self.change_id.strip():
            raise ValueError("an intervention needs an identifier")


@dataclass(frozen=True, slots=True)
class Control:
    """A candidate comparison series.

    `attested_unexposed` is a human's assertion, and there is no way to compute it. Brodersen et
    al. assume "a set control time series that were themselves not affected by the intervention";
    Abadie says units that adopted a similar intervention "should not be included in the donor
    pool". Neither is checkable from the numbers.
    """

    name: str
    #: Pre-period, seasonally-adjusted correlation with the treated series.
    correlation: float
    #: Whether a human has attested this series was not exposed to the change.
    attested_unexposed: bool

    def admissible(self, rho_min: float = DEFAULT_RHO_MIN) -> bool:
        return self.attested_unexposed and self.correlation >= rho_min


@dataclass(frozen=True, slots=True)
class Identifiability:
    """The gate's verdict on whether a causal claim is available."""

    checks: tuple[GateCheck, ...]
    #: The smallest p-value the available inference route could produce, or None when neither
    #: route is available.
    attainable_p: float | None = None
    #: Minimum detectable effect at the requested alpha, in units per day.
    minimum_detectable_effect: float | None = None
    #: The separation, in days, that the observed effect size would need between candidates.
    required_separation: int | None = None
    sensitivity_ratio: float | None = None

    @property
    def refusals(self) -> tuple[GateCheck, ...]:
        """Checks that failed and are not advisory. Worst-domain: any one of these decides."""
        return tuple(c for c in self.checks if not c.passed and not c.advisory)

    @property
    def causal_claim_available(self) -> bool:
        return not self.refusals

    def summary(self) -> str:
        """What to say. Never "caused by", even when everything passes."""
        if self.causal_claim_available:
            return (
                "No reason was found that a cause cannot be established here. That supports "
                "'consistent with', not 'caused by' -- passing these checks removes obstacles "
                "and establishes nothing on its own."
            )
        lines = ["A cause cannot be established from this data."]
        for check in self.refusals:
            lines.append(f"  {check.gate.value}: {check.detail}")
            if check.lifts_it:
                lines.append(f"      would be lifted by: {check.lifts_it}")
        return "\n".join(lines)


def _z(p: float) -> float:
    """Inverse standard normal CDF.

    Acklam's rational approximation, refined once by Halley's method. Accurate to well beyond
    what a threshold needs, and it keeps scipy out of the dependency list for the sake of two
    quantiles.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p <= high:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    # One Halley refinement, using the error function for the CDF.
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def minimum_separation(
    shift: float, sigma: float, *, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> int:
    """Days of separation two candidate causes need before they can be told apart.

    `d_min = ceil( 2 * ((z_{1-alpha/2} + z_{power}) * sigma / shift)^2 )`

    **The power term is not optional.** Verified by simulation in the research: the naive
    z-only form predicted 2 and 4 days where the empirical requirement was 5 and 11, so
    omitting it produces a gate that passes cases it should refuse. The same applies, less
    dramatically, to using 0.80 power instead of 0.90 -- see `DEFAULT_POWER`.

    Reproduces the validated table exactly at the default: shifts of 125/60/30/20/12 per day
    against noise scales near 14-19 require 1/2/5/11/29 days of separation.

    The encouraging half: at our own effect size -- a 115/day shift against a noise scale near
    15 -- a *single day* of separation suffices. Two deploys 24 hours apart around 2026-06-17
    are distinguishable.
    """
    if shift == 0 or sigma <= 0:
        return 1
    ratio = (_z(1 - alpha / 2) + _z(power)) * sigma / abs(shift)
    return max(1, math.ceil(2 * ratio * ratio))


def _rank(rows: list[list[float]]) -> int:
    """Rank by Gaussian elimination with partial pivoting."""
    if not rows:
        return 0
    matrix = [row[:] for row in rows]
    height, width = len(matrix), len(matrix[0])
    rank, pivot_row = 0, 0
    for column in range(width):
        pivot = None
        best = 1e-9
        for r in range(pivot_row, height):
            if abs(matrix[r][column]) > best:
                best, pivot = abs(matrix[r][column]), r
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        for r in range(height):
            if r != pivot_row and abs(matrix[r][column]) > 1e-12:
                factor = matrix[r][column] / matrix[pivot_row][column]
                for c in range(column, width):
                    matrix[r][c] -= factor * matrix[pivot_row][c]
        rank += 1
        pivot_row += 1
        if pivot_row == height:
            break
    return rank


def assess_identifiability(
    calendar: Calendar,
    *,
    intervention: Intervention | None,
    shift: float | None,
    sigma: float,
    days_before: int,
    days_after: int,
    candidates: tuple[date, ...] = (),
    controls: tuple[Control, ...] = (),
    ledger: tuple[str, ...] | None = None,
    placebo_effect: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    rho_min: float = DEFAULT_RHO_MIN,
) -> Identifiability:
    """Run G0-G7.

    `ledger=None` means nobody looked; `ledger=()` means somebody looked and found nothing.
    Those are different claims and produce different verdicts, which is why the parameter is
    optional-with-a-meaningful-None rather than defaulted to empty.
    """
    checks: list[GateCheck] = []

    # ---- G0. Completeness. First, because every later check silently produces a number on a
    # zero-filled array.
    freshness = calendar.freshness
    if freshness.complete:
        checks.append(GateCheck(Gate.COMPLETENESS, True, "every day in the range was measured"))
    else:
        checks.append(
            GateCheck(
                Gate.COMPLETENESS,
                False,
                freshness.note() or "the series has gaps",
                lifts_it="fix the loader, or narrow the window to the measured span",
            )
        )

    # ---- G1. A named intervention. The system must not invent one.
    if intervention is None:
        checks.append(
            GateCheck(
                Gate.INTERVENTION,
                False,
                "no change was named, so there is nothing to test for a causal effect. We can "
                "say when the metric moved, not why",
                lifts_it="a caller naming (change, date, scope)",
            )
        )
    else:
        checks.append(
            GateCheck(
                Gate.INTERVENTION,
                True,
                f"testing '{intervention.change_id}' on {intervention.at.isoformat()} "
                f"({intervention.scope})",
            )
        )

    # ---- G2. Separability. (a) is exact; (b) is a power calculation.
    required = None
    if len(candidates) > 1:
        design = _separability_design(calendar, candidates)
        if design and _rank(design) < len(design[0]):
            checks.append(
                GateCheck(
                    Gate.SEPARABILITY,
                    False,
                    "two or more candidate causes produce identical indicator columns, so the "
                    "design matrix is rank-deficient. This is not 'hard to separate' -- no "
                    "estimator distinguishes them, exactly",
                    lifts_it="a finer timestamp than the sampling period, or an experiment",
                )
            )
        else:
            closest = min(
                abs((a - b).days) for i, a in enumerate(candidates) for b in candidates[i + 1 :]
            )
            required = minimum_separation(shift or 0.0, sigma, alpha=alpha, power=power)
            if closest < required or required > max(days_after - 1, 0):
                checks.append(
                    GateCheck(
                        Gate.SEPARABILITY,
                        False,
                        f"candidates are {closest} day(s) apart; separating a shift this size "
                        f"needs {required}",
                        lifts_it="a larger contrast, more post-period days, or a finer timestamp",
                    )
                )
            else:
                checks.append(
                    GateCheck(
                        Gate.SEPARABILITY,
                        True,
                        f"{closest} day(s) apart against {required} required",
                    )
                )
    else:
        checks.append(
            GateCheck(Gate.SEPARABILITY, True, "one candidate, so nothing to separate it from")
        )

    # ---- G3. Control admissibility. The attestation is not automatable.
    admissible = tuple(c for c in controls if c.admissible(rho_min))
    unattested = tuple(c for c in controls if not c.attested_unexposed)
    if admissible:
        checks.append(
            GateCheck(
                Gate.CONTROLS,
                True,
                f"{len(admissible)} admissible control series",
            )
        )
    else:
        detail = "no admissible control series"
        if unattested:
            detail += (
                f" -- {len(unattested)} candidate(s) correlate but nobody has attested they "
                "were unexposed, and correlation alone selects contaminated controls "
                "preferentially, because a series that moved because of the change correlates "
                "beautifully"
            )
        checks.append(
            GateCheck(
                Gate.CONTROLS,
                False,
                detail,
                lifts_it=(
                    "one comparison series a human attests the change did not touch. Nineteen "
                    "would be needed for p <= 0.05 by the synthetic-control route"
                ),
            )
        )

    # ---- G4. Attainable p floor. Shapes only, no values.
    total = days_before + days_after
    conformal_floor = 1.0 / total if total else None
    synthetic_floor = 1.0 / (len(admissible) + 1)
    attainable = min(f for f in (conformal_floor, synthetic_floor) if f is not None)
    if attainable > alpha:
        checks.append(
            GateCheck(
                Gate.P_FLOOR,
                False,
                f"the smallest p either route could produce is {attainable:.4f}, above the "
                f"requested {alpha}. No significance claim is available at any effect size; "
                "report descriptively instead",
                lifts_it="a longer window (conformal floor is 1/T) or control series",
            )
        )
    else:
        checks.append(
            GateCheck(
                Gate.P_FLOOR,
                True,
                f"attainable p floor {attainable:.4f} against alpha {alpha}",
            )
        )

    # ---- G5. Is the change itself established? Ask that before asking about its cause.
    mde = None
    if sigma > 0 and days_before and days_after:
        mde = _z(1 - alpha / 2) * sigma * math.sqrt(1 / days_before + 1 / days_after)
        if shift is None or abs(shift) < mde:
            checks.append(
                GateCheck(
                    Gate.DETECTION_FLOOR,
                    False,
                    f"the movement is within noise (minimum detectable effect {mde:.1f}/day"
                    + (f" against an observed {abs(shift):.1f}/day" if shift else "")
                    + "). There may be nothing to explain",
                    lifts_it="more data, or a coarser period",
                )
            )
        else:
            checks.append(
                GateCheck(
                    Gate.DETECTION_FLOOR,
                    True,
                    f"observed {abs(shift):.1f}/day is {abs(shift) / mde:.1f}x the minimum "
                    f"detectable effect of {mde:.1f}/day",
                )
            )
    else:
        checks.append(
            GateCheck(
                Gate.DETECTION_FLOOR,
                False,
                "cannot compute a detection floor without a noise scale and both windows",
                lifts_it="a measured span long enough to estimate noise on",
            )
        )

    # ---- G6. Sensitivity. Always reported, never pass/fail.
    ratio = None
    if shift is not None and placebo_effect:
        ratio = abs(shift) / max(abs(placebo_effect), 1e-9)
        checks.append(
            GateCheck(
                Gate.SENSITIVITY,
                ratio >= 1.0,
                f"the effect is {ratio:.1f}x the largest in-time placebo "
                f"({abs(placebo_effect):.1f}/day)"
                + (
                    ". Below 1 means the conclusion does not survive a violation the size of "
                    "one already seen in the pre-period; report bounds, not a point"
                    if ratio < 1.0
                    else ""
                ),
                advisory=True,
            )
        )
    else:
        checks.append(
            GateCheck(
                Gate.SENSITIVITY,
                True,
                "no in-time placebo was run, so no sensitivity ratio is reported",
                advisory=True,
            )
        )

    # ---- G7. Confounder ledger. None and empty are different claims.
    if ledger is None:
        checks.append(
            GateCheck(
                Gate.CONFOUNDER_LEDGER,
                False,
                "no confounder ledger. Nobody has said what else happened in this window, and "
                "an empty ledger is a claim rather than a default",
                lifts_it="someone attesting which other changes landed in the window",
            )
        )
    elif len(ledger) > 1:
        checks.append(
            GateCheck(
                Gate.CONFOUNDER_LEDGER,
                False,
                f"{len(ledger)} other changes landed in this window: {', '.join(ledger)}. "
                "Report the set, ranked by nothing",
                lifts_it="separating them in time, or an experiment",
            )
        )
    else:
        checks.append(
            GateCheck(
                Gate.CONFOUNDER_LEDGER,
                True,
                "a human attests nothing else landed in this window"
                if not ledger
                else f"one other change in the window: {ledger[0]}",
            )
        )

    return Identifiability(
        checks=tuple(checks),
        attainable_p=attainable,
        minimum_detectable_effect=mde,
        required_separation=required,
        sensitivity_ratio=ratio,
    )


def _separability_design(calendar: Calendar, candidates: tuple[date, ...]) -> list[list[float]]:
    """`[ seasonal dummies | 1{t >= t_k} for each candidate ]` over the measured days."""
    days = [day for day in calendar.days if day.measured]
    if not days:
        return []
    rows: list[list[float]] = []
    for day in days:
        weekday = [1.0] + [0.0] * 6
        if day.on.weekday() < 6:
            weekday[day.on.weekday() + 1] = 1.0
        indicators = [1.0 if day.on >= at else 0.0 for at in candidates]
        rows.append(weekday + indicators)
    return rows
