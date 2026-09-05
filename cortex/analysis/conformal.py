"""Is that break real, or is it noise -- answered without a control series.

**This is the only significance statement available to us, and the reason is arithmetic.**
Abadie's placebo p-value for synthetic control is `p = (1/(J+1)) * sum I+(r_j - r_1)`. The
`j = 1` term is always 1, so `p >= 1/(J+1)`, and with `J = 0` donor series `p = 1` identically.
We have zero admissible control series, so **no synthetic-control claim is available at any
effect size, ever.** Getting to `p <= 0.05` needs `J = 19`: about twenty comparable,
independently-unaffected series, plus somebody willing to attest that the change did not touch
them. That is an infrastructure requirement, not a modelling one.

Nor can we fall back on standard errors. Alvarez, Ferman and Wuethrich (arXiv:2504.19841):
"it is not possible to consistently estimate the variances of eps_j,t and tau_j,t using only
information from the treated units." Asymptotic standard errors with one treated unit are not
conservative, they are meaningless.

The way out trades the cross-section for the time axis. Chernozhukov, Wuethrich and Zhu, "An
Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls," JASA
116(536):1849-1864, 2021 (arXiv:1712.09089). Their framing is the bridge this needed: "We
recast the causal inference problem as a counterfactual prediction and a structural breaks
testing problem." And section 2.4.1 covers our case explicitly: "If no control units are
available, one can use time series models for the single unit exposed to the intervention."

Two details are load-bearing and easy to get wrong:

- **Fit the proxy on all T periods, under the null** -- not on the pre-period alone.
  "Estimation under the null guarantees the exact finite sample validity of our procedures if
  the data are iid or exchangeable." Fitting on the pre-period only "does not yield procedures
  with exact finite sample validity, not even with iid data".
- **The p-value floor is `1/|Pi|`**, and with the T moving-block permutations that is `1/T`. A
  70-day window can reach `p = 0.0143` and no lower. Reporting the floor alongside the p-value
  is part of the claim, not a footnote -- a p at the floor means "as significant as this window
  can resolve", not "p = 0.014 exactly".

**A power cliff, found by building this rather than by reading the paper.** The test has no
power unless the post period is *shorter* than the pre period. Measured: power 1.000 at 70 pre
against 20 post, and 0.000 at 20 pre against 70 post, for the same 110/day shift -- while Type I
error stays at 1.7-3.3% against alpha = 0.05 throughout. So it is valid everywhere and useful
only on one side, because the intercept-plus-weekday proxy fits the weighted average of the two
regimes and the longer regime gets the smaller residuals. `ConformalResult.resolvable` reports
that as an inability rather than as a non-significant result, because "p = 1.0" from a window
that could never have rejected is worse than saying nothing.

What this does not do: it says a break is real. It says nothing whatever about what caused it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from cortex.analysis.changepoints import Segmentation
from cortex.analysis.series import Calendar

__all__ = ["ConformalResult", "significance_of", "significance_of_each"]

#: Exponent in the test statistic. q=1 is recommended for heavy tails, which count data has.
DEFAULT_Q = 1.0


@dataclass(frozen=True, slots=True)
class ConformalResult:
    """The outcome of testing one candidate break date."""

    at: date
    p_value: float
    #: `1/|Pi|`. The smallest p this window could have produced.
    p_floor: float
    statistic: float
    pre_days: int
    post_days: int
    total_days: int

    @property
    def resolvable(self) -> bool:
        """Whether this window could have rejected at all.

        **Measured, not assumed.** With the intercept-plus-weekday proxy the fitted level sits
        at the weighted average of the two regimes, so the *longer* regime pulls the mean
        towards it and gets the smaller residuals. The statistic is computed over the post
        window, so it is extreme only when the post period is the shorter one.

        Type I error is fine in every configuration -- 1.7% to 3.3% against alpha = 0.05 on
        pure noise -- so this is a power cliff, not an invalidity. But the power either side of
        it is 1.000 and 0.000 for the same 110/day shift, which makes a non-significant result
        from a long post window meaningless rather than informative: the test could not have
        rejected whatever the data did.

        So it is reported as an inability rather than as a negative, on the same principle as
        the attainable-p floor in `cortex.analysis.identifiability` (G4). The research behind
        this method validated only at 42 pre against 28 post and never exercised the other
        direction.
        """
        return self.post_days < self.pre_days

    @property
    def at_floor(self) -> bool:
        """Whether the p-value is the smallest the window can resolve.

        Distinguished because "p = 0.014" and "p = 0.014, which is the floor" support different
        sentences. The second one cannot be compared to a p from a longer window.
        """
        return abs(self.p_value - self.p_floor) < 1e-12

    def significant(self, alpha: float = 0.05) -> bool:
        """Significant only where significance was reachable. See `resolvable`."""
        return self.resolvable and self.p_value <= alpha

    def note(self) -> str:
        """One sentence, with the resolution limit disclosed."""
        if not self.resolvable:
            return (
                f"No significance can be established for the break at {self.at.isoformat()}: "
                f"the period after it ({self.post_days} days) is not shorter than the period "
                f"before it ({self.pre_days} days), and this test has no power in that "
                "configuration -- it could not have rejected whatever the data showed. This is "
                "an inability to test, not evidence the change is unreal. A longer run of "
                "comparable days before the break would resolve it."
            )
        # The floor is stated on every result, not only when p has landed on it. A p of 0.20
        # against a floor of 0.14 reads as "not significant" and is mostly a statement about the
        # window: no evidence, however strong, could have produced much less. Naming the smallest
        # attainable value is what separates "we looked and found nothing" from "this window
        # could not have found much", and it costs a clause.
        floor = (
            f", which is the floor for a {self.total_days}-day window"
            if self.at_floor
            else f" (the smallest attainable here was {self.p_floor:.4f})"
        )
        return (
            f"Permutation p = {self.p_value:.4f}{floor} for a break at "
            f"{self.at.isoformat()} ({self.pre_days} days before, {self.post_days} after). "
            "This establishes that the series changed, and says nothing about why."
        )


def significance_of(
    calendar: Calendar,
    at: date,
    *,
    q: float = DEFAULT_Q,
    seasonal: bool = True,
) -> ConformalResult | None:
    """Test whether the level change at `at` is distinguishable from noise.

    Returns None when the window cannot support a test at all -- no measured days on one side,
    or too few to fit the proxy.

    `at` is the first day of the post period, matching `Segmentation.breaks`.

    **The calendar must contain exactly one regime change: the one being tested.** This is not
    a stylistic preference and getting it wrong does not raise -- it silently weakens the test.
    A second break inside the post period lands in the residuals, inflates the statistic under
    every permutation as well as the observed one, and the p-value collapses towards 1. Handed
    our own two-break replica whole, this returns `p = 0.29` for a break that scores `p = 0.014`
    on its own window.

    Prefer `significance_of_each`, which derives the correct window from a `Segmentation` so the
    mistake is not available.
    """
    days = [day for day in calendar.days if day.measured and day.value is not None]
    if not days:
        return None

    pre = [day for day in days if day.on < at]
    post = [day for day in days if day.on >= at]
    if not pre or not post:
        return None

    total = len(pre) + len(post)
    # Intercept plus six day-of-week dummies needs a handful of spare observations to mean
    # anything. Below that the proxy fits the noise and the residuals go to zero.
    columns = 7 if seasonal else 1
    if total < columns + 2:
        return None

    values = [day.value for day in pre + post]
    design = [_row(day.on, seasonal) for day in pre + post]

    # Step 2: fit on ALL T periods, under the null. Under H0 the effect is zero, so the
    # observed series *is* the imputed one and no adjustment is needed before fitting.
    coefficients = _least_squares(design, values)
    if coefficients is None:
        return None
    residuals = [
        value - sum(c * x for c, x in zip(coefficients, row, strict=True))
        for value, row in zip(values, design, strict=True)
    ]

    post_start = len(pre)
    observed = _statistic(residuals, post_start, len(post), q)

    # Step 4: the T moving-block permutations, pi_j(i) = i + j mod T. j=0 is the identity, which
    # is what puts the floor at 1/T.
    at_least_as_extreme = 0
    for shift in range(total):
        rotated = residuals[shift:] + residuals[:shift]
        if _statistic(rotated, post_start, len(post), q) >= observed:
            at_least_as_extreme += 1

    return ConformalResult(
        at=at,
        p_value=at_least_as_extreme / total,
        p_floor=1.0 / total,
        statistic=observed,
        pre_days=len(pre),
        post_days=len(post),
        total_days=total,
    )


def _statistic(residuals: list[float], start: int, length: int, q: float) -> float:
    """`S_q(u) = ( T*^(-1/2) * sum_{t in post} |u_t|^q )^(1/q)`."""
    window = residuals[start : start + length]
    total = sum(abs(value) ** q for value in window)
    return (total / (length**0.5)) ** (1.0 / q)


def _row(on: date, seasonal: bool) -> list[float]:
    """Intercept, plus one-hot day-of-week with Sunday folded into the intercept.

    Day-of-week rather than a trend because the movement we are testing *is* a level change; a
    trend term would absorb part of it into the proxy and shrink the very residuals the test
    measures.
    """
    if not seasonal:
        return [1.0]
    row = [1.0] + [0.0] * 6
    weekday = on.weekday()
    if weekday < 6:
        row[weekday + 1] = 1.0
    return row


def _least_squares(design: list[list[float]], target: list[float]) -> list[float] | None:
    """Normal equations, solved by Gaussian elimination with partial pivoting.

    Stdlib only. The design matrix is 7 columns wide, so this is a 7x7 solve and numpy would
    be a dependency bought for nothing.

    Returns None on a singular system, which happens when the window is too short to contain
    every weekday -- a real condition rather than a numerical accident, and one the caller must
    treat as "no test available" rather than as a zero.
    """
    width = len(design[0])
    matrix = [[0.0] * (width + 1) for _ in range(width)]
    for row, value in zip(design, target, strict=True):
        for i in range(width):
            for j in range(width):
                matrix[i][j] += row[i] * row[j]
            matrix[i][width] += row[i] * value

    for column in range(width):
        pivot = max(range(column, width), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) < 1e-10:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        for row_index in range(column + 1, width):
            factor = matrix[row_index][column] / matrix[column][column]
            for col in range(column, width + 1):
                matrix[row_index][col] -= factor * matrix[column][col]

    solution = [0.0] * width
    for row_index in reversed(range(width)):
        total = matrix[row_index][width] - sum(
            matrix[row_index][c] * solution[c] for c in range(row_index + 1, width)
        )
        solution[row_index] = total / matrix[row_index][row_index]
    return solution


def significance_of_each(
    calendar: Calendar,
    segmentation: Segmentation,
    *,
    q: float = DEFAULT_Q,
    seasonal: bool = True,
) -> tuple[ConformalResult, ...]:
    """Test every break in a segmentation, each on its own single-regime window.

    **The window is the point of this function.** CWZ tests one break with `T0` pre-periods and
    `T*` post-periods, and the argument requires both sides to be one regime. So each break is
    tested with the *preceding segment* as its pre period and *its own segment* as its post
    period -- never the whole series, which would put neighbouring breaks in the residuals and
    silently destroy the test's power.

    A consequence worth stating: the window for a break is bounded by its neighbours, so the
    p-value floor `1/T` is set by how far apart the breaks are. Two breaks a fortnight apart
    cannot produce a small p-value no matter how large the movement between them, and that is
    a real limit rather than a weak result.
    """
    results: list[ConformalResult] = []
    for index, segment in enumerate(segmentation.segments):
        if index == 0:
            continue
        previous = segmentation.segments[index - 1]
        window = Calendar(
            days=tuple(day for day in calendar.days if previous.start <= day.on <= segment.end),
            freshness=calendar.freshness,
        )
        result = significance_of(window, segment.start, q=q, seasonal=seasonal)
        if result is not None:
            results.append(result)
    return tuple(results)
