"""Labelled evaluation cases generated from real series, with computed labels.

**Why this exists.** The hand-written suite in `fixtures.py` has nine scenarios, and a
measured run over them found a blocking gap: *no scenario makes `unverifiable` the correct
premise verdict*. A three-way decision cannot be calibrated on a set where one class never
appears, so conformal abstention had nothing to fit a threshold against.

Hand-writing more scenarios does not fix that at the scale calibration needs, and every
hand-written scenario is also a fabricated world. This module takes the other route: real
extracted series, and questions whose answer is **arithmetic over those series** rather
than anyone's judgement. "Did this event's daily rate fall between June and July" is a
subtraction. So is the false-premise version where it did not, and so is the unverifiable
version where the data does not reach the months asked about.

**The property that makes the set worth calibrating on.** For each comparison the generator
computes the real direction and then emits *both* readings -- the one that matches and the
one that does not -- in identical wording. A case labelled `false` is textually
indistinguishable from one labelled `holds`; only the underlying data differs. Without that,
a model could score well by reading the question's surface rather than the evidence, and the
fitted threshold would be worthless.

**The discipline that makes the labels trustworthy.** A wrong label is worse than a missing
one: it does not merely waste a case, it miscalibrates the threshold that governs every
future abstention. Candidates are free and the extract is large, so this module *discards
aggressively*. A comparison is emitted only when the movement clears both a statistical and
a plain-language bar (see `separated`), both windows are nearly fully observed, and the rates
are large enough for a ratio to mean anything. Most candidates do not survive, by design.

Nothing here reads the filesystem or depends on any particular extract; `real_data.py` loads
series, and the tests drive these functions with series they construct. Real business data
never enters the repository.
"""

from __future__ import annotations

import calendar
import statistics
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from hashlib import sha1

from cortex.eval.fixtures import DailyTruth, Difficulty, GroundTruth, Requirement, Scenario

#: How far apart two window means must be, in standard errors of their difference.
#:
#: This is the guard against labelling noise as movement. `variance.py` exists because an
#: earlier version of this project called a 3% weekly wobble a regression; the same mistake
#: made here would be baked into a calibration constant rather than a single answer.
#:
#: Three rather than the conventional two: a candidate that fails costs nothing, and the
#: extract yields far more candidates than the calibration needs.
SEPARATION_FLOOR = 3.0

#: How large the change must be relative to the earlier window, independent of significance.
#:
#: Needed *as well as* `SEPARATION_FLOOR` because the two fail in opposite directions. A
#: high-volume event makes a 1% difference statistically overwhelming, and no analyst reading
#: plain English would call that "a fall" -- a report describing it as flat would be right,
#: and scoring it wrong would teach the threshold to punish honesty.
RELATIVE_FLOOR = 0.15

#: Daily rate below which a ratio stops meaning anything. Two events one day and one the next
#: is a 50% fall by arithmetic and nothing at all in fact.
RATE_FLOOR = 5.0

#: Observed days over calendar days, per window. A month seen on 40% of its days has a total
#: that is not comparable with a fully observed month's, and a rate whose denominator is doing
#: the work. At 0.9 a 30-day month must be seen on at least 27 days.
DENSITY_FLOOR = 0.9

#: Days a window must clear the series' covered range by, for `unverifiable` to be the
#: unambiguous answer. A month ending the day before collection starts is a boundary
#: judgement, not a clean absence.
UNREACHABLE_MARGIN_DAYS = 7

#: Minimum observed days in a window before its variance is meaningful at all.
MIN_WINDOW_DAYS = 14


@dataclass(frozen=True, slots=True)
class RealSeries:
    """One event's real daily counts, exactly as extracted.

    `days` holds only the days the source reported. Gaps are real and are not filled with
    zeros: a missing day and a day with no events are different facts, and inventing the
    second from the first is the fabrication this project exists to avoid.
    """

    source: str
    project: str
    event: str
    days: tuple[tuple[date, int], ...]

    @property
    def covered(self) -> tuple[date, date] | None:
        """First and last day with an observation, or None for an empty series."""
        if not self.days:
            return None
        return self.days[0][0], self.days[-1][0]

    @property
    def key(self) -> str:
        return f"{self.source}:{self.project}:{self.event}"


@dataclass(frozen=True, slots=True)
class WindowStats:
    """What one window of a series supports as a claim."""

    month: date
    observed: int
    calendar_days: int
    mean: float
    #: Sample variance of the daily counts. Inflated by day-of-week seasonality, which makes
    #: every comparison built on it conservative -- the direction this should err in.
    variance: float

    @property
    def density(self) -> float:
        return self.observed / self.calendar_days


@dataclass(frozen=True, slots=True)
class LabelledQuestion:
    """One generated case, with the arithmetic that produced its label.

    The computed fields are kept so a human can audit any label without re-deriving it, and
    so a disagreement between the analyst and the label can be settled by reading rather
    than by re-running the generator.
    """

    name: str
    question: str
    #: The comparison this came from. Both readings of one month pair share it, and
    #: `balanced` samples by group so a matched pair is never split -- keeping one reading
    #: and dropping its twin would destroy the indistinguishability the pairing exists for.
    group: str
    #: `holds`, `false`, `unverifiable` or `none_asserted` -- the value of `PremiseVerdict`
    #: a correct report should reach. Stored as a string so this module does not depend on
    #: the report schema.
    premise_verdict: str
    series: RealSeries
    #: Why the label is what it is, in one line, from the numbers.
    basis: str
    required_signals: tuple[Requirement, ...] = ()
    refutation_signals: tuple[Requirement, ...] = ()
    windows: tuple[WindowStats, ...] = field(default_factory=tuple)

    #: The other events measured in the same project, planted alongside this one.
    #:
    #: **Why the case needs a world and not just its series.** With one event planted, every
    #: question is asked of a tenant that measures exactly one thing, and an `unverifiable`
    #: case becomes answerable by noticing the connector is nearly empty. With the project's
    #: catalogue planted, other events *do* have data in the window being asked about, so
    #: telling "this event is not collected here" apart from "collection was down" takes
    #: actually checking -- which is the discrimination the class is meant to measure.
    #:
    #: Empty means the case stands alone, which is what the unit tests use.
    world: tuple[RealSeries, ...] = field(default_factory=tuple)

    #: The months the question asks about, for cases whose windows carry no statistics.
    asked_months: tuple[date, ...] = field(default_factory=tuple)

    #: Verdicts that are defensible here besides `premise_verdict`, scored as neither right
    #: nor wrong.
    #:
    #: **Only where the evidence genuinely admits two readings.** An `unverifiable` case whose
    #: window the project *was* collecting through is the one that does: the event has no rows
    #: while its siblings have plenty, so either it was not instrumented yet -- the data cannot
    #: say what it did -- or it was instrumented and never fired, in which case the asserted
    #: movement demonstrably did not occur. PostHog cannot distinguish those, and the report
    #: schema's own definition of `false` covers a movement that is an artefact of "a changed
    #: definition", which an event beginning in July is.
    #:
    #: Measured, not assumed: 5 of the first 6 such cases answered `false`, each correctly
    #: describing the absence first -- "there is no baseline to fall from". Scoring that as a
    #: defect would fit a threshold to a distinction the evidence does not support.
    #:
    #: Empty where the window is outside the project's collection entirely. There nothing was
    #: measured, "we cannot tell" is the only reading, and `false` really is wrong.
    also_acceptable: tuple[str, ...] = field(default_factory=tuple)

    #: Explanations the data refutes, which a report must not name as the cause.
    #:
    #: **The error class this catches is a right verdict reached the wrong way.** The first
    #: live run answered an `unverifiable` case correctly and justified it with "no events of
    #: any kind are recorded in that period in the connected PostHog project" -- true of a
    #: tenant with one series planted, false once the project's real catalogue is there, and
    #: scored identically either way. Verdict accuracy alone cannot see it.
    #:
    #: Planted only when a sibling event demonstrably has data in the asked window, so the
    #: decoy is refuted by this case's own evidence rather than by an assumption.
    decoys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def difficulty(self) -> Difficulty:
        return {
            "false": Difficulty.FALSE_PREMISE,
            "unverifiable": Difficulty.UNANSWERABLE,
        }.get(self.premise_verdict, Difficulty.STRAIGHTFORWARD)

    def to_scenario(self) -> Scenario:
        """Build a runnable `Scenario` whose world is this series and nothing else.

        The series is planted as `DailyTruth`, so a request outside it is clamped rather than
        answered -- which is what makes an `unverifiable` case behave like one at the tool
        boundary instead of merely being labelled as one.
        """
        return Scenario(
            name=self.name,
            question=self.question,
            difficulty=self.difficulty,
            ground_truth=GroundTruth(
                cause=self.basis,
                required_signals=self.required_signals,
                required_capabilities=("posthog__event_trend",),
                decoys=self.decoys,
                is_unanswerable=self.premise_verdict == "unverifiable",
                is_false_premise=self.premise_verdict == "false",
                refutation_signals=self.refutation_signals,
            ),
            daily_truth={
                # This case's own series first, so it is the one matched when an unnamed
                # event falls back to the head of the tuple.
                "posthog__event_trend": tuple(
                    DailyTruth(event=s.event, days=s.days)
                    for s in (self.series, *(w for w in self.world if w.key != self.series.key))
                )
            },
        )


def _month_bounds(month: date) -> tuple[date, date]:
    last = calendar.monthrange(month.year, month.month)[1]
    return month.replace(day=1), month.replace(day=last)


def _month_name(month: date) -> str:
    return f"{calendar.month_name[month.month]} {month.year}"


def months_spanned(series: RealSeries) -> tuple[date, ...]:
    """Every calendar month the series has any observation in, first days, ordered."""
    seen: dict[date, None] = {}
    for day, _ in series.days:
        seen.setdefault(day.replace(day=1), None)
    return tuple(seen)


def window_stats(series: RealSeries, month: date) -> WindowStats | None:
    """Stats for one calendar month, or None when the month cannot support a claim.

    Returns None rather than a low-confidence figure: a caller that received a number would
    have to remember to check its density, and one that forgot would emit a label derived
    from four observed days of a thirty-day month.
    """
    lo, hi = _month_bounds(month)
    counts = [count for day, count in series.days if lo <= day <= hi]
    span = (hi - lo).days + 1
    if len(counts) < MIN_WINDOW_DAYS or len(counts) / span < DENSITY_FLOOR:
        return None
    return WindowStats(
        month=lo,
        observed=len(counts),
        calendar_days=span,
        mean=statistics.fmean(counts),
        variance=statistics.variance(counts) if len(counts) > 1 else 0.0,
    )


def separated(earlier: WindowStats, later: WindowStats) -> bool:
    """Whether the two windows differ by enough for a direction to be stated as fact.

    Both bars must clear, because they catch opposite errors: the standard-error bar rejects
    a large-looking move inside a noisy series, and the relative bar rejects a tiny move made
    significant by volume alone. See the constants for why each exists.
    """
    if earlier.mean < RATE_FLOOR or later.mean < RATE_FLOOR:
        return False
    if abs(later.mean - earlier.mean) / earlier.mean < RELATIVE_FLOOR:
        return False
    standard_error = (earlier.variance / earlier.observed + later.variance / later.observed) ** 0.5
    if standard_error == 0.0:
        # No within-window variation at all: the relative bar above is the whole test.
        return True
    return abs(later.mean - earlier.mean) / standard_error >= SEPARATION_FLOOR


#: How a report describes each direction, any one of which counts.
#:
#: Keyed by the direction *the data actually moved*, which is the only thing a report can be
#: required to say. An earlier version keyed a table by the asserted verb and indexed it with
#: the actual one, so a case built on a rise required the report to contain "fell" -- the
#: refutation wording was right and the required-signal wording was its exact inverse.
_DESCRIBES = {
    "fall": ("fell", "declin", "lower", "dropped", "down"),
    "rise": ("rose", "increase", "higher", "grew", "up"),
}


def comparison_cases(series: RealSeries) -> tuple[LabelledQuestion, ...]:
    """Both readings of every adjacent month pair whose movement is unambiguous.

    Emits the matching assertion as `holds` and the opposing one as `false`, in the same
    words. That pairing is the point: it is what stops a model scoring well from the
    question's surface, and it balances the two classes by construction rather than by a
    quota applied afterwards.
    """
    out: list[LabelledQuestion] = []
    months = months_spanned(series)
    for earlier_month, later_month in zip(months, months[1:], strict=False):
        if (later_month - earlier_month).days > 31:
            continue  # not adjacent; a gap month sits between them
        earlier = window_stats(series, earlier_month)
        later = window_stats(series, later_month)
        if earlier is None or later is None or not separated(earlier, later):
            continue

        actual = "fall" if later.mean < earlier.mean else "rise"
        out.extend(
            _comparison(series, earlier, later, asserted, actual) for asserted in ("fall", "rise")
        )
        out.append(_which_month(series, earlier, later))
    return tuple(out)


def _which_month(series: RealSeries, earlier: WindowStats, later: WindowStats) -> LabelledQuestion:
    """The same pair asked neutrally: answerable, asserting nothing.

    `none_asserted` is the ordinary case in production and was the scarcest class here --
    superlatives need three well-observed months, which most series do not have. This asks
    one already-qualified pair which way round it goes, so every pair that supports a
    `holds`/`false` twin also supports the neutral reading, at no extra bar.

    Keeping the class well populated matters: a calibration set holding only the three
    interesting verdicts would be fitted on a population the product never sees, and would
    price abstention against questions nobody asks.
    """
    higher = later if later.mean > earlier.mean else earlier
    return LabelledQuestion(
        name=_case_name(series, f"{earlier.month:%Y%m}_which"),
        question=(
            f"Between {_month_name(earlier.month)} and {_month_name(later.month)}, which "
            f"had the higher daily rate of {series.event}?"
        ),
        group=f"{series.key}|{earlier.month:%Y%m}|which",
        premise_verdict="none_asserted",
        series=series,
        basis=(
            f"{_month_name(higher.month)}: {higher.mean:.1f}/day against "
            f"{(earlier if higher is later else later).mean:.1f}/day."
        ),
        required_signals=(calendar.month_name[higher.month.month],),
        windows=(earlier, later),
    )


def _comparison(
    series: RealSeries,
    earlier: WindowStats,
    later: WindowStats,
    asserted: str,
    actual: str,
) -> LabelledQuestion:
    holds = asserted == actual
    change = (later.mean - earlier.mean) / earlier.mean
    basis = (
        f"{series.event} averaged {earlier.mean:.1f}/day in {_month_name(earlier.month)} "
        f"and {later.mean:.1f}/day in {_month_name(later.month)}, "
        f"a {abs(change):.0%} {'fall' if actual == 'fall' else 'rise'}."
    )
    return LabelledQuestion(
        name=_case_name(series, f"{earlier.month:%Y%m}_{asserted}"),
        question=(
            f"Why did {series.event} {asserted} between {_month_name(earlier.month)} "
            f"and {_month_name(later.month)}?"
        ),
        group=f"{series.key}|{earlier.month:%Y%m}|moved",
        premise_verdict="holds" if holds else "false",
        series=series,
        basis=basis,
        # Required of both readings, not only the false one. A `holds` case whose ground
        # truth demanded nothing would pass on any report at all, including one that named
        # the movement backwards -- the class would have been scored vacuously.
        required_signals=(_DESCRIBES[actual],),
        # A refutation is only required of the case whose premise is false, and it is checked
        # against the executive summary: reaching the right answer in a fourth bullet has
        # already misinformed every reader who stopped at the first.
        refutation_signals=(_DESCRIBES[actual],) if not holds else (),
        windows=(earlier, later),
    )


def unreachable_cases(
    series: RealSeries, plausible_months: frozenset[date]
) -> tuple[LabelledQuestion, ...]:
    """Questions about months this series cannot speak to, worded like any other.

    `plausible_months` should be months some *other* series in the same workspace does cover.
    That restriction is what separates a real `unverifiable` case from a toy one: asking about
    a month no data anywhere reaches is answerable from the question alone, while asking about
    a month that is ordinary for the workspace and absent for this event requires actually
    checking. The extract's 48 distinct series start dates supply these for free.
    """
    covered = series.covered
    if covered is None:
        return ()
    # The series must be able to answer *something*, or the case tests the wrong skill. A
    # series with four days of data makes every month pair unverifiable for a trivial reason,
    # and a model could clear it by noticing the tool returned almost nothing. The case worth
    # calibrating on is the one where the event is well measured and the *asked window* still
    # is not covered -- absence that is specific rather than total. Unguarded, near-empty
    # series supplied the bulk of this class.
    if not any(window_stats(series, month) for month in months_spanned(series)):
        return ()
    first, last = covered
    out: list[LabelledQuestion] = []
    ordered = sorted(plausible_months)

    def unreachable(month: date) -> bool:
        lo, hi = _month_bounds(month)
        return (first - hi).days >= UNREACHABLE_MARGIN_DAYS or (
            lo - last
        ).days >= UNREACHABLE_MARGIN_DAYS

    for earlier_month, later_month in zip(ordered, ordered[1:], strict=False):
        if (later_month - earlier_month).days > 31:
            continue
        # Both months must be out of reach. A pair straddling the boundary has a partial
        # answer available, and the honest response there is a disclosed partial comparison
        # rather than `unverifiable` -- a different case, and not this one.
        if not (unreachable(earlier_month) and unreachable(later_month)):
            continue
        for asserted in ("fall", "rise"):
            out.append(
                LabelledQuestion(
                    # The identical template to `_comparison`. An `unverifiable` case that
                    # announced itself by shape -- one month and "during", where the others
                    # take two months and "between" -- would let a model match the wording
                    # instead of checking the evidence, and the fitted threshold would be
                    # measuring that shortcut.
                    name=_case_name(series, f"{earlier_month:%Y%m}_{asserted}_absent"),
                    group=f"{series.key}|{earlier_month:%Y%m}|absent",
                    question=(
                        f"Why did {series.event} {asserted} between "
                        f"{_month_name(earlier_month)} and {_month_name(later_month)}?"
                    ),
                    premise_verdict="unverifiable",
                    series=series,
                    asked_months=(earlier_month, later_month),
                    basis=(
                        f"{series.event} has no data in {_month_name(earlier_month)} or "
                        f"{_month_name(later_month)}; collection runs {first.isoformat()} "
                        f"to {last.isoformat()}."
                    ),
                    required_signals=(("no data", "no events", "does not cover", "outside"),),
                    refutation_signals=(("no data", "no events", "does not cover", "outside"),),
                )
            )
    return tuple(out)


def superlative_cases(series: RealSeries) -> tuple[LabelledQuestion, ...]:
    """ "Which month had the highest daily rate" -- answerable, and asserting nothing.

    Emitted only when the top month is separated from the *runner-up* by the same bar a
    comparison must clear. A superlative over two months that are level is a coin flip, and
    scoring a coin flip trains the threshold on noise.

    These carry `none_asserted`: the ordinary case, and the fourth value of the premise enum.
    A calibration set holding only the three interesting classes would be fitted on a
    population the product never sees.
    """
    ranked = sorted(
        (stats for month in months_spanned(series) if (stats := window_stats(series, month))),
        key=lambda s: s.mean,
        reverse=True,
    )
    if len(ranked) < 3 or not separated(ranked[1], ranked[0]):
        return ()
    best = ranked[0]
    return (
        LabelledQuestion(
            name=_case_name(series, "peak"),
            group=f"{series.key}|peak",
            question=f"Which month had the highest daily rate of {series.event}?",
            premise_verdict="none_asserted",
            series=series,
            basis=(
                f"{_month_name(best.month)} at {best.mean:.1f}/day, ahead of "
                f"{_month_name(ranked[1].month)} at {ranked[1].mean:.1f}/day."
            ),
            required_signals=(calendar.month_name[best.month.month],),
            windows=tuple(ranked),
        ),
    )


#: Longest case name the harness can turn into a tenant slug.
#:
#: It builds `eval-{name, underscores as hyphens}-{6 hex}` and a slug may be 63 characters,
#: which leaves 51. Real event names are long enough to break that: `automation_api_dispatch
#: _automation` produced a 64-character name, and the case errored before drafting with
#: `InvalidTenantSlug`. Unbounded, 268 of 770 cases on the current extract would have failed
#: that way -- a third of a calibration run, lost to something that reads like an outage
#: rather than a naming bug, on cases the eval had already paid to build.
MAX_CASE_NAME = 51


def _case_name(series: RealSeries, suffix: str) -> str:
    """`gen_<project>_<event>_<suffix>`, bounded so a tenant slug can be built from it."""
    cleaned = "".join(c if c.isalnum() else "_" for c in series.event).strip("_").lower()
    head = f"gen_{series.project}_"
    room = MAX_CASE_NAME - len(head) - len(suffix) - 1
    if len(cleaned) > room:
        # Truncated with a digest of the full name rather than cut short. Two events sharing
        # a prefix -- `automation_api_dispatch_automation` and `automation_api_dispatch_run`
        # -- would otherwise take the same name, and two cases with one name overwrite each
        # other's bundle, which loses a result silently instead of noisily.
        digest = sha1(series.event.encode()).hexdigest()[:4]
        cleaned = f"{cleaned[: max(1, room - 5)]}_{digest}"
    return f"{head}{cleaned}_{suffix}"


def generate(all_series: tuple[RealSeries, ...]) -> tuple[LabelledQuestion, ...]:
    """Every case the extract supports, across all four premise verdicts."""
    plausible = frozenset(month for series in all_series for month in months_spanned(series))
    by_project: dict[str, tuple[RealSeries, ...]] = {}
    for series in all_series:
        by_project[series.project] = (*by_project.get(series.project, ()), series)

    out: list[LabelledQuestion] = []
    for series in all_series:
        cases = (
            *comparison_cases(series),
            *unreachable_cases(series, plausible),
            *superlative_cases(series),
        )
        # Each case is asked of the project that actually measures it, rather than of a
        # tenant that measures one event. See `LabelledQuestion.world`.
        world = by_project[series.project]
        for case in cases:
            if collection_failed(case, world):
                continue
            refuted = _refuted_by(case, world)
            out.append(
                replace(
                    case,
                    world=world,
                    decoys=refuted,
                    # The same condition that makes the "nothing was collected" claim false
                    # makes `false` a defensible verdict: both follow from the project having
                    # been collecting through a window this event has no rows in.
                    also_acceptable=("false",) if refuted else (),
                )
            )
    return tuple(out)


def balanced(
    cases: tuple[LabelledQuestion, ...], *, per_class: int | None = None, seed: int = 0
) -> tuple[LabelledQuestion, ...]:
    """Sample down to an even mix of premise verdicts, deterministically.

    **Why balancing is not optional here.** Run over a real extract the raw generator is
    roughly three-quarters `unverifiable`, because every month outside a series' coverage is
    a candidate and the extract's series start on 48 different dates. Nobody asks questions
    in that proportion. A conformal threshold fitted on it would be fitted to a population
    the product never meets, and would over-abstain on the questions users actually ask --
    which is precisely the failure the calibration is meant to prevent.

    Sampling is by `group`, so both readings of a comparison survive or neither does. Losing
    one half of a matched pair would leave a set where wording correlates with the label
    again, undoing the reason the pairs are generated together.

    `seed` makes the selection reproducible: a calibration fitted on a set that cannot be
    rebuilt is not auditable.
    """
    import random

    by_class: dict[str, dict[str, list[LabelledQuestion]]] = {}
    for case in cases:
        by_class.setdefault(case.premise_verdict, {}).setdefault(case.group, []).append(case)
    if not by_class:
        return ()

    # Each class is capped independently. Letting the scarcest class set the cap for all of
    # them -- the obvious reading of "balanced" -- threw away about 1,200 of 1,229 real cases
    # because one class had nine. A cap the caller chooses, with `class_counts` reporting
    # what each class actually reached, keeps the proportions deliberate without paying that.
    target = (
        per_class
        if per_class is not None
        else min(sum(len(g) for g in groups.values()) for groups in by_class.values())
    )

    out: list[LabelledQuestion] = []
    for verdict in sorted(by_class):
        groups = sorted(by_class[verdict])
        random.Random(f"{seed}:{verdict}").shuffle(groups)
        taken = 0
        for group in groups:
            members = by_class[verdict][group]
            if taken + len(members) > target:
                continue
            out.extend(members)
            taken += len(members)
    return tuple(sorted(out, key=lambda c: c.name))


#: Wordings of the claim that nothing at all was collected, which an `unverifiable` case
#: invites and its own siblings refute. Matched as substrings against a report's conclusion.
_NOTHING_COLLECTED = (
    "no events of any kind",
    "no data of any kind",
    "collection stopped",
    "tracking stopped",
)


def _refuted_by(case: LabelledQuestion, world: tuple[RealSeries, ...]) -> tuple[str, ...]:
    """Explanations this case's own world contradicts.

    Only for `unverifiable` cases, and only when a sibling event really does have data in
    the months asked about -- otherwise "nothing was collected then" is a fair reading of the
    evidence, and penalising it would punish an honest answer.

    A single observed day per month is enough, because that is exactly what the claim denies:
    "no events of any kind" is refuted by one event, not by a well-observed month. Requiring a
    dense month here is a different and stricter test, and it planted the decoy on 8 cases
    where this plants it on the ones the evidence actually contradicts.
    """
    if case.premise_verdict != "unverifiable" or not case.asked_months:
        return ()
    wanted = {(month.year, month.month) for month in case.asked_months}
    for sibling in world:
        if sibling.key == case.series.key:
            continue
        seen = {(day.year, day.month) for day, count in sibling.days if count}
        if wanted <= seen:
            return _NOTHING_COLLECTED
    return ()


#: Share of a project's comparable events that must collapse together before a fall is read
#: as the pipeline stopping rather than the metric moving.
PROJECT_WIDE_SHARE = 0.6

#: How far a series must fall to count as collapsed rather than merely down. Collection
#: stopping takes an event to near-zero; a bad month does not.
COLLAPSE = 0.9


def _collapsed(series: RealSeries, earlier: date, later: date) -> bool:
    """Whether this series is effectively dead by the end of the later month.

    Measured on the tail rather than on the month's mean, because a pipeline that stops
    mid-month leaves a month that merely looks bad: that event lost 98% of its rate after
    2026-08-11 and still averaged only 81% down across August, under any threshold that would
    separate a collapse from a steep decline. The last week is what says whether anything is
    still arriving.
    """
    before = window_stats(series, earlier)
    if before is None or before.mean < RATE_FLOOR:
        return False
    lo, hi = _month_bounds(later)
    tail = [count for day, count in series.days if hi - timedelta(days=6) <= day <= hi]
    if not tail:
        return False
    return statistics.fmean(tail) <= before.mean * (1 - COLLAPSE)


def collection_failed(case: LabelledQuestion, world: tuple[RealSeries, ...]) -> bool:
    """Whether this fall is the whole project going dark rather than this metric moving.

    **The case that made this necessary.** one event in a three-project workspace fell 81%
    from July to August 2026, so arithmetic labelled it `holds` -- the premise asserts a fall
    and a fall is what the numbers show. The analyst answered `false`, because every other
    event in that project fell 98-100% across the same days: collection had stopped. The
    analyst was right, the label was wrong, and the report schema agrees -- `false` covers a
    movement that is "an artefact of an incomplete period or a changed definition".

    **Narrow on purpose.** A first version dropped any movement most of the project shared,
    which also removed sixty perfectly good cases: fourteen of seventeen events rose together
    from January to February, and there the premise really does hold -- the metric really rose,
    and "everything rose" is an answer to *why*, not a reason the premise fails. What makes the
    August case different is not that the movement was shared but that it was a collapse to
    near-zero, which is what a pipeline stopping looks like and what growth never does.

    Dropped rather than relabelled `false`: a disclosure about collection is equally defensible
    there, and a calibration set should not contain a question with two right answers.
    """
    if len(case.windows) != 2:
        return False
    earlier, later = case.windows
    if not _collapsed(case.series, earlier.month, later.month):
        # This event is still alive at the end of the window, so whatever the rest of the
        # project did, the movement being asked about is not a dead pipeline.
        return False
    collapsed = comparable = 0
    for sibling in world:
        if sibling.key == case.series.key:
            continue
        before = window_stats(sibling, earlier.month)
        if before is None or before.mean < RATE_FLOOR:
            continue
        comparable += 1
        collapsed += _collapsed(sibling, earlier.month, later.month)
    # With no comparable sibling there is no control group, so a lone fall stands as measured.
    return comparable > 0 and collapsed / comparable >= PROJECT_WIDE_SHARE


def class_counts(cases: tuple[LabelledQuestion, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.premise_verdict] = counts.get(case.premise_verdict, 0) + 1
    return counts
