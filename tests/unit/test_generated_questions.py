"""The generated calibration set is only as good as its labels and its wording.

Two failures would make the set actively harmful rather than merely useless, and both are
silent, so both are tested here rather than left to inspection.

**A label that is wrong** does not waste one case: it miscalibrates the abstention threshold
fitted on the whole set, and that threshold governs every future answer. So the bars that
decide whether a case is emitted at all — noise, effect size, density, reachability — are
tested at their boundaries.

**Wording that correlates with the label** lets a model score well without reading any
evidence. The first working generator had this defect: its `unverifiable` cases all asked
about one month with "during", while every other class asked about two with "between", so
the class was legible from the question alone. `TestTheQuestionDoesNotGiveAwayTheAnswer` is
the regression test for it, and it is the most important test in this file.

Every series here is constructed. The real extract lives outside the repository and no test
may depend on it.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from cortex.eval.generated import (
    MAX_CASE_NAME,
    RealSeries,
    _collapsed,
    balanced,
    class_counts,
    collection_failed,
    comparison_cases,
    generate,
    separated,
    superlative_cases,
    unreachable_cases,
    window_stats,
)
from cortex.memory.naming import validate_slug

#: The shape every comparison question takes, whatever its label. See the class below.
COMPARISON = re.compile(r"^Why did (.+) (fall|rise) between (\w+ \d{4}) and (\w+ \d{4})\?$")


def series(
    start: date, days: int, value, *, event: str = "signups", project: str = "p1"
) -> RealSeries:
    """A daily series whose count for day `i` is `value(i)`, or a constant."""
    if not callable(value):
        constant = value
        value = lambda _: constant  # noqa: E731
    return RealSeries(
        source="posthog",
        project=project,
        event=event,
        days=tuple((start + timedelta(days=i), value(i)) for i in range(days)),
    )


def two_months(january: int, february: int, *, jitter=lambda i: 0, **kw) -> RealSeries:
    """January and February at the given daily rates, fully observed."""
    return series(
        date(2026, 1, 1),
        59,
        lambda i: (january if i < 31 else february) + jitter(i),
        **kw,
    )


class TestAMovementMustBeRealBeforeItIsStatedAsFact:
    """The bars that decide whether arithmetic is allowed to become a label."""

    def test_a_clear_change_is_emitted(self):
        cases = comparison_cases(two_months(100, 200))
        assert {c.premise_verdict for c in cases} == {"holds", "false", "none_asserted"}

    def test_a_change_smaller_than_the_noise_is_discarded(self):
        # 100 -> 112 is a 12% move, under the relative floor, and swamped by a swing of
        # +/-60 a day. Calling it a fall is the mistake `variance.py` exists to prevent.
        noisy = two_months(100, 112, jitter=lambda i: 60 if i % 2 else -60)
        assert comparison_cases(noisy) == ()

    def test_a_tiny_change_is_discarded_however_significant_it_looks(self):
        # Perfectly flat within each month, so the standard error is zero and any difference
        # is "significant". A 2% move is still not a fall in any language a reader speaks,
        # and scoring a report that calls it flat as wrong would teach the threshold to
        # punish the honest answer.
        assert comparison_cases(two_months(10_000, 10_200)) == ()

    def test_a_change_between_trickles_is_discarded(self):
        # 2/day to 4/day doubles by arithmetic and means nothing in fact.
        assert comparison_cases(two_months(2, 4)) == ()

    def test_a_sparsely_observed_month_supports_no_claim(self):
        sparse = RealSeries(
            source="posthog",
            project="p1",
            event="signups",
            # January seen on 6 of 31 days; February fully observed.
            days=tuple((date(2026, 1, 1) + timedelta(days=i * 5), 100) for i in range(6))
            + tuple((date(2026, 2, 1) + timedelta(days=i), 300) for i in range(28)),
        )
        assert window_stats(sparse, date(2026, 1, 1)) is None
        assert comparison_cases(sparse) == ()

    def test_separation_needs_both_bars(self):
        flat_low = window_stats(two_months(100, 100), date(2026, 1, 1))
        flat_high = window_stats(two_months(100, 200), date(2026, 2, 1))
        assert separated(flat_low, flat_high)
        assert not separated(flat_low, flat_low)


class TestBothReadingsOfOneComparisonAreEmitted:
    """The matched pair is what balances the classes by construction."""

    def test_the_true_and_false_readings_differ_only_in_the_verb(self):
        cases = comparison_cases(two_months(100, 300))
        holds = next(c for c in cases if c.premise_verdict == "holds")
        false = next(c for c in cases if c.premise_verdict == "false")
        assert holds.question.replace("rise", "@") == false.question.replace("fall", "@")

    def test_the_reading_that_matches_the_data_is_the_one_that_holds(self):
        rose = comparison_cases(two_months(100, 300))
        assert next(c for c in rose if c.premise_verdict == "holds").question.count("rise")
        fell = comparison_cases(two_months(300, 100))
        assert next(c for c in fell if c.premise_verdict == "holds").question.count("fall")

    def test_the_false_reading_must_be_refuted_in_the_summary(self):
        false = next(
            c for c in comparison_cases(two_months(100, 300)) if c.premise_verdict == "false"
        )
        # Asserted a fall against a rise, so a correct report says so up front.
        assert "rose" in false.refutation_signals[0]


class TestUnverifiableMeansTheWindowIsOutOfReach:
    def test_months_before_collection_started_are_unverifiable(self):
        # Well measured from June; asked about January and February.
        covered = series(date(2026, 6, 1), 100, 500)
        cases = unreachable_cases(covered, frozenset({date(2026, 1, 1), date(2026, 2, 1)}))
        assert cases and {c.premise_verdict for c in cases} == {"unverifiable"}

    def test_a_pair_straddling_the_boundary_is_not_unverifiable(self):
        # February is covered, so a partial answer exists and the honest response is a
        # disclosed partial comparison rather than a refusal.
        covered = series(date(2026, 2, 1), 200, 500)
        assert unreachable_cases(covered, frozenset({date(2026, 1, 1), date(2026, 2, 1)})) == ()

    def test_a_month_ending_days_before_collection_is_too_close_to_call(self):
        covered = series(date(2026, 2, 3), 200, 500)
        assert unreachable_cases(covered, frozenset({date(2026, 1, 1), date(2026, 2, 1)})) == ()

    def test_a_near_empty_series_generates_nothing(self):
        """The class must test window-specific absence, not an empty connector.

        A series with four days of data makes every month unverifiable for a reason that has
        nothing to do with the window asked about, and a model could clear those cases by
        noticing the tool returned almost nothing. Unguarded, such series produced the
        majority of this class on the real extract.
        """
        thin = series(date(2026, 6, 1), 4, 500)
        assert unreachable_cases(thin, frozenset({date(2026, 1, 1), date(2026, 2, 1)})) == ()


class TestTheQuestionDoesNotGiveAwayTheAnswer:
    """The invariant the whole set rests on.

    If wording predicts the label, a model can score well without consulting any evidence and
    the fitted threshold measures that shortcut instead of the model's judgement. This is a
    regression test: the first working generator asked `unverifiable` cases about one month
    with "during" while every other class asked about two with "between".
    """

    def test_all_three_premise_verdicts_share_one_question_template(self):
        # Rises then falls, so both classes carry both verbs. One month pair could only ever
        # put one verb in each class, which would leave the invariant untested rather than
        # satisfied: what must not predict the label is the *verb*, across the set.
        moved = series(
            date(2026, 1, 1),
            90,
            lambda i: 100 if i < 31 else (300 if i < 59 else 100),
            event="signups",
        )
        # A second, later series so the months asked about are ordinary for the workspace
        # while being absent for this event -- the property that makes the case non-trivial.
        absent = series(date(2026, 6, 1), 100, 500, event="signups", project="p2")
        cases = generate((moved, absent))

        by_verdict: dict[str, set[str]] = {}
        for case in cases:
            if match := COMPARISON.match(case.question):
                by_verdict.setdefault(case.premise_verdict, set()).add(match.group(2))

        assert {"holds", "false", "unverifiable"} <= set(by_verdict)
        # Every class uses both verbs, so neither the template nor the verb carries signal.
        for verdict, verbs in by_verdict.items():
            assert verbs == {"fall", "rise"}, verdict

    def test_no_question_states_its_own_answer(self):
        for case in generate((two_months(100, 300), series(date(2026, 6, 1), 100, 500))):
            lowered = case.question.lower()
            for giveaway in ("no data", "unverifiable", "did not", "actually", "incomplete"):
                assert giveaway not in lowered, case.question


class TestSuperlativesNeedAClearWinner:
    def test_a_runaway_month_is_named(self):
        rising = series(date(2026, 1, 1), 90, lambda i: 100 if i < 31 else (200 if i < 59 else 900))
        cases = superlative_cases(rising)
        assert cases and cases[0].required_signals == ("March",)

    def test_a_photo_finish_is_discarded(self):
        level = series(date(2026, 1, 1), 90, lambda i: 100 if i < 31 else (300 if i < 59 else 302))
        assert superlative_cases(level) == ()


class TestBalancingIsDeliberateAndReproducible:
    def _mixed(self):
        return generate((two_months(100, 300), series(date(2026, 6, 1), 100, 500, project="p2")))

    def test_no_class_exceeds_the_cap(self):
        counts = class_counts(balanced(self._mixed(), per_class=2))
        assert counts and all(n <= 2 for n in counts.values())

    def test_a_matched_pair_is_never_split(self):
        """Keeping one reading and dropping its twin would undo the pairing's whole purpose."""
        kept = balanced(self._mixed(), per_class=2)
        groups = {c.group for c in kept}
        for case in self._mixed():
            if case.group in groups:
                assert case in kept, case.name

    def test_the_same_seed_selects_the_same_cases(self):
        cases = self._mixed()
        assert [c.name for c in balanced(cases, per_class=2, seed=7)] == [
            c.name for c in balanced(cases, per_class=2, seed=7)
        ]


class TestACaseIsRunnable:
    def test_a_generated_case_becomes_a_scenario_carrying_the_real_series(self):
        case = next(
            c for c in comparison_cases(two_months(100, 300)) if c.premise_verdict == "false"
        )
        scenario = case.to_scenario()
        assert scenario.question == case.question
        assert scenario.ground_truth.is_false_premise
        planted = scenario.daily_truth["posthog__event_trend"][0]
        assert planted.days == case.series.days

    def test_an_unverifiable_case_declines_a_cause(self):
        case = unreachable_cases(
            series(date(2026, 6, 1), 100, 500), frozenset({date(2026, 1, 1), date(2026, 2, 1)})
        )[0]
        assert case.to_scenario().ground_truth.declines_a_cause


class TestTheToolBoundaryAgreesWithTheLabel:
    """A case is only `unverifiable` if the tool really returns nothing for that window.

    The label is arithmetic over the series; what the analyst actually sees is whatever the
    fixture serves for the window it asks about. If those two disagree the set is scoring
    something other than what it claims to, and nothing else in this file would notice --
    every other test reasons about the series directly and never goes through `to_scenario`.

    Written after a probe appeared to show an `unverifiable` case being handed 89 rows of
    data from months it had not asked about. That turned out to be the probe passing the
    wrong parameter names, but the property it was checking was load-bearing and untested.
    """

    WINDOW = {"start_date": "2026-01-01", "end_date": "2026-02-28", "interval": "day"}

    def test_an_unverifiable_case_is_served_nothing_for_the_window_it_asks_about(self):
        covered = series(date(2026, 6, 1), 100, 500)
        case = unreachable_cases(covered, frozenset({date(2026, 1, 1), date(2026, 2, 1)}))[0]
        payload = case.to_scenario().response_for(
            "posthog__event_trend", {"event": covered.event, **self.WINDOW}
        )
        assert payload["series"] == []

    def test_a_comparison_case_is_served_only_the_window_it_asks_about(self):
        moved = two_months(100, 300)
        case = next(c for c in comparison_cases(moved) if c.premise_verdict == "holds")
        payload = case.to_scenario().response_for(
            "posthog__event_trend", {"event": moved.event, **self.WINDOW}
        )
        buckets = [row["bucket"][:10] for row in payload["series"]]
        assert len(buckets) == 59
        assert buckets[0] == "2026-01-01"
        assert buckets[-1] == "2026-02-28"


class TestAReportIsRequiredToNameTheMovementThatHappened:
    """Both readings require the *actual* direction, which is the only checkable fact.

    Regression test: the required-signal table was keyed by the asserted verb and indexed
    with the actual one, so a case built on a rise required the report to say "fell". The
    refutation wording was correct, which is why nothing caught it -- that field was tested
    and this one was not. A `holds` case additionally required nothing at all, so it would
    have passed on a report that named the movement backwards.
    """

    def test_a_rise_requires_rise_wording_whichever_way_it_was_asked(self):
        for case in comparison_cases(two_months(100, 300)):
            if case.premise_verdict in ("holds", "false"):
                assert "rose" in case.required_signals[0], case.name
                assert "fell" not in case.required_signals[0], case.name

    def test_a_fall_requires_fall_wording_whichever_way_it_was_asked(self):
        for case in comparison_cases(two_months(300, 100)):
            if case.premise_verdict in ("holds", "false"):
                assert "fell" in case.required_signals[0], case.name
                assert "rose" not in case.required_signals[0], case.name


class TestACaseIsAskedOfAWorldNotOfOneSeries:
    """An `unverifiable` case must not be answerable by noticing the tenant is empty.

    With one event planted, a tenant measures exactly one thing, and "no data for this
    window" is indistinguishable from "this connector has nothing". The discrimination the
    class exists to measure is the other one: siblings *do* have data for the very window
    being asked about, so collection being down is refutable from evidence and the only
    honest answer left is that this event is not measured then.
    """

    WINDOW = {"start_date": "2026-01-01", "end_date": "2026-02-28", "interval": "day"}

    def _world(self):
        # Two events in one project: one measured all year, one only from June.
        return (
            series(date(2026, 1, 1), 250, 400, event="all_year"),
            series(date(2026, 6, 1), 100, 500, event="from_june"),
        )

    def test_every_case_carries_its_project(self):
        for case in generate(self._world()):
            assert {s.event for s in case.world} == {"all_year", "from_june"}

    def test_the_sibling_answers_the_window_the_asked_event_cannot(self):
        case = next(
            c
            for c in generate(self._world())
            if c.premise_verdict == "unverifiable" and c.series.event == "from_june"
        )
        scenario = case.to_scenario()
        asked = scenario.response_for("posthog__event_trend", {"event": "from_june", **self.WINDOW})
        sibling = scenario.response_for(
            "posthog__event_trend", {"event": "all_year", **self.WINDOW}
        )
        assert asked["series"] == []
        assert len(sibling["series"]) == 59

    def test_the_tenant_advertises_every_event_it_measures(self):
        case = generate(self._world())[0]
        assert case.to_scenario().events_described() == frozenset({"all_year", "from_june"})

    def test_a_case_built_without_a_world_still_plants_its_own_series(self):
        case = next(c for c in comparison_cases(two_months(100, 300)))
        planted = case.to_scenario().daily_truth["posthog__event_trend"]
        assert [t.event for t in planted] == ["signups"]


class TestARightVerdictReachedTheWrongWayIsStillCaught:
    """Verdict accuracy cannot see a correct answer justified by a false claim.

    The first live run answered an `unverifiable` case correctly and explained it with "no
    events of any kind are recorded in that period in the connected PostHog project". With
    the project's real catalogue planted that sentence is false, and scoring the verdict
    alone rates it identically to a report that checked. So the claim is planted as a decoy
    -- but only where this case's own siblings refute it.
    """

    def _world(self):
        return (
            series(date(2026, 1, 1), 250, 400, event="all_year"),
            series(date(2026, 6, 1), 100, 500, event="from_june"),
        )

    def _absent(self, world):
        return next(
            c
            for c in generate(world)
            if c.premise_verdict == "unverifiable" and c.series.event == "from_june"
        )

    def test_the_claim_is_planted_when_a_sibling_refutes_it(self):
        case = self._absent(self._world())
        assert "no events of any kind" in case.decoys
        assert case.to_scenario().ground_truth.decoys == case.decoys

    def test_one_observed_day_is_enough_to_refute_it(self):
        """The claim denies any event at all, so one event contradicts it.

        Requiring a well-observed month instead is a stricter and different test, and it
        planted the decoy on a twelfth as many cases.
        """
        sparse = RealSeries(
            source="posthog",
            project="p1",
            event="all_year",
            days=((date(2026, 1, 9), 3), (date(2026, 2, 9), 3))
            + tuple((date(2026, 6, 1) + timedelta(days=i), 400) for i in range(120)),
        )
        case = self._absent((sparse, series(date(2026, 6, 1), 100, 500, event="from_june")))
        assert case.asked_months[:2] == (date(2026, 1, 1), date(2026, 2, 1))
        assert "no events of any kind" in case.decoys

    def test_nothing_is_planted_where_the_claim_would_be_fair(self):
        """Penalising an honest reading of the evidence is worse than not scoring it."""
        # January is plausible because *another project* measures it, so the case is still
        # generated -- but nothing in this project can refute "nothing was collected then",
        # which makes that a fair reading of the evidence the analyst actually has.
        both_late = (
            series(date(2026, 1, 1), 250, 400, event="elsewhere", project="p2"),
            series(date(2026, 6, 1), 100, 400, event="also_from_june"),
            series(date(2026, 6, 1), 100, 500, event="from_june"),
        )
        assert self._absent(both_late).decoys == ()

    def test_answerable_cases_carry_no_such_decoy(self):
        for case in generate(self._world()):
            if case.premise_verdict != "unverifiable":
                assert case.decoys == (), case.name


class TestACaseNameCanBecomeATenantSlug:
    """A name too long to address is a case that errors before it drafts anything.

    The harness gives each attempt its own tenant, named `eval-{case name}-{6 hex}`, and a
    slug may be 63 characters. Real event names are long enough to break that: a 64-character
    name failed with `InvalidTenantSlug` after the eval had already built the case, which
    reads like an outage rather than a naming bug. Unbounded, 268 of 770 cases on the real
    extract failed that way -- a third of a calibration run.
    """

    LONG = "automation_api_dispatch_automation_with_a_very_long_tail"

    def _cases(self, event: str, project: str = "414029"):
        # Rises then falls, so comparison and superlative cases are actually generated; a
        # flat series produces none and would make every assertion here vacuous.
        return generate(
            (
                series(
                    date(2026, 1, 1),
                    90,
                    lambda i: 100 if i < 31 else (400 if i < 59 else 100),
                    event=event,
                    project=project,
                ),
            )
        )

    def test_every_generated_name_fits_the_slug_budget(self):
        for case in self._cases(self.LONG):
            assert len(case.name) <= MAX_CASE_NAME, f"{case.name} is {len(case.name)}"

    def test_the_slug_the_harness_builds_is_valid(self):
        for case in self._cases(self.LONG):
            validate_slug(f"eval-{case.name.replace('_', '-')}-{'a' * 6}")

    def test_two_events_sharing_a_prefix_do_not_collide(self):
        """Truncation without a digest gives two cases one name, and one bundle overwrites
        the other -- losing a result silently rather than noisily."""
        first = self._cases(self.LONG + "_alpha")
        second = self._cases(self.LONG + "_beta")
        assert {c.name for c in first}.isdisjoint({c.name for c in second})

    def test_a_short_name_is_left_readable(self):
        names = {c.name for c in self._cases("signups")}
        assert any(n.startswith("gen_414029_signups_") for n in names)


class TestAFallThatIsThePipelineStopping:
    """Arithmetic cannot tell a metric falling from the pipeline that measures it stopping.

    `$pageleave` fell 81% from July to August 2026, so arithmetic labelled the case `holds`.
    The analyst answered `false`, because every other event in that project fell 98-100% over
    the same days -- collection had stopped. The analyst was right and the label was wrong,
    and the report schema agrees: `false` covers a movement that is an artefact.

    The check has to be narrow. A first version dropped any movement most of the project
    shared, which also removed sixty good cases where fourteen of seventeen events rose
    together -- there the metric really did rise and the premise really does hold.
    """

    def _project(self, subject_tail: int, sibling_tail: int):
        """July healthy for everyone; August healthy until the 11th, then at the given tails."""

        def shaped(tail: int, event: str):
            days = []
            for i in range(31):
                days.append((date(2026, 7, 1) + timedelta(days=i), 7000))
            for i in range(31):
                day = date(2026, 8, 1) + timedelta(days=i)
                days.append((day, 7000 if day.day <= 11 else tail))
            return RealSeries("posthog", "p1", event, tuple(days))

        return (
            shaped(subject_tail, "subject"),
            shaped(sibling_tail, "sib_a"),
            shaped(sibling_tail, "sib_b"),
        )

    def _july_case(self, world):
        return [
            c
            for c in comparison_cases(world[0])
            if c.premise_verdict == "holds" and "202607" in c.name
        ]

    def test_a_project_wide_collapse_drops_the_case(self):
        world = self._project(subject_tail=60, sibling_tail=60)
        case = self._july_case(world)[0]
        assert collection_failed(case, world)
        assert not [c for c in generate(world) if "202607" in c.name]

    def test_one_event_collapsing_alone_is_a_real_fall_and_is_kept(self):
        """The siblings are the control group. If only this metric died, that is the answer."""
        world = self._project(subject_tail=60, sibling_tail=7000)
        assert not collection_failed(self._july_case(world)[0], world)

    def test_shared_growth_is_not_a_collapse(self):
        rising = tuple(
            RealSeries(
                "posthog",
                "p1",
                name,
                tuple(
                    (date(2026, 1, 1) + timedelta(days=i), 100 if i < 31 else 400)
                    for i in range(59)
                ),
            )
            for name in ("subject", "sib_a", "sib_b")
        )
        kept = [c for c in generate(rising) if c.premise_verdict in ("holds", "false")]
        assert kept, "platform-wide growth must not be mistaken for a dead pipeline"

    def test_a_collapse_is_read_from_the_tail_not_the_month(self):
        """A pipeline stopping mid-month leaves a month that merely looks bad.

        These series average only ~55% down across August while ending at 1% of July's rate.
        A threshold on the monthly mean cannot separate that from a steep decline.
        """
        world = self._project(subject_tail=60, sibling_tail=60)
        august = [c for d, c in world[0].days if d.month == 8]
        assert sum(august) / len(august) > 0.35 * 7000
        assert _collapsed(world[0], date(2026, 7, 1), date(2026, 8, 1))
