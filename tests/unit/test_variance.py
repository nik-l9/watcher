"""Telling a change from the dice.

Written because every claim made about an intervention in this project has been undermined by
variance. Three runs of one scenario produced overall 0.86, 0.89 and a third figure again, with
`tool_selection` at 1.00 then 0.50, on unchanged code. A FAIL became a PASS between two runs and
was reported as an intervention working; the captured bundle showed the intervention had not
fired at all.

The arithmetic is tested here. `measure_spread` itself needs bundles and a session, and is
exercised by `TestTheVarianceInstrumentActuallyRuns` in `tests/db/test_eval.py`.

That claim used to be made here without being true, and it cost the module its only coverage:
`replay_bundle` grew a sixth return value when the sufficiency gate landed, `measure_spread`
still unpacked five, and every `--variance` invocation died on the tuple unpack for as long as
anybody wanted the numbers. A comment asserting coverage that does not exist is worse than no
comment, because it stops the next reader looking.
"""

from __future__ import annotations

import pytest

from cortex.eval.variance import ALPHA, POWER, DimensionSpread, Spread, render_spread


def _dimension(
    name: str = "accuracy",
    *,
    sigma_attempt: float = 0.1,
    sigma_scenario: float = 0.0,
    scenarios: int = 5,
    stable: int = 0,
    mean: float = 0.8,
    attempts: int = 15,
) -> DimensionSpread:
    return DimensionSpread(
        name=name,
        mean=mean,
        sigma_attempt=sigma_attempt,
        sigma_scenario=sigma_scenario,
        stable_scenarios=stable,
        scenarios=scenarios,
        attempts=attempts,
    )


class TestTheDetectableDifference:
    def test_more_scenarios_detect_a_smaller_shift(self) -> None:
        """The only lever available without reducing the noise itself."""
        few = _dimension(scenarios=4).paired_mde
        many = _dimension(scenarios=16).paired_mde
        assert many < few
        # Halving the MDE takes four times the scenarios, which is the sqrt in the formula and
        # the reason "add a couple more scenarios" is not a plan.
        assert few / many == pytest.approx(2.0, rel=0.01)

    def test_a_noisier_dimension_needs_a_bigger_shift(self) -> None:
        assert _dimension(sigma_attempt=0.2).paired_mde > _dimension(sigma_attempt=0.05).paired_mde

    def test_a_deterministic_dimension_can_detect_anything(self) -> None:
        """Zero variance means any difference is real, so the floor is zero rather than a small
        number that would imply a threshold where there is none."""
        dimension = _dimension(sigma_attempt=0.0)
        assert dimension.deterministic
        assert dimension.paired_mde == 0.0

    def test_no_scenarios_means_no_answer_rather_than_a_division_error(self) -> None:
        assert _dimension(scenarios=0).paired_mde == 0.0

    def test_the_coefficient_matches_the_stated_confidence_and_power(self) -> None:
        """Pinned so a later edit to ALPHA or POWER cannot silently change every published MDE
        while the docstring keeps claiming the old numbers."""
        from cortex.analysis.identifiability import _z

        dimension = _dimension(sigma_attempt=0.1, scenarios=8)
        expected = (_z(1 - ALPHA / 2) + _z(POWER)) * 0.1 * (2 / 8) ** 0.5
        assert dimension.paired_mde == pytest.approx(expected)
        assert (ALPHA, POWER) == (0.05, 0.80)


class TestTheReportSaysWhatToDoWithIt:
    def test_it_names_the_noisiest_dimension_and_its_floor(self) -> None:
        spread = Spread(
            dimensions=(
                _dimension("accuracy", sigma_attempt=0.02),
                _dimension("tool_selection", sigma_attempt=0.25),
            ),
            scenarios=("a", "b", "c", "d", "e"),
            attempts_per_scenario={"a": 3, "b": 3, "c": 3, "d": 3, "e": 3},
        )
        rendered = render_spread(spread)
        assert "Noisiest: tool_selection" in rendered
        assert "Anything smaller is the dice." in rendered

    def test_it_lists_the_dimensions_that_never_vary(self) -> None:
        """Worth saying explicitly: on these, a difference of any size is a real difference, and
        a reader who assumes everything is noisy will discard a true result."""
        spread = Spread(
            dimensions=(
                _dimension("grounding", sigma_attempt=0.0, stable=5),
                _dimension("accuracy", sigma_attempt=0.1),
            ),
            scenarios=("a",),
            attempts_per_scenario={"a": 3},
        )
        rendered = render_spread(spread)
        assert "Deterministic across every attempt: grounding" in rendered
        assert "always real" in rendered

    def test_it_explains_that_scenario_spread_is_not_noise(self) -> None:
        """The distinction that decides whether a comparison is worth anything. Treating
        `sig_scen` as noise leads to averaging it away; it is difficulty, and the fix is pairing
        rather than more attempts."""
        rendered = render_spread(
            Spread(
                dimensions=(_dimension(sigma_scenario=0.2),),
                scenarios=("a", "b"),
                attempts_per_scenario={"a": 3, "b": 3},
            )
        )
        assert "scenario difficulty, not noise" in rendered
        assert "must be paired" in rendered

    def test_an_empty_directory_says_what_to_do(self) -> None:
        rendered = render_spread(Spread(dimensions=(), scenarios=(), attempts_per_scenario={}))
        assert "--repeat" in rendered
        assert "single attempt per scenario has no spread" in rendered

    def test_dimensions_are_ordered_noisiest_first(self) -> None:
        """The only order anyone reads this in."""
        spread = Spread(
            dimensions=(
                _dimension("quiet", sigma_attempt=0.01),
                _dimension("loud", sigma_attempt=0.3),
                _dimension("middling", sigma_attempt=0.1),
            ),
            scenarios=("a",),
            attempts_per_scenario={"a": 3},
        )
        rendered = render_spread(spread)
        assert rendered.index("loud") < rendered.index("middling") < rendered.index("quiet")


class TestTheLayerThatPredictsWhetherTheAnswerIsRight:
    """Route stability, measured because the literature says it is the layer that matters.

    *How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step
    Tool-Calling Pipelines* (arXiv 2605.28840) measures three layers of consistency and finds
    only one carries signal: attempts whose tool sequences agreed were 90.2% correct against
    61.2% for those that did not (d = 0.81), while argument variance predicted nothing
    (r = 0.12, n.s.) and *final wording matched under 5% of the time even when the route was
    identical*.

    That last figure is why this class exists. "It gives a different answer every time" is the
    complaint that starts a reliability investigation, and prose variation is the expected
    behaviour of the thing rather than evidence against it. What has to be measured instead is
    whether the route was stable and whether the conclusion was right.
    """

    def test_the_survey_prefix_is_not_counted_as_a_decision(self) -> None:
        """Every investigation opens by calling each discovery capability. It is the loop's
        behaviour, not the model's choice, so counting it adds the same constant to every
        similarity score and hides what is being measured."""
        from cortex.eval.variance import _trajectory

        calls = [
            {"tool_name": "github", "capability": "list_repositories"},
            {"tool_name": "posthog", "capability": "list_events"},
            {"tool_name": "posthog", "capability": "list_projects"},
            {"tool_name": "ga4", "capability": "get_funnel"},
            {"tool_name": "github", "capability": "commits"},
        ]
        assert _trajectory(calls) == ["ga4__get_funnel", "github__commits"]

    def test_a_listing_asked_for_later_is_a_decision(self) -> None:
        """Only the opening run is the survey. An analyst that goes back to the catalogue
        mid-investigation has chosen to, and that choice is part of the route."""
        from cortex.eval.variance import _trajectory

        calls = [
            {"tool_name": "posthog", "capability": "list_events"},
            {"tool_name": "posthog", "capability": "event_trend"},
            {"tool_name": "posthog", "capability": "list_events"},
        ]
        assert _trajectory(calls) == ["posthog__event_trend", "posthog__list_events"]

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (["a", "b"], ["a", "b"], 1.0),
            (["a", "b"], ["a", "c"], 0.5),
            (["a", "b"], ["b", "a"], 0.0),
            (["a", "b"], ["a"], 0.5),
            ([], [], 1.0),
            ([], ["a"], 0.0),
        ],
    )
    def test_sequence_similarity_is_order_sensitive(
        self, left: list[str], right: list[str], expected: float
    ) -> None:
        """Order matters, and that is the point of using edit distance rather than set overlap:
        two attempts that called the same tools in a different order took different routes."""
        from cortex.eval.variance import _sequence_similarity

        assert _sequence_similarity(left, right) == pytest.approx(expected)

    def test_argument_similarity_is_set_overlap_on_key_value_pairs(self) -> None:
        from cortex.eval.variance import _argument_similarity

        assert _argument_similarity({"repo": "a"}, {"repo": "a"}) == 1.0
        assert _argument_similarity({"repo": "a"}, {"repo": "b"}) == 0.0
        assert _argument_similarity({}, {}) == 1.0
        assert _argument_similarity(
            {"repo": "a", "since": "2026-06-01"}, {"repo": "a"}
        ) == pytest.approx(0.5)

    def test_the_split_replicates_the_papers_comparison(self) -> None:
        """One route against several, which is the comparison arXiv 2605.28840 makes. Returned
        rather than asserted, so a run where every scenario took one route says so instead of
        inventing a contrast."""
        from cortex.eval.variance import TrajectorySpread

        def _t(scenario: str, routes: int, accuracy: float) -> TrajectorySpread:
            return TrajectorySpread(
                scenario=scenario,
                attempts=5,
                tool_sequence_similarity=1.0 if routes == 1 else 0.4,
                argument_consistency=1.0,
                distinct_routes=routes,
                accuracy_rate=accuracy,
                unanimous=accuracy in (0.0, 1.0),
            )

        spread = Spread(
            dimensions=(),
            scenarios=("a", "b"),
            attempts_per_scenario={"a": 5, "b": 5},
            trajectories=(_t("a", 1, 1.0), _t("b", 3, 0.6)),
        )
        assert spread.route_accuracy_split == (1.0, 0.6)

    def test_no_contrast_is_reported_when_every_route_was_stable(self) -> None:
        from cortex.eval.variance import TrajectorySpread

        spread = Spread(
            dimensions=(),
            scenarios=("a",),
            attempts_per_scenario={"a": 5},
            trajectories=(
                TrajectorySpread(
                    scenario="a",
                    attempts=5,
                    tool_sequence_similarity=1.0,
                    argument_consistency=1.0,
                    distinct_routes=1,
                    accuracy_rate=1.0,
                    unanimous=True,
                ),
            ),
        )
        assert spread.route_accuracy_split is None

    def test_the_report_leads_with_disagreement_about_the_answer(self) -> None:
        """The number a reader asking "why does it answer differently each time" needs, stated
        as such: a scenario that reached different answers, not one that used different words."""
        from cortex.eval.variance import TrajectorySpread

        spread = Spread(
            dimensions=(_dimension(),),
            scenarios=("wobbly",),
            attempts_per_scenario={"wobbly": 5},
            trajectories=(
                TrajectorySpread(
                    scenario="wobbly",
                    attempts=5,
                    tool_sequence_similarity=0.42,
                    argument_consistency=0.9,
                    distinct_routes=4,
                    accuracy_rate=0.6,
                    unanimous=False,
                ),
            ),
        )
        report = render_spread(spread)
        assert "Disagreed with itself about the answer: wobbly (60% right)" in report
        assert "attempts reaching different answers are not" in report

    def test_it_says_so_when_every_attempt_agreed(self) -> None:
        from cortex.eval.variance import TrajectorySpread

        spread = Spread(
            dimensions=(_dimension(),),
            scenarios=("steady",),
            attempts_per_scenario={"steady": 5},
            trajectories=(
                TrajectorySpread(
                    scenario="steady",
                    attempts=5,
                    tool_sequence_similarity=0.5,
                    argument_consistency=1.0,
                    distinct_routes=3,
                    accuracy_rate=1.0,
                    unanimous=True,
                ),
            ),
        )
        report = render_spread(spread)
        assert "reached the same answer on every attempt" in report
        assert "neither is a defect on its own" in report
