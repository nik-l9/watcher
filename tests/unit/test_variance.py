"""Telling a change from the dice.

Written because every claim made about an intervention in this project has been undermined by
variance. Three runs of one scenario produced overall 0.86, 0.89 and a third figure again, with
`tool_selection` at 1.00 then 0.50, on unchanged code. A FAIL became a PASS between two runs and
was reported as an intervention working; the captured bundle showed the intervention had not
fired at all.

The arithmetic is tested here. `measure_spread` itself needs bundles and a session and is
exercised in `tests/db/`.
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
