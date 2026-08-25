"""The analysis layer as a connector emits it.

Computed inside `event_trend` rather than left to the investigation, for the reason seven
previous fixes established: a fact the analyst has to make a second call to obtain is a fact it
will not obtain. The series and the description of how it moved travel in one payload.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from cortex.analysis.movement import MIN_DAYS, describe_movement
from tests.analysis.conftest import WEEKDAY_MULTIPLIER, replica_rows


def _rows(rows: dict[date, float] | None = None) -> list[dict[str, object]]:
    source = replica_rows() if rows is None else rows
    return [
        {"bucket": f"{on.isoformat()}T00:00:00Z", "value": value}
        for on, value in sorted(source.items())
    ]


def _describe(**overrides):
    kwargs = {
        "rows": _rows(),
        "start_date": "2026-05-06",
        "end_date": "2026-08-17",
        "interval": "day",
    }
    kwargs.update(overrides)
    return describe_movement(kwargs.pop("rows"), **kwargs)


class TestItDescribesTheRealMovement:
    def test_both_level_shifts_are_reported_with_their_magnitudes(self) -> None:
        shifts = _describe()["movement"]["level_shifts"]
        assert [s["at"] for s in shifts] == ["2026-06-17", "2026-07-15"]
        first = shifts[0]
        assert first["change_per_day"] < -100
        assert first["change_pct"] < -50

    def test_each_shift_carries_its_significance_and_its_floor(self) -> None:
        first = _describe()["movement"]["level_shifts"][0]
        assert first["p_value"] == 0.0143
        assert first["p_floor"] == 0.0143
        assert first["p_at_floor"] is True
        assert first["established"] is True
        assert first["window_days"] == 70

    def test_the_noise_scale_is_reported_so_a_reader_can_judge_the_size(self) -> None:
        assert _describe()["movement"]["noise_scale_per_day"] > 0


class TestTheNoteRefusesToImplyACause:
    def test_it_says_outright_that_this_establishes_only_when(self) -> None:
        note = _describe()["movement_note"]
        assert "establishes when the series moved, and nothing about why" in note
        assert "Do not attribute any of these shifts to a cause" in note

    def test_it_names_the_three_things_a_causal_claim_needs(self) -> None:
        note = _describe()["movement_note"]
        for requirement in ("a named change", "a comparison series", "what else happened"):
            assert requirement in note

    def test_it_states_the_elimination_rule(self) -> None:
        """The one line from ADR 0005 that would have killed the wrong answer: a shift dated
        before a candidate cause rules that cause out."""
        note = _describe()["movement_note"]
        assert "precedes a candidate cause's own date rules that cause out" in note

    def test_it_explains_what_a_floored_p_means(self) -> None:
        """ "p = 0.0143" and "p = 0.0143, at the floor" support different sentences, and the
        second cannot be compared against a p from a longer window."""
        note = _describe()["movement_note"]
        assert "as significant as this window can resolve" in note

    def test_it_declares_the_duration_it_cannot_see(self) -> None:
        note = _describe()["movement_note"]
        assert "Nothing shorter than 14 days can appear here" in note


class TestFindingNothingIsAlsoAFinding:
    def test_a_flat_series_says_so_rather_than_omitting_the_field(self) -> None:
        """ "We looked for a level shift and found none" is what stops a reader reading noise
        as a trend."""
        generator = random.Random(5)
        flat = {
            date(2026, 5, 6) + timedelta(days=offset): 200.0
            * WEEKDAY_MULTIPLIER[(date(2026, 5, 6) + timedelta(days=offset)).weekday()]
            + generator.gauss(0, 10)
            for offset in range(90)
        }
        result = _describe(rows=_rows(flat), end_date="2026-08-03")
        assert result["movement"]["level_shifts"] == []
        assert "No sustained level shift was found" in result["movement_note"]
        assert "an incident rather than a level shift" in result["movement_note"]


class TestItRefusesWhereItIsNotCalibrated:
    """An absent field is honest. A field saying "not applicable" is noise on every ordinary
    call, and a disclosure that appears everywhere stops being read on the one that needed it.
    """

    def test_a_weekly_series_gets_nothing(self) -> None:
        """`min_len` is a count of days tuned against a weekly cycle. A weekly series has
        neither the resolution nor the seasonality, and running anyway would produce a
        confident answer from an uncalibrated method."""
        assert _describe(interval="week") is None
        assert _describe(interval="month") is None

    def test_a_broken_down_series_gets_nothing(self) -> None:
        """A breakdown is several series interleaved, not one."""
        assert _describe(breakdown_property="$current_url") is None

    def test_too_few_days_gets_nothing(self) -> None:
        short = {date(2026, 5, 6) + timedelta(days=o): 100.0 for o in range(MIN_DAYS - 1)}
        assert _describe(rows=_rows(short), end_date="2026-06-01") is None

    def test_an_unparseable_bucket_gets_nothing_rather_than_a_guess(self) -> None:
        rows = _rows()
        rows[3] = {"bucket": "not-a-date", "value": 10}
        assert _describe(rows=rows) is None

    def test_a_non_numeric_value_gets_nothing(self) -> None:
        rows = _rows()
        rows[3] = {"bucket": "2026-05-09T00:00:00Z", "value": "many"}
        assert _describe(rows=rows) is None

    def test_an_inverted_range_gets_nothing_rather_than_raising(self) -> None:
        assert _describe(start_date="2026-08-17", end_date="2026-05-06") is None

    def test_a_missing_bucket_key_gets_nothing(self) -> None:
        assert _describe(rows=[{"value": 1} for _ in range(MIN_DAYS + 5)]) is None


class TestTheGapDoesNotCorruptTheAnalysis:
    def test_a_trailing_outage_does_not_become_a_level_shift(self) -> None:
        """The range runs fourteen days past the last row. That is a freshness fact, already
        reported by `series_ends_early`, and filing it as a level shift would put it in the
        same category as a real movement."""
        shifts = _describe()["movement"]["level_shifts"]
        assert "2026-08-04" not in [s["at"] for s in shifts]
        assert len(shifts) == 2
