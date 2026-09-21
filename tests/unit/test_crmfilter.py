"""The guard over model-authored CRM filters.

Every test here is a way a filter can be wrong *and still return HTTP 200 with zero rows*,
which is the failure this module exists to convert into an error. A test that only checked
"valid input produces valid output" would pass against no guard at all.
"""

from __future__ import annotations

import pytest

from cortex.tools.base import InvalidParams
from cortex.tools.crmfilter import (
    MAX_FILTERS,
    MAX_IN_VALUES,
    describe,
    epoch_ms,
    validate,
)


class TestTheAllowlist:
    def test_a_misspelt_property_is_refused_rather_than_returning_nothing(self) -> None:
        # HubSpot answers 200 with zero results for an unknown property. Passing it
        # through would hand the analyst an empty observation indistinguishable from
        # "this portal has no such deals".
        with pytest.raises(InvalidParams) as caught:
            validate("deals", [{"property": "hs_is_closed_wonn", "value": True}])
        assert "hs_is_closed_wonn" in str(caught.value)

    def test_the_error_lists_what_is_available_so_the_retry_can_succeed(self) -> None:
        with pytest.raises(InvalidParams) as caught:
            validate("deals", [{"property": "nope", "value": 1}])
        # The analyst reads this and retries; an error it cannot act on costs a loop step.
        assert "hs_is_closed_lost" in str(caught.value)

    def test_a_property_of_another_object_is_not_borrowed(self) -> None:
        with pytest.raises(InvalidParams):
            validate("contacts", [{"property": "hs_is_closed_won", "value": True}])

    def test_an_unknown_object_type_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate("tickets", [{"property": "createdate", "value": "2026-01-01"}])

    def test_an_unknown_operator_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate("deals", [{"property": "amount", "operator": "REGEX", "value": "x"}])

    def test_an_empty_filter_list_is_refused(self) -> None:
        # An unfiltered search over a real portal is a different operation, and an
        # expensive one. It is not reachable by omission.
        with pytest.raises(InvalidParams):
            validate("deals", [])

    def test_too_many_filters_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate(
                "deals",
                [
                    {"property": "amount", "operator": "GT", "value": n}
                    for n in range(MAX_FILTERS + 1)
                ],
            )


class TestDatesAreConverted:
    def test_a_calendar_date_becomes_epoch_milliseconds(self) -> None:
        # Written as YYYY-MM-DD against a property HubSpot stores as epoch ms, the filter
        # matches nothing at all. Silently.
        out = validate(
            "deals", [{"property": "closedate", "operator": "GTE", "value": "2026-06-21"}]
        )
        assert out[0]["value"] == epoch_ms("2026-06-21")
        assert out[0]["value"].isdigit()

    def test_between_pushes_the_upper_bound_to_end_of_day(self) -> None:
        out = validate(
            "deals",
            [
                {
                    "property": "closedate",
                    "operator": "BETWEEN",
                    "value": "2026-06-21",
                    "high_value": "2026-09-21",
                }
            ],
        )
        # Without this a range ending today excludes everything that happened today.
        assert out[0]["highValue"] == epoch_ms("2026-09-21", end_of_day=True)
        assert int(out[0]["highValue"]) > int(out[0]["value"])

    def test_an_epoch_the_analyst_already_has_is_not_rewritten(self) -> None:
        already = "1750464000000"
        out = validate("deals", [{"property": "closedate", "operator": "GTE", "value": already}])
        assert out[0]["value"] == already

    def test_a_backwards_range_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate(
                "deals",
                [
                    {
                        "property": "closedate",
                        "operator": "BETWEEN",
                        "value": "2026-09-21",
                        "high_value": "2026-06-21",
                    }
                ],
            )

    @pytest.mark.parametrize(
        "value",
        [
            # Nine characters, so a "looks like a date" length check waved it through and
            # HubSpot returned 200 with nothing. That is how this test was earned.
            "June 2026",
            "2026-13-01",
            "21-09-2026",
            "2026/09/21",
            "last quarter",
            "",
        ],
    )
    def test_a_value_that_is_neither_a_date_nor_an_epoch_is_refused(self, value: str) -> None:
        with pytest.raises(InvalidParams):
            validate("deals", [{"property": "closedate", "operator": "GTE", "value": value}])

    def test_a_malformed_bound_is_refused_in_a_range_too(self) -> None:
        with pytest.raises(InvalidParams):
            validate(
                "deals",
                [
                    {
                        "property": "closedate",
                        "operator": "BETWEEN",
                        "value": "2026-06-21",
                        "high_value": "whenever",
                    }
                ],
            )

    def test_a_backwards_range_is_caught_across_mixed_forms(self) -> None:
        # A calendar date against an epoch: string comparison ordered these wrongly, so the
        # interval was accepted backwards and matched nothing.
        with pytest.raises(InvalidParams):
            validate(
                "deals",
                [
                    {
                        "property": "closedate",
                        "operator": "BETWEEN",
                        "value": "2026-09-21",
                        "high_value": epoch_ms("2026-06-21"),
                    }
                ],
            )


class TestValueShapes:
    def test_between_without_an_upper_bound_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate("deals", [{"property": "amount", "operator": "BETWEEN", "value": 100}])

    def test_in_with_a_scalar_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate("deals", [{"property": "dealstage", "operator": "IN", "value": "won"}])

    def test_in_with_an_empty_list_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate("deals", [{"property": "dealstage", "operator": "IN", "values": []}])

    def test_too_many_in_values_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate(
                "deals",
                [
                    {
                        "property": "dealstage",
                        "operator": "IN",
                        "values": [str(n) for n in range(MAX_IN_VALUES + 1)],
                    }
                ],
            )

    def test_an_existence_check_takes_no_value(self) -> None:
        out = validate("deals", [{"property": "hubspot_owner_id", "operator": "HAS_PROPERTY"}])
        assert out == [{"propertyName": "hubspot_owner_id", "operator": "HAS_PROPERTY"}]
        with pytest.raises(InvalidParams):
            validate(
                "deals",
                [{"property": "hubspot_owner_id", "operator": "HAS_PROPERTY", "value": "x"}],
            )

    def test_a_boolean_becomes_hubspots_lowercase_string(self) -> None:
        # `str(True)` is "True", which HubSpot does not recognise. bool is also an int in
        # Python, so the order of the isinstance checks is load-bearing.
        out = validate("deals", [{"property": "hs_is_closed_won", "value": True}])
        assert out[0]["value"] == "true"
        out = validate("deals", [{"property": "hs_is_closed_won", "value": False}])
        assert out[0]["value"] == "false"

    def test_a_number_becomes_a_string(self) -> None:
        out = validate("deals", [{"property": "amount", "operator": "GTE", "value": 1000}])
        assert out[0]["value"] == "1000"

    def test_a_nested_object_as_a_value_is_refused(self) -> None:
        with pytest.raises(InvalidParams):
            validate("deals", [{"property": "amount", "operator": "EQ", "value": {"$ne": 1}}])

    def test_the_default_operator_is_equality(self) -> None:
        out = validate("deals", [{"property": "dealtype", "value": "newbusiness"}])
        assert out[0]["operator"] == "EQ"


class TestTheQuestionThatPromptedThis:
    """A closed-won rate, which the five named capabilities could not express."""

    def test_both_halves_of_a_win_rate_are_reachable(self) -> None:
        period = {"operator": "BETWEEN", "value": "2026-06-21", "high_value": "2026-09-21"}
        won = validate(
            "deals",
            [{"property": "hs_is_closed_won", "value": True}, {"property": "closedate", **period}],
        )
        lost = validate(
            "deals",
            [{"property": "hs_is_closed_lost", "value": True}, {"property": "closedate", **period}],
        )
        assert won[0] == {"propertyName": "hs_is_closed_won", "operator": "EQ", "value": "true"}
        assert lost[0] == {"propertyName": "hs_is_closed_lost", "operator": "EQ", "value": "true"}
        # The denominator is only meaningful if both halves cover the same window.
        assert won[1]["value"] == lost[1]["value"]
        assert won[1]["highValue"] == lost[1]["highValue"]


class TestTheSourceRef:
    def test_the_same_search_renders_the_same_reference_whatever_the_argument_order(
        self,
    ) -> None:
        # A source_ref is what a reader clicks to check a claim. Two identical searches
        # that render differently would look like two different pieces of evidence.
        a = describe(
            "deals",
            [
                {"property": "hs_is_closed_won", "value": True},
                {
                    "property": "closedate",
                    "operator": "BETWEEN",
                    "value": "2026-06-21",
                    "high_value": "2026-09-21",
                },
            ],
        )
        b = describe(
            "deals",
            [
                {
                    "property": "closedate",
                    "operator": "BETWEEN",
                    "value": "2026-06-21",
                    "high_value": "2026-09-21",
                },
                {"property": "hs_is_closed_won", "value": True},
            ],
        )
        assert a == b

    def test_the_reference_shows_dates_a_human_can_read(self) -> None:
        ref = describe(
            "deals",
            [
                {
                    "property": "closedate",
                    "operator": "BETWEEN",
                    "value": "2026-06-21",
                    "high_value": "2026-09-21",
                }
            ],
        )
        # Not the thirteen-digit epochs the API actually receives.
        assert "closedate=2026-06-21..2026-09-21" in ref
        assert "1750" not in ref

    def test_different_searches_do_not_collide(self) -> None:
        won = describe("deals", [{"property": "hs_is_closed_won", "value": True}])
        lost = describe("deals", [{"property": "hs_is_closed_lost", "value": True}])
        assert won != lost
