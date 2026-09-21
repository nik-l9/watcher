"""Validation for model-authored CRM filters.

A fixed set of named capabilities cannot answer questions nobody anticipated. HubSpot's
connector shipped five — `pipeline`, `closed_won`, `contacts`, `companies`, `activities` —
each a thin wrapper that hard-codes a filter list over one general search endpoint. A live
investigation then asked for the closed-won *rate*, which needs lost deals, and there was no
capability for those. The analyst did the honest thing and disclosed the gap, but the number
it led with divided a three-month flow by a point-in-time stock, because that was the only
denominator its tools could reach.

That is not a missing capability. It is the cross-product of every field against every
question, and hand-writing it never converges: a rate needs lost deals, "by rep" needs owner,
"by source" needs attribution. So the analyst authors the filter, and the job of this module
is to make that safe enough — and, just as importantly, *legible* enough — to point at a real
portal.

**Why a guard at all, when HubSpot would reject a bad filter itself.** It mostly would not.
An unknown `propertyName` does not error; the search returns zero results. So does a date
filter written as `2026-06-21` against a property HubSpot stores in epoch milliseconds. Both
hand the analyst an empty result indistinguishable from a real absence, which is the bug this
codebase has now shipped four times and which `Capability.result_key` exists to stop. A guard
that refuses an unknown property is not being strict for its own sake — it is converting a
silent wrong answer into a loud error.

Three properties are enforced:

  - **Known properties only.** Per object type, from an allowlist. An unknown property is
    refused rather than passed through to return nothing.
  - **Known operators only, with the value shape each one requires.** `BETWEEN` without a
    second bound, or `IN` with a scalar, are refused here rather than silently matching
    nothing upstream.
  - **Dates are converted, not trusted.** A date property filtered with `YYYY-MM-DD` is
    rewritten to epoch milliseconds, and a `BETWEEN` upper bound is pushed to end-of-day so
    a range ending today does not exclude today.

Bounds on filter count and `IN` list length are here for the same reason `_MAX_PAGES` is in
the connector: a model-authored filter can be arbitrarily large, and a request that cannot
finish is worse than one that is refused.

This module knows nothing about HubSpot's client, only its filter grammar, so it is
importable from the connector without a cycle and testable without a network.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from typing import Any

from cortex.tools.base import InvalidParams

#: Object types a tenant may search. Deliberately not every CRM object: each entry needs a
#: property allowlist below, and an object with no allowlist would admit anything.
OBJECT_TYPES = ("deals", "contacts", "companies")

#: Properties that may appear in a filter, per object type.
#:
#: Wider than the properties each capability *returns*, because the useful filters are on
#: fields nobody wants in the output -- `hs_is_closed_lost` is the whole point of this module
#: and appears in no result payload. Narrower than HubSpot's full schema, because an
#: allowlist that admits anything cannot turn a typo into an error.
FILTERABLE: dict[str, frozenset[str]] = {
    "deals": frozenset(
        {
            "dealname",
            "amount",
            "dealstage",
            "pipeline",
            "closedate",
            "createdate",
            "hs_lastmodifieddate",
            "hs_deal_stage_probability",
            "hubspot_owner_id",
            "dealtype",
            "hs_analytics_source",
            "hs_analytics_source_data_1",
            "num_associated_contacts",
            # The three that make win rate computable. HubSpot maintains these itself from
            # the stage, so they are reliable in a way that matching stage labels is not:
            # a portal can rename "Closed Won" and these keep working.
            "hs_is_closed",
            "hs_is_closed_won",
            "hs_is_closed_lost",
        }
    ),
    "contacts": frozenset(
        {
            "email",
            "firstname",
            "lastname",
            "company",
            "jobtitle",
            "lifecyclestage",
            "hs_lead_status",
            "createdate",
            "lastmodifieddate",
            "hubspot_owner_id",
        }
    ),
    "companies": frozenset(
        {
            "name",
            "domain",
            "industry",
            "numberofemployees",
            "annualrevenue",
            "lifecyclestage",
            "createdate",
            "country",
            "hubspot_owner_id",
        }
    ),
}

#: Properties HubSpot stores as epoch milliseconds. A filter on one of these written as a
#: calendar date matches nothing, silently, which is why the conversion is compulsory rather
#: than offered.
DATE_PROPERTIES = frozenset({"closedate", "createdate", "hs_lastmodifieddate", "lastmodifieddate"})

#: Operators, mapped to the value shape each requires.
#:
#: "one"  -- a scalar in `value`
#: "two"  -- `value` and `high_value`, a closed interval
#: "list" -- a non-empty list in `values`
#: "none" -- an existence check, no value at all
OPERATORS: dict[str, str] = {
    "EQ": "one",
    "NEQ": "one",
    "LT": "one",
    "LTE": "one",
    "GT": "one",
    "GTE": "one",
    "CONTAINS_TOKEN": "one",
    "NOT_CONTAINS_TOKEN": "one",
    "BETWEEN": "two",
    "IN": "list",
    "NOT_IN": "list",
    "HAS_PROPERTY": "none",
    "NOT_HAS_PROPERTY": "none",
}

#: A filter list longer than this is a sign the analyst is enumerating rather than filtering.
MAX_FILTERS = 8

#: HubSpot's own ceiling on `IN` values is higher, but a list this long in a generated filter
#: is a pasted export, not a query.
MAX_IN_VALUES = 50


def epoch_ms(date: str, *, end_of_day: bool = False) -> str:
    """A calendar date as HubSpot's epoch milliseconds.

    `end_of_day` makes a BETWEEN upper bound inclusive of the final day; without it a range
    ending today silently excludes everything that happened today.
    """
    try:
        year, month, day = (int(part) for part in date.split("-"))
        moment = datetime.combine(
            datetime(year, month, day, tzinfo=UTC).date(),
            time.max if end_of_day else time.min,
            tzinfo=UTC,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidParams(f"{date!r} is not a date of the form YYYY-MM-DD") from exc
    return str(int(moment.timestamp() * 1000))


def _date_value(value: Any, *, prop: str, end_of_day: bool = False) -> str:
    """A filter value for an epoch-milliseconds property: a date, or an epoch already.

    **Exactly two forms are admitted, and everything else is refused.** An earlier version
    asked whether the value *looked like* a date -- ten characters with dashes in the right
    places -- and passed anything else straight through. `"June 2026"` is nine characters,
    so it went to HubSpot verbatim, matched nothing, and returned 200 with an empty list:
    the silent-empty bug this module was written to prevent, reintroduced by the check
    meant to prevent it. A heuristic that falls back to "pass it on" cannot guard anything.
    """
    text = str(value)
    if text.isdigit():
        # An epoch the analyst already holds -- from a previous observation, say. Rewriting
        # it would corrupt it, so it is taken as given.
        return text
    return epoch_ms(text, end_of_day=end_of_day)


def _scalar(value: Any, *, operator: str, prop: str) -> str:
    """HubSpot compares against strings, including for numbers and booleans."""
    if isinstance(value, bool):
        # Before the numeric branch: bool is an int in Python, and `str(True)` is "True",
        # which HubSpot does not recognise for its boolean properties.
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    raise InvalidParams(
        f"filter on {prop} with operator {operator} needs a string, number or boolean, "
        f"not {type(value).__name__}"
    )


def validate(object_type: str, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Model-authored filters as HubSpot search filters, or `InvalidParams`.

    Returns a new list; the input is not modified. Every message names the offending
    property and what was expected, because the analyst reads the error and retries, and an
    error it cannot act on costs a whole loop step.
    """
    if object_type not in OBJECT_TYPES:
        raise InvalidParams(
            f"{object_type!r} is not a searchable object; choose one of {', '.join(OBJECT_TYPES)}"
        )
    if not isinstance(filters, list) or not filters:
        raise InvalidParams("filters must be a non-empty list")
    if len(filters) > MAX_FILTERS:
        raise InvalidParams(f"at most {MAX_FILTERS} filters, got {len(filters)}")

    allowed = FILTERABLE[object_type]
    out: list[dict[str, Any]] = []

    for index, raw in enumerate(filters):
        if not isinstance(raw, dict):
            raise InvalidParams(f"filter {index} must be an object")

        prop = raw.get("property")
        if prop not in allowed:
            raise InvalidParams(
                f"{prop!r} is not a filterable property of {object_type}. "
                f"Available: {', '.join(sorted(allowed))}"
            )

        operator = raw.get("operator", "EQ")
        shape = OPERATORS.get(operator)
        if shape is None:
            raise InvalidParams(
                f"{operator!r} is not a supported operator; choose one of "
                f"{', '.join(sorted(OPERATORS))}"
            )

        built: dict[str, Any] = {"propertyName": prop, "operator": operator}
        is_date = prop in DATE_PROPERTIES

        if shape == "none":
            if "value" in raw or "values" in raw or "high_value" in raw:
                raise InvalidParams(f"{operator} on {prop} takes no value")

        elif shape == "two":
            if raw.get("value") is None or raw.get("high_value") is None:
                raise InvalidParams(
                    f"BETWEEN on {prop} needs both value and high_value "
                    "(a closed interval, both bounds inclusive)"
                )
            low, high = raw["value"], raw["high_value"]
            if is_date:
                built["value"] = _date_value(low, prop=prop)
                built["highValue"] = _date_value(high, prop=prop, end_of_day=True)
                # Compared after conversion: comparing the raw values ordered correctly
                # only when both were calendar dates, and a date against an epoch would
                # have silently accepted a backwards interval.
                if int(built["value"]) > int(built["highValue"]):
                    raise InvalidParams(f"BETWEEN on {prop}: {low} is after {high}")
            else:
                built["value"] = _scalar(low, operator=operator, prop=prop)
                built["highValue"] = _scalar(high, operator=operator, prop=prop)

        elif shape == "list":
            values = raw.get("values")
            if not isinstance(values, list) or not values:
                raise InvalidParams(f"{operator} on {prop} needs a non-empty values list")
            if len(values) > MAX_IN_VALUES:
                raise InvalidParams(
                    f"{operator} on {prop}: at most {MAX_IN_VALUES} values, got {len(values)}"
                )
            built["values"] = [
                _date_value(v, prop=prop) if is_date else _scalar(v, operator=operator, prop=prop)
                for v in values
            ]

        else:  # "one"
            if "value" not in raw or raw["value"] is None:
                raise InvalidParams(f"{operator} on {prop} needs a value")
            value = raw["value"]
            if is_date:
                built["value"] = _date_value(value, prop=prop)
            else:
                built["value"] = _scalar(value, operator=operator, prop=prop)

        out.append(built)

    return out


def describe(object_type: str, filters: list[dict[str, Any]]) -> str:
    """A stable, human-followable rendering of a search, for `source_ref`.

    **Why not the raw filter JSON.** A `source_ref` is the thing a reader clicks to check a
    claim, so it has to be the same string for the same search however the analyst happened
    to order its arguments. Filters are therefore sorted, and the epoch milliseconds are
    *not* used -- the caller passes the filters as written, so the reference reads
    `closedate=2026-06-21..2026-09-21` rather than a pair of thirteen-digit integers nobody
    can check by eye.
    """

    def show(value: Any) -> str:
        # `str(True)` is "True", which is neither what HubSpot receives nor what a reader
        # would type to reproduce the search. The reference has to be followable.
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    parts: list[str] = []
    for raw in filters:
        prop = raw.get("property")
        operator = raw.get("operator", "EQ")
        if operator == "BETWEEN":
            parts.append(f"{prop}={show(raw.get('value'))}..{show(raw.get('high_value'))}")
        elif operator in ("IN", "NOT_IN"):
            joined = ",".join(show(v) for v in raw.get("values") or [])
            parts.append(f"{prop}{'!' if operator == 'NOT_IN' else ''}in[{joined}]")
        elif operator in ("HAS_PROPERTY", "NOT_HAS_PROPERTY"):
            parts.append(f"{'!' if operator.startswith('NOT') else ''}{prop}?")
        elif operator == "EQ":
            parts.append(f"{prop}={show(raw.get('value'))}")
        else:
            parts.append(f"{prop}[{operator}]={show(raw.get('value'))}")
    return f"hubspot://crm/v3/objects/{object_type}/search?" + "&".join(sorted(parts))
