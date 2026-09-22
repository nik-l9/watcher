"""What the verifier is shown when a payload is too large to show whole.

A payload of sixty deals does not fit in the budget, so something is cut. Which part gets
cut decides whether a claim about a total can be judged at all, and for a while the answer
was "the total".
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any

from cortex.reports.verifier import render_evidence


@dataclass
class _Row:
    """The fields `render_evidence` reads, without a database."""

    payload: dict[str, Any]
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    tool_name: str = "hubspot"
    capability: str = "search"
    params: dict[str, Any] = field(default_factory=dict)
    observed_at: datetime.datetime | None = None
    from_cache: bool = False


def _deals(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": str(1000 + n),
            "name": f"Acme subsidiary {n} renewal negotiation",
            "amount": 50_000,
            "stage": "closedlost",
            "close_date": "2026-08-01T00:00:00Z",
            "owner": "A Rep With A Reasonably Long Name",
        }
        for n in range(count)
    ]


class TestTruncationCutsRowsNotTotals:
    def test_the_aggregate_survives_a_payload_far_over_budget(self) -> None:
        # The regression: keys were rendered alphabetically, so `records` came before
        # `total_matching` and the cut landed inside the rows. Asked whether "9 won out of
        # 71 closed" held, the verifier answered that the evidence never showed 62 lost
        # deals -- true, because it had been cut off before reaching the number.
        row = _Row(payload={"records": _deals(62), "total_matching": 62, "returned": 62})
        [block] = render_evidence([row], max_chars=600)
        assert "total_matching" in block
        assert "62" in block

    def test_the_rows_are_what_gets_cut(self) -> None:
        row = _Row(payload={"records": _deals(62), "total_matching": 62})
        [block] = render_evidence([row], max_chars=600)
        # Not every deal can fit in 600 characters; that is the point of the budget.
        assert block.count("Acme subsidiary") < 62
        assert "TRUNCATED" in block

    def test_every_scalar_survives_not_only_the_first(self) -> None:
        row = _Row(
            payload={
                "records": _deals(80),
                "total_matching": 80,
                "returned": 80,
                "truncated": False,
                "object_type": "deals",
                "total_amount": 4_000_000,
            }
        )
        [block] = render_evidence([row], max_chars=700)
        for key in ("total_matching", "returned", "truncated", "object_type", "total_amount"):
            assert key in block, key

    def test_a_payload_within_budget_is_not_marked_truncated(self) -> None:
        row = _Row(payload={"total_matching": 2, "records": _deals(2)})
        [block] = render_evidence([row], max_chars=100_000)
        assert "TRUNCATED" not in block
        assert block.count("Acme subsidiary") == 2

    def test_the_rendering_is_deterministic(self) -> None:
        # Two renderings of one payload must be byte-identical: a verifier that read
        # different text on a second reading would disagree with itself for no reason,
        # and the second reading exists precisely to break ties.
        payload = {"records": _deals(30), "total_matching": 30, "object_type": "deals"}
        first = render_evidence([_Row(payload=dict(payload))], max_chars=500)[0]
        second = render_evidence([_Row(payload=dict(payload))], max_chars=500)[0]
        assert first.split("---", 2)[-1] == second.split("---", 2)[-1]

    def test_a_payload_with_no_collections_is_unchanged_in_content(self) -> None:
        row = _Row(payload={"b": 2, "a": 1})
        [block] = render_evidence([row], max_chars=10_000)
        assert '"a": 1' in block
        assert '"b": 2' in block

    def test_nested_collections_are_not_reordered_internally(self) -> None:
        # Only the top level is reordered. A record's own field order is the connector's
        # business, and rewriting it would make payloads harder to compare against the API.
        row = _Row(payload={"records": [{"z": 1, "a": 2}], "total_matching": 1})
        [block] = render_evidence([row], max_chars=10_000)
        assert block.index('"z": 1') < block.index('"a": 2')


class TestTheVerifierSeesRealCharacters:
    """The verifier judges a claim against the evidence text it is shown.

    Escaped non-ASCII makes that comparison harder than it needs to be: a claim quoting a
    customer's name as "Schonherr" with an o-umlaut, checked against evidence rendering it
    as `Sch\\u00f6nherr`, is a string mismatch invented by the renderer.
    """

    def test_a_payload_with_non_ascii_is_shown_unescaped(self) -> None:
        row = _Row(payload={"total_matching": 1, "records": [{"name": "Mölnlycke — Q3"}]})
        [block] = render_evidence([row], max_chars=10_000)
        assert "Mölnlycke — Q3" in block
        assert "\\u" not in block

    def test_parameters_with_non_ascii_are_shown_unescaped(self) -> None:
        row = _Row(payload={"rows": []}, params={"query": "Ørsted"})
        [block] = render_evidence([row], max_chars=10_000)
        assert "Ørsted" in block
        assert "\\u00d8" not in block
