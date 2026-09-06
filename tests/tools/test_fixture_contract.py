"""One set of properties, asserted against both the fixture and the real connector.

**Why this file exists.** *Software Engineering at Google* Ch. 13 states the problem exactly:
"with stubbing, there is no way to ensure the function being stubbed behaves like the real
implementation… there is no way to guarantee that the contract is correct." Its remedy is the
one applied here — "writing tests against the API's public interface and running those tests
against **both** the real implementation and the fake (these are known as **contract tests**)."

Six defects in these fixtures in a single day were all one thing: the fixture behaved unlike the
connector it stands in for. Every guard written so far checks the *fixture* in isolation, which
can only ever encode what somebody already believed the connector does.

**No credentials are needed, and that is the point.** The real side runs the connector's own
code — its request construction, its bucketing, its disclosure computation — against a mock
transport, so only the socket is synthetic. That makes this cheap enough to run on every commit
rather than nightly, which is what the Google text assumes you cannot do.

Where the two sides legitimately differ, the difference is named. Where they must agree, they are
asserted with the same code.
"""

from __future__ import annotations

from typing import Any

import pytest

from cortex.eval.fixtures import by_name
from cortex.tools.base import ToolContext
from cortex.tools.posthog import PostHogTool

#: One planted world, used by both sides. Thirty days at a steady rate with a step down, which is
#: the shape every scenario in the suite uses and the shape the properties below are about.
_DAYS = [f"2026-06-{day:02d}" for day in range(1, 31)]
_VALUES = [56 if index < 14 else 33 for index in range(30)]


def _query_response(columns: list[str], results: list[list[Any]]) -> dict[str, Any]:
    return {"columns": columns, "results": results, "hogql": "SELECT ..."}


def _rows_between(start: str, end: str) -> list[list[Any]]:
    """The rows a real PostHog query would return for this window — the database does the
    filtering, so the mock transport must too, or the real side is not being tested."""
    return [
        [f"{day}T00:00:00", value]
        for day, value in zip(_DAYS, _VALUES, strict=True)
        if start <= day <= end
    ]


@pytest.fixture
def tool() -> PostHogTool:
    return PostHogTool()


@pytest.fixture
def ctx(tenant: Any) -> ToolContext:
    # A project mapping is required: the connector refuses a credential that lists none, which
    # is itself a contract worth honouring here rather than routing around.
    return ToolContext(
        tenant=tenant,
        credential="phx-not-a-real-key",
        credential_metadata={"projects": "100001:web-app"},
    )


@pytest.fixture
def real_trend(tool: PostHogTool, ctx: ToolContext, patch_client: Any) -> Any:
    """The real connector, socket replaced. Returns an async callable taking the same params the
    fixture's `response_for` takes."""

    async def _call(**params: Any) -> dict[str, Any]:
        start = params.get("start_date", _DAYS[0])
        end = params.get("end_date", _DAYS[-1])
        patch_client(
            tool,
            {"/query/": _query_response(["bucket", "value"], _rows_between(start, end))},
            is_async_factory=True,
        )
        result = await tool.event_trend(ctx, **params)
        payload = result.payload if hasattr(result, "payload") else result
        return dict(payload)

    return _call


def _fixture_trend(**params: Any) -> dict[str, Any]:
    """The fixture side, through the same entry point the eval uses."""
    scenario = by_name("campaign_traffic_drop")
    return dict(scenario.response_for("posthog__event_trend", params))


class TestBothSidesHonourTheRequestedWindow:
    """The **subset** property, asserted identically on each side.

    An ignored date range is the defect that made every scenario report a collection failure to
    an analyst who asked wider than was planted. The fixture side is guarded elsewhere; this is
    what checks the guard encodes the connector's actual behaviour rather than a belief about it.
    """

    WIDE = {"start_date": "2026-06-01", "end_date": "2026-06-30"}
    NARROW = {"start_date": "2026-06-10", "end_date": "2026-06-20"}

    async def test_the_real_connector_narrows(self, real_trend: Any) -> None:
        wide = await real_trend(event="user signed up", interval="day", **self.WIDE)
        narrow = await real_trend(event="user signed up", interval="day", **self.NARROW)
        assert self._buckets(narrow) < self._buckets(wide)
        assert self._buckets(narrow) <= self._buckets(wide)

    def test_the_fixture_narrows_the_same_way(self) -> None:
        wide = _fixture_trend(event="user signed up", interval="day", **self.WIDE)
        narrow = _fixture_trend(event="user signed up", interval="day", **self.NARROW)
        assert self._buckets(narrow) < self._buckets(wide)
        assert self._buckets(narrow) <= self._buckets(wide)

    @staticmethod
    def _buckets(payload: dict[str, Any]) -> set[str]:
        return {row["bucket"][:10] for row in payload.get("series") or []}


class TestBothSidesReturnTheIntervalAsked:
    """The **complete** property. Asking for days and receiving weeks cost score in two
    scenarios before anything checked it, and it was only ever fixed one fixture at a time."""

    WINDOW = {"start_date": "2026-06-01", "end_date": "2026-06-28"}

    @pytest.mark.parametrize("interval", ["day", "week", "month"])
    async def test_the_real_connector_echoes_the_interval(
        self, real_trend: Any, interval: str
    ) -> None:
        payload = await real_trend(event="user signed up", interval=interval, **self.WINDOW)
        assert payload["interval"] == interval

    @pytest.mark.parametrize("interval", ["day", "week", "month"])
    def test_the_fixture_echoes_the_interval(self, interval: str) -> None:
        payload = _fixture_trend(event="user signed up", interval=interval, **self.WINDOW)
        assert payload["interval"] == interval

    @pytest.mark.parametrize("interval", ["day", "week", "month"])
    def test_the_fixture_total_equals_its_series(self, interval: str) -> None:
        """A canned payload could declare a total its own series contradicted, and one did, by
        18%. Deriving makes the two the same number."""
        payload = _fixture_trend(event="user signed up", interval=interval, **self.WINDOW)
        assert payload["total"] == sum(row["value"] for row in payload["series"])


class TestBothSidesCarryTheSameEnvelope:
    """The fields the analyst reads to know *what it was given* must exist on both sides.

    `ScenarioTool` already mirrors the capability's name, description, schema, `result_key` and
    discovery flag, so the model sees production's surface. This asserts the payload shape
    matches too — a fixture missing `interval` or `end_date` produces an observation the analyst
    reads differently from the real one, which is drift the mirroring cannot catch.
    """

    ENVELOPE = ("event", "measure", "interval", "start_date", "end_date", "series", "total")

    async def test_the_real_connector_carries_the_envelope(self, real_trend: Any) -> None:
        payload = await real_trend(
            event="user signed up", interval="day", start_date="2026-06-01", end_date="2026-06-30"
        )
        missing = [field for field in self.ENVELOPE if field not in payload]
        assert not missing, missing

    def test_the_fixture_carries_the_same_envelope(self) -> None:
        payload = _fixture_trend(
            event="user signed up", interval="day", start_date="2026-06-01", end_date="2026-06-30"
        )
        missing = [field for field in self.ENVELOPE if field not in payload]
        assert not missing, missing


class TestWhereTheTwoSidesLegitimatelyDiffer:
    """Named, so a difference is a decision rather than a discovery.

    The fixture serves one planted world and the connector serves whatever the tenant has, so
    *values* cannot be compared between the two -- only shapes and relations. That is the same
    limitation the Google text notes for fakes, and it is why every assertion above is a subset,
    an echo, or a field's presence.

    Comparing a value would make this file fail whenever a fixture's numbers were tuned, and a
    test that fails for a legitimate edit gets deleted rather than fixed.
    """
