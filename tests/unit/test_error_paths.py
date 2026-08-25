"""Error and edge paths across the connectors, stores and helpers.

These are the branches that fire when an upstream returns a shape the happy-path
fixtures do not contain — a null where a number was expected, a bare array where an
object was, a timeout instead of a response. They were the residual coverage gap,
and every one of them is a path that runs in production the first time an API
behaves slightly differently than its documentation.

Grouped by module rather than by feature: each is a small independent check, and
scattering them into the feature files would bury the behaviour those files exist
to describe.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from cortex.tools.base import ToolContext, UpstreamError
from cortex.tools.http import request_json


class TestGA4Parsing:
    async def test_reversed_range_is_rejected_by_every_dated_capability(self) -> None:
        """Each capability validates independently; one of them missing the check
        would send a nonsensical window to Google and get an opaque 400 back."""
        from cortex.tools.base import InvalidParams
        from cortex.tools.ga4 import GA4Tool

        tool = GA4Tool()
        ctx = ToolContext(
            tenant=None,  # type: ignore[arg-type]
            credential="{}",
            credential_metadata={"property_id": "123456789"},
        )
        for call in (tool.get_funnel, tool.top_pages, tool.get_sessions):
            with pytest.raises(InvalidParams, match="after end_date"):
                await call(ctx, start_date="2026-07-31", end_date="2026-07-01")

    def test_unparseable_metric_becomes_none_not_zero(self) -> None:
        """A zero would read as a real measurement of nothing."""
        from cortex.tools.ga4 import _coerce

        assert _coerce("not-a-number") is None
        assert _coerce("") is None
        assert _coerce(None) is None
        assert _coerce("12") == 12
        assert _coerce("1.5") == 1.5

    def test_comparison_without_a_date_range_dimension_still_splits(self) -> None:
        """GA4 omits the synthetic dateRange dimension in some shapes. Rows are then
        all treated as current rather than silently dropped."""
        from cortex.tools.ga4 import _split_by_range

        raw = {
            "dimensionHeaders": [{"name": "deviceCategory"}],
            "metricHeaders": [{"name": "sessions"}],
            "rows": [
                {
                    "dimensionValues": [{"value": "mobile"}],
                    "metricValues": [{"value": "900"}],
                }
            ],
        }
        current, previous = _split_by_range(raw, ["deviceCategory"], ["sessions"])
        assert current == {("mobile",): {"sessions": 900}}
        assert previous == {}


class TestBigQueryParsing:
    @staticmethod
    def _ctx() -> ToolContext:
        return ToolContext(
            tenant=None,  # type: ignore[arg-type]
            credential="{}",
            credential_metadata={
                "project_id": "cortex-test-project",
                "dataset": "analytics",
                "events_table": "events",
            },
        )

    async def test_a_query_with_an_unknown_placeholder_is_named_not_opaque(self) -> None:
        """`str.format` raises on an unknown placeholder rather than leaving it in
        the string, so this is where that authoring mistake has to be caught. Without
        it the loop receives a generic failure it cannot act on."""
        from cortex.tools.base import ToolError
        from cortex.tools.bigquery import NAMED_QUERIES, BigQueryTool, NamedQuery

        broken = NamedQuery(
            name="broken",
            description="A query referencing a placeholder Cortex does not supply." + "x" * 10,
            sql="SELECT 1 FROM `{project}.{dataset}.{table}` WHERE x = {not_supplied}",
            params_schema={"type": "object", "additionalProperties": False, "properties": {}},
            table_key="events_table",
        )
        NAMED_QUERIES["broken"] = broken
        try:
            with pytest.raises(ToolError, match="does not provide"):
                await BigQueryTool().run_named_query(self._ctx(), query_name="broken", params={})
        finally:
            del NAMED_QUERIES["broken"]

    async def test_an_escaped_brace_surviving_into_sql_is_refused(self) -> None:
        """The path `format()` does not catch: a doubled brace collapses to a literal
        rather than raising, and stray braces should not reach BigQuery."""
        from cortex.tools.base import ToolError
        from cortex.tools.bigquery import NAMED_QUERIES, BigQueryTool, NamedQuery

        braced = NamedQuery(
            name="braced",
            description="A query whose escaped brace survives formatting intact." + "x" * 10,
            sql="SELECT 1 FROM `{project}.{dataset}.{table}` WHERE json = '{{}}'",
            params_schema={"type": "object", "additionalProperties": False, "properties": {}},
            table_key="events_table",
        )
        NAMED_QUERIES["braced"] = braced
        try:
            with pytest.raises(ToolError, match="unexpected brace"):
                await BigQueryTool().run_named_query(self._ctx(), query_name="braced", params={})
        finally:
            del NAMED_QUERIES["braced"]

    def test_parameter_types_cover_float_and_bool(self) -> None:
        """Types come from the schema, not from Python inference, so a value cannot
        change how it binds."""
        from cortex.tools.bigquery import _query_parameters

        schema = {
            "properties": {
                "rate": {"type": "number"},
                "flag": {"type": "boolean"},
                "count": {"type": "integer"},
            }
        }
        params = {
            p["name"]: p for p in _query_parameters({"rate": 0.5, "flag": True, "count": 3}, schema)
        }
        assert params["rate"]["parameterType"]["type"] == "FLOAT64"
        assert params["rate"]["parameterValue"]["value"] == "0.5"
        assert params["flag"]["parameterType"]["type"] == "BOOL"
        assert params["flag"]["parameterValue"]["value"] == "true"
        assert params["count"]["parameterType"]["type"] == "INT64"

    def test_false_boolean_binds_as_false(self) -> None:
        from cortex.tools.bigquery import _query_parameters

        params = _query_parameters({"flag": False}, {"properties": {"flag": {"type": "boolean"}}})
        assert params[0]["parameterValue"]["value"] == "false"

    @pytest.mark.parametrize(
        ("value", "field_type", "expected"),
        [
            ("not-a-number", "FLOAT64", None),
            ("not-an-int", "INT64", None),
            ("true", "BOOL", True),
            ("false", "BOOL", False),
            (None, "INT64", None),
            ("text", "STRING", "text"),
            ("3.5", "NUMERIC", 3.5),
        ],
    )
    def test_coerce_handles_every_declared_type(
        self, value: object, field_type: str, expected: object
    ) -> None:
        """BigQuery returns every scalar as a string; a bad value must become None
        rather than crash the parse or default to zero."""
        from cortex.tools.bigquery import _coerce

        assert _coerce(value, field_type) == expected


class TestConnectorHelpers:
    @pytest.mark.parametrize(
        "helper",
        [
            "cortex.tools.github._as_list",
            "cortex.tools.hubspot._as_list",
            "cortex.tools.slack._as_list",
        ],
    )
    def test_as_list_tolerates_a_non_list(self, helper: str) -> None:
        """Upstreams occasionally return an object where an array is documented.
        Coercing to empty beats raising: an empty result is a valid observation."""
        import importlib

        module_name, name = helper.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_name), name)
        assert fn({"not": "a list"}) == []
        assert fn(None) == []
        assert fn("string") == []
        # Non-dict members are filtered out rather than passed through.
        assert fn([{"a": 1}, "junk", None]) == [{"a": 1}]

    def test_github_release_notes_none_stays_none(self) -> None:
        from cortex.tools.github import _truncate_notes

        assert _truncate_notes(None) is None
        assert _truncate_notes("") is None
        assert _truncate_notes("short") == "short"

    def test_slack_timestamp_helpers_tolerate_junk(self) -> None:
        from cortex.tools.slack import _iso, _truncate

        assert _iso(None) is None
        assert _iso("") is None
        assert _iso("not-a-timestamp") is None
        assert _iso("1784889840.000100") is not None
        assert _truncate(None) is None
        assert _truncate("") is None

    def test_hubspot_number_helper_tolerates_junk(self) -> None:
        """HubSpot returns numerics as strings and unset values as empty strings."""
        from cortex.tools.hubspot import _number

        assert _number("") is None
        assert _number(None) is None
        assert _number("not-a-number") is None
        assert _number([]) is None
        assert _number("48000") == 48000
        assert _number("0.8") == 0.8

    def test_hubspot_engagement_summary_without_content(self) -> None:
        from cortex.tools.hubspot import _engagement_summary

        assert _engagement_summary("notes", {}) is None
        assert _engagement_summary("unknown_type", {"hs_note_body": "x"}) is None
        assert _engagement_summary("notes", {"hs_note_body": "  spaced   out  "}) == "spaced out"

    async def test_hubspot_pipeline_forwards_a_minimum_amount(self) -> None:
        """The optional filter path: without it a `min_amount` argument would be
        accepted and silently ignored."""
        from cortex.tools.hubspot import HubSpotTool

        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            # Deals now come with an owner lookup, which resolves `hubspot_owner_id` to a
            # name so results can be segmented by rep. It is a GET with no body, so
            # parsing every request as JSON fails on it.
            if "search" not in str(request.url):
                return httpx.Response(200, json={"results": []})
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"total": 0, "results": []})

        tool = HubSpotTool()
        transport = httpx.MockTransport(handler)
        original = tool._client

        def _client(ctx: ToolContext) -> httpx.AsyncClient:
            real = original(ctx)
            return httpx.AsyncClient(
                transport=transport, base_url=real.base_url, headers=real.headers
            )

        tool._client = _client  # type: ignore[method-assign]
        await tool.pipeline(
            ToolContext(tenant=None, credential="pat"),  # type: ignore[arg-type]
            min_amount=10000,
        )
        filters = seen[0]["filterGroups"][0]["filters"]
        amount = next(f for f in filters if f["propertyName"] == "amount")
        assert amount == {"propertyName": "amount", "operator": "GTE", "value": "10000"}

    async def test_slack_find_decision_forwards_the_after_filter(self) -> None:
        """Slack carries date bounds in the query string, so an unused `after` would
        silently widen the search."""
        from cortex.tools.slack import SlackTool

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"ok": True, "messages": {"total": 0, "matches": []}})

        tool = SlackTool()
        transport = httpx.MockTransport(handler)
        original = tool._client

        def _client(ctx: ToolContext) -> httpx.AsyncClient:
            real = original(ctx)
            return httpx.AsyncClient(
                transport=transport, base_url=real.base_url, headers=real.headers
            )

        tool._client = _client  # type: ignore[method-assign]
        await tool.find_decision(
            ToolContext(tenant=None, credential="xoxb"),  # type: ignore[arg-type]
            topic="onboarding",
            after="2026-07-01",
        )
        assert "after:2026-07-01" in seen[0].url.params["query"]


class TestHTTPTransportFailures:
    async def test_a_timeout_becomes_an_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.example.com"
        ) as client:
            with pytest.raises(UpstreamError, match="timeout"):
                await request_json(client, "GET", "/r", tool="probe")

    async def test_a_connection_failure_becomes_an_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns failure", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.example.com"
        ) as client:
            with pytest.raises(UpstreamError, match="transport error"):
                await request_json(client, "GET", "/r", tool="probe")

    async def test_transport_errors_do_not_leak_the_query_string(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.example.com"
        ) as client:
            with pytest.raises(UpstreamError) as exc:
                await request_json(
                    client, "GET", "/r", tool="probe", params={"token": "SECRET-VALUE"}
                )
        assert "SECRET-VALUE" not in str(exc.value)


class TestVaultShortCiphertext:
    def test_a_truncated_blob_is_refused(self) -> None:
        """Shorter than the nonce: rejected as a length error rather than reaching
        AES-GCM, where the failure would be less diagnosable."""
        from cortex.security.vault import VaultError, decrypt_credential, encrypt_credential

        tenant = uuid.uuid4()
        wrapped, _ = encrypt_credential(tenant, "ga4", "secret")
        with pytest.raises(VaultError, match="too short"):
            decrypt_credential(tenant, "ga4", wrapped, b"tiny")


class TestEmployeeLoading:
    def test_a_yaml_file_that_is_not_a_mapping_is_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from cortex.agents.employee import EmployeeNotFound, load_employee

        (tmp_path / "listy.yaml").write_text("- just\n- a\n- list\n")
        with pytest.raises(EmployeeNotFound, match="does not contain a mapping"):
            load_employee("listy", directory=tmp_path)

    def test_a_declared_role_must_match_the_filename(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Otherwise the two ways of naming an employee resolve differently."""
        from cortex.agents.employee import EmployeeNotFound, gtm_data_analyst, load_employee

        contract = gtm_data_analyst().model_dump()
        contract["role"] = "someone_else"
        import yaml

        (tmp_path / "mismatched.yaml").write_text(yaml.safe_dump(contract))
        with pytest.raises(EmployeeNotFound, match="declares role"):
            load_employee("mismatched", directory=tmp_path)


class TestRecordedProviderExhaustion:
    async def test_an_exhausted_completion_script_raises(self) -> None:
        """A loop that ran more turns than the test anticipated is a test failure
        worth being loud about, not a silent empty response."""
        from cortex.agents.llm import LLMError, RecordedLLM

        with pytest.raises(LLMError, match="no completion left"):
            await RecordedLLM().complete(system="s", messages=[])
