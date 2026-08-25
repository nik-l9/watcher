"""BigQuery connector.

The property this file exists to defend: **the agent cannot author SQL.** Every
test below either proves a value was bound as a query parameter rather than
interpolated, or proves an identifier came from tenant configuration rather than
from the model.

Also pinned: cost controls, since a runaway scan against a production warehouse is
a real bill rather than a failed test.
"""

from __future__ import annotations

import httpx
import pytest

from cortex.tools.base import InvalidParams, ToolContext, ToolError
from cortex.tools.bigquery import (
    MAX_BYTES_BILLED,
    NAMED_QUERIES,
    BigQueryTool,
    NamedQuery,
)

METADATA = {
    "project_id": "cortex-test-project",
    "dataset": "analytics",
    "events_table": "events",
}


def _response(fields: list[tuple[str, str]], rows: list[list[str | None]], **extra: object) -> dict:
    return {
        "jobComplete": True,
        "schema": {"fields": [{"name": n, "type": t} for n, t in fields]},
        "rows": [{"f": [{"v": v} for v in row]} for row in rows],
        "totalRows": str(len(rows)),
        "totalBytesProcessed": "1048576",
        "cacheHit": False,
        **extra,
    }


@pytest.fixture
def tool() -> BigQueryTool:
    return BigQueryTool()


@pytest.fixture
def ctx(tenant: object) -> ToolContext:
    return ToolContext(
        tenant=tenant,  # type: ignore[arg-type]
        credential="{}",
        credential_metadata=dict(METADATA),
    )


class TestNoAgentAuthoredSql:
    def test_no_capability_accepts_sql(self, tool: BigQueryTool) -> None:
        """The decisive check: nowhere can a model hand us a query string."""
        for name in tool.capability_names:
            schema = tool.capability(name).params_schema
            properties = set(schema.get("properties", {}))
            assert not properties & {"sql", "query", "statement", "q"}, (
                f"{name} exposes a SQL-shaped parameter"
            )

    def test_query_name_is_a_closed_enum(self, tool: BigQueryTool) -> None:
        schema = tool.capability("run_named_query").params_schema
        assert schema["properties"]["query_name"]["enum"] == sorted(NAMED_QUERIES)

    async def test_unknown_query_name_is_rejected(
        self, tool: BigQueryTool, ctx: ToolContext
    ) -> None:
        with pytest.raises(InvalidParams, match="unknown query"):
            await tool.run_named_query(ctx, query_name="drop_everything", params={})

    async def test_list_queries_states_the_restriction(
        self, tool: BigQueryTool, ctx: ToolContext
    ) -> None:
        """The model needs to know arbitrary SQL is unavailable, or it will keep
        trying and burn steps."""
        result = await tool.list_queries(ctx)
        assert result.payload["arbitrary_sql_supported"] is False
        assert result.payload["count"] == len(NAMED_QUERIES)
        assert all("params_schema" in q for q in result.payload["queries"])

    def test_named_queries_do_not_interpolate_values(self) -> None:
        for query in NAMED_QUERIES.values():
            assert "%s" not in query.sql
            assert "{value" not in query.sql
            # Every value placeholder must be a bound @parameter.
            assert "@" in query.sql or query.name == "list_queries"

    def test_named_query_rejects_value_interpolation_at_definition(self) -> None:
        """The guard fires at import time, so a bad query cannot ship."""
        with pytest.raises(ValueError, match="interpolate"):
            NamedQuery(
                name="bad",
                description="x",
                sql="SELECT * FROM t WHERE x = %s",
                params_schema={"type": "object", "additionalProperties": False},
                table_key="events_table",
            )


class TestParameterBinding:
    async def test_values_are_bound_not_interpolated(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(
            tool,
            {
                "/queries": _response(
                    [("day", "DATE"), ("events", "INTEGER")], [["2026-07-20", "5"]]
                )
            },
            is_async_factory=True,
        )
        await tool.run_named_query(
            ctx,
            query_name="daily_metric",
            params={
                "event_name": "signup_completed",
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
            },
        )
        body = transport.request_bodies()[0]
        assert body["parameterMode"] == "NAMED"
        names = {p["name"] for p in body["queryParameters"]}
        assert names == {"event_name", "start_date", "end_date"}
        # The value must not appear in the SQL text at all.
        assert "signup_completed" not in body["query"]

    async def test_parameter_types_come_from_the_schema(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Types are declared, not inferred, so a value cannot change how it binds."""
        transport = patch_client(
            tool,
            {"/queries": _response([("dimension_value", "STRING")], [])},
            is_async_factory=True,
        )
        await tool.run_named_query(
            ctx,
            query_name="metric_by_dimension",
            params={
                "event_name": "signup_completed",
                "dimension": "platform",
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
                "row_limit": 10,
            },
        )
        by_name = {p["name"]: p for p in transport.request_bodies()[0]["queryParameters"]}
        assert by_name["row_limit"]["parameterType"]["type"] == "INT64"
        assert by_name["start_date"]["parameterType"]["type"] == "DATE"
        assert by_name["event_name"]["parameterType"]["type"] == "STRING"

    async def test_sql_injection_in_a_value_stays_a_value(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        hostile = "x'; DROP TABLE events; --"
        transport = patch_client(
            tool, {"/queries": _response([("day", "DATE")], [])}, is_async_factory=True
        )
        await tool.run_named_query(
            ctx,
            query_name="daily_metric",
            params={
                "event_name": hostile,
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
            },
        )
        body = transport.request_bodies()[0]
        assert "DROP TABLE" not in body["query"]
        bound = {p["name"]: p["parameterValue"]["value"] for p in body["queryParameters"]}
        assert bound["event_name"] == hostile

    async def test_per_query_params_are_validated(
        self, tool: BigQueryTool, ctx: ToolContext
    ) -> None:
        """The outer schema can only say params is an object, so the inner schema
        has to be enforced separately."""
        with pytest.raises(InvalidParams, match="daily_metric"):
            await tool.run_named_query(
                ctx, query_name="daily_metric", params={"event_name": "signup_completed"}
            )

    async def test_unknown_params_are_rejected(self, tool: BigQueryTool, ctx: ToolContext) -> None:
        with pytest.raises(InvalidParams):
            await tool.run_named_query(
                ctx,
                query_name="daily_metric",
                params={
                    "event_name": "x",
                    "start_date": "2026-07-01",
                    "end_date": "2026-07-31",
                    "extra": "hallucinated",
                },
            )

    async def test_dimension_must_come_from_the_enum(
        self, tool: BigQueryTool, ctx: ToolContext
    ) -> None:
        """The dimension becomes a column name, so its values are a closed set."""
        with pytest.raises(InvalidParams):
            await tool.run_named_query(
                ctx,
                query_name="metric_by_dimension",
                params={
                    "event_name": "x",
                    "dimension": "user_id) FROM other_table --",
                    "start_date": "2026-07-01",
                    "end_date": "2026-07-31",
                },
            )


class TestTargetResolution:
    async def test_identifiers_come_from_connector_metadata(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(
            tool, {"/queries": _response([("day", "DATE")], [])}, is_async_factory=True
        )
        result = await tool.run_named_query(
            ctx,
            query_name="daily_metric",
            params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
        )
        assert result.payload["target"] == "cortex-test-project.analytics.events"
        assert "`cortex-test-project.analytics.events`" in transport.request_bodies()[0]["query"]

    async def test_missing_metadata_is_a_clear_error(
        self, tool: BigQueryTool, tenant: object
    ) -> None:
        ctx = ToolContext(tenant=tenant, credential="{}", credential_metadata={})  # type: ignore[arg-type]
        with pytest.raises(InvalidParams, match="project_id"):
            await tool.run_named_query(
                ctx,
                query_name="daily_metric",
                params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
            )

    @pytest.mark.parametrize(
        "bad_dataset",
        ["analytics; DROP", "analytics`", "an-alytics", "1analytics", "", "analytics.other"],
    )
    async def test_hostile_dataset_identifier_is_rejected(
        self, tool: BigQueryTool, tenant: object, bad_dataset: str
    ) -> None:
        """Identifiers are interpolated because the API cannot bind them, which is
        exactly why they are validated."""
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="{}",
            credential_metadata={**METADATA, "dataset": bad_dataset},
        )
        with pytest.raises(InvalidParams):
            await tool.run_named_query(
                ctx,
                query_name="daily_metric",
                params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
            )

    @pytest.mark.parametrize("bad_project", ["A-Project", "x", "proj`ect", "project.other"])
    async def test_hostile_project_id_is_rejected(
        self, tool: BigQueryTool, tenant: object, bad_project: str
    ) -> None:
        ctx = ToolContext(
            tenant=tenant,  # type: ignore[arg-type]
            credential="{}",
            credential_metadata={**METADATA, "project_id": bad_project},
        )
        with pytest.raises(InvalidParams):
            await tool.run_named_query(
                ctx,
                query_name="daily_metric",
                params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
            )


class TestCostControls:
    async def test_every_job_caps_bytes_billed(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A runaway scan must fail the job rather than arrive as an invoice."""
        transport = patch_client(
            tool, {"/queries": _response([("day", "DATE")], [])}, is_async_factory=True
        )
        await tool.run_named_query(
            ctx,
            query_name="daily_metric",
            params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
        )
        body = transport.request_bodies()[0]
        assert body["maximumBytesBilled"] == str(MAX_BYTES_BILLED)
        assert body["useQueryCache"] is True
        assert body["useLegacySql"] is False
        assert body["timeoutMs"] <= 60_000

    async def test_reports_bytes_processed(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/queries": _response([("day", "DATE")], [])}, is_async_factory=True)
        result = await tool.run_named_query(
            ctx,
            query_name="daily_metric",
            params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
        )
        assert result.payload["bytes_processed"] == 1048576


class TestResults:
    async def test_types_values_from_the_schema(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/queries": _response(
                    [("day", "DATE"), ("events", "INTEGER"), ("rate", "FLOAT")],
                    [["2026-07-20", "1200", "0.029"], ["2026-07-21", None, None]],
                )
            },
            is_async_factory=True,
        )
        result = await tool.run_named_query(
            ctx,
            query_name="daily_metric",
            params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
        )
        first, second = result.payload["rows"]
        assert first["events"] == 1200 and isinstance(first["events"], int)
        assert first["rate"] == 0.029
        # A NULL must stay None, not become zero.
        assert second["events"] is None

    async def test_incomplete_job_is_an_error(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Returning partial results as complete would let a report cite a number
        that is still being computed."""
        patch_client(tool, {"/queries": {"jobComplete": False}}, is_async_factory=True)
        with pytest.raises(ToolError, match="did not complete"):
            await tool.run_named_query(
                ctx,
                query_name="daily_metric",
                params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
            )

    async def test_empty_result_is_a_valid_observation(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/queries": _response([("day", "DATE")], [])}, is_async_factory=True)
        result = await tool.run_named_query(
            ctx,
            query_name="daily_metric",
            params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
        )
        assert result.payload["row_count"] == 0


class TestNamedQueryDefinitions:
    def test_every_query_declares_a_closed_schema(self) -> None:
        for query in NAMED_QUERIES.values():
            assert query.params_schema["type"] == "object", query.name
            assert query.params_schema["additionalProperties"] is False, query.name

    def test_every_query_has_a_table_key(self) -> None:
        for query in NAMED_QUERIES.values():
            assert query.table_key, query.name

    def test_every_query_has_a_usable_description(self) -> None:
        for query in NAMED_QUERIES.values():
            assert len(query.description) >= 40, query.name

    async def test_no_query_leaves_unfilled_placeholders(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """An unfilled {placeholder} reaching BigQuery would be a syntax error at
        best and an injection surface at worst."""
        transport = patch_client(
            tool, {"/queries": _response([("x", "STRING")], [])}, is_async_factory=True
        )
        cases = {
            "daily_metric": {
                "event_name": "x",
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
            },
            "metric_by_dimension": {
                "event_name": "x",
                "dimension": "country",
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
            },
            "funnel_step_conversion": {
                "from_step": "a",
                "to_step": "b",
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
            },
        }
        assert set(cases) == set(NAMED_QUERIES), "add a case for every named query"

        for name, params in cases.items():
            await tool.run_named_query(ctx, query_name=name, params=params)

        for body in transport.request_bodies():
            assert "{" not in body["query"], body["query"]


class TestFailures:
    async def test_upstream_error_is_mapped(
        self, tool: BigQueryTool, ctx: ToolContext, patch_client: object
    ) -> None:
        from cortex.tools.http import AuthRejected

        patch_client(
            tool,
            {"/queries": httpx.Response(403, json={"error": "denied"})},
            is_async_factory=True,
        )
        with pytest.raises(AuthRejected):
            await tool.run_named_query(
                ctx,
                query_name="daily_metric",
                params={"event_name": "x", "start_date": "2026-07-01", "end_date": "2026-07-31"},
            )
