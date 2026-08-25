"""BigQuery connector — warehouse metrics.

**The agent cannot author SQL.** V1 exposes a fixed set of named, reviewed queries
whose only variable inputs are typed query parameters. This is deliberate and is
the single most important property of this connector:

  - A model writing SQL against a production warehouse can trivially produce a
    query that scans terabytes and costs real money, or one that is subtly wrong in
    a way no reviewer sees because it was generated at runtime.
  - Generated SQL is unreviewable. A named query can be read, tested, and reasoned
    about once; an infinite family of generated queries cannot.
  - Parameters go through BigQuery's named-parameter binding, so a value can never
    become syntax.

Table identifiers are the one part of a query that cannot be parameterized by the
API, so they come from the tenant's connector metadata — never from the model — and
are validated against a strict identifier pattern before interpolation.

`use_query_cache` and `maximum_bytes_billed` are set on every job: a runaway scan
should fail loudly rather than appear on an invoice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from cortex.db.models import CredentialProvider
from cortex.tools.base import (
    NEVER_EMPTY,
    Capability,
    InvalidParams,
    ToolContext,
    ToolError,
    ToolResult,
)
from cortex.tools.base import Tool as BaseTool
from cortex.tools.google_auth import BIGQUERY_SCOPES, access_token
from cortex.tools.http import DEFAULT_TIMEOUT, request_json

API_ROOT = "https://bigquery.googleapis.com/bigquery/v2"

# A hard ceiling on any single job. Exceeding it fails the job rather than billing
# for it.
MAX_BYTES_BILLED = 20 * 1024**3  # 20 GiB

_MAX_ROWS = 1000

# Identifiers are interpolated, so they are validated rather than trusted.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_PROJECT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")

_DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}


@dataclass(frozen=True, slots=True)
class NamedQuery:
    """One reviewed query.

    `sql` may reference {project}.{dataset}.{table} placeholders, filled from
    validated connector metadata, and @named parameters, bound by BigQuery.
    """

    name: str
    description: str
    sql: str
    params_schema: dict[str, Any]
    #: Which metadata table key this query reads, e.g. "events_table".
    table_key: str

    def __post_init__(self) -> None:
        # A named query containing string interpolation of a parameter would defeat
        # the entire point, so the shape is checked at import time.
        if "%s" in self.sql or "{value" in self.sql:
            raise ValueError(f"named query {self.name!r} appears to interpolate values")


NAMED_QUERIES: dict[str, NamedQuery] = {
    "daily_metric": NamedQuery(
        name="daily_metric",
        description=(
            "Daily totals for one event type over a date range. Use to establish a "
            "time series and locate the day a change began."
        ),
        table_key="events_table",
        sql="""
            SELECT
              DATE(event_timestamp) AS day,
              COUNT(*) AS events,
              COUNT(DISTINCT user_id) AS users
            FROM `{project}.{dataset}.{table}`
            WHERE event_name = @event_name
              AND DATE(event_timestamp) BETWEEN @start_date AND @end_date
            GROUP BY day
            ORDER BY day
        """,
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["event_name", "start_date", "end_date"],
            "properties": {
                "event_name": {"type": "string", "minLength": 1, "maxLength": 128},
                "start_date": _DATE,
                "end_date": _DATE,
            },
        },
    ),
    "metric_by_dimension": NamedQuery(
        name="metric_by_dimension",
        description=(
            "One event type broken down by a dimension over a date range, to find "
            "which segment drove a change."
        ),
        table_key="events_table",
        sql="""
            SELECT
              {dimension_column} AS dimension_value,
              COUNT(*) AS events,
              COUNT(DISTINCT user_id) AS users
            FROM `{project}.{dataset}.{table}`
            WHERE event_name = @event_name
              AND DATE(event_timestamp) BETWEEN @start_date AND @end_date
            GROUP BY dimension_value
            ORDER BY events DESC
            LIMIT @row_limit
        """,
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["event_name", "dimension", "start_date", "end_date"],
            "properties": {
                "event_name": {"type": "string", "minLength": 1, "maxLength": 128},
                # An enum, not a free string: the column name is interpolated, so
                # the set of legal values is closed.
                "dimension": {
                    "type": "string",
                    "enum": ["platform", "country", "device_category", "channel", "plan"],
                },
                "start_date": _DATE,
                "end_date": _DATE,
                "row_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_ROWS,
                    "default": 50,
                },
            },
        },
    ),
    "funnel_step_conversion": NamedQuery(
        name="funnel_step_conversion",
        description=(
            "Users reaching each of two funnel steps, and the conversion between "
            "them, over a date range."
        ),
        table_key="events_table",
        sql="""
            WITH step_users AS (
              SELECT
                event_name,
                COUNT(DISTINCT user_id) AS users
              FROM `{project}.{dataset}.{table}`
              WHERE event_name IN (@from_step, @to_step)
                AND DATE(event_timestamp) BETWEEN @start_date AND @end_date
              GROUP BY event_name
            )
            SELECT
              (SELECT users FROM step_users WHERE event_name = @from_step) AS from_users,
              (SELECT users FROM step_users WHERE event_name = @to_step) AS to_users
        """,
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["from_step", "to_step", "start_date", "end_date"],
            "properties": {
                "from_step": {"type": "string", "minLength": 1, "maxLength": 128},
                "to_step": {"type": "string", "minLength": 1, "maxLength": 128},
                "start_date": _DATE,
                "end_date": _DATE,
            },
        },
    ),
}

_DIMENSION_COLUMNS = {
    "platform": "platform",
    "country": "geo_country",
    "device_category": "device_category",
    "channel": "traffic_source",
    "plan": "user_plan",
}


class BigQueryTool(BaseTool):
    name = "bigquery"
    provider = CredentialProvider.BIGQUERY

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="list_queries",
                description=(
                    "List the named warehouse queries available to this tenant, with "
                    "their parameters. Call this first — arbitrary SQL is not "
                    "available."
                ),
                params_schema={"type": "object", "additionalProperties": False, "properties": {}},
                handler=self.list_queries,
                result_key=f"{NEVER_EMPTY}: the allowlist is compiled in, so it is never empty",
                # Enumerates what exists. Run before the first step -- see Capability.discovery.
                discovery=True,
            ),
            Capability(
                name="run_named_query",
                description=(
                    "Run one reviewed, named warehouse query with typed parameters. "
                    "Arbitrary SQL is deliberately not supported: a generated query "
                    "cannot be reviewed and can scan terabytes."
                ),
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query_name", "params"],
                    "properties": {
                        "query_name": {"type": "string", "enum": sorted(NAMED_QUERIES)},
                        "params": {
                            "type": "object",
                            "description": (
                                "Parameters for the chosen query. See list_queries for "
                                "the schema of each."
                            ),
                        },
                    },
                },
                handler=self.run_named_query,
                result_key="rows",
            ),
        ]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _target(ctx: ToolContext, table_key: str) -> tuple[str, str, str]:
        """Resolve project, dataset and table from connector metadata.

        Table identifiers cannot be bound as query parameters, so they are
        interpolated — which is exactly why they come from tenant configuration
        rather than from the model, and are validated before use.
        """
        meta = ctx.credential_metadata
        project = str(meta.get("project_id", "")).strip()
        dataset = str(meta.get("dataset", "")).strip()
        table = str(meta.get(table_key, "")).strip()

        missing = [
            key
            for key, value in (("project_id", project), ("dataset", dataset), (table_key, table))
            if not value
        ]
        if missing:
            raise InvalidParams(
                f"BigQuery credential metadata is missing {', '.join(missing)}; "
                "reconnect the integration with the warehouse location"
            )
        if not _PROJECT_ID.match(project):
            raise InvalidParams(f"invalid BigQuery project_id: {project!r}")
        for label, value in (("dataset", dataset), (table_key, table)):
            if not _IDENTIFIER.match(value):
                raise InvalidParams(f"invalid BigQuery {label}: {value!r}")
        return project, dataset, table

    async def _client(self, ctx: ToolContext) -> httpx.AsyncClient:
        # Token minting is awaited rather than done synchronously: a blocking
        # refresh would stall the event loop on every tool call.
        token = await access_token(str(ctx.credential), BIGQUERY_SCOPES)
        return httpx.AsyncClient(
            base_url=API_ROOT,
            timeout=DEFAULT_TIMEOUT,
            headers={"Authorization": f"Bearer {token}"},
        )

    # ------------------------------------------------------------------ capabilities

    async def list_queries(self, ctx: ToolContext) -> ToolResult:
        del ctx
        return ToolResult(
            payload={
                "arbitrary_sql_supported": False,
                "reason": (
                    "Named queries only. Generated SQL is unreviewable and can scan "
                    "terabytes of a production warehouse."
                ),
                "count": len(NAMED_QUERIES),
                "queries": [
                    {
                        "name": q.name,
                        "description": q.description,
                        "params_schema": q.params_schema,
                    }
                    for q in (NAMED_QUERIES[n] for n in sorted(NAMED_QUERIES))
                ],
            },
            source_ref="cortex://bigquery/named-queries",
        )

    async def run_named_query(
        self, ctx: ToolContext, *, query_name: str, params: dict[str, Any]
    ) -> ToolResult:
        query = NAMED_QUERIES.get(query_name)
        if query is None:
            raise InvalidParams(
                f"unknown query {query_name!r}; available: {', '.join(sorted(NAMED_QUERIES))}"
            )

        # The per-query schema is validated here because the outer capability
        # schema can only say "params is an object" — it cannot vary by query name.
        import jsonschema

        try:
            jsonschema.validate(params, query.params_schema)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(p) for p in exc.absolute_path) or "(root)"
            raise InvalidParams(f"{query_name}: params.{path}: {exc.message}") from None

        project, dataset, table = self._target(ctx, query.table_key)

        try:
            sql = query.sql.format(
                project=project,
                dataset=dataset,
                table=table,
                # Resolved from the closed enum, so a column name can never arrive
                # unvalidated from the model.
                dimension_column=_DIMENSION_COLUMNS.get(str(params.get("dimension")), ""),
            )
        except (KeyError, IndexError, ValueError) as exc:
            # A named query referencing a placeholder this code does not supply.
            # str.format raises rather than leaving the placeholder in place, so this
            # is where that class of authoring mistake surfaces — reported by name
            # instead of as an opaque failure the loop cannot act on.
            raise ToolError(
                f"named query {query_name!r} references a placeholder Cortex does not "
                f"provide ({type(exc).__name__}: {exc}); fix the query definition"
            ) from exc

        if "{" in sql or "}" in sql:
            # Reachable via an escaped brace ({{ or }}), which format() collapses to a
            # literal rather than rejecting. A stray brace in generated SQL is not
            # worth sending to BigQuery either way.
            raise ToolError(
                f"named query {query_name!r} produced SQL containing an unexpected brace"
            )

        body = {
            "query": " ".join(sql.split()),
            "useLegacySql": False,
            "useQueryCache": True,
            # A runaway scan fails the job instead of arriving as an invoice.
            "maximumBytesBilled": str(MAX_BYTES_BILLED),
            "timeoutMs": 30_000,
            "parameterMode": "NAMED",
            "queryParameters": _query_parameters(params, query.params_schema),
        }

        async with await self._client(ctx) as client:
            raw = await request_json(
                client, "POST", f"/projects/{project}/queries", tool=self.name, json_body=body
            )

        if not raw.get("jobComplete", False):
            # Returning partial results as though complete would be a grounding
            # failure: the report would cite a number that is still being computed.
            raise ToolError(
                f"bigquery: {query_name} did not complete within the timeout; narrow the date range"
            )

        rows = _parse_rows(raw)
        return ToolResult(
            payload={
                "query_name": query_name,
                "params": params,
                "target": f"{project}.{dataset}.{table}",
                "row_count": len(rows),
                "rows": rows,
                "bytes_processed": _int(raw.get("totalBytesProcessed")),
                "cache_hit": bool(raw.get("cacheHit")),
            },
            source_ref=f"bigquery://{project}.{dataset}.{table}#{query_name}",
            meta={"total_rows": _int(raw.get("totalRows"))},
        )


# ---------------------------------------------------------------------- parsing


def _query_parameters(params: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Build BigQuery named parameters from validated values.

    Types come from the query's own schema rather than from Python inference, so a
    value cannot change the parameter type it binds as.
    """
    properties = schema.get("properties", {})
    out = []
    for key, value in params.items():
        spec = properties.get(key, {})
        json_type = spec.get("type")
        if json_type == "integer":
            bq_type, raw = "INT64", str(int(value))
        elif json_type == "number":
            bq_type, raw = "FLOAT64", str(float(value))
        elif json_type == "boolean":
            bq_type, raw = "BOOL", "true" if value else "false"
        elif spec.get("pattern") == _DATE["pattern"]:
            bq_type, raw = "DATE", str(value)
        else:
            bq_type, raw = "STRING", str(value)
        out.append(
            {
                "name": key,
                "parameterType": {"type": bq_type},
                "parameterValue": {"value": raw},
            }
        )
    return out


def _parse_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten BigQuery's schema-plus-rows response into dicts of typed values."""
    fields = [f for f in (raw.get("schema") or {}).get("fields", []) if isinstance(f, dict)]
    names = [f.get("name", "") for f in fields]
    types = [f.get("type", "STRING") for f in fields]

    rows = []
    for row in raw.get("rows", []):
        values = [cell.get("v") for cell in row.get("f", []) if isinstance(cell, dict)]
        rows.append(
            {
                name: _coerce(value, field_type)
                for name, field_type, value in zip(names, types, values, strict=False)
            }
        )
    return rows


def _coerce(value: Any, field_type: str) -> Any:
    """BigQuery returns every scalar as a string; typed values are what charts need."""
    if value is None:
        return None
    if field_type in ("INTEGER", "INT64"):
        return _int(value)
    if field_type in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if field_type in ("BOOLEAN", "BOOL"):
        return str(value).lower() == "true"
    return value


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
