"""DABstep — real data-analysis tasks from Adyen's operational workloads.

The closest public analogue to what Cortex does: multi-step questions over a real payments
dataset, answered as factoids and graded against a key. Frontier reasoning agents score
around 16%, which is the main reason it is worth running — a benchmark everything passes
measures nothing.

What it does and does not tell us. It exercises the analyst's ability to reason over data
it has never seen, using SQL it writes itself. It does **not** exercise the product's
actual claim — multi-source causal investigation with a resolvable citation behind every
sentence — because DABstep asks for a value, not an argument. So the score belongs on the
component list, never presented as a product score.

**The dev split, and why.** The 450-task set withholds answers for leaderboard submission;
the dev split publishes them, so it can be graded locally without an upload. Ten tasks is
a small sample and the harness prints the denominator on every line for exactly that
reason.

**One adapter decision worth stating.** Cortex answers with a report, DABstep wants a
formatted value ("NL", "B. BE"). So after the investigation a single structured call
extracts the answer from the report. That call does no analysis — it reformats a conclusion
already reached — and separating it keeps a formatting mismatch from being scored as an
analysis failure. It is disclosed here because it is the one place the adapter could be
accused of flattering the result.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker

from cortex.agents.anthropic_llm import DEFAULT_EFFORT, DEFAULT_MODEL, AnthropicLLM
from cortex.agents.employee import gtm_data_analyst
from cortex.agents.investigator import Investigator
from cortex.agents.llm import Message, Usage
from cortex.bench.harness import Task, TaskResult
from cortex.db.models import Investigation as InvestigationRow
from cortex.db.models import Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.runtime.resources import Resources
from cortex.tenancy.context import TenantContext
from cortex.tools.base import ToolRegistry
from cortex.tools.executor import ToolExecutor
from cortex.tools.sql import SQLTool

TASKS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=adyen%2FDABstep&config=tasks&split=dev&offset=0&length=100"
)

CONTEXT_BASE = "https://huggingface.co/datasets/adyen/DABstep/resolve/main/data/context"

#: Files the questions query. The two markdown documents are domain documentation rather
#: than data — DABstep intends them to be read, since the fee rules they describe cannot be
#: inferred from the tables — so they are given to the analyst as text rather than loaded
#: as tables.
DATA_FILES = (
    "payments.csv",
    "fees.json",
    "merchant_data.json",
    "merchant_category_codes.csv",
    "acquirer_countries.csv",
)
DOC_FILES = ("manual.md", "payments-readme.md")

#: Per-request deadline for benchmark work, above the product's 120 seconds.
#:
#: The product ceiling is right for the product: a GTM loop turn reads an API response and
#: takes about ten seconds, so 120 bounds a hung call without cutting off real work. A
#: benchmark task aggregating over 22MB is a different shape, and one task errored on the
#: first run with "no response within 120s" — the same mis-tuning as F-18, one constant
#: applied to calls of very different length, appearing in a new place.
_BENCH_TIMEOUT_SECONDS = 300.0

_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "The final answer, formatted exactly as the guidelines require and "
                "nothing else. No units, no explanation, no restatement of the question."
            ),
        }
    },
}


class DABstepAdapter:
    """Runs Cortex against DABstep, one task per investigation."""

    name = "dabstep-dev"

    def __init__(self, resources: Resources, workspace: Path) -> None:
        self._resources = resources
        self._workspace = Path(workspace)
        # A longer per-request deadline than the product default. 120 seconds is right for
        # the GTM loop, where a turn reads an API response and takes about ten; a single
        # aggregation over 22MB of payments data legitimately exceeds it, and one task in
        # the first run errored with "no response within 120s". Raised here rather than in
        # the provider, so the product keeps the ceiling that suits the product.
        self._llm = AnthropicLLM(
            model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, timeout=_BENCH_TIMEOUT_SECONDS
        )
        self._docs = _read_docs(self._workspace)

    # --------------------------------------------------------------------- loading

    def load(self) -> list[Task]:
        cache = self._workspace / "tasks.json"
        if cache.is_file():
            rows = json.loads(cache.read_text())
        else:
            response = httpx.get(TASKS_URL, timeout=60.0)
            response.raise_for_status()
            rows = [row["row"] for row in response.json()["rows"]]
            cache.write_text(json.dumps(rows, indent=1))

        return [
            Task(
                task_id=str(row["task_id"]),
                question=row["question"],
                payload={
                    "answer": str(row["answer"]),
                    "guidelines": row.get("guidelines", ""),
                    "level": row.get("level", ""),
                },
            )
            for row in rows
        ]

    # --------------------------------------------------------------------- solving

    async def solve(self, task: Task) -> TaskResult:
        registry = ToolRegistry()
        # SQL only. The GTM connectors would need credentials and have nothing to say
        # about a payments dataset, and offering tools that always fail would waste the
        # agent's steps discovering that.
        registry.register(SQLTool(workspace=self._workspace))

        investigator = Investigator(
            llm=self._llm,
            registry=registry,
            executor=ToolExecutor(registry),
            employee=gtm_data_analyst(),
        )

        maker = async_sessionmaker(self._resources.engine, expire_on_commit=False)
        async with maker() as session:
            tenant, investigation_id = await _provision(session, task)
            try:
                investigation = await investigator.investigate(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    question=_prompt(task, self._docs),
                )
            finally:
                # Rolled back: a benchmark must not leave a tenant and its evidence in
                # whatever database it was pointed at.
                await session.rollback()

        answer, extract_usage = await self._extract(task, investigation.report)
        expected = str(task.payload["answer"])
        return TaskResult(
            task_id=task.task_id,
            correct=_matches(answer, expected),
            answer=answer,
            expected=expected,
            usage=investigation.usage + extract_usage,
        )

    async def _extract(self, task: Task, report: Any) -> tuple[str, Usage]:
        """Pull the formatted answer out of the report. No analysis happens here."""
        summary = " ".join(claim.text for claim in report.executive_summary)
        findings = " ".join(claim.text for finding in report.findings for claim in finding.claims)
        payload, usage = await self._llm.structured(
            system=(
                "You extract a final answer from an analyst's report. You do no analysis "
                "and add no information: if the report does not contain the answer, say "
                "exactly 'Not Applicable'."
            ),
            messages=[
                Message(
                    role="user",
                    content=(
                        f"QUESTION: {task.question}\n\n"
                        f"REQUIRED FORMAT: {task.payload.get('guidelines', '')}\n\n"
                        f"REPORT:\n{summary}\n{findings}"[:20000]
                    ),
                )
            ],
            schema=_ANSWER_SCHEMA,
            max_tokens=512,
        )
        return str(payload.get("answer", "")).strip(), usage


def _provision(session: Any, task: Task) -> Any:
    async def _run() -> tuple[TenantContext, uuid.UUID]:
        tenant_id = uuid.uuid4()
        slug = f"bench-dabstep-{tenant_id.hex[:8]}"
        graph_name = graph_name_for_new_tenant(slug, tenant_id)
        session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
        await session.flush()
        row = InvestigationRow(tenant_id=tenant_id, question=task.question[:2000])
        session.add(row)
        await session.flush()
        return (
            TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name),
            row.id,
        )

    return _run()


def _read_docs(workspace: Path) -> str:
    """The domain documentation, concatenated.

    Given as text rather than loaded as tables because it is prose: the fee rules in
    `manual.md` cannot be derived from the data, and DABstep intends them to be read.
    """
    parts: list[str] = []
    for name in DOC_FILES:
        path = workspace / name
        if path.is_file():
            parts.append(f"--- {name} ---\n{path.read_text(errors='replace')}")
    return "\n\n".join(parts)


def _prompt(task: Task, docs: str) -> str:
    guidelines = task.payload.get("guidelines", "")
    return (
        f"{task.question}\n\n"
        f"ANSWER FORMAT REQUIRED: {guidelines}\n\n"
        "The data is available through the `sql` tool. Call `sql__list_tables` first to "
        "see the tables and columns — do not guess names.\n\n"
        "Domain documentation follows. It defines rules that cannot be inferred from the "
        "tables, so read it before writing queries.\n\n"
        f"{docs}"
    )


def _matches(answer: str, expected: str) -> bool:
    """Compare an answer to the key, tolerating formatting but not content.

    DABstep grades factoids with some tolerance for formatting, and the same is needed
    here: "NL" and "nl" are the same answer, and a trailing period is not a wrong answer.
    What is deliberately *not* tolerated is a substring match — "BE" appearing inside a
    sentence is not the same as answering "BE", and accepting it would inflate the score
    on exactly the tasks where the agent failed to commit to a value.
    """
    return _normalise(answer) == _normalise(expected)


def _normalise(value: str) -> str:
    cleaned = value.strip().lower().rstrip(".").strip()
    # Thousands separators and currency symbols vary between a model's prose and a key.
    for junk in ("$", ",", "%"):
        cleaned = cleaned.replace(junk, "")
    return " ".join(cleaned.split())
