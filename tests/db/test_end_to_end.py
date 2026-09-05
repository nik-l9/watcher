"""The end-to-end path: question in, grounded report out.

Everything up to now tested components. This tests the sequence — service
orchestration, the worker that drives it, and the endpoints that accept a question and
serve the result. It is the difference between "the pieces are verified" and "the
product runs".

Deterministic: a scripted provider stands in for the model, and Celery is bypassed by
invoking the task function directly, so the harness proves the wiring rather than the
broker's delivery semantics (which `test_worker_tasks.py` covers separately).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.investigator import InvestigationFailed
from cortex.agents.llm import LLMResponse, RecordedLLM, ToolRequest, Usage
from cortex.agents.service import InvestigationService
from cortex.config.settings import Settings, get_settings
from cortex.db.models import (
    Credential,
    CredentialProvider,
    Evidence,
    Investigation,
    InvestigationStatus,
    Report,
    Tenant,
    ToolCall,
    User,
)
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.reports.gate import ReportRejected
from cortex.runtime.resources import open_resources
from cortex.security.vault import encrypt_credential
from cortex.tenancy.context import TenantContext


def _in_own_loop(dsn: str, work):  # type: ignore[no-untyped-def]
    """Run an async seeding step on its own engine and event loop.

    The `session` fixture belongs to the test's event loop. Reusing it inside
    `asyncio.run` attaches its connection to a second loop and raises "attached to a
    different loop", so sync tests that need committed rows build their own engine.
    """
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    async def _main():  # type: ignore[no-untyped-def]
        engine = create_async_engine(dsn)
        try:
            async with async_sessionmaker(engine, expire_on_commit=False)() as s:
                result = await work(s)
                await s.commit()
                return result
        finally:
            await engine.dispose()

    return asyncio.run(_main())


# --------------------------------------------------------------------- scripting


class _Analyst(RecordedLLM):
    """A provider that cites the evidence the loop actually gathered."""

    def __init__(self, completions, report_builder, verdict: str = "supported") -> None:  # type: ignore[no-untyped-def]
        super().__init__(completions=completions)
        self._build = report_builder
        self._verdict = verdict
        self._ids: list[uuid.UUID] = []
        self.verdict_calls = 0

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        for message in kwargs.get("messages", []):
            for rendered in message.tool_results.values():
                for line in rendered.splitlines():
                    if line.startswith("evidence_id: "):
                        self._ids.append(uuid.UUID(line.split(": ", 1)[1].strip()))
        return await super().complete(**kwargs)

    async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
        is_verdict = "verdict" in kwargs.get("schema", {}).get("properties", {})
        # Recorded explicitly: this override does not call super(), so without this the
        # call log would be empty and any assertion about call counts would pass
        # vacuously.
        self.calls.append({"kind": "structured", "verdict_call": is_verdict})
        if is_verdict:
            self.verdict_calls += 1
            return {"verdict": self._verdict, "reason": "scripted"}, Usage(
                input_tokens=60, output_tokens=15
            )
        return self._build(list(dict.fromkeys(self._ids))), Usage(
            input_tokens=400, output_tokens=180
        )


def _tool_turn() -> LLMResponse:
    return LLMResponse(
        text="Checking sessions.",
        tool_requests=[
            ToolRequest(
                id=f"t_{uuid.uuid4().hex[:6]}",
                name="ga4__get_sessions",
                arguments={"start_date": "2026-07-01", "end_date": "2026-07-21"},
            )
        ],
        usage=Usage(input_tokens=500, output_tokens=120),
    )


def _done() -> LLMResponse:
    return LLMResponse(text="Concluding.", usage=Usage(input_tokens=200, output_tokens=50))


def _report(ids: list[uuid.UUID]) -> dict:
    return {
        "question": "Why did signups fall last week?",
        "executive_summary": [
            {
                "text": "Signups fell 18%, concentrated in mobile.",
                "evidence_ids": [str(i) for i in ids[:1]],
            }
        ],
        "hypotheses": [
            {
                "statement": "A deploy caused the drop",
                "verdict": "supported",
                "supporting_evidence_ids": [str(ids[0])],
                "reasoning": "The decline begins the day after the deploy.",
            }
        ],
        "recommendations": [
            {
                "action": "Roll back the onboarding modal",
                "rationale": "It is the only change coincident with the drop.",
                "evidence_ids": [str(ids[0])],
            }
        ],
        "risks": [{"description": "GA4 figures may be sampled."}],
        "confidence": "high",
    }


def _fabricating(ids: list[uuid.UUID]) -> dict:
    body = _report(ids)
    body["executive_summary"].append(
        {"text": "Revenue also fell 40%.", "evidence_ids": [str(uuid.uuid4())]}
    )
    return body


# --------------------------------------------------------------------- fixtures


async def _tenant(session: AsyncSession, slug: str = "e2e") -> TenantContext:
    """A tenant with a GA4 credential connected.

    The credential matters: the executor resolves and decrypts one before dispatching,
    so a tenant without it gets CredentialMissing and the investigation gathers nothing.
    That is the real behaviour — an unconnected connector is unavailable — so the
    fixture connects one rather than the executor being bypassed.
    """
    slug = f"{slug}-{uuid.uuid4().hex[:6]}"
    tenant_id = uuid.uuid4()
    session.add(
        Tenant(
            id=tenant_id,
            slug=slug,
            name=slug,
            graph_name=graph_name_for_new_tenant(slug, tenant_id),
        )
    )
    await session.flush()

    wrapped, ciphertext = encrypt_credential(
        tenant_id, CredentialProvider.GA4.value, '{"type": "service_account"}'
    )
    session.add(
        Credential(
            tenant_id=tenant_id,
            provider=CredentialProvider.GA4,
            wrapped_data_key=wrapped,
            ciphertext=ciphertext,
            metadata_={"property_id": "123456789"},
        )
    )
    await session.flush()

    return TenantContext(
        tenant_id=tenant_id,
        tenant_slug=slug,
        graph_name=graph_name_for_new_tenant(slug, tenant_id),
    )


async def _queued(session: AsyncSession, ctx: TenantContext) -> uuid.UUID:
    row = Investigation(tenant_id=ctx.tenant_id, question="Why did signups fall last week?")
    session.add(row)
    await session.flush()
    return row.id


@pytest.fixture(autouse=True)
def _stub_ga4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve GA4 from a fixture instead of Google.

    Only the connector's transport is replaced — the real tool, the real executor and
    the real evidence write all run, because the grounding gate resolves against the
    rows the executor writes.
    """
    from cortex.tools import ga4 as ga4_module
    from cortex.tools.base import ToolResult

    async def _get_sessions(self, ctx, *, start_date, end_date, dimensions=None, limit=50):  # type: ignore[no-untyped-def]
        del self, ctx, dimensions, limit
        return ToolResult(
            payload={
                "start_date": start_date,
                "end_date": end_date,
                "totals": {"sessions": 41200},
                "rows": [
                    {
                        "dimensions": {"deviceCategory": "mobile"},
                        "metrics": {"sessions": 9800, "conversions": 284},
                    }
                ],
            },
            source_ref="ga4://fixture",
        )

    monkeypatch.setattr(ga4_module.GA4Tool, "get_sessions", _get_sessions)


# --------------------------------------------------------------------- service


class TestInvestigationService:
    async def test_a_completed_run_persists_a_grounded_report(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _queued(session, ctx)
        llm = _Analyst([_tool_turn(), _done()], _report)

        completed = await InvestigationService(llm=llm).run(
            session, ctx, investigation_id=investigation_id
        )

        row = await session.get(Investigation, investigation_id)
        assert row is not None
        assert row.status is InvestigationStatus.COMPLETED
        assert row.started_at is not None and row.completed_at is not None
        assert row.tokens_used > 0
        assert row.steps_used > 0
        # The hypotheses the loop tested are kept, including what was ruled out.
        assert row.hypotheses and row.hypotheses[0]["verdict"] == "supported"

        report = await session.get(Report, completed.report_id)
        assert report is not None
        assert report.body["executive_summary"][0]["text"].startswith("Signups fell 18%")
        assert report.confidence == pytest.approx(0.9)
        # Sources are derived by the gate, never authored by the model.
        assert report.body["sources"], "the gate should have populated sources"
        assert completed.hallucinations == 0

    async def test_evidence_is_written_through_the_real_executor(
        self, session: AsyncSession
    ) -> None:
        """If the executor were stubbed the grounding score would be meaningless, since
        the gate resolves against the rows it writes."""
        ctx = await _tenant(session)
        investigation_id = await _queued(session, ctx)

        await InvestigationService(llm=_Analyst([_tool_turn(), _done()], _report)).run(
            session, ctx, investigation_id=investigation_id
        )

        evidence = (
            (
                await session.execute(
                    select(Evidence).where(Evidence.investigation_id == investigation_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(evidence) == 1
        assert evidence[0].tool_name == "ga4"
        assert evidence[0].payload["totals"]["sessions"] == 41200

    async def test_a_fabricated_citation_is_removed_and_disclosed(
        self, session: AsyncSession
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _queued(session, ctx)

        completed = await InvestigationService(
            llm=_Analyst([_tool_turn(), _done()], _fabricating)
        ).run(session, ctx, investigation_id=investigation_id)

        report = await session.get(Report, completed.report_id)
        assert report is not None
        texts = [c["text"] for c in report.body["executive_summary"]]
        assert "Revenue also fell 40%." not in texts
        assert completed.hallucinations >= 1
        # Recorded on the row, so a reader can see something was removed.
        assert report.gate_rejections

    async def test_a_verifier_rejection_removes_the_claim(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _queued(session, ctx)
        llm = _Analyst([_tool_turn(), _done()], _report, verdict="unsupported")

        with pytest.raises(ReportRejected):
            await InvestigationService(llm=llm).run(session, ctx, investigation_id=investigation_id)

        row = await session.get(Investigation, investigation_id)
        assert row is not None
        assert row.status is InvestigationStatus.FAILED
        assert "verification rejected" in (row.error or "")

    async def test_verification_can_be_skipped(self, session: AsyncSession) -> None:
        """Useful when iterating on the loop; the gate still runs."""
        ctx = await _tenant(session)
        investigation_id = await _queued(session, ctx)
        llm = _Analyst([_tool_turn(), _done()], _report, verdict="unsupported")

        completed = await InvestigationService(llm=llm, verify=False).run(
            session, ctx, investigation_id=investigation_id
        )
        report = await session.get(Report, completed.report_id)
        assert report is not None
        assert report.verifier_rejections == []

    async def test_a_failed_loop_records_the_failure_on_the_row(
        self, session: AsyncSession
    ) -> None:
        """An investigation that dies must not leave a row still claiming to be queued."""
        ctx = await _tenant(session)
        investigation_id = await _queued(session, ctx)

        with pytest.raises(InvestigationFailed):
            await InvestigationService(llm=RecordedLLM()).run(
                session, ctx, investigation_id=investigation_id
            )

        row = await session.get(Investigation, investigation_id)
        assert row is not None
        assert row.status is InvestigationStatus.FAILED
        assert row.error
        assert row.completed_at is not None

    async def test_another_tenants_investigation_cannot_be_run(self, session: AsyncSession) -> None:
        """Same rule as F-01: a redelivered or hand-crafted message must not be able to
        run another tenant's investigation."""
        victim = await _tenant(session, "e2e-victim")
        attacker = await _tenant(session, "e2e-attacker")
        victim_investigation = await _queued(session, victim)

        with pytest.raises(InvestigationFailed, match="not available to this tenant"):
            await InvestigationService(llm=_Analyst([_tool_turn(), _done()], _report)).run(
                session, attacker, investigation_id=victim_investigation
            )

    async def test_the_gate_runs_before_the_verifier(self, session: AsyncSession) -> None:
        """Verification costs a model call per claim, so there is no point paying to
        read a claim whose citation does not exist."""
        ctx = await _tenant(session)
        investigation_id = await _queued(session, ctx)
        llm = _Analyst([_tool_turn(), _done()], _fabricating)

        await InvestigationService(llm=llm).run(session, ctx, investigation_id=investigation_id)

        # Two summary claims were drafted; the gate removed the fabricated one, so
        # only the surviving claim was sent for verification.
        assert llm.verdict_calls == 1


# --------------------------------------------------------------------- worker


class TestWorkerRunsItForReal:
    _dsn: str = ""

    @pytest.fixture(autouse=True)
    def _local(self, _test_database: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from cortex.runtime import resources as resources_module

        settings = get_settings().model_copy(update={"postgres_dsn": _test_database})
        monkeypatch.setattr(resources_module, "get_settings", lambda: settings)
        type(self)._dsn = _test_database

    def test_the_task_completes_an_investigation(
        self, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The task body drives the real service — no `not_implemented` stub."""
        from services.investigation_worker import worker

        # Committed on its own engine, since the task opens its own session and loop.
        async def _seed(s: AsyncSession) -> tuple[uuid.UUID, uuid.UUID]:
            ctx = await _tenant(s)
            return ctx.tenant_id, await _queued(s, ctx)

        tenant_id, investigation_id = _in_own_loop(self._dsn, _seed)

        monkeypatch.setattr(
            worker, "build_llm", lambda *a, **k: _Analyst([_tool_turn(), _done()], _report)
        )
        result = worker.run_investigation(
            {"tenant_id": str(tenant_id), "investigation_id": str(investigation_id)}
        )

        assert result["status"] == "completed"
        assert result["hallucinations"] == 0
        assert result["report_id"]
        assert result["tokens"] > 0

    def test_an_unknown_tenant_is_skipped_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A queued message can outlive its tenant. Retrying would never succeed."""
        from services.investigation_worker import worker

        result = worker.run_investigation(
            {"tenant_id": str(uuid.uuid4()), "investigation_id": str(uuid.uuid4())}
        )
        assert result["status"] == "skipped"
        assert "tenant unavailable" in result["reason"]

    def test_a_refused_report_is_reported_not_raised(
        self, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raising would make Celery retry, spending the tokens again to reach the same
        refusal."""
        from services.investigation_worker import worker

        async def _seed(s: AsyncSession) -> tuple[uuid.UUID, uuid.UUID]:
            ctx = await _tenant(s)
            return ctx.tenant_id, await _queued(s, ctx)

        tenant_id, investigation_id = _in_own_loop(self._dsn, _seed)
        monkeypatch.setattr(worker, "build_llm", lambda *a, **k: RecordedLLM())

        result = worker.run_investigation(
            {"tenant_id": str(tenant_id), "investigation_id": str(investigation_id)}
        )
        assert result["status"] == "failed"
        assert result["reason"]


# --------------------------------------------------------------------- endpoints


@pytest.fixture
def app(_test_database: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    """The real gateway, with the queue publish captured instead of sent."""
    from services.gateway import investigations as endpoints
    from services.gateway.app import create_app

    settings: Settings = get_settings().model_copy(
        update={"postgres_dsn": _test_database, "env": "test"}
    )

    sent: list[dict] = []
    monkeypatch.setattr(
        endpoints._producer,
        "send_task",
        lambda name, args=None, **kw: sent.append(
            {"name": name, "args": args, "queue": kw.get("queue")}
        ),
    )

    application = create_app()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with open_resources(settings) as resources:
            _app.state.resources = resources
            yield

    application.router.lifespan_context = lifespan  # type: ignore[assignment]
    application.state.published = sent
    yield application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded(_test_database: str) -> str:
    """A committed tenant the HTTP request can resolve."""

    async def _seed(s: AsyncSession) -> str:
        ctx = await _tenant(s, "api")
        s.add(User(tenant_id=ctx.tenant_id, external_id="api_user", email="a@example.com"))
        return ctx.tenant_slug

    return _in_own_loop(_test_database, _seed)


class TestEndpoints:
    def test_asking_a_question_returns_202_and_queues_it(
        self, client: TestClient, seeded: str
    ) -> None:
        """202, not 200: the work is accepted, not performed."""
        response = client.post(
            "/investigations",
            headers={"X-Cortex-Tenant": seeded},
            json={"question": "Why did signups fall last week?"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert uuid.UUID(body["id"])

        published = client.app.state.published  # type: ignore[attr-defined]
        assert len(published) == 1
        assert published[0]["name"] == "cortex.investigation.run"
        assert published[0]["queue"] == "cortex.investigation"
        assert published[0]["args"][0]["investigation_id"] == body["id"]

    def test_the_row_exists_before_the_message_is_published(
        self, client: TestClient, seeded: str
    ) -> None:
        """Publishing first would race: a fast worker could look for a row the request
        transaction has not written yet."""
        response = client.post(
            "/investigations",
            headers={"X-Cortex-Tenant": seeded},
            json={"question": "Why did signups fall last week?"},
        )
        investigation_id = response.json()["id"]
        state = client.get(
            f"/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert state.status_code == 200
        assert state.json()["status"] == "queued"

    def test_a_short_question_is_refused(self, client: TestClient, seeded: str) -> None:
        response = client.post(
            "/investigations", headers={"X-Cortex-Tenant": seeded}, json={"question": "why"}
        )
        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self, client: TestClient, seeded: str) -> None:
        response = client.post(
            "/investigations",
            headers={"X-Cortex-Tenant": seeded},
            json={"question": "Why did signups fall?", "invented": True},
        )
        assert response.status_code == 422

    def test_an_unknown_investigation_is_404(self, client: TestClient, seeded: str) -> None:
        response = client.get(
            f"/investigations/{uuid.uuid4()}", headers={"X-Cortex-Tenant": seeded}
        )
        assert response.status_code == 404

    def test_an_unknown_report_is_404(self, client: TestClient, seeded: str) -> None:
        response = client.get(f"/reports/{uuid.uuid4()}", headers={"X-Cortex-Tenant": seeded})
        assert response.status_code == 404

    def test_a_missing_tenant_header_is_422(self, client: TestClient) -> None:
        response = client.post(
            "/investigations", json={"question": "Why did signups fall last week?"}
        )
        assert response.status_code == 422


class TestEndpointTenantScoping:
    def test_another_tenants_investigation_is_404_not_403(
        self, client: TestClient, _test_database: str, seeded: str
    ) -> None:
        """One response for absent and foreign, so ids cannot be enumerated."""

        async def _other(s: AsyncSession) -> uuid.UUID:
            other = await _tenant(s, "api-other")
            return await _queued(s, other)

        foreign_id = _in_own_loop(_test_database, _other)

        response = client.get(f"/investigations/{foreign_id}", headers={"X-Cortex-Tenant": seeded})
        assert response.status_code == 404

    def test_another_tenants_report_is_404(
        self, client: TestClient, _test_database: str, seeded: str
    ) -> None:
        async def _other(s: AsyncSession) -> uuid.UUID:
            other = await _tenant(s, "api-other")
            investigation_id = await _queued(s, other)
            report = Report(
                tenant_id=other.tenant_id,
                investigation_id=investigation_id,
                body={"secret": "another tenant's report"},
            )
            s.add(report)
            await s.flush()
            return report.id

        foreign_report = _in_own_loop(_test_database, _other)

        response = client.get(f"/reports/{foreign_report}", headers={"X-Cortex-Tenant": seeded})
        assert response.status_code == 404
        assert "another tenant" not in response.text


class TestReportRetrieval:
    def test_a_completed_report_is_served_with_its_rejections(
        self, client: TestClient, _test_database: str, seeded: str
    ) -> None:
        """Removals are surfaced, not hidden: a reader is entitled to know claims were
        removed before this was shown to them."""

        async def _complete(s: AsyncSession) -> tuple[uuid.UUID, uuid.UUID]:
            tenant_row = (await s.execute(select(Tenant).where(Tenant.slug == seeded))).scalar_one()
            ctx = TenantContext(
                tenant_id=tenant_row.id,
                tenant_slug=tenant_row.slug,
                graph_name=tenant_row.graph_name,
            )
            investigation_id = await _queued(s, ctx)
            completed = await InvestigationService(
                llm=_Analyst([_tool_turn(), _done()], _fabricating)
            ).run(s, ctx, investigation_id=investigation_id)
            return investigation_id, completed.report_id

        investigation_id, report_id = _in_own_loop(_test_database, _complete)

        state = client.get(
            f"/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        ).json()
        assert state["status"] == "completed"
        assert state["report_id"] == str(report_id)

        report = client.get(f"/reports/{report_id}", headers={"X-Cortex-Tenant": seeded})
        assert report.status_code == 200
        body = report.json()
        assert body["report"]["executive_summary"]
        assert body["removed_by_grounding"], "a removed claim must be disclosed"
        assert body["confidence"] is not None


class TestCancellation:
    """`CancelInvestigation` was a declared contract message and `CANCELLED` a declared status,
    and nothing implemented either — there was no way to stop a running investigation, or to
    stop it spending tokens.

    Cooperative rather than pre-emptive. The usual shape is a threading flag on the
    conversation, which works when the loop and the cancel request share a process. These do
    not: the loop runs in a Celery worker and the request arrives at the gateway, so the signal
    has to cross a process boundary. It travels through the investigation row, which
    this codebase already treats as the authoritative state.
    """

    def test_cancelling_marks_the_row(self, client: TestClient, seeded: str) -> None:
        created = client.post(
            "/investigations",
            json={"question": "Why did signups fall last week?"},
            headers={"X-Cortex-Tenant": seeded},
        )
        investigation_id = created.json()["id"]

        response = client.post(
            f"/investigations/{investigation_id}/cancel", headers={"X-Cortex-Tenant": seeded}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        # Recorded on the row rather than left blank, so the audit trail distinguishes a
        # cancellation from a crash without cross-referencing anything.
        assert "cancelled by request" in (response.json()["error"] or "")

    def test_cancelling_twice_is_not_an_error(self, client: TestClient, seeded: str) -> None:
        """A user pressing a button twice has not done anything wrong."""
        created = client.post(
            "/investigations",
            json={"question": "Why did signups fall last week?"},
            headers={"X-Cortex-Tenant": seeded},
        )
        investigation_id = created.json()["id"]

        first = client.post(
            f"/investigations/{investigation_id}/cancel", headers={"X-Cortex-Tenant": seeded}
        )
        second = client.post(
            f"/investigations/{investigation_id}/cancel", headers={"X-Cortex-Tenant": seeded}
        )

        assert first.status_code == second.status_code == 200
        assert second.json()["status"] == "cancelled"

    def test_another_tenant_cannot_cancel(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The same 404 as an absent investigation, so ids cannot be enumerated by watching
        which cancellations are refused."""
        created = client.post(
            "/investigations",
            json={"question": "Why did signups fall last week?"},
            headers={"X-Cortex-Tenant": seeded},
        )
        investigation_id = created.json()["id"]

        async def _other(s: AsyncSession) -> str:
            ctx = await _tenant(s, "api-other")
            return ctx.tenant_slug

        other_tenant = _in_own_loop(_test_database, _other)
        response = client.post(
            f"/investigations/{investigation_id}/cancel",
            headers={"X-Cortex-Tenant": other_tenant},
        )

        assert response.status_code == 404

    def test_an_unknown_investigation_is_404(self, client: TestClient, seeded: str) -> None:
        response = client.post(
            f"/investigations/{uuid.uuid4()}/cancel", headers={"X-Cortex-Tenant": seeded}
        )
        assert response.status_code == 404


class TestTheList:
    """The endpoint that makes the product navigable.

    Until this existed an investigation could only be read by an id the caller already held,
    so nothing could show what had been asked before. The tests that matter are the boundary
    ones: a list is where tenant isolation and pagination both fail quietly.
    """

    def _ask(self, client: TestClient, tenant: str, question: str) -> str:
        response = client.post(
            "/investigations", headers={"X-Cortex-Tenant": tenant}, json={"question": question}
        )
        assert response.status_code == 202
        return response.json()["id"]

    def test_it_returns_this_tenants_investigations_with_titles(
        self, client: TestClient, seeded: str
    ) -> None:
        self._ask(client, seeded, "Why did signups fall last week?")
        response = client.get("/investigations", headers={"X-Cortex-Tenant": seeded})

        assert response.status_code == 200
        body = response.json()
        assert body["total"] >= 1
        row = body["investigations"][0]
        # The title is what a list renders. A row showing the raw question is the reason
        # `cortex/db/titles.py` exists.
        assert row["title"] == "Why did signups fall last week"
        assert row["status"] == "queued"
        assert row["report_id"] is None

    def test_newest_first(self, client: TestClient, seeded: str) -> None:
        self._ask(client, seeded, "Why did signups fall in June?")
        second = self._ask(client, seeded, "Why did signups fall in July?")

        body = client.get("/investigations", headers={"X-Cortex-Tenant": seeded}).json()
        assert body["investigations"][0]["id"] == second

    def test_another_tenants_investigations_are_absent(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The failure this endpoint makes possible for the first time. Every other route
        takes an id the caller must already know; a list hands back rows it chose itself, so
        one missing predicate leaks a tenant's entire history."""

        async def _other(s: AsyncSession) -> str:
            ctx = await _tenant(s, "api-list-other")
            return ctx.tenant_slug

        other = _in_own_loop(_test_database, _other)
        mine = self._ask(client, seeded, "Why did my signups fall?")
        self._ask(client, other, "Why did their signups fall?")

        body = client.get("/investigations", headers={"X-Cortex-Tenant": seeded}).json()
        ids = [row["id"] for row in body["investigations"]]
        assert mine in ids
        assert body["total"] == len(ids) == 1

    def test_paging_does_not_repeat_or_drop_a_row(self, client: TestClient, seeded: str) -> None:
        """Rows created in the same millisecond are free to swap places between pages unless
        the ordering is total, which shows one row twice and silently drops another."""
        for index in range(5):
            self._ask(client, seeded, f"Why did signups fall in week {index}?")

        first = client.get(
            "/investigations?limit=2&offset=0", headers={"X-Cortex-Tenant": seeded}
        ).json()
        second = client.get(
            "/investigations?limit=2&offset=2", headers={"X-Cortex-Tenant": seeded}
        ).json()
        third = client.get(
            "/investigations?limit=2&offset=4", headers={"X-Cortex-Tenant": seeded}
        ).json()

        seen = [row["id"] for page in (first, second, third) for row in page["investigations"]]
        assert len(seen) == 5
        assert len(set(seen)) == 5
        assert first["total"] == 5

    def test_an_unbounded_limit_is_refused(self, client: TestClient, seeded: str) -> None:
        """An unbounded list endpoint is a way to turn one request into a table scan of a
        tenant's whole history."""
        response = client.get("/investigations?limit=5000", headers={"X-Cortex-Tenant": seeded})
        assert response.status_code == 422

    def test_a_negative_offset_is_refused(self, client: TestClient, seeded: str) -> None:
        response = client.get("/investigations?offset=-1", headers={"X-Cortex-Tenant": seeded})
        assert response.status_code == 422


class TestTheTrace:
    """What the analyst actually did, in order.

    Derived from the `tool_calls` audit rows rather than from a stored event stream, so it
    cannot disagree with what happened. These tests are about the two things a reader needs
    from it: stable order, and the difference between "found nothing" and "could not look".
    """

    def _ask(self, client: TestClient, tenant: str) -> str:
        response = client.post(
            "/investigations",
            headers={"X-Cortex-Tenant": tenant},
            json={"question": "Why did signups fall last week?"},
        )
        return response.json()["id"]

    def _record(
        self,
        database: str,
        tenant_slug: str,
        investigation_id: str,
        calls: list[dict],
    ) -> None:
        """Write audit rows directly. The loop's own writes are tested elsewhere; this is
        about what the endpoint makes of them."""

        async def _write(s: AsyncSession) -> None:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == tenant_slug))
            for call in calls:
                s.add(
                    ToolCall(
                        tenant_id=tenant_id,
                        investigation_id=uuid.UUID(investigation_id),
                        tool_name=call["tool"],
                        capability=call["capability"],
                        params=call.get("params", {}),
                        succeeded=call.get("succeeded", True),
                        error=call.get("error"),
                        duration_ms=call.get("duration_ms"),
                        evidence_id=None,
                    )
                )

        _in_own_loop(database, _write)

    def test_it_is_available_while_the_investigation_is_still_running(
        self, client: TestClient, seeded: str
    ) -> None:
        """The point of the endpoint: it answers "is it stuck or is it working", which is
        only a question before the report exists."""
        investigation_id = self._ask(client, seeded)
        response = client.get(
            f"/investigations/{investigation_id}/trace", headers={"X-Cortex-Tenant": seeded}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert response.json()["steps"] == []

    def test_the_calls_appear_in_order_with_their_parameters(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        investigation_id = self._ask(client, seeded)
        self._record(
            _test_database,
            seeded,
            investigation_id,
            [
                {"tool": "ga4", "capability": "get_sessions", "params": {"days": 7}},
                {"tool": "github", "capability": "commits", "params": {"since": "2026-07-01"}},
            ],
        )

        body = client.get(
            f"/investigations/{investigation_id}/trace", headers={"X-Cortex-Tenant": seeded}
        ).json()
        assert [(step["tool"], step["capability"]) for step in body["steps"]] == [
            ("ga4", "get_sessions"),
            ("github", "commits"),
        ]
        assert body["steps"][0]["params"] == {"days": 7}

    def test_a_failed_call_is_shown_as_failed(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """A trace that hid failures would show an investigation doing less work than it
        did, and would hide the difference between an empty source and an unreachable one."""
        investigation_id = self._ask(client, seeded)
        self._record(
            _test_database,
            seeded,
            investigation_id,
            [
                {
                    "tool": "github",
                    "capability": "find_feature",
                    "succeeded": False,
                    "error": "repository not found",
                }
            ],
        )

        body = client.get(
            f"/investigations/{investigation_id}/trace", headers={"X-Cortex-Tenant": seeded}
        ).json()
        assert body["steps"][0]["succeeded"] is False
        assert body["steps"][0]["error"] == "repository not found"

    def test_calls_that_produced_no_observation_are_counted(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """ "Nothing happened" and "I could not look" are different facts, and a count is
        what stops the second being read as the first."""
        investigation_id = self._ask(client, seeded)
        self._record(
            _test_database,
            seeded,
            investigation_id,
            [
                {"tool": "slack", "capability": "search_messages"},
                {"tool": "posthog", "capability": "event_trend", "succeeded": False},
            ],
        )

        body = client.get(
            f"/investigations/{investigation_id}/trace", headers={"X-Cortex-Tenant": seeded}
        ).json()
        # Both: one succeeded without evidence, one failed outright.
        assert body["empty_or_failed"] == 2

    def test_no_response_body_is_exposed(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The audit row deliberately stores params and not responses, and the trace must not
        reintroduce a second surface where customer data reaches a UI."""
        investigation_id = self._ask(client, seeded)
        self._record(
            _test_database,
            seeded,
            investigation_id,
            [{"tool": "slack", "capability": "search_messages", "params": {"query": "apollo"}}],
        )

        step = client.get(
            f"/investigations/{investigation_id}/trace", headers={"X-Cortex-Tenant": seeded}
        ).json()["steps"][0]
        assert "response" not in step
        assert "body" not in step

    def test_another_tenant_cannot_read_a_trace(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        investigation_id = self._ask(client, seeded)

        async def _other(s: AsyncSession) -> str:
            ctx = await _tenant(s, "api-trace-other")
            return ctx.tenant_slug

        other = _in_own_loop(_test_database, _other)
        response = client.get(
            f"/investigations/{investigation_id}/trace", headers={"X-Cortex-Tenant": other}
        )
        assert response.status_code == 404

    def test_an_unknown_investigation_is_404(self, client: TestClient, seeded: str) -> None:
        response = client.get(
            f"/investigations/{uuid.uuid4()}/trace", headers={"X-Cortex-Tenant": seeded}
        )
        assert response.status_code == 404


class TestTheReportView:
    """The HTML page — M7.

    Two things are worth asserting here and nothing else is. **Citations must be links**, because
    "every claim visibly clickable to its evidence" is the product's whole argument and a page
    that renders a claim without one is the failure the gate exists to prevent. And **model
    output must not become markup**: claim text, finding titles and chart titles are all written
    by a model, and "we escape" is the kind of claim that stays true until someone adds a field.
    """

    def _completed(self, database: str, tenant_slug: str, *, body: dict) -> str:
        """A completed investigation with a stored report."""

        async def _write(s: AsyncSession) -> str:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == tenant_slug))
            investigation = Investigation(
                tenant_id=tenant_id,
                question="Why did signups fall last week?",
                title="Why did signups fall last week",
                status=InvestigationStatus.COMPLETED,
            )
            s.add(investigation)
            await s.flush()
            s.add(
                Report(
                    tenant_id=tenant_id,
                    investigation_id=investigation.id,
                    body=body,
                    confidence=0.6,
                    gate_rejections=[{"claim": "dropped"}],
                    verifier_rejections=[],
                )
            )
            return str(investigation.id)

        return _in_own_loop(database, _write)

    @staticmethod
    def _body(**overrides: object) -> dict:
        evidence_id = str(uuid.uuid4())
        body: dict = {
            "question": "Why did signups fall last week?",
            "executive_summary": [
                {
                    "text": "Signups fell 12% after the 14 July deploy.",
                    "evidence_ids": [evidence_id],
                }
            ],
            "findings": [],
            "hypotheses": [],
            "charts": [],
            "confidence": "medium",
            "risks": [],
            "recommendations": [],
            "data_quality": [],
            "sources": [
                {
                    "evidence_id": evidence_id,
                    "tool_name": "ga4",
                    "capability": "get_sessions",
                    "source_ref": "ga4://properties/1",
                    "from_cache": False,
                }
            ],
        }
        body.update(overrides)
        return body

    def test_a_citation_is_a_link_to_its_source(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The page's reason to exist. A citation rendered as a bare uuid is technically
        grounded and practically unreadable."""
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )

        assert page.status_code == 200
        assert 'href="#e-' in page.text
        # Labelled with the capability that produced it, and anchored to a row that exists.
        assert "ga4.get_sessions" in page.text
        assert '<tr id="e-' in page.text

    def test_the_link_opens_with_no_tenant_at_all(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The bug a user hit twice: clicking the link Cortex posted into Slack.

        First it was ERR_NGROK_3200, which masked a 422 -- the page required a tenant and a
        browser cannot send a header. Then it was `?tenant=` in the URL, which broke whenever a
        chat client truncated the query string. The id is the only secret in that link, so the
        page resolves the tenant from it.
        """
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get(f"/ui/investigations/{investigation_id}")
        assert page.status_code == 200
        assert "Signups fell 12% after the 14 July deploy." in page.text

    def test_an_explicit_tenant_still_wins(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The fallback is reached only when nobody said who they were, so a real deployment
        behind Clerk behaves exactly as before."""
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert page.status_code == 200

    def test_a_wrong_tenant_is_still_refused(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """Naming a tenant is a claim about who you are, and a claim that does not match the
        investigation must fail rather than fall back to resolving it."""
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get(
            f"/ui/investigations/{investigation_id}",
            headers={"X-Cortex-Tenant": "some-other-tenant"},
        )
        assert page.status_code in (403, 404)

    def test_an_unknown_investigation_is_not_enumerable(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """One response for absent and foreign, so resolving the tenant from the id cannot be
        turned into a probe for which ids exist."""
        page = client.get(f"/ui/investigations/{uuid.uuid4()}")
        assert page.status_code == 404
        assert "unknown investigation" in page.text

    def test_the_all_investigations_link_carries_a_tenant(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The list route has no id to resolve from, so following the link out of a working
        report page would otherwise be a dead end one click deeper."""
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get(f"/ui/investigations/{investigation_id}")
        assert f'href="/ui/investigations?tenant={seeded}"' in page.text
        assert client.get(f"/ui/investigations?tenant={seeded}").status_code == 200

    def test_removed_claims_are_disclosed(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """A reader is entitled to know claims were removed before this was shown to them —
        the difference between a report that was checked and one that merely looks tidy."""
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert "were removed before this report was shown" in page.text
        # Counted, not printed: both columns store the rejections as JSON lists, so a naive
        # interpolation would put a Python list on the page.
        assert "[{" not in page.text

    def test_model_output_cannot_become_markup(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        hostile = "<img src=x onerror=alert(1)><script>alert(2)</script>"
        body = self._body()
        body["executive_summary"][0]["text"] = f"Signups fell {hostile}"
        body["findings"] = [
            {
                "title": hostile,
                "claims": [
                    {
                        "text": hostile,
                        "evidence_ids": [body["sources"][0]["evidence_id"]],
                    }
                ],
                "confidence": "medium",
            }
        ]
        investigation_id = self._completed(_test_database, seeded, body=body)

        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert "<script>alert(2)</script>" not in page.text
        assert "<img src=x" not in page.text
        assert "&lt;img src=x" in page.text

    def test_a_chart_renders_as_svg(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        body = self._body()
        evidence_id = body["sources"][0]["evidence_id"]
        body["charts"] = [
            {
                "type": "line",
                "title": "Signups per day",
                "x_label": "day",
                "y_label": "signups",
                "series": [
                    {
                        "name": "signups",
                        "points": [
                            {"x": "2026-07-13", "y": 100.0},
                            {"x": "2026-07-14", "y": 88.0},
                            {"x": "2026-07-15", "y": 84.0},
                        ],
                    }
                ],
                "annotations": [
                    {"x": "2026-07-14", "label": "deploy 91c3e4a", "evidence_id": evidence_id}
                ],
                "evidence_ids": [evidence_id],
            }
        ]
        investigation_id = self._completed(_test_database, seeded, body=body)

        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert "<svg" in page.text
        assert "deploy 91c3e4a" in page.text

    def test_a_malformed_chart_does_not_take_the_page_down(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The report page is the last step, after everything expensive has succeeded. A
        chart that cannot be parsed is dropped with a note rather than losing the report."""
        body = self._body()
        body["charts"] = [{"type": "line", "title": "broken"}]
        investigation_id = self._completed(_test_database, seeded, body=body)

        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert page.status_code == 200
        assert "could not be rendered" in page.text
        assert "Signups fell 12%" in page.text

    def test_a_running_investigation_refreshes_and_shows_its_trace(
        self, client: TestClient, seeded: str
    ) -> None:
        """The complaint that started this work: a 63-second wait printing nothing was
        indistinguishable from a hang."""
        created = client.post(
            "/investigations",
            headers={"X-Cortex-Tenant": seeded},
            json={"question": "Why did signups fall last week?"},
        )
        page = client.get(
            f"/ui/investigations/{created.json()['id']}", headers={"X-Cortex-Tenant": seeded}
        )
        assert 'http-equiv="refresh"' in page.text
        assert "Still working" in page.text
        assert "How this was investigated" in page.text

    def test_a_finished_investigation_stops_refreshing(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """A completed page that keeps reloading is a page that keeps costing requests to
        show the same thing."""
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert 'http-equiv="refresh"' not in page.text

    def test_the_list_links_to_each_investigation(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        investigation_id = self._completed(_test_database, seeded, body=self._body())
        page = client.get("/ui/investigations", headers={"X-Cortex-Tenant": seeded})
        assert f"/ui/investigations/{investigation_id}" in page.text
        assert "Why did signups fall last week" in page.text

    def test_another_tenant_cannot_read_the_page(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        investigation_id = self._completed(_test_database, seeded, body=self._body())

        async def _other(s: AsyncSession) -> str:
            ctx = await _tenant(s, "api-ui-other")
            return ctx.tenant_slug

        other = _in_own_loop(_test_database, _other)
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": other}
        )
        assert page.status_code == 404

    def test_the_page_is_not_in_the_openapi_document(self, client: TestClient) -> None:
        """It is a rendering of the JSON endpoints, not a second API. Listing it as one
        would invite a client to scrape HTML for data the JSON already carries."""
        paths = client.app.openapi()["paths"]  # type: ignore[attr-defined]
        assert not any(path.startswith("/ui") for path in paths)


class TestAbandonedInvestigations:
    """A queued row nothing will ever pick up.

    Found by looking at real data rather than by reasoning: eleven of twelve investigations in
    the development database sit at `queued` permanently, because `POST /investigations`
    enqueues work and no worker consumed it. On a page "queued" and "queued since Tuesday"
    render identically, so a list of mostly-abandoned rows reads as a broken product rather
    than one whose worker is not running.

    Disclosed, never corrected. Marking the row FAILED during a page render would mean a GET
    that mutates state, and the gateway does not get to decide a worker is dead.
    """

    def _queued(self, database: str, tenant_slug: str, *, age_minutes: int) -> str:
        async def _write(s: AsyncSession) -> str:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == tenant_slug))
            row = Investigation(
                tenant_id=tenant_id,
                question="Why did signups fall last week?",
                title="Why did signups fall last week",
                status=InvestigationStatus.QUEUED,
                created_at=datetime.now(UTC) - timedelta(minutes=age_minutes),
            )
            s.add(row)
            await s.flush()
            return str(row.id)

        return _in_own_loop(database, _write)

    def test_a_long_queued_investigation_is_disclosed_as_abandoned(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        investigation_id = self._queued(_test_database, seeded, age_minutes=120)
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert "almost certainly been abandoned" in page.text
        assert "Still working" not in page.text

    def test_an_abandoned_page_stops_refreshing(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """Reloading every five seconds forever to show the same dead row is the one behaviour
        that turns a disclosure into a cost."""
        investigation_id = self._queued(_test_database, seeded, age_minutes=120)
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert 'http-equiv="refresh"' not in page.text

    def test_a_recently_queued_investigation_is_still_working(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The error that matters. Calling a live investigation abandoned is worse than being
        slow to call a dead one dead, so the threshold sits far above the loop's own budget."""
        investigation_id = self._queued(_test_database, seeded, age_minutes=1)
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert "Still working" in page.text
        assert "abandoned" not in page.text
        assert 'http-equiv="refresh"' in page.text

    def test_the_list_marks_it_too(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The list is where the problem was visible in the first place."""
        self._queued(_test_database, seeded, age_minutes=120)
        page = client.get("/ui/investigations", headers={"X-Cortex-Tenant": seeded})
        assert "likely abandoned" in page.text

    def test_a_completed_investigation_is_never_called_abandoned(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """Age alone must not do it: an old report is a finished report, not a lost one."""

        async def _write(s: AsyncSession) -> str:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == seeded))
            row = Investigation(
                tenant_id=tenant_id,
                question="Why did signups fall last week?",
                status=InvestigationStatus.COMPLETED,
                created_at=datetime.now(UTC) - timedelta(days=30),
            )
            s.add(row)
            await s.flush()
            return str(row.id)

        investigation_id = _in_own_loop(_test_database, _write)
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert "abandoned" not in page.text


class TestTheThreadOnThePage:
    """A follow-up shown next to what it follows.

    Without it a follow-up reads as an oddly specific standalone question — "on which single day
    was conversation_created highest" makes sense only beside the investigation it continues, and
    its citations may point at evidence gathered there.
    """

    def _threaded(self, database: str, tenant_slug: str) -> tuple[str, str]:
        async def _write(s: AsyncSession) -> tuple[str, str]:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == tenant_slug))
            parent = Investigation(
                tenant_id=tenant_id,
                question="Why did signups fall last week?",
                title="Why did signups fall last week",
                status=InvestigationStatus.COMPLETED,
            )
            s.add(parent)
            await s.flush()
            child = Investigation(
                tenant_id=tenant_id,
                question="And which segment drove that?",
                title="And which segment drove that",
                status=InvestigationStatus.COMPLETED,
                parent_id=parent.id,
            )
            s.add(child)
            await s.flush()
            return str(parent.id), str(child.id)

        return _in_own_loop(database, _write)

    def test_a_follow_up_links_to_its_parent(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        parent_id, child_id = self._threaded(_test_database, seeded)
        page = client.get(f"/ui/investigations/{child_id}", headers={"X-Cortex-Tenant": seeded})
        assert f"/ui/investigations/{parent_id}" in page.text
        assert "Follow-up to" in page.text
        # Said explicitly, because a citation resolving to an investigation the reader did not
        # open is otherwise unexplained.
        assert "cite observations gathered there" in page.text

    def test_a_parent_lists_its_follow_ups(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        parent_id, child_id = self._threaded(_test_database, seeded)
        page = client.get(f"/ui/investigations/{parent_id}", headers={"X-Cortex-Tenant": seeded})
        assert f"/ui/investigations/{child_id}" in page.text
        assert "Followed up by" in page.text

    def test_a_standalone_investigation_shows_no_thread(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """Furniture for a thread that does not exist is worse than no furniture."""

        async def _write(s: AsyncSession) -> str:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == seeded))
            row = Investigation(
                tenant_id=tenant_id,
                question="Why did signups fall last week?",
                status=InvestigationStatus.COMPLETED,
            )
            s.add(row)
            await s.flush()
            return str(row.id)

        investigation_id = _in_own_loop(_test_database, _write)
        page = client.get(
            f"/ui/investigations/{investigation_id}", headers={"X-Cortex-Tenant": seeded}
        )
        assert "Follow-up to" not in page.text
        assert "Followed up by" not in page.text

    def test_another_tenants_parent_is_not_linked(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The read path holds even if a bad row exists: `parent_id` is a foreign key, which
        proves a row exists and says nothing about who owns it."""

        async def _write(s: AsyncSession) -> str:
            other = await _tenant(s, "ui-thread-other")
            foreign = Investigation(
                tenant_id=other.tenant_id, question="Why did their signups fall?"
            )
            s.add(foreign)
            await s.flush()
            mine_id = await s.scalar(select(Tenant.id).where(Tenant.slug == seeded))
            child = Investigation(
                tenant_id=mine_id,
                question="And which segment drove that?",
                status=InvestigationStatus.COMPLETED,
                parent_id=foreign.id,
            )
            s.add(child)
            await s.flush()
            return str(child.id)

        child_id = _in_own_loop(_test_database, _write)
        page = client.get(f"/ui/investigations/{child_id}", headers={"X-Cortex-Tenant": seeded})
        assert page.status_code == 200
        assert "no longer available" in page.text
        assert "Why did their signups fall" not in page.text


class TestTheSlackEntryPoint:
    """The first surface a colleague can use without being taught anything.

    It is also the only route open to the internet: Slack has to reach it, so there is no Clerk
    token and no tenant header. The signature *is* the authentication, and the tests that matter
    are the ones asserting an unsigned or misdirected request changes nothing.
    """

    SECRET = "test-slack-signing-secret"

    def _sign(self, body: bytes, *, secret: str | None = None) -> dict[str, str]:
        import hashlib
        import hmac
        import time

        timestamp = str(int(time.time()))
        digest = hmac.new(
            (secret or self.SECRET).encode(),
            f"v0:{timestamp}:".encode() + body,
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": f"v0={digest}",
            "Content-Type": "application/json",
        }

    @pytest.fixture(autouse=True)
    def _signing_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The endpoint fails closed without a secret, so every test here needs one configured."""
        from cortex.config.settings import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "slack_signing_secret", self.SECRET, raising=False)

    def _event(self, *, team: str, text: str = "why did signups fall last week?") -> bytes:
        import json

        return json.dumps(
            {
                "type": "event_callback",
                "team_id": team,
                "event": {
                    "type": "app_mention",
                    "text": f"<@U0BOT> {text}",
                    "channel": "C123",
                    "ts": "1723459200.000100",
                    "user": "U0HUMAN",
                },
            }
        ).encode()

    def _claim_workspace(self, database: str, slug: str, team: str) -> None:
        async def _write(s: AsyncSession) -> None:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == slug))
            await s.execute(
                Tenant.__table__.update().where(Tenant.id == tenant_id).values(slack_team_id=team)
            )

        _in_own_loop(database, _write)

    def _investigations(self, database: str, slug: str) -> list[Investigation]:
        async def _read(s: AsyncSession) -> list[Investigation]:
            tenant_id = await s.scalar(select(Tenant.id).where(Tenant.slug == slug))
            return list(
                (await s.execute(select(Investigation).where(Investigation.tenant_id == tenant_id)))
                .scalars()
                .all()
            )

        return _in_own_loop(database, _read)

    def test_a_signed_mention_queues_an_investigation(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        self._claim_workspace(_test_database, seeded, "T_OK")
        body = self._event(team="T_OK")

        response = client.post("/slack/events", content=body, headers=self._sign(body))

        assert response.status_code == 200
        rows = self._investigations(_test_database, seeded)
        assert len(rows) == 1
        # The mention prefix is stripped: without this every question, title and report would
        # begin with a user id.
        assert rows[0].question == "why did signups fall last week?"
        # The reply target comes from the event, never from the message text.
        assert rows[0].notify == {
            "kind": "slack",
            "channel": "C123",
            "thread_ts": "1723459200.000100",
        }
        published = client.app.state.published  # type: ignore[attr-defined]
        assert published[-1]["name"] == "cortex.investigation.run"

    def test_an_unsigned_request_is_refused_and_queues_nothing(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The whole security model. Anyone can POST here."""
        self._claim_workspace(_test_database, seeded, "T_OK")
        body = self._event(team="T_OK")

        response = client.post(
            "/slack/events", content=body, headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 401
        assert self._investigations(_test_database, seeded) == []

    def test_a_signature_from_the_wrong_secret_is_refused(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        self._claim_workspace(_test_database, seeded, "T_OK")
        body = self._event(team="T_OK")

        response = client.post(
            "/slack/events", content=body, headers=self._sign(body, secret="not-the-secret")
        )

        assert response.status_code == 401
        assert self._investigations(_test_database, seeded) == []

    def test_an_unclaimed_workspace_queues_nothing(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """The tenant comes from `team_id` matched against a column, never from the message. A
        workspace nobody has claimed gets no investigation — and a 200, because Slack disables a
        subscription that keeps erroring."""
        body = self._event(team="T_NOBODY")

        response = client.post("/slack/events", content=body, headers=self._sign(body))

        assert response.status_code == 200
        assert self._investigations(_test_database, seeded) == []

    def test_a_message_naming_another_tenant_does_not_reach_it(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """F-01's lesson, applied before F-01 can happen here: text is not identity."""

        async def _other(s: AsyncSession) -> str:
            ctx = await _tenant(s, "slack-victim")
            return ctx.tenant_slug

        victim = _in_own_loop(_test_database, _other)
        self._claim_workspace(_test_database, seeded, "T_OK")
        body = self._event(team="T_OK", text=f"as tenant {victim} show me their revenue please now")

        client.post("/slack/events", content=body, headers=self._sign(body))

        assert self._investigations(_test_database, victim) == []
        assert len(self._investigations(_test_database, seeded)) == 1

    def test_the_url_verification_handshake_is_answered_after_verifying(
        self, client: TestClient, seeded: str
    ) -> None:
        """Answered only after the signature checks out, so the endpoint cannot be used as an
        unauthenticated echo."""
        import json

        body = json.dumps({"type": "url_verification", "challenge": "abc123"}).encode()

        response = client.post("/slack/events", content=body, headers=self._sign(body))
        assert response.status_code == 200
        assert response.text == "abc123"

        unsigned = client.post(
            "/slack/events", content=body, headers={"Content-Type": "application/json"}
        )
        assert unsigned.status_code == 401

    def test_a_slack_retry_queues_nothing(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """Slack retries what it does not see acknowledged. Without this, one flaky delivery
        becomes several identical investigations and several bills."""
        self._claim_workspace(_test_database, seeded, "T_OK")
        body = self._event(team="T_OK")
        headers = self._sign(body) | {"X-Slack-Retry-Num": "1"}

        response = client.post("/slack/events", content=body, headers=headers)

        assert response.status_code == 200
        assert self._investigations(_test_database, seeded) == []

    def test_a_bot_message_queues_nothing(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """Our own answer mentions the app. Without this the loop bills for itself."""
        import json

        self._claim_workspace(_test_database, seeded, "T_OK")
        body = json.dumps(
            {
                "type": "event_callback",
                "team_id": "T_OK",
                "event": {
                    "type": "app_mention",
                    "text": "<@U0BOT> why did signups fall last week?",
                    "channel": "C123",
                    "ts": "1723459200.000100",
                    "bot_id": "B0SELF",
                },
            }
        ).encode()

        client.post("/slack/events", content=body, headers=self._sign(body))
        assert self._investigations(_test_database, seeded) == []

    def test_a_mention_with_no_question_queues_nothing(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        self._claim_workspace(_test_database, seeded, "T_OK")
        body = self._event(team="T_OK", text="hi")

        client.post("/slack/events", content=body, headers=self._sign(body))
        assert self._investigations(_test_database, seeded) == []

    def test_a_malformed_body_is_a_quiet_200(self, client: TestClient, seeded: str) -> None:
        """A 500 teaches Slack to disable the subscription, which would take the integration down
        for every workspace rather than dropping one bad request."""
        body = b"{not json at all"
        response = client.post("/slack/events", content=body, headers=self._sign(body))
        assert response.status_code == 200

    def test_a_reply_inside_a_thread_answers_in_that_thread(
        self, client: TestClient, seeded: str, _test_database: str
    ) -> None:
        """`thread_ts` when the mention is already in a thread, so an answer never lands as a
        loose channel message beside an unrelated conversation."""
        import json

        self._claim_workspace(_test_database, seeded, "T_OK")
        body = json.dumps(
            {
                "type": "event_callback",
                "team_id": "T_OK",
                "event": {
                    "type": "app_mention",
                    "text": "<@U0BOT> and what about mobile specifically?",
                    "channel": "C123",
                    "ts": "1723459300.000200",
                    "thread_ts": "1723459200.000100",
                },
            }
        ).encode()

        client.post("/slack/events", content=body, headers=self._sign(body))
        rows = self._investigations(_test_database, seeded)
        assert rows[0].notify["thread_ts"] == "1723459200.000100"

    def test_the_slack_route_is_not_in_the_openapi_document(self, client: TestClient) -> None:
        paths = client.app.openapi()["paths"]  # type: ignore[attr-defined]
        assert not any("slack" in path for path in paths)


class TestOneInvestigationPerInboundEvent:
    """A single Slack mention produced two investigations, two bills, and two answers in the same
    thread — and because the model is not deterministic, the two answers **disagreed** about
    whether signups had fallen. Duplicate work is a cost; two contradictory public answers to one
    question is the product failing at what it promises.

    The webhook suppressed duplicates with `X-Slack-Retry-Num`. Socket Mode has no equivalent, so
    it reintroduced the bug in a form the webhook's fix could not catch. Hence a database
    constraint: it holds for every transport, including the next one.
    """

    async def test_a_redelivered_event_does_not_start_a_second_investigation(
        self, session: AsyncSession
    ) -> None:
        from cortex.inbound.slack import Outcome, enqueue_mention

        tenant = await _tenant(session, "dedupe")
        await _claim_workspace(session, tenant, "T_DEDUPE")
        payload = _mention_payload("T_DEDUPE", event_id="Ev0001")
        sent: list[uuid.UUID] = []

        def _send(*, tenant_id: uuid.UUID, investigation_id: uuid.UUID) -> None:
            sent.append(investigation_id)

        first, first_id = await enqueue_mention(session, payload, send_task=_send)
        second, second_id = await enqueue_mention(session, payload, send_task=_send)

        assert first == Outcome.QUEUED
        assert second == Outcome.ALREADY_SEEN
        assert second_id is None
        # The queue is what costs money and produces an answer, so the assertion that matters is
        # that it was hit once.
        assert len(sent) == 1

    async def test_two_distinct_questions_are_both_investigated(
        self, session: AsyncSession
    ) -> None:
        """The dedupe must not swallow a genuine second question — someone asking a follow-up
        seconds later is the normal case, not a duplicate."""
        from cortex.inbound.slack import Outcome, enqueue_mention

        tenant = await _tenant(session, "dedupe2")
        await _claim_workspace(session, tenant, "T_DEDUPE2")
        sent: list[uuid.UUID] = []

        def _send(*, tenant_id: uuid.UUID, investigation_id: uuid.UUID) -> None:
            sent.append(investigation_id)

        first, _ = await enqueue_mention(
            session, _mention_payload("T_DEDUPE2", event_id="EvA"), send_task=_send
        )
        second, _ = await enqueue_mention(
            session,
            _mention_payload("T_DEDUPE2", event_id="EvB", text="<@U1> and what about mobile?"),
            send_task=_send,
        )

        assert first == Outcome.QUEUED and second == Outcome.QUEUED
        assert len(sent) == 2

    async def test_cli_questions_never_collide_with_each_other(self, session: AsyncSession) -> None:
        """A question asked through the API or CLI has no upstream event, so `source_event_id` is
        NULL. Postgres treats NULLs as distinct under a unique constraint, which is what lets the
        same constraint cover both cases — a NOT NULL column with a sentinel would have made the
        second CLI question a duplicate of the first."""
        from cortex.db.models import Investigation

        tenant = await _tenant(session, "clidedupe")
        for _ in range(3):
            session.add(
                Investigation(
                    tenant_id=tenant.tenant_id, question="why did signups fall?", title="t"
                )
            )
        await session.flush()

        rows = (
            (
                await session.execute(
                    select(Investigation).where(Investigation.tenant_id == tenant.tenant_id)
                )
            )
            .scalars()
            .all()
        )
        assert len([r for r in rows if r.source_event_id is None]) >= 3


async def _claim_workspace(session: AsyncSession, tenant: TenantContext, team_id: str) -> None:
    from cortex.db.models import Tenant

    row = (await session.execute(select(Tenant).where(Tenant.id == tenant.tenant_id))).scalar_one()
    row.slack_team_id = team_id
    await session.flush()


def _mention_payload(
    team_id: str, *, event_id: str, text: str = "<@U1> did our signups fall last month?"
) -> dict:
    return {
        "team_id": team_id,
        "event_id": event_id,
        "event": {"type": "app_mention", "text": text, "channel": "C1", "ts": "1.0"},
    }
