"""Worker task bodies, executed for real.

`test_worker_tasks.py` covers registration, routing and broker safety by
inspection. This file actually *runs* the task functions, so the synchronous
Celery entrypoint, the async bridge, and the resource lifecycle inside a task are
exercised rather than assumed.

That bridge is worth executing: `run_with_resources` builds a fresh event loop and
a fresh set of resources per call, and the bug it exists to prevent — a client
cached against a dead loop — only appears when a task is invoked more than once in
one process.
"""

from __future__ import annotations

import uuid

import pytest

from cortex.config.settings import Settings, get_settings
from cortex.contracts.messages import RunInvestigation, SyncConnector, SyncScope
from cortex.runtime import resources as resources_module
from services.ingest_worker import worker as ingest
from services.investigation_worker import worker as investigation


@pytest.fixture
def local_settings(_test_database: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point the worker's resources at the test database."""
    settings = get_settings().model_copy(update={"postgres_dsn": _test_database})
    monkeypatch.setattr(resources_module, "get_settings", lambda: settings)
    return settings


class TestInvestigationWorker:
    def test_task_runs_and_returns_a_correlated_result(self, local_settings: Settings) -> None:
        tenant_id, investigation_id = uuid.uuid4(), uuid.uuid4()
        message = RunInvestigation(tenant_id=tenant_id, investigation_id=investigation_id)

        result = investigation.run_investigation(message.model_dump(mode="json"))

        assert result["tenant_id"] == str(tenant_id)
        assert result["investigation_id"] == str(investigation_id)
        # Propagated so one investigation is traceable across services.
        assert result["correlation_id"] == str(message.correlation_id)
        # No such tenant, so the message is skipped rather than failed: it outlived its
        # tenant, which is not an error and must not be retried.
        assert result["status"] == "skipped"
        assert "tenant unavailable" in result["reason"]

    def test_running_twice_in_one_process_works(self, local_settings: Settings) -> None:
        """The regression the per-process resource design exists to prevent: a client
        cached against the first task's event loop fails on the second."""
        message = RunInvestigation(tenant_id=uuid.uuid4(), investigation_id=uuid.uuid4())
        payload = message.model_dump(mode="json")

        first = investigation.run_investigation(payload)
        second = investigation.run_investigation(payload)

        assert first["status"] == second["status"] == "skipped"


class TestIngestWorker:
    def test_dispatch_nightly_reports_what_it_dispatched(self, local_settings: Settings) -> None:
        """Names the targets rather than only counting them: "0 dispatched" is equally
        consistent with no tenants, no credentials, and no syncer for the providers that
        are connected — three different problems that would look identical in a log."""
        result = ingest.dispatch_nightly()
        assert "dispatched" in result
        assert isinstance(result["targets"], list)
        assert result["dispatched"] == len(result["targets"])

    def test_a_provider_without_a_syncer_fails_loudly(self, local_settings: Settings) -> None:
        """GA4 is read live during an investigation and has no nightly syncer. A message
        for it should fail with a message naming what does have one, rather than
        succeeding with a sync that wrote nothing."""
        message = SyncConnector(tenant_id=uuid.uuid4(), provider="ga4", scope=SyncScope.BACKFILL)
        with pytest.raises(KeyError, match="no ingest syncer"):
            ingest.sync_connector(message.model_dump(mode="json"))

    def test_an_unknown_tenant_is_refused(self, local_settings: Settings) -> None:
        """A message for a deleted tenant must not provision a graph for it."""
        message = SyncConnector(tenant_id=uuid.uuid4(), provider="github")
        with pytest.raises(LookupError, match="does not exist"):
            ingest.sync_connector(message.model_dump(mode="json"))

    def test_the_contract_defaults_to_incremental(self) -> None:
        """The nightly default. A backfill fired every night would re-pull history."""
        assert SyncConnector(tenant_id=uuid.uuid4(), provider="github").scope is (
            SyncScope.INCREMENTAL
        )

    def test_only_a_total_failure_is_retried(self) -> None:
        """One failed stream is already recorded with its watermark left where it was, so
        the next night re-reads it. Retrying the whole provider would re-read the streams
        that succeeded, whose watermarks have already advanced."""
        assert ingest._all_streams_failed({"streams": {"a": {"ok": False}}}) is True
        assert ingest._all_streams_failed({"streams": {"a": {"ok": True}, "b": {"ok": False}}}) is (
            False
        )
        # No streams at all is not a failure to retry: it means the syncer declared none,
        # which retrying cannot fix.
        assert ingest._all_streams_failed({"streams": {}}) is False
