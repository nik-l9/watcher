"""Worker task registration and message handling.

These do not execute investigations — that is M2. They pin the wiring that is easy
to get wrong and expensive to discover in production: each worker consumes only its
own queue, tasks reject payloads they cannot understand, and nothing but JSON is
accepted off the broker.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest

from cortex.contracts.messages import (
    QUEUE_INGEST,
    QUEUE_INVESTIGATION,
    UnsupportedSchemaVersion,
)
from services.ingest_worker import worker as ingest
from services.investigation_worker import worker as investigation

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTaskRegistration:
    def test_investigation_task_is_registered(self) -> None:
        assert "cortex.investigation.run" in investigation.celery.tasks

    def test_ingest_tasks_are_registered(self) -> None:
        for name in ("cortex.ingest.dispatch_nightly", "cortex.ingest.sync_connector"):
            assert name in ingest.celery.tasks

    def test_task_names_are_prefixed_by_owning_service(self) -> None:
        """Routing is by name prefix, so the name is what puts a task on a queue.

        Note: Celery's task registry is process-global, so importing both worker
        modules here registers every task on both apps. That is a test artifact —
        in production each container imports only its own module. The isolation
        that actually holds is the --queues flag, asserted in
        TestWorkerQueueBindings below.
        """
        assert investigation.run_investigation.name == "cortex.investigation.run"
        assert ingest.dispatch_nightly.name == "cortex.ingest.dispatch_nightly"
        assert ingest.sync_connector.name == "cortex.ingest.sync_connector"


class TestQueueRouting:
    @pytest.mark.parametrize(
        ("app", "task", "expected"),
        [
            ("investigation", "cortex.investigation.run", QUEUE_INVESTIGATION),
            ("ingest", "cortex.ingest.sync_connector", QUEUE_INGEST),
            ("ingest", "cortex.ingest.dispatch_nightly", QUEUE_INGEST),
        ],
    )
    def test_routes_to_the_expected_queue(self, app: str, task: str, expected: str) -> None:
        celery = investigation.celery if app == "investigation" else ingest.celery
        routes = celery.conf.task_routes
        matched = next(
            (dest["queue"] for pattern, dest in routes.items() if _matches(pattern, task)), None
        )
        assert matched == expected, f"{task} -> {matched}"


def _matches(pattern: str, name: str) -> bool:
    return name.startswith(pattern.removesuffix("*")) if pattern.endswith("*") else pattern == name


class TestBrokerSafety:
    @pytest.mark.parametrize("celery", [investigation.celery, ingest.celery])
    def test_only_json_is_accepted(self, celery: object) -> None:
        """A pickled payload would let a producer execute code inside a worker."""
        conf = celery.conf  # type: ignore[attr-defined]
        assert conf.accept_content == ["json"]
        assert conf.task_serializer == "json"
        assert "pickle" not in conf.accept_content

    @pytest.mark.parametrize("celery", [investigation.celery, ingest.celery])
    def test_acks_late_so_a_killed_worker_redelivers(self, celery: object) -> None:
        conf = celery.conf  # type: ignore[attr-defined]
        assert conf.task_acks_late is True
        assert conf.task_reject_on_worker_lost is True

    @pytest.mark.parametrize("celery", [investigation.celery, ingest.celery])
    def test_time_limits_are_set(self, celery: object) -> None:
        """A hung third-party API must not pin a worker slot forever."""
        conf = celery.conf  # type: ignore[attr-defined]
        assert conf.task_soft_time_limit > 0
        assert conf.task_time_limit > conf.task_soft_time_limit


class TestMessageValidation:
    def test_investigation_rejects_unknown_schema_version(self) -> None:
        payload = {
            "schema_version": 99,
            "tenant_id": str(uuid.uuid4()),
            "investigation_id": str(uuid.uuid4()),
        }
        with pytest.raises(UnsupportedSchemaVersion):
            investigation.run_investigation(payload)

    def test_sync_rejects_unknown_schema_version(self) -> None:
        payload = {"schema_version": 99, "tenant_id": str(uuid.uuid4()), "provider": "ga4"}
        with pytest.raises(UnsupportedSchemaVersion):
            ingest.sync_connector(payload)

    def test_investigation_rejects_missing_tenant(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            investigation.run_investigation({"investigation_id": str(uuid.uuid4())})

    def test_sync_rejects_unknown_fields(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ingest.sync_connector(
                {"tenant_id": str(uuid.uuid4()), "provider": "ga4", "provdier": "typo"}
            )


class TestWorkerQueueBindings:
    """The deployed isolation guarantee.

    Each worker container consumes exactly one queue. This is what makes ingest
    load and investigation load independent — a nightly backfill cannot occupy the
    slots a waiting user needs. Asserted against the Dockerfiles because that is
    where the guarantee actually lives.
    """

    @pytest.mark.parametrize(
        ("target", "queue", "forbidden"),
        [
            ("investigation-worker", QUEUE_INVESTIGATION, QUEUE_INGEST),
            ("ingest-worker", QUEUE_INGEST, QUEUE_INVESTIGATION),
        ],
    )
    def test_consumes_only_its_own_queue(self, target: str, queue: str, forbidden: str) -> None:
        cmd = _cmd(target)
        assert queue in cmd, f"{target} CMD must consume {queue}: {cmd}"
        assert forbidden not in cmd, f"{target} CMD must not consume {forbidden}: {cmd}"

    def test_investigation_worker_has_lower_concurrency(self) -> None:
        """Each investigation is an expensive LLM loop; throughput comes from
        replicas, not from packing more tasks into one process."""
        assert _concurrency(_cmd("investigation-worker")) < _concurrency(_cmd("ingest-worker"))

    def test_only_the_scheduler_runs_beat(self) -> None:
        """Two beat processes double-fire every nightly sync."""
        running_beat = [t for t in _stage_names() if '"beat"' in _cmd(t)]
        assert running_beat == ["scheduler"], running_beat

    def test_every_compose_target_exists_in_the_dockerfile(self) -> None:
        """Guards against a compose target silently falling back to `base`."""
        compose = (REPO_ROOT / "docker-compose.yml").read_text()
        declared = set(re.findall(r"target:\s*([a-z-]+)", compose))
        assert declared <= set(_stage_names()), declared - set(_stage_names())


DOCKERFILE = REPO_ROOT / "infra" / "docker" / "Dockerfile"


def _stage_names() -> list[str]:
    return re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", DOCKERFILE.read_text(), flags=re.M)


def _cmd(name: str) -> str:
    """The CMD of one build stage, with line continuations joined.

    Deliberately the CMD alone rather than the whole stage: stage text would also
    capture the next stage's leading comments, and a comment mentioning a queue is
    not the same as consuming it.
    """
    text = DOCKERFILE.read_text()
    stage = re.search(
        rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}$(.*?)(?=^FROM\s|\Z)", text, flags=re.M | re.S
    )
    assert stage, f"no stage named {name} in {DOCKERFILE}"
    body = stage.group(1).replace("\\\n", " ")
    cmd = re.search(r"^CMD\s+(\[.*?\])\s*$", body, flags=re.M | re.S)
    return cmd.group(1) if cmd else ""


def _concurrency(cmd_text: str) -> int:
    match = re.search(r'"--concurrency",\s*"(\d+)"', cmd_text)
    assert match, "no --concurrency declared"
    return int(match.group(1))


class TestSchedule:
    def test_nightly_sync_is_scheduled(self) -> None:
        schedule = ingest.celery.conf.beat_schedule
        assert "nightly-connector-sync" in schedule
        assert schedule["nightly-connector-sync"]["task"] == "cortex.ingest.dispatch_nightly"

    def test_investigation_worker_has_no_schedule(self) -> None:
        """Beat runs alongside ingest only. Two schedulers would double-fire."""
        assert not investigation.celery.conf.beat_schedule
