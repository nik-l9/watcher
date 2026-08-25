"""Service boundary guard.

Microservices stop being microservices the moment one imports another — you get a
distributed monolith: separate deploys that must nonetheless ship together. The
only permitted sharing is the `cortex` library; services communicate through the
contracts in cortex.contracts.

Also guards the direction of dependency: shared domain code must not import web or
worker framework wiring, or it cannot be reused by the other service.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICES_DIR = REPO_ROOT / "services"
CORTEX_DIR = REPO_ROOT / "cortex"

SERVICE_NAMES = ("gateway", "investigation_worker", "ingest_worker")


def _imports(path: Path) -> list[str]:
    """Every module name imported by a file, as dotted strings."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def _service_files(service: str) -> list[Path]:
    return sorted((SERVICES_DIR / service).rglob("*.py"))


class TestServicesDoNotImportEachOther:
    @pytest.mark.parametrize("service", SERVICE_NAMES)
    def test_no_sibling_imports(self, service: str) -> None:
        siblings = [s for s in SERVICE_NAMES if s != service]
        offenders: list[str] = []
        for path in _service_files(service):
            for imported in _imports(path):
                for sibling in siblings:
                    if imported.startswith(f"services.{sibling}"):
                        offenders.append(f"{path.relative_to(REPO_ROOT)} -> {imported}")
        assert not offenders, (
            "Services must communicate through cortex.contracts, not imports. "
            "A direct import means these two can no longer deploy independently:\n  "
            + "\n  ".join(offenders)
        )

    def test_every_service_exists(self) -> None:
        """Guards against the guard silently passing because a path was renamed."""
        for service in SERVICE_NAMES:
            assert _service_files(service), f"no Python files found for services/{service}"


class TestSharedLibraryStaysFrameworkAgnostic:
    """cortex/ is installed into every service image, so it must not drag one
    service's framework into another's."""

    @pytest.mark.parametrize("banned", ["fastapi", "starlette", "uvicorn"])
    def test_no_web_framework_in_shared_code(self, banned: str) -> None:
        offenders: list[str] = []
        for path in CORTEX_DIR.rglob("*.py"):
            for imported in _imports(path):
                if imported == banned or imported.startswith(f"{banned}."):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} -> {imported}")
        assert not offenders, (
            f"cortex/ must not import {banned}; HTTP concerns belong in "
            "services/gateway/:\n  " + "\n  ".join(offenders)
        )

    def test_shared_code_never_imports_a_service(self) -> None:
        """The dependency arrow points one way: services -> cortex, never back."""
        offenders: list[str] = []
        for path in CORTEX_DIR.rglob("*.py"):
            for imported in _imports(path):
                if imported.startswith("services"):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} -> {imported}")
        assert not offenders, "cortex/ must not depend on any service:\n  " + "\n  ".join(offenders)

    def test_celery_is_confined_to_the_runtime_module(self) -> None:
        """Celery config is shared, but domain logic must stay runnable without a
        broker so it can be tested and reused directly."""
        offenders: list[str] = []
        allowed = CORTEX_DIR / "runtime" / "celery_app.py"
        for path in CORTEX_DIR.rglob("*.py"):
            if path == allowed:
                continue
            for imported in _imports(path):
                if imported == "celery" or imported.startswith("celery."):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} -> {imported}")
        assert not offenders, (
            "Celery belongs in cortex/runtime/celery_app.py only:\n  " + "\n  ".join(offenders)
        )


class TestNoModuleLevelResourceSingletons:
    """An async client cached at module scope binds to the event loop that created
    it, so a process running more than one loop — a Celery worker, a test suite —
    reuses it against a dead loop and fails with "Event loop is closed". Resources
    must be created per process lifecycle instead."""

    def test_no_module_level_graph_store(self) -> None:
        offenders: list[str] = []
        for path in list(CORTEX_DIR.rglob("*.py")) + list(SERVICES_DIR.rglob("*.py")):
            if path.name == "resources.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:  # module level only
                if not isinstance(node, ast.Assign):
                    continue
                call = node.value
                if isinstance(call, ast.Call):
                    func = call.func
                    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                    if name in ("FalkorDBGraphStore", "create_async_engine", "async_sessionmaker"):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> {name}()")
        assert not offenders, (
            "Construct these inside open_resources(), not at module scope:\n  "
            + "\n  ".join(offenders)
        )
