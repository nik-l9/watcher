"""Every command-line entrypoint imports.

Written after shipping `from cortex.db.titles import Tenant, title_for` — `Tenant` lives in
`cortex.db.models`, so `python -m cortex.ask` died on an ImportError before parsing its
arguments. 1,325 tests passed over that commit, because not one of them imports a module
whose only job is to be run as `__main__`.

That is the gap this file closes, and it is the cheapest test in the suite: importing a
module executes its imports and its module-level code, which is exactly where a broken
entrypoint fails. It asserts nothing about behaviour — argument parsing and output are
tested elsewhere — only that the door opens.

`runpy` is deliberately not used: it would execute `main()`. The import is the check.
"""

from __future__ import annotations

import importlib

import pytest

#: Every module a human or a cron job invokes with `python -m`.
ENTRYPOINTS = [
    # The dispatcher `watcher` resolves to, and the one a broken import would take every other
    # command down with.
    "cortex.cli",
    "cortex.ask",
    "cortex.connect",
    "cortex.spend",
    "cortex.bench.__main__",
    "cortex.eval.__main__",
    "cortex.ingest.__main__",
]


@pytest.mark.parametrize("module", ENTRYPOINTS)
def test_it_imports(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_the_list_covers_every_entrypoint_in_the_tree() -> None:
    """A new CLI added without a line here would be unguarded, and the failure would again
    only appear when somebody ran it."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "cortex"
    found = {
        f"cortex.{path.stem}"
        for path in root.glob("*.py")
        if path.stem not in {"__init__"} and "def main(" in path.read_text()
    } | {f"cortex.{path.parent.name}.__main__" for path in root.glob("*/__main__.py")}
    assert found == set(ENTRYPOINTS)
