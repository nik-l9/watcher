"""Every third-party import must be declared in `pyproject.toml`.

`cortex/tools/sqlguard.py` imported `sqlglot` at module level and nothing declared it. The suite
passed anyway, because the development venv had the package installed by hand — so the gap was
invisible for as long as nobody built a fresh environment. It surfaced when one was: two test
modules failed to collect on a clean install, straight after a setup that printed success.

The instance is one line of `pyproject.toml`. The class is that a dependency set and an import
graph can disagree indefinitely while every test passes, and that is what this file closes.

Reads the *installed* distribution metadata rather than a hand-written import→package map, so a
package that renames its import (`pyyaml` → `yaml`, `pyjwt` → `jwt`) needs no entry here and
cannot be got wrong.
"""

from __future__ import annotations

import ast
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
#: Trees whose imports must be covered. `tests` is included because a test-only dependency is
#: still a dependency — the undeclared import that started this was found by a test module.
SEARCHED = ("cortex", "services", "tests", "alembic")
FIRST_PARTY = {"cortex", "services", "tests", "alembic"}


def _declared() -> set[str]:
    """Distribution names in `dependencies` and every optional group, normalised.

    Extras are stripped (`uvicorn[standard]` is the `uvicorn` distribution) and names are
    lowercased with `_`/`.` folded to `-`, which is how packaging compares them.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data.get("project", {})
    raw = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        raw.extend(group)
    names = set()
    for spec in raw:
        head = spec.split(";")[0].strip()
        for boundary in ("[", ">", "<", "=", "!", "~", " "):
            head = head.split(boundary)[0]
        if head:
            names.add(head.strip().lower().replace("_", "-").replace(".", "-"))
    return names


def _imported() -> dict[str, set[Path]]:
    """Top-level module name → the files importing it, third-party only.

    Only module-level imports would be enough for the defect that prompted this, but a function
    -level import of an undeclared package fails just as hard when the function runs, so the walk
    does not distinguish them.
    """
    found: dict[str, set[Path]] = {}
    for tree in SEARCHED:
        for path in (ROOT / tree).rglob("*.py"):
            try:
                parsed = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not our source
                continue
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    heads = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    # Relative imports have no module to resolve against a distribution.
                    heads = [node.module.split(".")[0]] if node.module and not node.level else []
                else:
                    continue
                for head in heads:
                    found.setdefault(head, set()).add(path.relative_to(ROOT))
    import sys

    return {
        name: files
        for name, files in found.items()
        if name not in sys.stdlib_module_names and name not in FIRST_PARTY
    }


class TestTheDependencySetCoversTheImportGraph:
    def test_every_third_party_import_is_declared(self) -> None:
        declared = _declared()
        provided = packages_distributions()
        undeclared: list[str] = []
        for name, files in sorted(_imported().items()):
            dists = {d.lower().replace("_", "-").replace(".", "-") for d in provided.get(name, ())}
            if not dists:
                # Installed-but-unmapped or not installed at all. Either way this test cannot
                # say, and saying nothing is better than a false accusation — the collection
                # error a missing package causes is loud on its own.
                continue
            if not dists & declared:
                where = ", ".join(str(f) for f in sorted(files)[:3])
                undeclared.append(f"{name} (from {'/'.join(sorted(dists))}) imported by {where}")
        assert not undeclared, "imported but not in pyproject.toml:\n  " + "\n  ".join(undeclared)

    def test_the_check_can_actually_fail(self) -> None:
        """A guard whose negative case is never exercised is a guard that might match nothing.

        `sqlglot` is the package the defect was about, so it stands in for the failure: remove it
        from the declared set and the comparison must object.
        """
        declared = _declared()
        assert "sqlglot" in declared, "sqlglot must stay declared; sqlguard imports it"
        provided = packages_distributions()
        assert "sqlglot" in provided, "sqlglot must be installed for this test to mean anything"
        assert not {d.lower() for d in provided["sqlglot"]} & (declared - {"sqlglot"})


@pytest.mark.parametrize("module", ["sqlglot"])
def test_a_named_guard_dependency_imports(module: str) -> None:
    """The specific regression. `tests/tools/test_sql.py` and `test_sqlguard.py` failed to
    collect without this on a clean install, which is how the gap was found at all."""
    __import__(module)
