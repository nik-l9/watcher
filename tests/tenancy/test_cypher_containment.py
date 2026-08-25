"""Cypher containment guard.

Isolation holds because every graph access goes through cortex/memory/. That
property is only true as long as nobody writes Cypher elsewhere, so it is
asserted rather than documented. Runs in CI as an ordinary test — no separate
grep step to forget.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PKG = REPO_ROOT / "cortex" / "memory"

# Clause keywords that only appear in Cypher.
#
# Each is prefixed with (?<![.\w]) so an attribute access is not a match: Python's
# `re.match(...)` and `_REPO_RE.match(...)` would otherwise trip the MATCH pattern
# on every file that does regex work. Matching stays case-insensitive so
# lowercase Cypher is still caught — only method calls are excluded.
_NOT_AN_ATTRIBUTE = r"(?<![.\w])"

CYPHER_PATTERNS = (
    re.compile(_NOT_AN_ATTRIBUTE + r"MATCH\s*\(", re.IGNORECASE),
    re.compile(_NOT_AN_ATTRIBUTE + r"MERGE\s*\(", re.IGNORECASE),
    re.compile(_NOT_AN_ATTRIBUTE + r"CREATE\s*\((?!\s*\))", re.IGNORECASE),
    re.compile(_NOT_AN_ATTRIBUTE + r"DETACH\s+DELETE\b", re.IGNORECASE),
    re.compile(_NOT_AN_ATTRIBUTE + r"UNWIND\b", re.IGNORECASE),
    re.compile(_NOT_AN_ATTRIBUTE + r"RETURN\s+(DISTINCT\s+)?[a-z]\b"),
)

# Graph client entry points that would bypass the GraphStore interface.
CLIENT_PATTERNS = (
    re.compile(r"\bselect_graph\s*\("),
    re.compile(r"\bro_query\s*\("),
    re.compile(r"from\s+falkordb"),
    re.compile(r"\bimport\s+falkordb\b"),
)


def _python_files_outside_memory() -> list[Path]:
    files = []
    for path in (REPO_ROOT / "cortex").rglob("*.py"):
        if MEMORY_PKG in path.parents or path == MEMORY_PKG:
            continue
        files.append(path)
    return files


def test_no_cypher_outside_memory_package() -> None:
    offenders: list[str] = []
    for path in _python_files_outside_memory():
        text = path.read_text(encoding="utf-8")
        for pattern in CYPHER_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {match.group(0)!r}")
    assert not offenders, (
        "Cypher must live only in cortex/memory/, where tenant graph selection is "
        "enforced. Move these into a GraphStore method:\n  " + "\n  ".join(offenders)
    )


def test_no_direct_graph_client_use_outside_memory_package() -> None:
    offenders: list[str] = []
    for path in _python_files_outside_memory():
        text = path.read_text(encoding="utf-8")
        for pattern in CLIENT_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {match.group(0)!r}")
    assert not offenders, (
        "Import the GraphStore interface instead of a graph client directly:\n  "
        + "\n  ".join(offenders)
    )


def test_graph_name_is_only_constructed_in_naming_module() -> None:
    """No caller may build a graph name; only cortex.memory.naming may."""
    naming = MEMORY_PKG / "naming.py"
    offenders: list[str] = []
    for path in (REPO_ROOT / "cortex").rglob("*.py"):
        if path == naming:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"[\"']cortex_g_", text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}")
    assert not offenders, (
        "Graph names must come from cortex.memory.naming, not string literals:\n  "
        + "\n  ".join(offenders)
    )
