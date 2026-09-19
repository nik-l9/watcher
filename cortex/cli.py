"""`watcher <command>` — one command with subcommands, rather than six binaries.

**Why a dispatcher rather than six entry points.** The audience is an individual founder, and
`watcher ask "why did signups fall last week?"` is a thing you can guess at; `watcher-ask` is a
thing you have to be told. Six sibling binaries also make `watcher --help` impossible, so there
is no way to find the other five once you know one.

The subcommands are the existing `main(argv)` functions, imported lazily. That laziness is not
tidiness: `ask` pulls in the whole investigation stack and `migrate` pulls in alembic, and paying
for both on every `watcher --help` would make the one command a new user runs the slowest.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

#: Subcommand -> the module path and function that implements it, plus one line of help.
#:
#: `migrate` is first because the others need a schema, and an install with no repository
#: checked out has no `alembic.ini` to run against -- the migrations ship inside the package and
#: that command points alembic at them.
COMMANDS: dict[str, tuple[str, str, str]] = {
    "migrate": (
        "cortex.db.migrate",
        "main",
        "Create or update the database schema. Run this first.",
    ),
    "ask": ("cortex.ask", "main", "Ask a question and read the investigation."),
    "connect": ("cortex.connect", "main", "Store a tenant's connector credentials in the vault."),
    "ingest": (
        "cortex.ingest.__main__",
        "main",
        "Sync a connector now rather than waiting for the nightly run.",
    ),
    "spend": ("cortex.spend", "main", "What each tenant is costing."),
    "eval": ("cortex.eval.__main__", "main", "Score the analyst against labelled fixtures."),
}


def _resolve(command: str) -> Callable[[list[str] | None], int]:
    module_path, attribute, _ = COMMANDS[command]
    from importlib import import_module

    return getattr(import_module(module_path), attribute)


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "watcher — an AI GTM data analyst that investigates rather than reporting numbers.",
        "",
        "usage: watcher <command> [options]",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, _, help_text) in COMMANDS.items()]
    lines += [
        "",
        "`watcher <command> --help` for a command's own options.",
        "",
        "Start here, and it needs no connector credentials:",
        "  watcher ask --dataset campaign_traffic_drop --show-truth",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a subcommand, or explain what the subcommands are.

    An unknown command exits 2 with the list rather than a stack trace: a typo is the most
    likely reason to arrive here, and the useful response to a typo is the menu.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        sys.stdout.write(_usage())
        return 0
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        sys.stderr.write(f"watcher: unknown command {command!r}\n\n")
        sys.stderr.write(_usage())
        return 2
    return _resolve(command)(rest)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
