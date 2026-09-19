"""Apply the schema from an installed copy of this package.

**Why this exists rather than `alembic upgrade head`.** That command needs an `alembic.ini` and a
`migrations/` directory in the working directory, which a `pip install` has no reason to contain.
So the migrations ship inside the package and this points alembic at them.

**Why not `Base.metadata.create_all`.** It is one line, and it is a second source of truth for the
schema. Nothing checks it against the migration history, so the two drift silently -- and the
drift surfaces as a production database no migration describes, which is discovered during the
first upgrade that matters. The tests use `create_all` deliberately, for speed on a database they
throw away; anything that persists gets the migrations.
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from alembic import command
from alembic.config import Config

from cortex.config.settings import get_settings

#: Where `force-include` puts them in the wheel. Both are resolved through `importlib.resources`
#: so a zip-safe install or a different site-packages layout still finds them.
_SCRIPTS = "cortex/_migrations"
_INI = "cortex/_alembic.ini"


def _packaged(relative: str) -> Path | None:
    """The installed path, or None when running from a source checkout that has no copy."""
    package, _, name = relative.partition("/")
    try:
        root = resources.files(package)
    except ModuleNotFoundError:  # pragma: no cover - cortex is always importable here
        return None
    candidate = Path(str(root)) / name
    return candidate if candidate.exists() else None


def alembic_config() -> Config:
    """Prefers the repository's own config, falls back to the packaged copy.

    That order matters for a contributor: running `cortex-migrate` inside a checkout should apply
    the migrations they are editing, not the ones that were installed.
    """
    local = Path("alembic.ini")
    if local.exists():
        config = Config(str(local))
    else:
        ini = _packaged(_INI)
        scripts = _packaged(_SCRIPTS)
        if ini is None or scripts is None:
            raise SystemExit(
                "no migrations found: this looks like neither a source checkout nor a complete "
                "install. Run from a clone, or reinstall the package."
            )
        config = Config(str(ini))
        config.set_main_option("script_location", str(scripts))
    # Never read from the ini: the DSN belongs to the environment, and a committed ini carrying a
    # connection string is how credentials end up in version control.
    config.set_main_option("sqlalchemy.url", get_settings().postgres_dsn)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="watcher migrate", description="Apply the Cortex database schema."
    )
    parser.add_argument(
        "revision",
        nargs="?",
        default="head",
        help="target revision (default: head)",
    )
    parser.add_argument(
        "--downgrade",
        action="store_true",
        help="move down to the given revision instead of up. Requires an explicit revision.",
    )
    args = parser.parse_args(argv)

    if args.downgrade and args.revision == "head":
        parser.error("--downgrade needs an explicit revision, not 'head'")

    config = alembic_config()
    if args.downgrade:
        command.downgrade(config, args.revision)
    else:
        command.upgrade(config, args.revision)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
