"""Applying the schema from an installed copy.

`cortex-migrate` exists because `alembic upgrade head` needs an `alembic.ini` and a `migrations/`
directory in the working directory, and a `pip install` has no reason to contain either. The
migrations ship inside the wheel and this points alembic at them.

The resolution order is the part worth testing: a contributor running the command inside a
checkout must get the migrations they are editing, not the ones that happen to be installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.db import migrate


class TestWhichMigrationsGetApplied:
    def test_a_checkout_uses_its_own(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A local `alembic.ini` wins. Editing a migration and then applying a different one is
        the kind of confusion that costs an afternoon."""
        (tmp_path / "alembic.ini").write_text("[alembic]\nscript_location = migrations\n")
        monkeypatch.chdir(tmp_path)
        config = migrate.alembic_config()
        assert config.get_main_option("script_location") == "migrations"

    def test_an_install_uses_the_packaged_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ini in the working directory, so the packaged path is used — and it is absolute,
        because alembic resolves a relative script_location against the cwd."""
        monkeypatch.chdir(tmp_path)
        packaged = tmp_path / "packaged"
        (packaged / "migrations").mkdir(parents=True)
        (packaged / "alembic.ini").write_text("[alembic]\n")
        monkeypatch.setattr(
            migrate,
            "_packaged",
            lambda rel: packaged / ("migrations" if rel.endswith("_migrations") else "alembic.ini"),
        )
        config = migrate.alembic_config()
        assert Path(config.get_main_option("script_location") or "").is_absolute()

    def test_neither_is_an_error_naming_both_causes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing schema is worth an explanation. "no such table" three commands later is not
        one, and the two situations that produce it need different fixes."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(migrate, "_packaged", lambda rel: None)
        with pytest.raises(SystemExit) as caught:
            migrate.alembic_config()
        assert "source checkout" in str(caught.value)
        assert "reinstall" in str(caught.value)

    def test_the_dsn_never_comes_from_the_ini(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A committed ini carrying a connection string is how credentials reach version control,
        so the environment is the only source and it overrides whatever the file said."""
        (tmp_path / "alembic.ini").write_text(
            "[alembic]\nscript_location = migrations\nsqlalchemy.url = postgresql://in-the-file/x\n"
        )
        monkeypatch.chdir(tmp_path)
        url = migrate.alembic_config().get_main_option("sqlalchemy.url") or ""
        assert "in-the-file" not in url


class TestTheCommandLine:
    def test_a_downgrade_needs_an_explicit_revision(self) -> None:
        """`--downgrade head` is a no-op that reads like a rollback. Refused rather than run."""
        with pytest.raises(SystemExit):
            migrate.main(["--downgrade"])
