"""Settings loading.

Settings are constructed directly rather than through get_settings() so the tests
do not depend on the developer's .env, and _env_file=None keeps a local .env from
leaking real values into assertions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cortex.config.settings import Settings, get_settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestDefaults:
    def test_local_defaults_point_at_the_compose_stack(self) -> None:
        s = _settings()
        assert s.env == "local"
        # Non-default ports: a developer machine usually already owns 5432/6379.
        assert ":5433/" in s.postgres_dsn
        assert s.falkordb_port == 6381
        assert s.redis_url.endswith(":6380/0")
        assert s.qdrant_url == "http://localhost:6333"

    def test_no_default_vault_key(self) -> None:
        """A missing key must stay empty so the vault fails loudly rather than
        falling back to something weak."""
        assert _settings().vault_master_key == ""

    def test_investigation_budgets_are_bounded_by_default(self) -> None:
        s = _settings()
        assert s.max_investigation_steps > 0
        assert s.max_investigation_seconds > 0


class TestEnvOverrides:
    def test_reads_prefixed_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORTEX_FALKORDB_HOST", "graph.internal")
        monkeypatch.setenv("CORTEX_FALKORDB_PORT", "7000")
        monkeypatch.setenv("CORTEX_ENV", "staging")
        s = _settings()
        assert (s.falkordb_host, s.falkordb_port, s.env) == ("graph.internal", 7000, "staging")

    def test_ignores_unprefixed_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unrelated FALKORDB_HOST in the environment must not be picked up."""
        monkeypatch.setenv("FALKORDB_HOST", "wrong.host")
        assert _settings().falkordb_host == "localhost"

    def test_rejects_unknown_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORTEX_ENV", "prod")  # must be "production"
        with pytest.raises(ValidationError):
            _settings()

    def test_rejects_non_integer_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORTEX_FALKORDB_PORT", "not-a-port")
        with pytest.raises(ValidationError):
            _settings()

    def test_extra_env_vars_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unrecognized CORTEX_* vars must not crash startup."""
        monkeypatch.setenv("CORTEX_SOMETHING_FUTURE", "1")
        assert _settings().env == "local"


class TestProductionFlag:
    @pytest.mark.parametrize(
        ("env", "expected"),
        [("local", False), ("test", False), ("staging", False), ("production", True)],
    )
    def test_is_production(self, env: str, expected: bool) -> None:
        assert _settings(env=env).is_production is expected


class TestSecretHandling:
    def test_vault_key_is_not_in_repr(self) -> None:
        """Settings get logged during startup diagnostics; the key must not ride along."""
        s = _settings(vault_master_key="SUPER-SECRET-KEY")
        assert "SUPER-SECRET-KEY" not in repr(s)


class TestCaching:
    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()
