"""`python -m cortex.connect` — the one place a real credential is stored.

Tested because of what it is, not because of its size. Three of its behaviours are
security properties rather than conveniences:

  - a secret is read from the environment, never from an argument, so it cannot land in
    shell history or the process table;
  - the tenancy isolation suite must pass *before* anything is stored, which is a gate in
    the plan rather than a step;
  - nothing it prints contains a secret.

The isolation suite is stubbed here rather than run: this file tests that the gate is
consulted and obeyed, which is a different question from whether the suite passes.
"""

from __future__ import annotations

import subprocess

import pytest

from cortex import connect
from cortex.db.models import CredentialProvider


class TestMetadataParsing:
    def test_key_value_pairs_are_parsed_and_trimmed(self) -> None:
        assert connect._parse_meta(["project_id = 100001", " host=https://eu.posthog.com "]) == {
            "project_id": "100001",
            "host": "https://eu.posthog.com",
        }

    def test_a_value_containing_an_equals_sign_survives(self) -> None:
        """A project allowlist is `id:label,id:label`, and a host may carry a query
        string. Splitting on every `=` would silently truncate either."""
        assert connect._parse_meta(["projects=100001:saas,100002:oss"]) == {
            "projects": "100001:saas,100002:oss"
        }
        assert connect._parse_meta(["url=https://x.example?a=b"]) == {
            "url": "https://x.example?a=b"
        }

    def test_a_malformed_pair_is_refused_rather_than_ignored(self) -> None:
        """Silently dropping it would store a credential whose configuration is missing,
        and the failure would surface later as an unexplained upstream error."""
        with pytest.raises(SystemExit, match="KEY=VALUE"):
            connect._parse_meta(["projects"])

    def test_no_metadata_is_an_empty_mapping(self) -> None:
        assert connect._parse_meta(None) == {}


class TestEveryProviderDeclaresItsEnvironmentVariable:
    def test_each_provider_has_a_named_variable_and_a_description(self) -> None:
        """Named per provider rather than one generic variable, so connecting HubSpot
        cannot store a GitHub token under the HubSpot provider — a mistake that surfaces
        much later as a confusing upstream 401."""
        for provider in CredentialProvider:
            env_var, description = connect._ENV_BY_PROVIDER[provider]
            assert env_var.isupper() and description

    def test_no_two_providers_share_a_variable(self) -> None:
        variables = [v for v, _ in connect._ENV_BY_PROVIDER.values()]
        assert len(variables) == len(set(variables))

    def test_slack_expects_a_user_token_not_a_bot_token(self) -> None:
        """Verified the hard way: search.messages, search.all and search.files all return
        `not_allowed_token_type` for a bot token even with `search:read.public` granted."""
        env_var, description = connect._ENV_BY_PROVIDER[CredentialProvider.SLACK]
        assert env_var == "SLACK_USER_TOKEN"
        assert "xoxp-" in description


class TestTheIsolationGate:
    def test_a_failing_suite_stores_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cross-tenant isolation is the precondition for real data. If the suite fails,
        the command must exit before it ever opens a database connection."""
        monkeypatch.setattr(connect, "_isolation_suite_passes", lambda: False)
        monkeypatch.setattr(
            connect, "get_settings", lambda: type("S", (), {"vault_master_key": "k"})()
        )

        def _must_not_open(*args: object, **kwargs: object) -> object:
            raise AssertionError("resources must not be opened when the gate fails")

        monkeypatch.setattr(connect, "open_resources", _must_not_open)

        assert connect.main(["--tenant", "acme", "--provider", "github"]) == 1

    def test_the_gate_runs_pytest_against_the_tenancy_suite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[list[str]] = []

        def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="12 passed\n", stderr="")

        monkeypatch.setattr(subprocess, "run", _run)
        assert connect._isolation_suite_passes() is True
        assert "tests/tenancy" in captured[0]

    def test_a_nonzero_exit_is_reported_as_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 1, stdout="1 failed\n", stderr="boom")

        monkeypatch.setattr(subprocess, "run", _run)
        assert connect._isolation_suite_passes() is False


class TestPreconditions:
    def test_an_absent_vault_key_fails_with_a_way_to_generate_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without it the credential cannot be sealed, and a default would silently fall
        back to something insecure."""
        monkeypatch.setattr(
            connect, "get_settings", lambda: type("S", (), {"vault_master_key": ""})()
        )
        assert connect.main(["--tenant", "acme", "--provider", "github"]) == 2
        assert "CORTEX_VAULT_MASTER_KEY" in capsys.readouterr().err

    def test_connecting_nothing_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--provider` is deliberately not argparse-required, because `--list` needs a
        tenant and nothing else — so the "nothing to do" case is checked here instead."""
        monkeypatch.setattr(
            connect, "get_settings", lambda: type("S", (), {"vault_master_key": "k"})()
        )
        assert connect.main(["--tenant", "acme"]) == 2
        assert "Nothing to do" in capsys.readouterr().err


class TestTheSecretNeverComesFromTheCommandLine:
    def test_the_parser_defines_no_option_that_could_carry_one(self) -> None:
        """A token passed as an argument lands in shell history and in the process table,
        where anyone on the machine can read it. The guarantee is that there is no option
        to pass one through, so it is asserted against the parser's own option strings
        rather than against one parsed invocation.
        """
        import argparse as _argparse

        registered: list[str] = []
        real_add = _argparse.ArgumentParser.add_argument

        def _record(self: _argparse.ArgumentParser, *args: object, **kwargs: object) -> object:
            registered.extend(a for a in args if isinstance(a, str) and a.startswith("-"))
            return real_add(self, *args, **kwargs)  # type: ignore[arg-type]

        _argparse.ArgumentParser.add_argument = _record  # type: ignore[method-assign]
        try:
            connect._parse(["--tenant", "acme", "--provider", "github"])
        finally:
            _argparse.ArgumentParser.add_argument = real_add  # type: ignore[method-assign]

        assert registered, "the parser should have registered options"
        for option in registered:
            assert not any(
                word in option for word in ("token", "secret", "password", "credential")
            ), option

    def test_metadata_is_documented_as_deliberately_plaintext(self) -> None:
        """`--meta` *is* a command-line argument, and that is a decision rather than an
        oversight: a project id is not a secret, and encrypting it would mean it could not
        be read without the vault key. Pinned so the distinction stays explicit."""
        # The reasoning lives in the help text the user actually sees.
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer), pytest.raises(SystemExit):
            connect._parse(["--help"])
        help_text = buffer.getvalue()
        assert "--meta" in help_text
        assert "not a secret" in help_text


class TestOneProviderTwoSecrets:
    """Slack needs two different tokens, and they must not be confusable.

    Search requires a *user* token (`xoxp-`): Slack's search endpoints return
    `not_allowed_token_type` for a bot token even with `search:read.public` granted. Posting an
    answer back into a thread requires a *bot* token (`xoxb-`) with `chat:write`.

    They live under the same provider with different labels, so the env var has to be chosen by
    (provider, label) rather than provider alone. Without that, connecting the bot would read
    `SLACK_USER_TOKEN` and store a search token under the `bot` label — and the failure would
    surface much later as a confusing 401 from `chat.postMessage`, which is precisely what the
    per-provider mapping exists to prevent.
    """

    def test_the_bot_label_reads_a_different_variable(self) -> None:
        from cortex.connect import _ENV_BY_PROVIDER, _ENV_BY_PROVIDER_LABEL
        from cortex.db.models import CredentialProvider

        default_var, _ = _ENV_BY_PROVIDER[CredentialProvider.SLACK]
        bot_var, _ = _ENV_BY_PROVIDER_LABEL[(CredentialProvider.SLACK, "bot")]

        assert default_var == "SLACK_USER_TOKEN"
        assert bot_var == "SLACK_BOT_TOKEN"
        assert bot_var != default_var

    def test_the_bot_description_names_the_scopes_it_needs(self) -> None:
        """The message a user sees when the variable is unset is the only documentation they will
        read at that moment."""
        from cortex.connect import _ENV_BY_PROVIDER_LABEL
        from cortex.db.models import CredentialProvider

        _, description = _ENV_BY_PROVIDER_LABEL[(CredentialProvider.SLACK, "bot")]
        assert "xoxb-" in description
        assert "chat:write" in description

    def test_every_override_names_a_provider_that_exists(self) -> None:
        from cortex.connect import _ENV_BY_PROVIDER, _ENV_BY_PROVIDER_LABEL

        for provider, _label in _ENV_BY_PROVIDER_LABEL:
            assert provider in _ENV_BY_PROVIDER
