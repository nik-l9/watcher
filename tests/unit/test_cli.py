"""One command with subcommands, because the audience has to be able to guess it.

`watcher ask "why did signups fall last week?"` is guessable; `watcher-ask` has to be told. Six
sibling binaries also make `watcher --help` impossible, so there is no way to discover the other
five once you know one — which for a solo founder installing this on a Tuesday evening is the
difference between it working and it not.
"""

from __future__ import annotations

import pytest

from cortex.cli import COMMANDS, main


class TestTheMenuIsDiscoverable:
    @pytest.mark.parametrize("flag", ["--help", "-h", "help"])
    def test_every_way_of_asking_gets_the_menu(self, flag: str, capsys) -> None:
        assert main([flag]) == 0
        printed = capsys.readouterr().out
        for command in COMMANDS:
            assert f"  {command}" in printed, command

    def test_no_arguments_is_a_question_not_an_error(self, capsys) -> None:
        """Typing the bare command is how someone finds out what it does, so it must not be
        punished with a stack trace or a non-zero exit."""
        assert main([]) == 0
        assert "usage: watcher <command>" in capsys.readouterr().out

    def test_it_names_a_first_command_that_needs_no_credentials(self, capsys) -> None:
        """The menu is where someone decides whether to keep going. A first step that needs a
        connector, a vault key and a tenant is a first step most people do not take."""
        main(["--help"])
        printed = capsys.readouterr().out
        assert "watcher ask --dataset" in printed

    def test_migrate_is_listed_first(self) -> None:
        """Everything else needs a schema, and dict order is the order the menu prints."""
        assert next(iter(COMMANDS)) == "migrate"


class TestATypoGetsTheMenuRatherThanATrace:
    def test_an_unknown_command_exits_two_and_explains(self, capsys) -> None:
        assert main(["aks"]) == 2
        captured = capsys.readouterr()
        assert "unknown command 'aks'" in captured.err
        assert "commands:" in captured.err

    def test_it_does_not_guess_what_was_meant(self) -> None:
        """A dispatcher that runs the nearest match runs something nobody asked for. The menu
        is the safe answer to an ambiguous input."""
        assert main(["as"]) == 2


class TestEverySubcommandResolves:
    """A menu entry pointing at a function that does not exist is worse than no entry: it is
    discoverable, and it fails at the moment of use rather than at import."""

    @pytest.mark.parametrize("command", sorted(COMMANDS))
    def test_the_target_is_importable_and_callable(self, command: str) -> None:
        from cortex.cli import _resolve

        assert callable(_resolve(command))

    @pytest.mark.parametrize("command", sorted(COMMANDS))
    def test_the_target_accepts_an_argv_list(self, command: str) -> None:
        """Dispatch passes the remaining arguments through, so every target must take them —
        a `main()` reading `sys.argv` directly would silently receive the wrong ones."""
        import inspect

        from cortex.cli import _resolve

        assert "argv" in inspect.signature(_resolve(command)).parameters


class TestNothingUserFacingSaysCortex:
    """The internal module name must not reach someone who typed `watcher`.

    The import name is still `cortex`, deliberately — renaming it touches 208 files and a
    collection prefix already written into live vector stores. But a founder who runs
    `watcher ask --help` and is shown `usage: cortex.ask` has been handed a word that means
    nothing to them, in the one place they went looking for help.
    """

    @pytest.mark.parametrize("command", sorted(COMMANDS))
    def test_each_subcommands_help_names_the_command_that_was_typed(
        self, command: str, capsys
    ) -> None:
        from cortex.cli import _resolve

        with pytest.raises(SystemExit):
            _resolve(command)(["--help"])
        printed = capsys.readouterr().out
        assert printed.startswith(f"usage: watcher {command}"), printed.splitlines()[:1]

    def test_the_menu_itself_never_says_cortex(self, capsys) -> None:
        main(["--help"])
        assert "cortex" not in capsys.readouterr().out.lower()
