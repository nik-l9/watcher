"""Where a captured run goes, and how a later command finds it again.

Bundles are the whole reason a scoring change can be re-measured without paying for another
run. The one time that mattered, it was forgotten: a fifteen-attempt variance run was written
to a session scratchpad, the scratchpad was cleared between sessions, and the run could not be
re-scored — defeating the entire purpose of having captured it, and forcing a fix to be verified
against a reconstructed sentence instead of the real report.

The fix is not remembering a flag. It is not needing to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cortex.eval.__main__ import CAPTURE_ROOT, _capture_directory, _resolve_run


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {"no_capture": False, "capture_to": None}
    return argparse.Namespace(**{**defaults, **overrides})


class TestCapturingIsTheDefault:
    def test_a_plain_run_captures_without_being_asked(self) -> None:
        directory = _capture_directory(_args())
        assert directory is not None
        assert directory.parts[0] == CAPTURE_ROOT.name

    def test_the_directory_is_repo_relative_not_temporary(self) -> None:
        """The original failure in one assertion: a session scratchpad does not survive the
        session, and neither did the run."""
        directory = _capture_directory(_args())
        assert directory is not None
        assert not directory.is_absolute()
        assert "tmp" not in str(directory)

    def test_each_run_gets_its_own_directory(self) -> None:
        """Timestamped rather than a fixed "latest", so a run cannot quietly overwrite the
        baseline someone is about to compare against."""
        directory = _capture_directory(_args())
        assert directory is not None
        stamp = directory.name
        assert len(stamp) == 15 and stamp[8] == "-", stamp

    def test_an_explicit_path_still_wins(self) -> None:
        assert _capture_directory(_args(capture_to="/somewhere/else")) == Path("/somewhere/else")

    def test_it_can_be_turned_off(self) -> None:
        """Only worth it for a throwaway run, and the help text says so."""
        assert _capture_directory(_args(no_capture=True)) is None

    def test_off_beats_an_explicit_path(self) -> None:
        assert _capture_directory(_args(no_capture=True, capture_to="/x")) is None


class TestFindingTheRunAgain:
    def test_latest_resolves_to_the_newest(self, tmp_path: Path, monkeypatch) -> None:
        """Typing a timestamp is how a comparison gets pointed at the wrong run."""
        root = tmp_path / "eval-runs"
        for name in ("20260101-000000", "20260820-120000", "20260501-090000"):
            (root / name).mkdir(parents=True)
        monkeypatch.setattr("cortex.eval.__main__.CAPTURE_ROOT", root)
        assert _resolve_run("latest").name == "20260820-120000"

    def test_an_explicit_directory_is_taken_as_given(self, tmp_path: Path) -> None:
        assert _resolve_run(str(tmp_path)) == tmp_path

    def test_latest_with_nothing_captured_says_so(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("cortex.eval.__main__.CAPTURE_ROOT", tmp_path / "eval-runs")
        with pytest.raises(SystemExit, match="no captured runs"):
            _resolve_run("latest")

    def test_a_file_is_not_mistaken_for_a_run(self, tmp_path: Path, monkeypatch) -> None:
        """A stray file beside the run directories must not be offered as one."""
        root = tmp_path / "eval-runs"
        root.mkdir()
        (root / "99999999-999999").write_text("not a directory")
        (root / "20260101-000000").mkdir()
        monkeypatch.setattr("cortex.eval.__main__.CAPTURE_ROOT", root)
        assert _resolve_run("latest").name == "20260101-000000"
