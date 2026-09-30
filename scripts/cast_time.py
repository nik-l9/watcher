#!/usr/bin/env python3
"""Print the time, in seconds, at which a line first appears in a recording.

A GIF cut out of a recording needs a time range, and the interesting boundaries are lines
rather than timestamps: where the loop stops and the report starts, where the answer key
begins. Both move every time a run is re-recorded, because the analyst takes a different
number of steps -- so the two README GIFs were cut by hand at timestamps read off a
player, and could not be regenerated without doing that again.

This turns an anchor into a time, so `make_readme_gifs.sh` can cut the same two frames
out of any recording of any scenario.

Usage:
    cast_time.py run.cast "WHAT THE DATA ACTUALLY CONTAINED"   # when the line appears
    cast_time.py run.cast --end                                # total duration
"""

from __future__ import annotations

import json
import sys


def _events(path: str) -> list[list]:
    lines = open(path, encoding="utf-8").read().splitlines()
    return [json.loads(line) for line in lines[1:] if line.strip()]


def when(path: str, anchor: str) -> float:
    """Seconds from the start until the first output line beginning with `anchor`."""
    elapsed = 0.0
    for event in _events(path):
        elapsed += float(event[0])
        if event[1] == "o" and event[2].strip().startswith(anchor):
            return elapsed
    raise SystemExit(f"anchor not found in {path}: {anchor!r}")


def duration(path: str) -> float:
    return sum(float(event[0]) for event in _events(path))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write(f'usage: {argv[0]} run.cast "anchor" | --end\n')
        return 2
    path, anchor = argv[1], argv[2]
    print(f"{duration(path) if anchor == '--end' else when(path, anchor):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
