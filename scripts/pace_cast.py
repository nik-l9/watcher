#!/usr/bin/env python3
"""Re-time a recording so a person can read it.

A recording of `watcher ask` is unwatchable as a demo, and the reason is in the product,
not in the recorder: the analyst works for ninety seconds printing four progress lines,
then prints its entire report in one burst. On playback that burst arrives in a few
hundred milliseconds. The viewer sees the screen fill and scroll, reaches the end, and has
read nothing.

This rewrites the *timing* and nothing else. Every byte of output is preserved in its
original order, byte for byte -- the assertion at the end of `pace()` enforces that. What
changes is when each line appears: one line at a time, dwelling on the sections a person
came to read and moving briskly through the ones they did not.

That is an edit, and the README says so. It is the same edit any screencast makes, and the
unedited `.cast` sits next to the video for anyone who wants to check it.

Usage:
    pace_cast.py in.cast out.cast
"""

from __future__ import annotations

import json
import math
import sys

#: Seconds a rendered row is left on screen before the next line appears, per section.
#:
#: These are reading rates, not decoration. The answer and the truth block are the two
#: things a viewer is actually there for -- one states a conclusion, the other states what
#: was really in the data -- so they get roughly two seconds per line. The source table and
#: the phase breakdown are reference material nobody reads from a video, so they scroll.
DWELL = {
    "ANSWER": 0.55,
    "FINDINGS": 0.32,
    "CHARTS": 0.30,
    "HYPOTHESES TESTED": 0.22,
    "RECOMMENDATIONS": 0.22,
    "RISKS AND CAVEATS": 0.18,
    "DATA QUALITY": 0.20,
    "SOURCES": 0.04,
    "WHAT THE DATA ACTUALLY CONTAINED": 0.60,
}

#: Seconds the whole preamble gets -- the question and every progress line before the report.
#:
#: A budget rather than a per-line dwell, because the number of lines changed by an order of
#: magnitude. It was 0.9s a line when a run printed four of them. Then the fixture path
#: started reporting each step and tool call, as the live path always had, and the same
#: constant turned forty-nine progress lines into forty-four seconds of watching a loop the
#: viewer has already understood by line five.
#:
#: The report keeps its per-line dwell. That is the part someone is reading rather than
#: watching, and it should not speed up because the loop happened to work harder.
PREAMBLE_BUDGET = 14.0

#: No line faster than this, however many there are, or the preamble becomes a flicker.
PREAMBLE_FLOOR = 0.12

#: Everything after the source table and before the truth block -- the phase breakdown.
DEFAULT_DWELL = 0.06

#: An extra beat when a new section opens, so its heading registers as a heading.
SECTION_PAUSE = 0.9

#: The report proper starts at a rule of equals signs.
REPORT_RULE = "=" * 78


def _rows(line: str, cols: int) -> int:
    """How many terminal rows this line occupies once wrapped."""
    return max(1, math.ceil(len(line) / cols)) if line else 1


def pace(source: str) -> str:
    lines = source.splitlines(keepends=False)
    header = json.loads(lines[0])
    cols = int(header.get("term", {}).get("cols", 100))

    parsed = [json.loads(line) for line in lines[1:] if line.strip()]
    # A cast holds more than output. The last event of a `-c` recording is `["x", "0"]`,
    # the exit status, and treating it as output printed a bare "0" under the report --
    # visible in the rendered video, which is how this was caught. Anything that is not an
    # output event is carried through untouched, after the output.
    text = "".join(event[2] for event in parsed if event[1] == "o")
    trailing = [event for event in parsed if event[1] != "o"]

    # `keepends` so the reassembled stream is identical to what was recorded, including the
    # \r\n pairs a terminal emits and any final line without a newline.
    out_lines = text.splitlines(keepends=True)

    # Everything before the report's opening rule is preamble, and it shares one budget.
    preamble_lines = next(
        (n for n, line in enumerate(out_lines) if line.strip() == REPORT_RULE), len(out_lines)
    )
    preamble_dwell = max(PREAMBLE_FLOOR, PREAMBLE_BUDGET / max(preamble_lines, 1))

    events: list[list[object]] = []
    section: str | None = None
    in_report = False
    pending = SECTION_PAUSE  # a beat before the first line, so playback does not start mid-word

    for line in out_lines:
        stripped = line.strip()

        if stripped == REPORT_RULE:
            in_report = True
        elif stripped in DWELL:
            # A section heading. The rule under it belongs to the same beat.
            section = stripped
            pending += SECTION_PAUSE

        if not in_report:
            dwell = preamble_dwell
        elif section is None:
            dwell = 0.25
        else:
            dwell = DWELL.get(section, DEFAULT_DWELL)

        events.append([round(pending, 3), "o", line])
        pending = dwell * _rows(stripped, cols)

    rebuilt = "".join(str(event[2]) for event in events)
    assert rebuilt == text, "pacing must not change a single byte of output"

    for event in trailing:
        events.append([round(pending, 3), event[1], event[2]])
        pending = 0.0

    header.pop("idle_time_limit", None)  # every interval here is deliberate
    return "\n".join([json.dumps(header), *(json.dumps(event) for event in events)]) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write(f"usage: {argv[0]} in.cast out.cast\n")
        return 2
    source = open(argv[1], encoding="utf-8").read()
    paced = pace(source)
    open(argv[2], "w", encoding="utf-8").write(paced)
    total = sum(float(json.loads(line)[0]) for line in paced.splitlines()[1:])
    sys.stderr.write(f"{argv[2]}: {total:.0f}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
