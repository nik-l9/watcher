#!/usr/bin/env bash
#
# Rebuild the two GIFs the README opens with, from a recording that already exists.
#
# They were cut by hand once, at timestamps read off a player, and could not be rebuilt
# without doing that again -- so when the phase breakdown was fixed, both GIFs went on
# showing the old one. The whole point of the README's first screen is that it is a real
# run; a real run that cannot be regenerated stops being one the moment the code changes.
#
# Two frames, for the two things the README claims:
#
#   investigating.gif  the loop working -- the question, each step, each tool call. Cut at
#                      the report's opening rule, because everything after it is output
#                      rather than investigation.
#   answer-key.gif     the end of the run, where the truth block states what was actually
#                      planted in the data underneath what the report just claimed.
#
# Both boundaries are found by anchor rather than by timestamp (scripts/cast_time.py), so
# a re-recorded run with a different number of steps still cuts in the right place.
#
# Rendered at the recording's own 100 columns, and sized by the font instead. Overriding
# --cols re-wraps a report written for 100 columns and wraps it mid-word; a smaller font
# shrinks the image without touching a single line break. 13px lands at 798px wide, which
# is about what the README displays it at, so the text stays sharp.
#
# Usage:
#   scripts/make_readme_gifs.sh                       # from won_accounts_not_activated
#   scripts/make_readme_gifs.sh onboarding_regression
set -euo pipefail

cd "$(dirname "$0")/.."

NAME="${1:-won_accounts_not_activated}"
IN="docs/case-studies/${NAME}.cast"
OUT=docs/case-studies/img
FONT="${FONT:-13}"
THEME="${THEME:-asciinema}"

#: Seconds of the answer key to show. The truth block is revealed a line at a time by the
#: pacer, and this is long enough to watch it arrive without the GIF becoming a video.
TAIL="${TAIL:-12}"

command -v agg >/dev/null 2>&1 || {
  echo "missing: agg (brew install agg)" >&2
  exit 1
}
[ -f "$IN" ] || {
  echo "no such recording: ${IN} -- run scripts/record_case_study.sh ${NAME}" >&2
  exit 1
}

mkdir -p "$OUT"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

paced="${tmp}/${NAME}.cast"
python3 scripts/pace_cast.py "$IN" "$paced" >/dev/null

rule=$(python3 scripts/cast_time.py "$paced" "$(printf '=%.0s' $(seq 1 78))")
truth=$(python3 scripts/cast_time.py "$paced" "WHAT THE DATA ACTUALLY CONTAINED")
end=$(python3 scripts/cast_time.py "$paced" --end)
start=$(python3 -c "print(max(0, ${truth} - 1))")

echo "investigating: 0..${rule}s   answer key: ${start}..${end}s" >&2

agg --font-size "$FONT" --theme "$THEME" --no-loop --fps-cap 8 \
  --select "0..${rule}" "$paced" "${OUT}/investigating.gif" >/dev/null 2>&1

# The last frame is the point of this one, so it is held rather than flashed past.
agg --font-size "$FONT" --theme "$THEME" --no-loop --fps-cap 8 --last-frame-duration "$TAIL" \
  --select "${start}..${end}" "$paced" "${OUT}/answer-key.gif" >/dev/null 2>&1

echo "wrote ${OUT}/investigating.gif and ${OUT}/answer-key.gif (from ${NAME})" >&2
du -h "${OUT}"/*.gif >&2
