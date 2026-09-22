#!/usr/bin/env bash
#
# Turn the recorded case studies into mp4s.
#
# The .cast files are the artifact; this is for putting one in a slide, a README or a
# message, where an asciinema player is not available. Nothing is re-run: the videos are
# rendered from recordings that already exist, so this costs no API calls and every video
# shows the same investigation its .cast and .txt do.
#
# The recording is first re-timed by scripts/pace_cast.py, and that step is not optional.
# Played back as recorded, the analyst prints its whole report in one burst: the screen
# fills, scrolls past and stops, and a viewer has read nothing. The pacer reveals the same
# bytes one line at a time, dwelling on the answer and the truth block and scrolling
# through the source table. It changes timing only -- it asserts the output is unchanged
# byte for byte -- and the unedited .cast sits next to the video for anyone checking.
#
# `agg` then renders the paced cast to a GIF and ffmpeg turns that into an mp4. The GIF is
# an intermediate and is deleted; terminal output is a handful of colours, so the
# 256-colour palette loses nothing.
#
# The final frame is held, because it is the point: the truth block, stating what was
# actually planted in the data next to the report that just claimed something about it.
#
# Knobs, as environment variables:
#   SPEED=1     playback multiplier -- 2 halves the running time
#   FONT=16     font size in px; drives the video's dimensions
#   HOLD=6      seconds to hold the last frame
#   THEME=asciinema   any agg theme (dracula, monokai, solarized-dark, ...)
#   PACED=0     skip the re-timing and render the raw recording instead
#
# Usage:
#   scripts/make_case_study_videos.sh                       # every recording
#   scripts/make_case_study_videos.sh onboarding_regression # named ones only
#   SPEED=2 scripts/make_case_study_videos.sh               # twice as fast
#
set -euo pipefail

cd "$(dirname "$0")/.."

IN=docs/case-studies
OUT="$IN/video"
SPEED="${SPEED:-1}"
FONT="${FONT:-16}"
HOLD="${HOLD:-6}"
THEME="${THEME:-asciinema}"

PACED="${PACED:-1}"

for binary in agg ffmpeg; do
  command -v "$binary" >/dev/null 2>&1 || {
    echo "missing: $binary (brew install $binary)" >&2
    exit 1
  }
done

if [ "$#" -gt 0 ]; then
  names="$*"
else
  names=$(find "$IN" -name '*.cast' -exec basename {} .cast \; | sort)
fi

[ -n "$names" ] || {
  echo "no recordings in ${IN}/ -- run scripts/record_case_study.sh first" >&2
  exit 1
}

mkdir -p "$OUT"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

count=0
for name in $names; do
  cast="${IN}/${name}.cast"
  [ -f "$cast" ] || {
    echo "no such recording: ${cast}" >&2
    exit 1
  }
  count=$((count + 1))
  echo "rendering ${name}..." >&2

  render="$cast"
  if [ "$PACED" != "0" ]; then
    python3 scripts/pace_cast.py "$cast" "${tmp}/${name}.cast" >/dev/null
    render="${tmp}/${name}.cast"
  fi

  # 8fps is plenty for text that changes a line at a time, and it keeps the GIF the
  # encoder has to chew through an order of magnitude smaller.
  agg --font-size "$FONT" --theme "$THEME" --speed "$SPEED" --no-loop --fps-cap 8 \
    "$render" "${tmp}/${name}.gif" >/dev/null 2>&1

  # yuv420p is what players outside a browser expect, and it requires both dimensions to
  # be even -- a terminal at an odd pixel height otherwise fails to encode. tpad holds the
  # closing frame; agg's own --last-frame-duration is scaled by --speed, so at SPEED=2 it
  # would halve the very pause that exists to be read.
  # -tune stillimage is exactly this case: a mostly static screen that changes in steps.
  # Without it the same video is four times the size at the same visual quality.
  ffmpeg -y -i "${tmp}/${name}.gif" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,tpad=stop_mode=clone:stop_duration=${HOLD}" \
    -pix_fmt yuv420p -movflags +faststart -r 12 \
    -c:v libx264 -preset veryslow -tune stillimage -crf 34 \
    "${OUT}/${name}.mp4" >/dev/null 2>&1
done

echo "wrote ${count} video(s) to ${OUT}/" >&2
du -sh "$OUT" >&2
