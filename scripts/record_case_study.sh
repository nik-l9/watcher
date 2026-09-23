#!/usr/bin/env bash
#
# Record one investigation as a terminal session, for publication as a case study.
#
# Runs against a *labelled* dataset, never `--real`. Two reasons, and the second is the
# one that matters:
#
#   1. A `--real` run puts customer names, deal amounts and owner names into a recording
#      that is meant to be published. There is no redacting a .cast after the fact.
#   2. A labelled run ends with `--show-truth`, which prints what was actually planted in
#      the data. That turns the recording from a demo into something a reader can mark:
#      the report claims X, the truth block says Y, and they can disagree on screen.
#
# Output per scenario, in docs/case-studies/:
#   <name>.cast  asciicast v3 -- replayable, diffable, and small enough to commit
#   <name>.txt   the same session as plain text, for reading in a diff or a README
#
# The .cast is the artifact; the .txt exists so a reviewer can see what changed in a pull
# request without an asciinema player.
#
# Usage:
#   scripts/record_case_study.sh                       # every scenario
#   scripts/record_case_study.sh tempting_coincidence  # named ones only
#
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=docs/case-studies
COLS=100
ROWS=45

# The command that appears *on screen*, which is the one a reader will try to repeat. It
# differs between the two repositories this script lives in, and getting it wrong produces
# a recording nobody can reproduce -- so it is detected rather than hardcoded.
if [ -x .venv/bin/watcher ]; then
  ASK=".venv/bin/watcher ask"
elif [ -x .venv/bin/cortex-ask ]; then
  ASK=".venv/bin/cortex-ask"
else
  ASK="${PY} -m cortex.ask"
fi

command -v asciinema >/dev/null 2>&1 || {
  echo "missing: asciinema (brew install asciinema)" >&2
  exit 1
}
[ -x "$PY" ] || {
  echo "missing: $PY -- run 'uv sync' first" >&2
  exit 1
}

# The per-request `llm.cache` diagnostic is one line per model call. Useful when watching
# your own run, unreadable interleaved with the report in something a stranger will read.
export CORTEX_LOG_LEVEL=info

# macOS ships bash 3.2, which has no `mapfile`, and trips `set -u` on `${empty[@]}`.
if [ "$#" -gt 0 ]; then
  scenarios="$*"
else
  scenarios=$("$PY" -c 'from cortex.eval.fixtures import SCENARIOS
for s in SCENARIOS: print(s.name)')
fi

mkdir -p "$OUT"

count=0
for name in $scenarios; do
  count=$((count + 1))
  echo "recording ${name}..." >&2
  asciinema rec \
    --headless \
    --window-size "${COLS}x${ROWS}" \
    --idle-time-limit 2 \
    --title "${name}" \
    --overwrite \
    "${OUT}/${name}.cast" \
    -c "${ASK} --dataset ${name} --show-truth"

  asciinema convert --output-format txt --overwrite \
    "${OUT}/${name}.cast" "${OUT}/${name}.txt"

  # A recording of a traceback is not a case study, and it looks exactly like one from the
  # outside: the file exists, it has the right name, and the script says it wrote it. Eleven
  # were produced in a row that way -- the repository this runs in has no ANTHROPIC_API_KEY,
  # by design, and every run died on the first model call while the loop reported success.
  if grep -q "^Traceback (most recent call last):" "${OUT}/${name}.txt"; then
    echo "REFUSING: ${name} recorded a traceback, not an investigation." >&2
    sed -n '/^Traceback/,$p' "${OUT}/${name}.txt" | tail -3 >&2
    rm -f "${OUT}/${name}.cast" "${OUT}/${name}.txt"
    exit 1
  fi
done

echo "wrote ${count} recording(s) to ${OUT}/" >&2
