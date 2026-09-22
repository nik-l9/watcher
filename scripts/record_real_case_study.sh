#!/usr/bin/env bash
#
# Record ONE investigation against live connectors and mask it in the same breath.
#
# Recording and masking are one step on purpose. `sensitive_terms.py` reads the *most
# recent* investigation, because a run does not print its own id, so a second run started
# before the first is masked would silently mask the wrong recording's terms. Doing both
# here makes that ordering impossible to get wrong.
#
# The unmasked recording is written to a temporary directory and deleted on exit, including
# on failure. It never exists inside a repository. The term list is deleted with it -- it
# is a list of the tenant's customers, which is exactly the thing being protected.
#
# Usage:
#   scripts/record_real_case_study.sh <name> "<question>"
#
#   TENANT=acme   OUT=../other/docs/case-studies   scripts/record_real_case_study.sh ...
#
set -euo pipefail

cd "$(dirname "$0")/.."

NAME="${1:?usage: record_real_case_study.sh <name> \"<question>\"}"
QUESTION="${2:?a question is required}"
TENANT="${TENANT:-openhands}"
OUT="${OUT:-docs/case-studies}"
PY=.venv/bin/python

if [ -x .venv/bin/watcher ]; then
  ASK=".venv/bin/watcher ask"
elif [ -x .venv/bin/cortex-ask ]; then
  ASK=".venv/bin/cortex-ask"
else
  ASK="${PY} -m cortex.ask"
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

export CORTEX_LOG_LEVEL=info

echo "running: ${QUESTION}" >&2
asciinema rec \
  --headless \
  --window-size 100x45 \
  --idle-time-limit 2 \
  --title "$NAME" \
  --overwrite \
  "${work}/raw.cast" \
  -c "${ASK} --real --tenant ${TENANT} \"${QUESTION}\""

# Terms come from the investigation that was just committed. The tenant slug is appended
# because it arrives as an argument rather than as evidence, so nothing else would catch it
# -- and it appears twice in the banner of every single run.
"$PY" scripts/sensitive_terms.py > "${work}/terms.txt"
printf '%s\n' "$TENANT" >> "${work}/terms.txt"

mkdir -p "$OUT"
"$PY" scripts/mask_cast.py "${work}/raw.cast" "${OUT}/${NAME}.cast" "${work}/terms.txt"
asciinema convert --output-format txt --overwrite \
  "${OUT}/${NAME}.cast" "${OUT}/${NAME}.txt" >/dev/null 2>&1

# Belt and braces. The masker asserts its own invariants, but this is the check that has
# actually caught things -- a tenant slug left in the cast header, among others.
if grep -qiE "${TENANT}" "${OUT}/${NAME}.cast"; then
  echo "REFUSING: tenant slug survived masking in ${OUT}/${NAME}.cast" >&2
  rm -f "${OUT}/${NAME}.cast" "${OUT}/${NAME}.txt"
  exit 1
fi

echo "wrote ${OUT}/${NAME}.cast" >&2
