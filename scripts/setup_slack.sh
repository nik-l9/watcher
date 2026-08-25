#!/usr/bin/env bash
#
# Store the @cortex Slack credentials for a tenant, and check them against Slack.
#
# Reads both secrets interactively rather than taking them as arguments, because a shell argument
# is visible in `ps` output and lands in shell history. This script therefore never puts either
# value on a command line, in a file, or in its own output.
#
#   ./scripts/setup_slack.sh acme
#
# What it does:
#   1. Reads the bot token and signing secret without echoing them.
#   2. Calls Slack's auth.test to confirm the token works and to read the workspace id.
#   3. Confirms the workspace id matches the one already claimed for this tenant, and refuses
#      loudly if it does not — a mismatch means answers would be posted into a workspace whose
#      data this tenant is not authorised to read.
#   4. Reports which of the three required scopes are actually granted.
#   5. Stores the bot token in the vault under label 'bot' via `python -m cortex.connect`.
#   6. Prints the one line to add to .env for the signing secret, which the gateway reads at
#      startup. Not written automatically: .env is yours, and a script that edits it silently is
#      a script that will one day overwrite something.

set -euo pipefail

TENANT="${1:-}"
if [[ -z "$TENANT" ]]; then
  echo "usage: $0 <tenant-slug>" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "no .venv here — run 'make install' first" >&2
  exit 2
fi

# -s suppresses echo. Both values are secrets and neither should appear on screen, in a scrollback
# buffer, or in a screen recording.
read -r -s -p "Bot User OAuth Token (xoxb-…): " SLACK_BOT_TOKEN
echo
read -r -s -p "Signing Secret: " SLACK_SIGNING_SECRET
echo

if [[ "$SLACK_BOT_TOKEN" != xoxb-* ]]; then
  # Named specifically, because pasting the user token by mistake is the likely error: search
  # needs xoxp- and posting needs xoxb-, and Slack's error for the wrong one arrives much later
  # as `not_allowed_token_type` from chat.postMessage.
  echo "that does not look like a bot token — a bot token starts with 'xoxb-'." >&2
  echo "the xoxp- user token is for search and is stored separately under the default label." >&2
  exit 2
fi
if [[ -z "$SLACK_SIGNING_SECRET" ]]; then
  echo "the signing secret is required: without it the endpoint refuses every request." >&2
  exit 2
fi

export SLACK_BOT_TOKEN SLACK_SIGNING_SECRET

echo
echo "Checking the token against Slack…"
SLACK_TENANT="$TENANT" .venv/bin/python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

token = os.environ["SLACK_BOT_TOKEN"]


def call(method: str) -> dict:
    request = urllib.request.Request(
        f"https://slack.com/api/{method}",
        data=b"",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read())
            # Scopes come back in a response header rather than the body, which is the only way to
            # see what was actually granted versus what the manifest asked for.
            body["_scopes"] = response.headers.get("x-oauth-scopes", "")
            return body
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"network: {exc}"}


result = call("auth.test")
if not result.get("ok"):
    # Slack's own error strings never echo the token, so this is safe to print.
    print(f"  token rejected: {result.get('error')}", file=sys.stderr)
    sys.exit(1)

team_id = result.get("team_id", "")
print(f"  workspace : {result.get('team')} ({team_id})")
print(f"  bot user  : {result.get('user')}")

granted = {scope.strip() for scope in result.get("_scopes", "").split(",") if scope.strip()}
required = ("app_mentions:read", "chat:write", "files:write")
missing = [scope for scope in required if scope not in granted]
for scope in required:
    print(f"  {'ok  ' if scope in granted else 'MISSING'} {scope}")
if missing:
    print(
        "\n  Add the missing scopes in OAuth & Permissions, then reinstall the app —"
        "\n  a scope added without reinstalling does not take effect.",
        file=sys.stderr,
    )

# The workspace this token belongs to must be the one already claimed for the tenant. A mismatch
# would mean posting answers into a workspace whose data this tenant is not authorised to read.
import asyncio  # noqa: E402 - only needed on the happy path

from sqlalchemy import select  # noqa: E402

from cortex.db.models import Tenant  # noqa: E402
from cortex.runtime.resources import open_resources  # noqa: E402


async def check_claim() -> int:
    slug = os.environ["SLACK_TENANT"]
    async with open_resources() as resources:
        async with resources.session() as session:
            row = (
                await session.execute(
                    select(Tenant.slack_team_id).where(Tenant.slug == slug)
                )
            ).scalar_one_or_none()
    if row is None:
        print(f"\n  no tenant named {slug!r}", file=sys.stderr)
        return 1
    if not row:
        print(
            f"\n  tenant {slug!r} has not claimed a workspace yet. Run:\n"
            f"    UPDATE tenants SET slack_team_id = '{team_id}' WHERE slug = '{slug}';",
            file=sys.stderr,
        )
        return 1
    if row != team_id:
        print(
            f"\n  MISMATCH: tenant {slug!r} is claimed by {row}, but this token belongs to "
            f"{team_id}.\n  Refusing: answers would be posted into a workspace whose data this "
            "tenant is not authorised to read.",
            file=sys.stderr,
        )
        return 1
    print(f"  claimed by: {slug}")
    return 0


sys.exit(asyncio.run(check_claim()))
PY

echo
echo "Storing the bot token in the vault (label 'bot')…"
.venv/bin/python -m cortex.connect --tenant "$TENANT" --provider slack --label bot

cat <<'EOF'

Two things left, both yours:

  1. Add the signing secret to .env — the gateway reads it at startup, and without it the
     endpoint refuses every request (verification fails closed rather than checking an
     internet-facing webhook against an empty key):

       CORTEX_SLACK_SIGNING_SECRET=<the signing secret you just pasted>
       CORTEX_PUBLIC_BASE_URL=<your public https URL>

     The second is only for the "full report" link. Unset means no link rather than a localhost
     one, because a message telling a colleague to open localhost reads as a broken feature.

  2. Set the app's Request URL to <your public URL>/slack/events and invite the bot:

       /invite @cortex

Then restart the gateway and mention it. See docs/running-it.md.
EOF
