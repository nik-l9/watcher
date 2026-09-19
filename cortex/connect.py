"""`python -m cortex.connect` — provision a tenant and store real credentials.

Bring-your-own credentials, per the plan: an operator supplies a token and it is
envelope-encrypted into the `credentials` table, scoped to one tenant. Nothing else in
the system can read it — the executor decrypts it per call, and only after checking that
the tool being invoked belongs to the tenant asking.

Secrets are read from the **environment**, never from an argument. A token passed on the
command line lands in shell history and in the process table, where anyone on the machine
can read it; an environment variable does neither. Nothing here prints a secret, and the
`credentials` table never holds plaintext.

Preconditions this enforces rather than assumes:

  - `CORTEX_VAULT_MASTER_KEY` must be set, or the credential cannot be sealed.
  - The tenancy isolation suite must pass before real data is connected. That is a gate
    in the plan, not a step, and it is checked here so the check cannot be skipped by
    someone in a hurry.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cortex.config.settings import get_settings
from cortex.db.models import Credential, CredentialProvider, Tenant
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.runtime.resources import open_resources
from cortex.security.vault import encrypt_credential
from cortex.tools.slack import IDENTITY_KEY

#: Which environment variable holds each provider's secret, and what the secret is.
#:
#: Named per provider rather than taken from one generic variable, so connecting HubSpot
#: cannot accidentally store a GitHub token under the HubSpot provider — a mistake that
#: would surface much later as a confusing upstream 401.
_ENV_BY_PROVIDER: dict[CredentialProvider, tuple[str, str]] = {
    CredentialProvider.HUBSPOT: ("HUBSPOT_SERVICE_KEY", "private-app access token"),
    CredentialProvider.GITHUB: ("GITHUB_TOKEN", "personal access token"),
    CredentialProvider.GA4: ("GA4_CREDENTIALS_JSON", "service-account JSON"),
    CredentialProvider.BIGQUERY: ("BIGQUERY_CREDENTIALS_JSON", "service-account JSON"),
    # A **user** token (`xoxp-`), not a bot token. Verified the hard way: Slack's
    # search.messages, search.all and search.files all return `not_allowed_token_type`
    # for a bot token even with `search:read.public` granted and the app reinstalled. The
    # scope can sit on a bot; the endpoint refuses it. A bot alternative exists —
    # conversations.history with `channels:history` — but it only sees channels the bot was
    # invited to, which structurally misses the channel nobody thought of.
    CredentialProvider.SLACK: ("SLACK_USER_TOKEN", "user token (xoxp-) with search:read"),
    CredentialProvider.POSTHOG: ("POSTHOG_API_KEY", "personal API key (phx-) with read scopes"),
    # Both analytics newcomers authenticate with HTTP Basic over a *pair*, so the secret is
    # stored as one colon-joined string. Naming the shape in the prompt matters more here than
    # elsewhere: pasting only one half yields a 401 that says nothing about which half is
    # missing, and both vendors call their halves something different from each other.
    CredentialProvider.MIXPANEL: (
        "MIXPANEL_SERVICE_ACCOUNT",
        "service account as 'username:secret' (a project token will not work); "
        "also pass --meta project_id=<id> and, outside the US, --meta region=eu|in",
    ),
    # One provider for every MCP server, because MCP is a transport rather than a vendor. Which
    # server a credential belongs to is the credential's `--label`, so connecting a second one is
    # `--provider mcp --label linear` with its own URL in `--meta`.
    CredentialProvider.MCP: (
        "MCP_SERVER_TOKEN",
        "bearer token for one MCP server (blank if it needs none); pass "
        "--meta url=https://... and --label <server-name>",
    ),
    CredentialProvider.AMPLITUDE: (
        "AMPLITUDE_API_KEY",
        "project credentials as 'api_key:secret_key'; outside the US also pass --meta region=eu",
    ),
}

#: Env var overrides keyed by (provider, label), for the case where one provider needs two
#: different secrets.
#:
#: Slack is that case, and the reason is in the comment above: search requires a *user* token, and
#: posting an answer back into a thread requires a *bot* token with `chat:write`. They are
#: different secrets with different scopes, stored under the same provider with different labels.
#: Without this, connecting the bot token would read SLACK_USER_TOKEN and store whatever was there
#: -- so the delivery path would try to post with a search token and fail with `not_allowed_token
#: _type`, which is exactly the confusing-401-much-later failure the per-provider mapping exists to
#: prevent.
_ENV_BY_PROVIDER_LABEL: dict[tuple[CredentialProvider, str], tuple[str, str]] = {
    (CredentialProvider.SLACK, "bot"): (
        "SLACK_BOT_TOKEN",
        "bot token (xoxb-) with chat:write and files:write, for posting answers back",
    ),
}


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="watcher connect", description=__doc__)
    parser.add_argument("--tenant", required=True, help="Tenant slug. Created if absent.")
    parser.add_argument(
        "--provider",
        action="append",
        # Not `required`: `--list` needs a tenant and nothing else, and argparse would
        # otherwise reject the one invocation that connects nothing.
        choices=[p.value for p in CredentialProvider],
        help="Provider to connect. Repeatable. Omit with --list.",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Non-secret configuration the connector needs alongside the credential, "
        "e.g. --meta project_id=100001 --meta host=https://us.posthog.com. Repeatable. "
        "Stored in plaintext deliberately: a project id is not a secret, and encrypting "
        "it would mean it could not be read without the vault key.",
    )
    parser.add_argument(
        "--label",
        default="default",
        help="Which credential this is, when a tenant has more than one per provider.",
    )
    parser.add_argument(
        "--skip-isolation-check",
        action="store_true",
        help="Do not run the tenancy isolation suite first. Only for a machine where it "
        "has just been run; the gate exists because a leak is the wrong risk to carry.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show what this tenant already has connected, and exit.",
    )
    return parser.parse_args(argv)


def _isolation_suite_passes() -> bool:
    """Run the cross-tenant isolation tests, and report honestly if they fail."""
    print("Running the tenancy isolation suite before touching real credentials...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/tenancy", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-2000:])
        return False
    print(f"  {result.stdout.strip().splitlines()[-1]}")
    return True


async def _tenant(session: AsyncSession, slug: str) -> Tenant:
    existing = await session.scalar(select(Tenant).where(Tenant.slug == slug))
    if existing is not None:
        return existing
    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        slug=slug,
        name=slug,
        graph_name=graph_name_for_new_tenant(slug, tenant_id),
    )
    session.add(tenant)
    await session.flush()
    print(f"Created tenant {slug} ({tenant_id})")
    return tenant


def _parse_meta(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--meta expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


async def _slack_identity(secret: str) -> str | None:
    """This token's own `auth.test` user id, or None.

    Its own function so the secret's lifetime stays inside it, and so a network failure here
    cannot take the connect command down -- `auth.test` needs no scope, but a revoked token or a
    Slack outage must not stop a credential being stored.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {secret}"},
            )
            body = response.json()
    except Exception:  # noqa: BLE001 - see the docstring
        return None
    user_id = body.get("user_id") if isinstance(body, dict) else None
    return user_id if isinstance(user_id, str) and user_id else None


async def _store(  # noqa: PLR0913
    session: AsyncSession,
    tenant: Tenant,
    provider: CredentialProvider,
    label: str,
    metadata: dict[str, str] | None = None,
) -> bool:
    env_var, description = _ENV_BY_PROVIDER_LABEL.get((provider, label), _ENV_BY_PROVIDER[provider])
    secret = os.environ.get(env_var)
    if not secret:
        print(f"  {provider.value}: skipped — {env_var} is not set (expects a {description})")
        return False

    # Slack's own identity for this token, recorded before the secret is dropped.
    #
    # Cortex reads Slack with one credential and posts its reports with another, and the reader
    # cannot run `auth.test` for a token it does not hold. So each credential records who it is
    # here, in metadata, and `cortex.tools.slack.self_ids` reads the whole set at query time --
    # which is what lets the reader recognise the poster's messages as Cortex's own. Without it
    # the analyst cited its own earlier report as having "independently reached the same
    # diagnosis", measured on real data.
    #
    # Best-effort: a token that cannot introspect itself is still a usable token, and refusing
    # to store it would trade a working connector for a provenance nicety.
    if provider is CredentialProvider.SLACK:
        identity = await _slack_identity(secret)
        if identity:
            metadata = {**(metadata or {}), IDENTITY_KEY: identity}
            print(f"  {provider.value}: identity for label {label!r} is {identity}")
        else:
            print(f"  {provider.value}: could not resolve an identity for label {label!r}")

    wrapped_data_key, ciphertext = encrypt_credential(
        tenant.id, provider.value, secret, label=label
    )
    # Dropped as soon as it is sealed. The window cannot be closed completely in Python,
    # but it can be kept short.
    del secret

    existing = await session.scalar(
        select(Credential).where(
            Credential.tenant_id == tenant.id,
            Credential.provider == provider,
            Credential.label == label,
        )
    )
    if existing is not None:
        existing.wrapped_data_key = wrapped_data_key
        existing.ciphertext = ciphertext
        if metadata:
            existing.metadata_ = {**(existing.metadata_ or {}), **metadata}
        print(f"  {provider.value}: rotated (label {label!r})")
    else:
        session.add(
            Credential(
                tenant_id=tenant.id,
                provider=provider,
                label=label,
                wrapped_data_key=wrapped_data_key,
                ciphertext=ciphertext,
                metadata_=metadata or {},
            )
        )
        print(f"  {provider.value}: connected (label {label!r})")
    return True


async def _main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    if not get_settings().vault_master_key:
        sys.stderr.write(
            "CORTEX_VAULT_MASTER_KEY is not set, so a credential cannot be sealed.\n"
            "Generate one with:\n"
            "  python -c 'import base64,os;"
            "print(base64.urlsafe_b64encode(os.urandom(32)).decode())'\n"
        )
        return 2

    if not args.list and not args.provider:
        sys.stderr.write("Nothing to do: pass --provider to connect, or --list to inspect.\n")
        return 2

    if not args.list and not args.skip_isolation_check and not _isolation_suite_passes():
        sys.stderr.write(
            "\nThe tenancy isolation suite failed. No credential was stored.\n"
            "Cross-tenant isolation is the precondition for real data, not a nicety.\n"
        )
        return 1

    async with open_resources() as resources:
        maker = async_sessionmaker(resources.engine, expire_on_commit=False)
        async with maker() as session:
            tenant = await _tenant(session, args.tenant)

            if args.list:
                rows = (
                    (
                        await session.execute(
                            select(Credential).where(Credential.tenant_id == tenant.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                print(f"Tenant {tenant.slug} ({tenant.id}) has {len(rows)} credential(s):")
                for row in rows:
                    print(f"  {row.provider.value:<10} label={row.label}")
                return 0

            stored = 0
            for value in args.provider:
                if await _store(
                    session,
                    tenant,
                    CredentialProvider(value),
                    args.label,
                    _parse_meta(args.meta),
                ):
                    stored += 1
            await session.commit()

    if not stored:
        sys.stderr.write("\nNothing was connected — no matching environment variable was set.\n")
        return 1
    print(f"\n{stored} credential(s) stored, encrypted, scoped to {args.tenant}.")
    print(f"Investigate real data with:  python -m cortex.ask --real --tenant {args.tenant}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
