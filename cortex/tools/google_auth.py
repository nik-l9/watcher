"""Google service-account token minting, shared by the GA4 and BigQuery connectors.

Both connectors authenticate the same way — a tenant pastes a service-account JSON,
which is exchanged for a short-lived OAuth2 access token. Doing that in one place
means one implementation to audit for credential handling, and one place where a
malformed key produces a clear error rather than a stack trace.

The exchange is implemented directly against the JWT-bearer grant rather than via
`google.auth.transport.requests`, for two reasons: that transport's `refresh()` is
synchronous and would block the event loop on every tool call, and it would pull in
a second HTTP library alongside httpx. Only the crypto primitives from google-auth
are used, so the JWT signing itself is not hand-rolled.

The service-account JSON is a private key. It is parsed, used, and dropped: never
logged, never written to disk, never echoed into a ToolResult.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from google.auth import crypt, jwt

from cortex.tools.base import ToolError

# Read-only scopes only. A write scope would make "read-only integrations" a
# promise about our code rather than a property of the granted access.
GA4_SCOPES = ("https://www.googleapis.com/auth/analytics.readonly",)
BIGQUERY_SCOPES = ("https://www.googleapis.com/auth/bigquery.readonly",)

_JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_ASSERTION_LIFETIME = 3600
_REQUIRED_FIELDS = ("type", "client_email", "private_key", "token_uri")


class GoogleCredentialInvalid(ToolError):
    """The stored service-account JSON is unusable. The tenant must reconnect."""


def parse_service_account(raw: str) -> dict[str, Any]:
    try:
        info = json.loads(raw)
    except ValueError as exc:
        raise GoogleCredentialInvalid(
            "stored Google credential is not valid JSON; expected a service-account key file"
        ) from exc
    if not isinstance(info, dict):
        raise GoogleCredentialInvalid("stored Google credential must be a JSON object")

    missing = [field for field in _REQUIRED_FIELDS if not info.get(field)]
    if missing:
        # Names the missing fields but never echoes a value — one of them is a
        # private key.
        raise GoogleCredentialInvalid(
            f"service-account key is missing required fields: {', '.join(missing)}"
        )
    if info.get("type") != "service_account":
        raise GoogleCredentialInvalid(
            f"expected a service_account key, got type={info.get('type')!r}"
        )
    return info


def service_account_email(raw_credential: str) -> str:
    """The service account's identity. Safe to display — it is not a secret.

    Worth surfacing: the most common GA4 and BigQuery setup failure is a valid key
    that has not been granted access, and the fix is to share the property or
    dataset with this address.
    """
    return str(parse_service_account(raw_credential)["client_email"])


def _signed_assertion(info: dict[str, Any], scopes: tuple[str, ...]) -> str:
    try:
        signer = crypt.RSASigner.from_service_account_info(info)
    except (ValueError, KeyError, TypeError) as exc:
        # The message is suppressed deliberately: cryptography libraries sometimes
        # include key material in parse errors.
        raise GoogleCredentialInvalid("service-account private key could not be loaded") from exc

    issued_at = int(time.time())
    payload = {
        "iss": info["client_email"],
        "scope": " ".join(scopes),
        "aud": info["token_uri"],
        "iat": issued_at,
        "exp": issued_at + _ASSERTION_LIFETIME,
    }
    return jwt.encode(signer, payload, key_id=info.get("private_key_id")).decode()


async def access_token(
    raw_credential: str,
    scopes: tuple[str, ...],
    *,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Mint a short-lived access token for the given read-only scopes.

    `client` exists so tests can supply a mock transport; production callers omit it.
    """
    info = parse_service_account(raw_credential)
    assertion = _signed_assertion(info, scopes)

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    try:
        response = await client.post(
            info["token_uri"],
            data={"grant_type": _JWT_BEARER_GRANT, "assertion": assertion},
        )
    except httpx.HTTPError as exc:
        raise GoogleCredentialInvalid(
            f"could not reach Google's token endpoint: {type(exc).__name__}"
        ) from exc
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code != 200:
        # Google's error body describes the key, not the request, and can be
        # verbose; the status plus its short error code is enough to act on.
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = f": {body.get('error', '')} {body.get('error_description', '')}".strip()
        except ValueError:
            detail = ""
        raise GoogleCredentialInvalid(
            f"Google rejected the service-account key ({response.status_code}){detail}"
        )

    try:
        token = response.json().get("access_token")
    except ValueError as exc:
        raise GoogleCredentialInvalid("Google's token endpoint returned non-JSON") from exc

    if not token:
        raise GoogleCredentialInvalid("Google returned no access token")
    return str(token)
