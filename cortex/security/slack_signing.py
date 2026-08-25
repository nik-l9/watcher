"""Verifying that a request really came from Slack.

The Slack entry point is the first endpoint Cortex exposes to the public internet that is not
behind Clerk. Anyone can POST to it. So the only thing standing between a stranger and "run an
investigation against this tenant's data, then post the answer into their Slack" is this file.

## The scheme

Slack signs every request with HMAC-SHA256 over `v0:{timestamp}:{body}`, keyed by the app's
signing secret, and sends the result as `X-Slack-Signature: v0=<hex>` alongside
`X-Slack-Request-Timestamp`. Verifying means recomputing it over the **raw** body.

## The four ways this is got wrong, each of which has a test

1. **Comparing with `==`.** String comparison short-circuits on the first differing byte, so its
   duration leaks how much of a guessed signature was correct. `hmac.compare_digest` is
   constant-time and is the only comparison used here.
2. **Not checking the timestamp.** A signature stays valid forever, so a captured request can be
   replayed indefinitely. Slack's own guidance is a five-minute window; this rejects anything
   outside it, in *either* direction — a timestamp far in the future is as suspicious as an old
   one and would otherwise grant an attacker a signature valid for as long as they chose.
3. **Verifying a re-serialised body.** JSON round-tripping changes key order and whitespace, and
   the signature is over bytes. The verifier therefore takes `bytes` and the endpoint must read
   the raw body before FastAPI parses it.
4. **Accepting a missing signature as valid.** An absent header, an empty secret, or a
   version prefix other than `v0` must all fail closed rather than skip the check.
"""

from __future__ import annotations

import hashlib
import hmac
import time

#: How far a request's timestamp may be from now, in seconds.
#:
#: Five minutes, matching Slack's own guidance. Applied in both directions: a timestamp in the
#: future is not a clock-skew nuisance to be tolerated but a way to mint a signature that stays
#: valid for as long as the attacker likes.
MAX_SKEW_SECONDS = 300

#: The only signature version this understands. A future `v1` must fail closed rather than be
#: compared against a v0 digest, which would never match but would waste the comparison and
#: obscure the reason.
_VERSION = "v0"


class SlackSignatureError(Exception):
    """The request did not come from Slack, or did not come recently.

    One exception for every cause — bad signature, missing header, stale timestamp — because the
    caller is unauthenticated and a specific reason tells an attacker which half of the scheme
    to work on.
    """


def verify_slack_request(
    *,
    body: bytes,
    timestamp: str | None,
    signature: str | None,
    signing_secret: str,
    now: float | None = None,
) -> None:
    """Raise `SlackSignatureError` unless this request is a recent, genuine Slack request.

    `body` must be the raw bytes as received. Re-serialised JSON will not verify, and that is
    the intended behaviour rather than a limitation.
    """
    if not signing_secret:
        # Fails closed. An unset secret in a misconfigured deployment would otherwise make every
        # request verify against an empty key, which is the worst possible default.
        raise SlackSignatureError("no Slack signing secret is configured")
    if not timestamp or not signature:
        raise SlackSignatureError("request is not signed")

    try:
        sent_at = float(timestamp)
    except ValueError as exc:
        raise SlackSignatureError("request timestamp is not a number") from exc

    moment = now if now is not None else time.time()
    if abs(moment - sent_at) > MAX_SKEW_SECONDS:
        raise SlackSignatureError("request timestamp is outside the accepted window")

    version, _, digest = signature.partition("=")
    if version != _VERSION or not digest:
        raise SlackSignatureError("unsupported signature format")

    expected = hmac.new(
        signing_secret.encode(),
        f"{_VERSION}:{timestamp}:".encode() + body,
        hashlib.sha256,
    ).hexdigest()

    # Constant-time. `==` would leak, through its duration, how many leading bytes of a guessed
    # signature were right, which is enough to reconstruct one byte at a time.
    if not hmac.compare_digest(expected, digest):
        raise SlackSignatureError("signature does not match")
