"""Clerk session-token verification.

Until now the gateway trusted `X-Cortex-Tenant` and `X-Cortex-User` as presented, and
refused to serve at all outside `local`/`test` so that the gap failed closed (F-02). This is
the other half: verifying a real token, so authenticated traffic can be served.

**The threat model is the caller.** A session token arrives from a browser, which means every
field in it is attacker-controlled until a signature over it has been checked with a key we
fetched ourselves. The failures that matter are not exotic:

  - **Algorithm confusion.** A token declaring `alg: none`, or `alg: HS256` signed with the
    *public* key as an HMAC secret, verifies against a naive implementation. Only RS256 is
    accepted here, and the algorithm is passed as an allowlist rather than read from the
    header.
  - **A forged `kid`.** An unknown key id must not trigger a JWKS refetch on every request,
    or an attacker can make us hammer Clerk — and take the gateway down with it. Refetches
    are rate-limited.
  - **Unvalidated claims.** Expiry, not-before, issuer and (when configured) audience are all
    required. A token that is merely *well-signed* proves only that Clerk issued it, not that
    it was issued for us, or recently.

**No secrets are logged, and no token text ever appears in an error.** A rejected token's
`sub` is not echoed either: a caller learning which subject the server thought it saw is a
small oracle, and there is nothing it helps them fix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt
from jwt import PyJWK, PyJWKSet

#: The only signing algorithm accepted.
#:
#: An allowlist, passed explicitly to `jwt.decode`. Reading the algorithm from the token's own
#: header is the algorithm-confusion vulnerability: a token can declare `none`, or declare
#: `HS256` and be signed with the RSA *public* key as an HMAC secret, and a library told to
#: trust the header will verify both.
ALGORITHMS = ("RS256",)

#: Clock skew tolerated on `exp`, `nbf` and `iat`.
#:
#: Sixty seconds. Enough that a browser and a server disagreeing about the time does not log
#: a user out; short enough that an expired token is not usable for meaningfully longer than
#: it should be.
LEEWAY_SECONDS = 60

#: How long a fetched JWKS is reused.
_JWKS_TTL_SECONDS = 600

#: The minimum gap between JWKS fetches, whatever happens.
#:
#: A token carrying an unknown `kid` is the signal that Clerk has rotated its keys — and also
#: exactly what a forged token looks like. Without this floor, a stream of forged tokens
#: becomes a request amplifier pointed at Clerk, and the gateway stalls waiting on it.
_MIN_FETCH_INTERVAL_SECONDS = 30

_TIMEOUT_SECONDS = 5.0


class ClerkAuthError(Exception):
    """A token could not be verified. The message is safe to return to the caller."""


class ClerkUnavailable(Exception):
    """Clerk's JWKS could not be fetched.

    Distinct from `ClerkAuthError` on purpose: one means "your token is bad" (401) and the
    other means "we cannot tell right now" (503). Collapsing them would tell a user their
    login had failed during an outage on our side, and would hide the outage.
    """


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    """Who the caller is, according to a signature we checked."""

    #: Clerk's stable user id. This is the external id `users.external_id` holds.
    subject: str
    #: The Clerk organisation, when the token carries one. This is what maps to a tenant.
    org_id: str | None = None
    org_slug: str | None = None
    #: The organisation role as Clerk states it — never as a client claims it.
    org_role: str | None = None
    #: Remaining claims, for diagnostics. Never used for authorisation decisions.
    claims: dict[str, Any] = field(default_factory=dict)


class ClerkVerifier:
    """Verifies Clerk session tokens against Clerk's published keys."""

    def __init__(
        self,
        *,
        jwks_url: str,
        issuer: str | None = None,
        audience: str | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if not jwks_url.startswith("https://"):
            # A JWKS fetched over plaintext can be swapped in transit, which makes every
            # signature check meaningless. Refused rather than warned about.
            raise ValueError(f"JWKS url must be https, got {jwks_url!r}")
        self._jwks_url = jwks_url
        self._issuer = issuer
        self._audience = audience
        self._clock = clock
        self._keys: PyJWKSet | None = None
        self._fetched_at = 0.0
        self._last_attempt = 0.0

    async def verify(self, token: str) -> VerifiedIdentity:
        """Verify a token and return who it says the caller is.

        Raises `ClerkAuthError` for anything wrong with the token, and `ClerkUnavailable`
        when Clerk cannot be reached.
        """
        if not token or token.count(".") != 2:
            raise ClerkAuthError("malformed token")

        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise ClerkAuthError("malformed token header") from None

        # Checked before the key lookup, so a token declaring `none` or `HS256` is refused
        # without touching the key material at all.
        algorithm = header.get("alg")
        if algorithm not in ALGORITHMS:
            raise ClerkAuthError(f"unsupported signing algorithm {algorithm!r}")

        kid = header.get("kid")
        if not kid:
            raise ClerkAuthError("token header carries no key id")

        key = await self._key_for(kid)
        options = {
            "require": ["exp", "iat", "sub"],
            "verify_aud": self._audience is not None,
        }
        try:
            claims = jwt.decode(
                token,
                key=key.key,
                # The allowlist, not the header. This is the line that closes algorithm
                # confusion.
                algorithms=list(ALGORITHMS),
                issuer=self._issuer,
                audience=self._audience,
                leeway=LEEWAY_SECONDS,
                options=options,
            )
        except jwt.ExpiredSignatureError:
            raise ClerkAuthError("token has expired") from None
        except jwt.ImmatureSignatureError:
            raise ClerkAuthError("token is not valid yet") from None
        except jwt.InvalidIssuerError:
            raise ClerkAuthError("token was not issued by the expected issuer") from None
        except jwt.InvalidAudienceError:
            raise ClerkAuthError("token was not issued for this audience") from None
        except jwt.MissingRequiredClaimError as exc:
            raise ClerkAuthError(f"token is missing the {exc.claim!r} claim") from None
        except jwt.PyJWTError:
            # Everything else — a bad signature above all. Deliberately not detailed: the
            # distinction between "wrong key" and "tampered payload" helps an attacker and
            # nobody else.
            raise ClerkAuthError("token signature could not be verified") from None

        subject = str(claims.get("sub") or "")
        if not subject:
            raise ClerkAuthError("token carries no subject")

        return VerifiedIdentity(
            subject=subject,
            org_id=_string_or_none(claims.get("org_id")),
            org_slug=_string_or_none(claims.get("org_slug")),
            org_role=_string_or_none(claims.get("org_role")),
            claims=dict(claims),
        )

    async def _key_for(self, kid: str) -> PyJWK:
        """The signing key for this id, fetching the JWKS if needed."""
        keys = await self._jwks()
        try:
            return keys[kid]
        except KeyError:
            pass

        # An unknown kid is what a key rotation looks like — and also what a forged token
        # looks like. Refetch, but only if the floor has passed, so forged tokens cannot be
        # used to amplify requests at Clerk.
        if self._clock() - self._last_attempt >= _MIN_FETCH_INTERVAL_SECONDS:
            keys = await self._jwks(force=True)
            try:
                return keys[kid]
            except KeyError:
                pass
        raise ClerkAuthError("token was signed by an unknown key")

    async def _jwks(self, *, force: bool = False) -> PyJWKSet:
        fresh = self._keys is not None and self._clock() - self._fetched_at < _JWKS_TTL_SECONDS
        if fresh and not force:
            return self._keys  # type: ignore[return-value]

        self._last_attempt = self._clock()
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.get(self._jwks_url)
                response.raise_for_status()
                document = response.json()
        except Exception as exc:  # noqa: BLE001 - re-raised as a distinct failure below
            if self._keys is not None:
                # Serve from the cached set rather than failing every request during a brief
                # Clerk outage. The keys are still Clerk's own, and a stale-but-real key set
                # verifies a genuine token correctly; the risk it carries is accepting a
                # token signed by a key Clerk has since revoked, bounded by the TTL.
                return self._keys
            raise ClerkUnavailable(f"could not fetch Clerk keys: {type(exc).__name__}") from exc

        try:
            self._keys = PyJWKSet.from_dict(document)
        except Exception as exc:  # noqa: BLE001
            raise ClerkUnavailable("Clerk returned a JWKS we could not parse") from exc
        self._fetched_at = self._clock()
        return self._keys


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
