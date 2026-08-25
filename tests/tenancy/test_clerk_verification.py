"""Clerk token verification.

Placed in `tests/tenancy` because this is the gate the plan puts before real credentials: it
is what stops one tenant reading another's data. The suite is written as the attacks it must
refuse, not as the happy path, because a verifier that accepts a good token and also accepts
a forged one passes any test written the other way round.

Keys are generated per run rather than fixtured, so no private key material lives in the
repository — and the forged tokens are genuinely signed by a key the verifier does not trust,
rather than being strings that merely look wrong.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from cortex.security.clerk import (
    ClerkAuthError,
    ClerkUnavailable,
    ClerkVerifier,
)

ISSUER = "https://clerk.example.com"


def _key(kid: str) -> tuple[Any, dict[str, Any]]:
    """A fresh RSA key and its JWKS entry."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    public_jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return private, public_jwk


def _token(
    private: Any,
    *,
    kid: str = "key-1",
    algorithm: str = "RS256",
    issuer: str | None = ISSUER,
    claims: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "sub": "user_2abc",
        "org_slug": "acme",
        "org_id": "org_1",
        "org_role": "org:member",
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
    }
    if issuer:
        payload["iss"] = issuer
    payload.update(claims or {})
    return jwt.encode(payload, private, algorithm=algorithm, headers={"kid": kid})


@pytest.fixture
def signing() -> tuple[Any, dict[str, Any]]:
    return _key("key-1")


@pytest.fixture(autouse=True)
def _stub_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the verifier's JWKS fetch through whatever the test installed."""
    real = httpx.AsyncClient

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        handler = _stub_httpx.handler  # type: ignore[attr-defined]
        kwargs.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("cortex.security.clerk.httpx.AsyncClient", _factory)


def _serve(jwks: dict[str, Any], *, status: int = 200, count: dict[str, int] | None = None) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if count is not None:
            count["n"] = count.get("n", 0) + 1
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(200, json=jwks)

    _stub_httpx.handler = _handler  # type: ignore[attr-defined]


def _plain(**kwargs: Any) -> ClerkVerifier:
    return ClerkVerifier(
        jwks_url="https://clerk.example.com/.well-known/jwks.json",
        issuer=ISSUER,
        **kwargs,
    )


class TestAValidTokenIsAccepted:
    async def test_the_identity_comes_from_the_verified_claims(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        private, jwk = signing
        _serve({"keys": [jwk]})

        identity = await _plain().verify(_token(private))

        assert identity.subject == "user_2abc"
        assert identity.org_slug == "acme"
        assert identity.org_role == "org:member"

    async def test_the_jwks_is_fetched_once_and_reused(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """A verifier constructed per request would fetch Clerk's keys on every call, turning
        our own traffic into an amplifier pointed at Clerk."""
        private, jwk = signing
        count: dict[str, int] = {}
        _serve({"keys": [jwk]}, count=count)

        verifier = _plain()
        for _ in range(5):
            await verifier.verify(_token(private))

        assert count["n"] == 1


class TestTokensItMustRefuse:
    async def test_a_token_signed_by_another_key_is_refused(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """The whole point. The attacker's key is a real RSA key and the token is genuinely
        signed — it is simply not signed by Clerk."""
        _, jwk = signing
        attacker, _ = _key("key-1")  # same kid, different key
        _serve({"keys": [jwk]})

        with pytest.raises(ClerkAuthError, match="signature could not be verified"):
            await _plain().verify(_token(attacker))

    async def test_alg_none_is_refused(self, signing: tuple[Any, dict[str, Any]]) -> None:
        """The classic. A token declaring `none` carries no signature at all, and a verifier
        that reads the algorithm from the header will happily accept it."""
        _, jwk = signing
        _serve({"keys": [jwk]})
        unsigned = jwt.encode(
            {"sub": "user_evil", "org_slug": "acme", "iat": int(time.time())},
            key=None,  # type: ignore[arg-type]
            algorithm="none",
            headers={"kid": "key-1"},
        )

        with pytest.raises(ClerkAuthError, match="unsupported signing algorithm"):
            await _plain().verify(unsigned)

    async def test_hs256_signed_with_the_public_key_is_refused(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """Algorithm confusion, the real version.

        An RSA public key is public, so an attacker can use its bytes as an HMAC secret. A
        verifier that trusts the header's `alg` computes the same HMAC and accepts it.

        Forged by hand rather than with `jwt.encode`, which refuses to use a PEM as an HMAC
        secret — that refusal is PyJWT protecting the *signer*, and relying on it would make
        this test prove nothing about whether *we* reject the token.
        """
        import base64
        import hashlib
        import hmac

        from cryptography.hazmat.primitives import serialization

        _, jwk = signing
        _serve({"keys": [jwk]})
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        secret = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        def _segment(payload: dict[str, Any]) -> bytes:
            raw = json.dumps(payload, separators=(",", ":")).encode()
            return base64.urlsafe_b64encode(raw).rstrip(b"=")

        header = _segment({"alg": "HS256", "typ": "JWT", "kid": "key-1"})
        body = _segment(
            {
                "sub": "user_evil",
                "org_slug": "acme",
                "iss": ISSUER,
                "iat": int(time.time()),
                "exp": int(time.time()) + 300,
            }
        )
        signature = base64.urlsafe_b64encode(
            hmac.new(secret, header + b"." + body, hashlib.sha256).digest()
        ).rstrip(b"=")
        forged = b".".join([header, body, signature]).decode()

        with pytest.raises(ClerkAuthError, match="unsupported signing algorithm"):
            await _plain().verify(forged)

    async def test_an_expired_token_is_refused(self, signing: tuple[Any, dict[str, Any]]) -> None:
        private, jwk = signing
        _serve({"keys": [jwk]})
        stale = _token(
            private,
            claims={"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},
        )

        with pytest.raises(ClerkAuthError, match="expired"):
            await _plain().verify(stale)

    async def test_a_token_from_another_issuer_is_refused(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """A well-signed token proves Clerk issued it, not that it was issued for us."""
        private, jwk = signing
        _serve({"keys": [jwk]})

        with pytest.raises(ClerkAuthError, match="issuer"):
            await _plain().verify(_token(private, issuer="https://evil.example.com"))

    async def test_a_token_without_an_expiry_is_refused(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """An immortal token is a permanent credential in a browser."""
        private, jwk = signing
        _serve({"keys": [jwk]})
        payload = {"sub": "u", "org_slug": "acme", "iat": int(time.time()), "iss": ISSUER}
        forever = jwt.encode(payload, private, algorithm="RS256", headers={"kid": "key-1"})

        with pytest.raises(ClerkAuthError, match="missing the 'exp'"):
            await _plain().verify(forever)

    async def test_an_unknown_key_id_is_refused(self, signing: tuple[Any, dict[str, Any]]) -> None:
        private, jwk = signing
        _serve({"keys": [jwk]})

        with pytest.raises(ClerkAuthError, match="unknown key"):
            await _plain().verify(_token(private, kid="key-does-not-exist"))

    @pytest.mark.parametrize("garbage", ["", "not-a-token", "a.b", "a.b.c.d"])
    async def test_malformed_tokens_are_refused_without_a_fetch(
        self, signing: tuple[Any, dict[str, Any]], garbage: str
    ) -> None:
        """Refused before any network call, so junk cannot be used to generate traffic."""
        _, jwk = signing
        count: dict[str, int] = {}
        _serve({"keys": [jwk]}, count=count)

        with pytest.raises(ClerkAuthError):
            await _plain().verify(garbage)
        assert count.get("n", 0) == 0

    async def test_an_audience_is_checked_when_configured(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        private, jwk = signing
        _serve({"keys": [jwk]})
        verifier = _plain(audience="cortex-api")

        with pytest.raises(ClerkAuthError, match="audience"):
            await verifier.verify(_token(private, claims={"aud": "someone-else"}))

        identity = await verifier.verify(_token(private, claims={"aud": "cortex-api"}))
        assert identity.subject == "user_2abc"


class TestKeyRotationAndOutages:
    async def test_an_unknown_kid_triggers_at_most_one_refetch(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """An unknown kid is what a key rotation looks like — and what a forged token looks
        like. Without a floor, forged tokens become a request amplifier pointed at Clerk and
        the gateway stalls waiting on it."""
        private, jwk = signing
        count: dict[str, int] = {}
        _serve({"keys": [jwk]}, count=count)

        clock = iter([float(n) for n in range(0, 200)])
        verifier = ClerkVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer=ISSUER,
            clock=lambda: next(clock),
        )
        for _ in range(4):
            with pytest.raises(ClerkAuthError):
                await verifier.verify(_token(private, kid="rotated"))

        # One initial fetch plus at most one refetch inside the interval.
        assert count["n"] <= 2, count

    async def test_a_rotated_key_is_picked_up(self) -> None:
        """The legitimate case the refetch exists for."""
        old, old_jwk = _key("key-old")
        new, new_jwk = _key("key-new")
        served = {"keys": [old_jwk]}
        _serve(served)

        clock = iter([float(n) * 100 for n in range(0, 50)])
        verifier = ClerkVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer=ISSUER,
            clock=lambda: next(clock),
        )
        assert (await verifier.verify(_token(old, kid="key-old"))).subject == "user_2abc"

        served["keys"] = [new_jwk]
        _serve(served)
        assert (await verifier.verify(_token(new, kid="key-new"))).subject == "user_2abc"

    async def test_an_outage_before_any_fetch_is_unavailable_not_unauthorised(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """One means "your token is bad" and the other means "we cannot tell". Collapsing
        them would tell a user their login failed during an outage on our side."""
        private, jwk = signing
        _serve({"keys": [jwk]}, status=500)

        with pytest.raises(ClerkUnavailable):
            await _plain().verify(_token(private))

    async def test_a_cached_key_set_survives_a_brief_outage(
        self, signing: tuple[Any, dict[str, Any]]
    ) -> None:
        """A stale-but-real key set verifies a genuine token correctly. Failing every request
        during a Clerk blip would be a worse trade than the bounded risk of accepting a token
        signed by a key Clerk has since revoked."""
        private, jwk = signing
        _serve({"keys": [jwk]})

        clock = iter([float(n) for n in range(0, 200)])
        verifier = ClerkVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer=ISSUER,
            clock=lambda: next(clock),
        )
        assert (await verifier.verify(_token(private))).subject == "user_2abc"

        _serve({"keys": [jwk]}, status=503)
        assert (await verifier.verify(_token(private))).subject == "user_2abc"


class TestConstruction:
    def test_a_plaintext_jwks_url_is_refused(self) -> None:
        """A JWKS fetched over plaintext can be swapped in transit, which makes every
        signature check meaningless."""
        with pytest.raises(ValueError, match="must be https"):
            ClerkVerifier(jwks_url="http://clerk.example.com/jwks.json")
