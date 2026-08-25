"""Google service-account handling.

The service-account JSON contains a private key, so the property that matters most
is that no error path ever echoes key material. Every rejection here is checked for
leakage, not just for raising.

The stub_google_token fixture from conftest is disabled in this module — these tests
are about the real exchange.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cortex.tools.google_auth import (
    BIGQUERY_SCOPES,
    GA4_SCOPES,
    GoogleCredentialInvalid,
    access_token,
    parse_service_account,
    service_account_email,
)

PRIVATE_KEY_MARKER = "SUPER-SECRET-KEY-MATERIAL"

VALID = json.dumps(
    {
        "type": "service_account",
        "project_id": "cortex-test",
        "private_key_id": "kid-1",
        "private_key": f"-----BEGIN PRIVATE KEY-----\n{PRIVATE_KEY_MARKER}\n"
        "-----END PRIVATE KEY-----\n",
        "client_email": "cortex@cortex-test.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)


@pytest.fixture(autouse=True)
def _no_token_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module tests the real exchange, so the conftest stub must not apply."""
    return None


class TestScopes:
    def test_are_read_only(self) -> None:
        """A write scope would make "read-only integrations" a promise about our
        code rather than a property of the granted access."""
        for scope in GA4_SCOPES + BIGQUERY_SCOPES:
            assert scope.endswith("readonly"), scope

    def test_are_distinct_per_product(self) -> None:
        assert set(GA4_SCOPES).isdisjoint(BIGQUERY_SCOPES)


class TestParsing:
    def test_accepts_a_complete_key(self) -> None:
        info = parse_service_account(VALID)
        assert info["client_email"].endswith("gserviceaccount.com")

    def test_rejects_non_json(self) -> None:
        with pytest.raises(GoogleCredentialInvalid, match="not valid JSON"):
            parse_service_account("not json at all")

    def test_rejects_a_json_array(self) -> None:
        with pytest.raises(GoogleCredentialInvalid, match="JSON object"):
            parse_service_account('["a"]')

    @pytest.mark.parametrize("field", ["type", "client_email", "private_key", "token_uri"])
    def test_names_missing_fields(self, field: str) -> None:
        payload = json.loads(VALID)
        del payload[field]
        with pytest.raises(GoogleCredentialInvalid, match=field):
            parse_service_account(json.dumps(payload))

    def test_rejects_a_user_oauth_credential(self) -> None:
        """Pasting the wrong kind of Google credential is a common setup mistake and
        deserves a specific message."""
        payload = json.loads(VALID)
        payload["type"] = "authorized_user"
        with pytest.raises(GoogleCredentialInvalid, match="authorized_user"):
            parse_service_account(json.dumps(payload))

    def test_rejects_an_empty_private_key(self) -> None:
        payload = json.loads(VALID)
        payload["private_key"] = ""
        with pytest.raises(GoogleCredentialInvalid, match="private_key"):
            parse_service_account(json.dumps(payload))


class TestNoKeyLeakage:
    """No error path may echo private key material."""

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.pop("client_email"),
            lambda p: p.pop("token_uri"),
            lambda p: p.update(type="authorized_user"),
        ],
    )
    def test_parse_errors_do_not_include_the_key(self, mutate: object) -> None:
        payload = json.loads(VALID)
        mutate(payload)  # type: ignore[operator]
        with pytest.raises(GoogleCredentialInvalid) as exc:
            parse_service_account(json.dumps(payload))
        assert PRIVATE_KEY_MARKER not in str(exc.value)

    async def test_signing_failure_does_not_include_the_key(self) -> None:
        """A malformed PEM makes cryptography raise, and its message can contain key
        material — which is why that message is suppressed."""
        with pytest.raises(GoogleCredentialInvalid) as exc:
            await access_token(VALID, GA4_SCOPES)
        assert PRIVATE_KEY_MARKER not in str(exc.value)
        assert "could not be loaded" in str(exc.value)


class TestServiceAccountEmail:
    def test_returns_the_identity(self) -> None:
        """Surfaced because the most common setup failure is a valid key that has
        not been granted access, and the fix is to share with this address."""
        assert service_account_email(VALID) == "cortex@cortex-test.iam.gserviceaccount.com"

    def test_rejects_a_malformed_credential(self) -> None:
        with pytest.raises(GoogleCredentialInvalid):
            service_account_email("{}")


class TestTokenExchange:
    """The exchange itself, with signing stubbed so these tests cover the HTTP half."""

    @pytest.fixture
    def signed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cortex.tools.google_auth as module

        monkeypatch.setattr(module, "_signed_assertion", lambda info, scopes: "signed.jwt.value")

    async def test_returns_the_access_token(self, signed: None) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"access_token": "ya29.token"})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            token = await access_token(VALID, GA4_SCOPES, client=client)
        assert token == "ya29.token"

    async def test_uses_the_jwt_bearer_grant(self, signed: None) -> None:
        seen: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.content)
            return httpx.Response(200, json={"access_token": "ya29.token"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await access_token(VALID, GA4_SCOPES, client=client)

        body = seen[0].decode()
        assert "grant-type%3Ajwt-bearer" in body
        assert "assertion=signed.jwt.value" in body

    async def test_rejection_is_reported_without_the_key(self, signed: None) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                400, json={"error": "invalid_grant", "error_description": "Invalid JWT"}
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(GoogleCredentialInvalid) as exc:
                await access_token(VALID, GA4_SCOPES, client=client)
        assert "invalid_grant" in str(exc.value)
        assert PRIVATE_KEY_MARKER not in str(exc.value)

    async def test_missing_token_in_response_is_an_error(self, signed: None) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(GoogleCredentialInvalid, match="no access token"):
                await access_token(VALID, GA4_SCOPES, client=client)

    async def test_non_json_response_is_an_error(self, signed: None) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>proxy</html>")
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(GoogleCredentialInvalid, match="non-JSON"):
                await access_token(VALID, GA4_SCOPES, client=client)

    async def test_transport_failure_is_wrapped(self, signed: None) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns failure")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(GoogleCredentialInvalid, match="could not reach"):
                await access_token(VALID, GA4_SCOPES, client=client)

    async def test_scopes_are_carried_into_the_assertion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A token minted for the wrong scope fails opaquely much later."""
        import cortex.tools.google_auth as module

        captured: list[tuple[str, ...]] = []

        def _fake(info: dict, scopes: tuple[str, ...]) -> str:
            captured.append(scopes)
            return "signed.jwt.value"

        monkeypatch.setattr(module, "_signed_assertion", _fake)
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"access_token": "t"})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            await access_token(VALID, BIGQUERY_SCOPES, client=client)
        assert captured == [BIGQUERY_SCOPES]


class TestRealJWTSigning:
    """The signing path with a genuine RSA key.

    Everything above stubs `_signed_assertion`, so without this the actual JWT
    construction — the part that decides whether Google accepts the credential at
    all — would be entirely untested.
    """

    @staticmethod
    def _real_key_credential() -> str:
        """A service-account key with a genuine, locally-generated RSA private key.

        Generated rather than committed: a real-looking private key in a repository
        is a liability even when it is worthless.
        """
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
        return json.dumps(
            {
                "type": "service_account",
                "project_id": "cortex-test",
                "private_key_id": "kid-real",
                "private_key": pem,
                "client_email": "cortex@cortex-test.iam.gserviceaccount.com",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )

    def test_produces_a_three_part_jwt(self) -> None:
        import cortex.tools.google_auth as module

        assertion = module._signed_assertion(
            module.parse_service_account(self._real_key_credential()), GA4_SCOPES
        )
        assert assertion.count(".") == 2, "a JWT is header.payload.signature"

    def test_claims_carry_the_scope_audience_and_issuer(self) -> None:
        """A token minted for the wrong scope or audience fails opaquely much later,
        at the first API call rather than at signing."""
        import base64

        import cortex.tools.google_auth as module

        assertion = module._signed_assertion(
            module.parse_service_account(self._real_key_credential()), BIGQUERY_SCOPES
        )
        payload_b64 = assertion.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))

        assert payload["scope"] == " ".join(BIGQUERY_SCOPES)
        assert payload["aud"] == "https://oauth2.googleapis.com/token"
        assert payload["iss"] == "cortex@cortex-test.iam.gserviceaccount.com"
        assert payload["exp"] > payload["iat"]

    def test_key_id_is_carried_in_the_header(self) -> None:
        """Google uses it to select the right public key; omitting it can fail
        verification after a key rotation."""
        import base64

        import cortex.tools.google_auth as module

        assertion = module._signed_assertion(
            module.parse_service_account(self._real_key_credential()), GA4_SCOPES
        )
        header_b64 = assertion.split(".")[0]
        header = json.loads(base64.urlsafe_b64decode(header_b64 + "=" * (-len(header_b64) % 4)))
        assert header["kid"] == "kid-real"

    async def test_end_to_end_exchange_with_a_real_signature(self) -> None:
        """The whole path: parse, sign with a real key, exchange, return the token.
        Nothing about the signing is stubbed."""
        seen: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.content)
            return httpx.Response(200, json={"access_token": "ya29.real-signature"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            token = await access_token(self._real_key_credential(), GA4_SCOPES, client=client)

        assert token == "ya29.real-signature"
        body = seen[0].decode()
        assert "grant-type%3Ajwt-bearer" in body
        # Three JWT segments reached the wire, so a real assertion was sent.
        assert body.split("assertion=")[1].count(".") == 2

    def test_a_malformed_pem_is_rejected_without_echoing_it(self) -> None:
        import cortex.tools.google_auth as module

        broken = json.loads(self._real_key_credential())
        broken["private_key"] = "-----BEGIN PRIVATE KEY-----\nGARBAGE\n-----END PRIVATE KEY-----\n"
        with pytest.raises(GoogleCredentialInvalid, match="could not be loaded") as exc:
            module._signed_assertion(module.parse_service_account(json.dumps(broken)), GA4_SCOPES)
        assert "GARBAGE" not in str(exc.value)
