"""Verifying that a request really came from Slack.

Written as the attacks it must refuse, because this is the first endpoint Cortex exposes that is
not behind Clerk. Anyone can POST to it, so this check is the only thing between a stranger and
"run an investigation against that tenant's data and post the answer into their Slack".

The positive case is one test. The rest are the four ways the scheme is got wrong.
"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from cortex.security.slack_signing import (
    MAX_SKEW_SECONDS,
    SlackSignatureError,
    verify_slack_request,
)

SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
BODY = b'{"type":"app_mention","text":"why did signups fall?","team_id":"T123"}'


def _signed(
    body: bytes = BODY, *, secret: str = SECRET, at: float | None = None
) -> tuple[str, str]:
    timestamp = str(int(at if at is not None else time.time()))
    digest = hmac.new(
        secret.encode(), f"v0:{timestamp}:".encode() + body, hashlib.sha256
    ).hexdigest()
    return timestamp, f"v0={digest}"


class TestAGenuineRequest:
    def test_it_verifies(self) -> None:
        timestamp, signature = _signed()
        verify_slack_request(
            body=BODY, timestamp=timestamp, signature=signature, signing_secret=SECRET
        )

    def test_a_body_with_awkward_bytes_verifies(self) -> None:
        """The signature is over bytes, and Slack sends UTF-8 text a user typed. An emoji or a
        non-Latin script must not break verification."""
        body = '{"text":"почему упали регистрации? 📉"}'.encode()
        timestamp, signature = _signed(body)
        verify_slack_request(
            body=body, timestamp=timestamp, signature=signature, signing_secret=SECRET
        )


class TestForgery:
    def test_a_wrong_signature_is_refused(self) -> None:
        timestamp, _ = _signed()
        with pytest.raises(SlackSignatureError):
            verify_slack_request(
                body=BODY,
                timestamp=timestamp,
                signature="v0=" + "0" * 64,
                signing_secret=SECRET,
            )

    def test_a_signature_from_a_different_secret_is_refused(self) -> None:
        """What an attacker who has seen the scheme but not the secret can produce."""
        timestamp, signature = _signed(secret="not-the-real-secret")
        with pytest.raises(SlackSignatureError):
            verify_slack_request(
                body=BODY, timestamp=timestamp, signature=signature, signing_secret=SECRET
            )

    def test_a_tampered_body_is_refused(self) -> None:
        """The whole point. A signature captured from a harmless message must not authenticate a
        different question against the same tenant."""
        timestamp, signature = _signed()
        with pytest.raises(SlackSignatureError):
            verify_slack_request(
                body=BODY.replace(b"signups fall", b"revenue leak"),
                timestamp=timestamp,
                signature=signature,
                signing_secret=SECRET,
            )

    def test_a_reserialised_body_is_refused(self) -> None:
        """JSON round-tripping changes key order and whitespace; the signature is over bytes. The
        endpoint must therefore read the raw body before anything parses it, and this asserts the
        verifier does not quietly tolerate the alternative."""
        import json

        timestamp, signature = _signed()
        reserialised = json.dumps(json.loads(BODY)).encode()
        assert reserialised != BODY
        with pytest.raises(SlackSignatureError):
            verify_slack_request(
                body=reserialised,
                timestamp=timestamp,
                signature=signature,
                signing_secret=SECRET,
            )


class TestReplay:
    def test_an_old_request_is_refused(self) -> None:
        """A signature never expires on its own, so a captured request would otherwise be
        replayable forever."""
        now = time.time()
        timestamp, signature = _signed(at=now - MAX_SKEW_SECONDS - 1)
        with pytest.raises(SlackSignatureError, match="window"):
            verify_slack_request(
                body=BODY,
                timestamp=timestamp,
                signature=signature,
                signing_secret=SECRET,
                now=now,
            )

    def test_a_future_request_is_refused(self) -> None:
        """The direction that is easy to forget. Tolerating a future timestamp lets an attacker
        who obtains one signature choose how long it stays valid."""
        now = time.time()
        timestamp, signature = _signed(at=now + MAX_SKEW_SECONDS + 60)
        with pytest.raises(SlackSignatureError, match="window"):
            verify_slack_request(
                body=BODY,
                timestamp=timestamp,
                signature=signature,
                signing_secret=SECRET,
                now=now,
            )

    def test_a_request_inside_the_window_is_accepted(self) -> None:
        now = time.time()
        timestamp, signature = _signed(at=now - MAX_SKEW_SECONDS + 30)
        verify_slack_request(
            body=BODY,
            timestamp=timestamp,
            signature=signature,
            signing_secret=SECRET,
            now=now,
        )


class TestItFailsClosed:
    def test_no_signing_secret_refuses_everything(self) -> None:
        """A misconfigured deployment must not verify every request against an empty key."""
        timestamp, signature = _signed()
        with pytest.raises(SlackSignatureError, match="signing secret"):
            verify_slack_request(
                body=BODY, timestamp=timestamp, signature=signature, signing_secret=""
            )

    @pytest.mark.parametrize(
        ("timestamp", "signature"),
        [(None, "v0=abc"), ("123", None), (None, None), ("", ""), ("123", "")],
    )
    def test_a_missing_header_is_refused(
        self, timestamp: str | None, signature: str | None
    ) -> None:
        with pytest.raises(SlackSignatureError):
            verify_slack_request(
                body=BODY, timestamp=timestamp, signature=signature, signing_secret=SECRET
            )

    def test_a_non_numeric_timestamp_is_refused(self) -> None:
        with pytest.raises(SlackSignatureError):
            verify_slack_request(
                body=BODY, timestamp="not-a-number", signature="v0=abc", signing_secret=SECRET
            )

    def test_an_unknown_signature_version_is_refused(self) -> None:
        """A future v1 must fail closed rather than be compared against a v0 digest."""
        timestamp, signature = _signed()
        with pytest.raises(SlackSignatureError, match="format"):
            verify_slack_request(
                body=BODY,
                timestamp=timestamp,
                signature=signature.replace("v0=", "v1="),
                signing_secret=SECRET,
            )

    def test_a_bare_digest_without_a_version_is_refused(self) -> None:
        timestamp, signature = _signed()
        with pytest.raises(SlackSignatureError, match="format"):
            verify_slack_request(
                body=BODY,
                timestamp=timestamp,
                signature=signature.removeprefix("v0="),
                signing_secret=SECRET,
            )


class TestTheComparisonIsConstantTime:
    def test_it_uses_compare_digest(self) -> None:
        """Asserted by reading the source, because timing is not reliably observable in a test
        on a shared machine. `==` short-circuits on the first differing byte, and its duration
        leaks how many leading bytes of a guessed signature were correct — enough to reconstruct
        one byte at a time."""
        from pathlib import Path

        source = Path("cortex/security/slack_signing.py").read_text()
        assert "hmac.compare_digest" in source
        # No equality comparison against the digest anywhere.
        assert "== digest" not in source
        assert "digest ==" not in source
