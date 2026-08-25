"""Delivering a finished report into Slack.

One rule shapes every line of `cortex/notify/deliver.py`: **delivery must never fail an
investigation.** The report is already committed, gated, verified and readable on its own page. If
Slack is down, the token was revoked, or the channel was archived between question and answer, the
right outcome is a logged reason and a completed investigation.

So the tests here are mostly failures, and each asserts the same two things: a string comes back,
and nothing raises.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import (
    Credential,
    CredentialProvider,
    Investigation,
    InvestigationStatus,
    Report,
    Tenant,
)
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.notify.deliver import BOT_LABEL, deliver_report
from cortex.notify.slack import post_answer, render_for_slack
from cortex.security.vault import encrypt_credential
from cortex.tenancy.context import TenantContext

BODY = {
    "question": "Why did signups fall?",
    "executive_summary": [
        {
            "text": "Signups fell 12% after the 14 July deploy.",
            "evidence_ids": ["11111111-1111-1111-1111-111111111111"],
        }
    ],
    "confidence": "medium",
    "sources": [
        {
            "evidence_id": "11111111-1111-1111-1111-111111111111",
            "tool_name": "posthog",
            "capability": "event_trend",
        }
    ],
}


async def _tenant(session: AsyncSession, slug: str) -> TenantContext:
    tenant_id = uuid.uuid4()
    graph_name = graph_name_for_new_tenant(slug, tenant_id)
    session.add(Tenant(id=tenant_id, slug=slug, name=slug, graph_name=graph_name))
    await session.flush()
    return TenantContext(tenant_id=tenant_id, tenant_slug=slug, graph_name=graph_name)


async def _investigation(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    notify: dict | None,
    with_report: bool = True,
) -> uuid.UUID:
    created = datetime.now(UTC) - timedelta(seconds=87)
    row = Investigation(
        tenant_id=tenant.tenant_id,
        question="Why did signups fall?",
        status=InvestigationStatus.COMPLETED,
        notify=notify,
        completed_at=datetime.now(UTC),
        created_at=created,
    )
    session.add(row)
    await session.flush()
    if with_report:
        session.add(
            Report(
                tenant_id=tenant.tenant_id,
                investigation_id=row.id,
                body=BODY,
                confidence=0.6,
            )
        )
        await session.flush()
    return row.id


async def _bot_token(
    session: AsyncSession, tenant: TenantContext, token: str = "xoxb-test"
) -> None:
    # Sealed with `label=BOT_LABEL`, exactly as `cortex.connect` does. The first version of this
    # helper omitted the label -- and so did the code under test, so the fixture reproduced the bug
    # and the test passed while every real decryption failed with "credential failed
    # authentication". The label is part of the AAD; a fixture that seals differently from
    # production is testing a path that does not exist.
    wrapped, ciphertext = encrypt_credential(
        tenant.tenant_id, CredentialProvider.SLACK.value, token, label=BOT_LABEL
    )
    session.add(
        Credential(
            tenant_id=tenant.tenant_id,
            provider=CredentialProvider.SLACK,
            label=BOT_LABEL,
            wrapped_data_key=wrapped,
            ciphertext=ciphertext,
        )
    )
    await session.flush()


class TestWhenThereIsNothingToDo:
    async def test_an_investigation_with_no_notify_target_is_skipped(
        self, session: AsyncSession
    ) -> None:
        """The common case. A question asked through the API or the CLI has nowhere to be
        delivered, and that is not a defect."""
        tenant = await _tenant(session, "deliver-none")
        investigation_id = await _investigation(session, tenant, notify=None)

        reason = await deliver_report(session, tenant, investigation_id=investigation_id)
        assert reason == "nothing to deliver to"

    async def test_an_unknown_notify_kind_is_skipped(self, session: AsyncSession) -> None:
        """Forward compatibility: an email target written by a newer deploy must not make an
        older worker try to post it to Slack."""
        tenant = await _tenant(session, "deliver-email")
        investigation_id = await _investigation(
            session, tenant, notify={"kind": "email", "to": "a@example.com"}
        )
        assert (
            await deliver_report(session, tenant, investigation_id=investigation_id)
            == "nothing to deliver to"
        )

    async def test_a_missing_report_is_reported_not_raised(self, session: AsyncSession) -> None:
        tenant = await _tenant(session, "deliver-noreport")
        investigation_id = await _investigation(
            session, tenant, notify={"kind": "slack", "channel": "C1"}, with_report=False
        )
        assert (
            await deliver_report(session, tenant, investigation_id=investigation_id)
            == "no report to deliver"
        )

    async def test_a_missing_bot_token_names_the_fix(self, session: AsyncSession) -> None:
        """The one failure an operator can fix, and the most likely on a first deployment. The
        reason names the tenant and the label rather than saying "unauthorized"."""
        tenant = await _tenant(session, "deliver-notoken")
        investigation_id = await _investigation(
            session, tenant, notify={"kind": "slack", "channel": "C1"}
        )

        reason = await deliver_report(session, tenant, investigation_id=investigation_id)
        assert "no Slack bot token" in reason
        assert BOT_LABEL in reason

    async def test_another_tenants_investigation_is_not_delivered(
        self, session: AsyncSession
    ) -> None:
        """Tenant-scoped even though the worker resolved the tenant already. One predicate is not
        a burden; a missing one is F-01."""
        mine = await _tenant(session, "deliver-mine")
        theirs = await _tenant(session, "deliver-theirs")
        foreign = await _investigation(session, theirs, notify={"kind": "slack", "channel": "C1"})

        assert (
            await deliver_report(session, mine, investigation_id=foreign) == "no such investigation"
        )


class TestTheCredentialIsTenantScoped:
    async def test_another_tenants_bot_token_is_not_used(self, session: AsyncSession) -> None:
        """A second decryption path is a second chance to resolve a credential across tenants,
        which is exactly the shape F-01 warned about."""
        mine = await _tenant(session, "deliver-cred-mine")
        theirs = await _tenant(session, "deliver-cred-theirs")
        await _bot_token(session, theirs, "xoxb-theirs")
        investigation_id = await _investigation(
            session, mine, notify={"kind": "slack", "channel": "C1"}
        )

        reason = await deliver_report(session, mine, investigation_id=investigation_id)
        assert "no Slack bot token" in reason


class TestPosting:
    async def test_a_successful_post_reports_posted(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer xoxb-test"
            return httpx.Response(200, json={"ok": True, "ts": "1.2"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            delivered, detail = await post_answer(
                token="xoxb-test",
                channel="C1",
                thread_ts="1.0",
                text="hello",
                client=client,
            )
        assert delivered is True
        assert detail == "posted"

    async def test_a_slack_error_inside_a_200_is_a_failure(self) -> None:
        """Slack reports application errors inside an HTTP 200 with `ok: false`, so the status
        code alone is not the outcome — the same trap as the provider's SSE errors."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            delivered, detail = await post_answer(
                token="xoxb-test", channel="C1", thread_ts=None, text="hello", client=client
            )
        assert delivered is False
        assert "channel_not_found" in detail

    async def test_the_token_never_appears_in_a_failure_reason(self) -> None:
        """The reason is logged. A secret in a log is a secret in a log aggregator."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            _, detail = await post_answer(
                token="xoxb-super-secret",
                channel="C1",
                thread_ts=None,
                text="hello",
                client=client,
            )
        assert "xoxb-super-secret" not in detail

    async def test_a_transport_failure_is_a_reason_not_an_exception(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            delivered, detail = await post_answer(
                token="xoxb-test", channel="C1", thread_ts=None, text="hello", client=client
            )
        assert delivered is False
        assert "transport" in detail

    async def test_a_non_json_body_does_not_raise(self) -> None:
        """An HTML error page from a proxy must not raise inside a path that is not allowed to
        fail."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="<html>bad gateway</html>")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            delivered, detail = await post_answer(
                token="xoxb-test", channel="C1", thread_ts=None, text="hello", client=client
            )
        assert delivered is False
        assert detail

    async def test_a_failed_chart_upload_still_counts_as_delivered(self) -> None:
        """The answer is already posted. A missing picture is a partial success, not a failure —
        reporting it as a failure would invite a retry that posts the answer twice."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("chat.postMessage"):
                return httpx.Response(200, json={"ok": True, "ts": "1.2"})
            return httpx.Response(200, json={"ok": False, "error": "upload_disabled"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            delivered, detail = await post_answer(
                token="xoxb-test",
                channel="C1",
                thread_ts=None,
                text="hello",
                chart_png=b"\x89PNG fake",
                client=client,
            )
        assert delivered is True
        assert "without chart" in detail


class TestTheMessage:
    def test_it_leads_with_the_question_and_the_answer(self) -> None:
        text = render_for_slack(question="Why did signups fall?", report=BODY)
        assert text.startswith("*Why did signups fall?*")
        assert "Signups fell 12%" in text

    def test_citations_survive_as_capability_names(self) -> None:
        """A grounded answer whose grounding is stripped for the channel is an assertion."""
        assert "posthog.event_trend" in render_for_slack(question="q?", report=BODY)

    def test_slacks_three_escapes_are_applied(self) -> None:
        text = render_for_slack(question="signups < 100 & falling > fast?", report=BODY)
        assert "&lt;" in text and "&amp;" in text and "&gt;" in text

    def test_a_long_report_is_truncated(self) -> None:
        from cortex.notify.slack import MAX_TEXT_CHARS

        body = dict(BODY)
        body["executive_summary"] = [{"text": "x" * 500, "evidence_ids": []} for _ in range(20)]
        text = render_for_slack(question="q?", report=body)
        assert len(text) <= MAX_TEXT_CHARS

    def test_no_base_url_means_no_link(self) -> None:
        """A message telling a colleague to open localhost reads as a broken feature rather than
        an unconfigured one."""
        assert "Full report" not in render_for_slack(question="q?", report=BODY)

    def test_a_base_url_produces_a_link(self) -> None:
        text = render_for_slack(
            question="q?", report=BODY, report_url="https://cortex.example.com/ui/x"
        )
        assert "https://cortex.example.com/ui/x" in text


class TestTheCredentialRoundTrips:
    """The bug the fixtures hid.

    `label` is part of the vault's authenticated data. The delivery path sealed under `bot` and
    opened under `default`, so every real decryption failed with "credential failed authentication
    — wrong master key, wrong tenant, or tampering" — the vault telling the exact truth. It was
    invisible in tests because the fixture omitted the label too, so both sides agreed on the wrong
    value and the test exercised a path that did not exist.

    These seal the way `cortex.connect` does and read the way the worker does, with nothing in
    between, so the two cannot drift apart again.
    """

    async def test_a_token_stored_by_connect_is_readable_by_the_worker(
        self, session: AsyncSession
    ) -> None:
        from cortex.notify.deliver import _bot_token as read_bot_token

        tenant = await _tenant(session, "deliver-roundtrip")
        await _bot_token(session, tenant, "xoxb-round-trip")

        assert await read_bot_token(session, tenant) == "xoxb-round-trip"

    async def test_a_token_sealed_under_the_wrong_label_is_refused(
        self, session: AsyncSession
    ) -> None:
        """Not silently mis-read. The AAD mismatch is what makes a credential stored for one
        purpose unusable for another — the property that stops a search token being used to post."""
        from cortex.notify.deliver import _bot_token as read_bot_token
        from cortex.security.vault import VaultError

        tenant = await _tenant(session, "deliver-wronglabel")
        # Sealed as the *default* (search) credential, then filed under the bot label.
        wrapped, ciphertext = encrypt_credential(
            tenant.tenant_id, CredentialProvider.SLACK.value, "xoxp-search-token"
        )
        session.add(
            Credential(
                tenant_id=tenant.tenant_id,
                provider=CredentialProvider.SLACK,
                label=BOT_LABEL,
                wrapped_data_key=wrapped,
                ciphertext=ciphertext,
            )
        )
        await session.flush()

        with pytest.raises(VaultError):
            await read_bot_token(session, tenant)

    async def test_a_vault_failure_is_reported_rather_than_raised(
        self, session: AsyncSession
    ) -> None:
        """The behaviour that saved the first live Slack question: the report was stored, gated,
        verified and readable on its page even though the answer never left the building."""
        tenant = await _tenant(session, "deliver-vaulterr")
        wrapped, ciphertext = encrypt_credential(
            tenant.tenant_id, CredentialProvider.SLACK.value, "xoxp-wrong"
        )
        session.add(
            Credential(
                tenant_id=tenant.tenant_id,
                provider=CredentialProvider.SLACK,
                label=BOT_LABEL,
                wrapped_data_key=wrapped,
                ciphertext=ciphertext,
            )
        )
        await session.flush()
        investigation_id = await _investigation(
            session, tenant, notify={"kind": "slack", "channel": "C1"}
        )

        reason = await deliver_report(session, tenant, investigation_id=investigation_id)
        assert "delivery failed" in reason
        assert "VaultError" in reason
