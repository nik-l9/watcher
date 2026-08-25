"""Schema behaviour: constraints, cascades, and the append-only evidence chain.

These assert the guarantees the rest of the system leans on — that a tenant's
data really disappears on offboarding, that duplicate credentials cannot exist,
and that evidence is always attached to an investigation.
"""

from __future__ import annotations

import hashlib
import json
import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import (
    AuditLog,
    Credential,
    CredentialProvider,
    Evidence,
    Investigation,
    InvestigationStatus,
    Report,
    Tenant,
    ToolCall,
    User,
)
from cortex.memory.naming import graph_name_for_new_tenant


async def _tenant(session: AsyncSession, slug: str = "acme-corp") -> Tenant:
    tid = uuid.uuid4()
    t = Tenant(id=tid, slug=slug, name="Acme", graph_name=graph_name_for_new_tenant(slug, tid))
    session.add(t)
    await session.flush()
    return t


async def _investigation(session: AsyncSession, tenant: Tenant) -> Investigation:
    inv = Investigation(tenant_id=tenant.id, question="Why did signups fall?")
    session.add(inv)
    await session.flush()
    return inv


class TestTenant:
    async def test_defaults(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        assert t.is_active is True
        assert t.created_at is not None

    async def test_slug_is_unique(self, session: AsyncSession) -> None:
        await _tenant(session, "acme-corp")
        session.add(
            Tenant(slug="acme-corp", name="Impostor", graph_name="cortex_g_acme-corp_other")
        )
        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_graph_name_is_unique(self, session: AsyncSession) -> None:
        """Two tenants sharing a graph name would silently share data."""
        t = await _tenant(session, "acme-corp")
        session.add(Tenant(slug="other-corp", name="Other", graph_name=t.graph_name))
        with pytest.raises(IntegrityError):
            await session.flush()


class TestUser:
    async def test_email_unique_within_tenant_only(self, session: AsyncSession) -> None:
        a = await _tenant(session, "tenant-one")
        b = await _tenant(session, "tenant-two")

        session.add(User(tenant_id=a.id, email="nik@example.com", external_id="clerk_1"))
        await session.flush()

        # Same email in a different tenant is legitimate.
        session.add(User(tenant_id=b.id, email="nik@example.com", external_id="clerk_2"))
        await session.flush()

        # Same email twice in one tenant is not.
        session.add(User(tenant_id=a.id, email="nik@example.com", external_id="clerk_3"))
        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_default_role_is_least_privilege(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        u = User(tenant_id=t.id, email="x@example.com")
        session.add(u)
        await session.flush()
        assert u.role == "member"


class TestCredential:
    async def test_stores_only_ciphertext(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        c = Credential(
            tenant_id=t.id,
            provider=CredentialProvider.GA4,
            wrapped_data_key=b"\x01wrapped",
            ciphertext=b"\x02cipher",
            metadata_={"property_id": "properties/12345"},
        )
        session.add(c)
        await session.flush()
        assert c.label == "default"
        assert c.last_sync_at is None
        # Non-secret connector metadata is the only readable field.
        assert c.metadata_["property_id"] == "properties/12345"

    async def test_one_credential_per_tenant_provider_label(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        for _ in range(2):
            session.add(
                Credential(
                    tenant_id=t.id,
                    provider=CredentialProvider.HUBSPOT,
                    wrapped_data_key=b"w",
                    ciphertext=b"c",
                )
            )
        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_distinct_labels_allow_multiple_accounts(self, session: AsyncSession) -> None:
        """A tenant may connect two GA4 properties."""
        t = await _tenant(session)
        for label in ("marketing-site", "app"):
            session.add(
                Credential(
                    tenant_id=t.id,
                    provider=CredentialProvider.GA4,
                    label=label,
                    wrapped_data_key=b"w",
                    ciphertext=b"c",
                )
            )
        await session.flush()
        count = await session.scalar(
            select(func.count()).select_from(Credential).where(Credential.tenant_id == t.id)
        )
        assert count == 2


class TestInvestigation:
    async def test_defaults(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        inv = await _investigation(session, t)
        assert inv.status is InvestigationStatus.QUEUED
        assert inv.employee == "gtm_data_analyst"
        assert inv.hypotheses == []
        assert inv.steps_used == 0

    async def test_hypotheses_roundtrip_as_jsonb(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        inv = await _investigation(session, t)
        inv.hypotheses = [
            {
                "claim": "mobile conversion dropped after deploy 91c3e",
                "supporting_evidence_ids": [str(uuid.uuid4())],
                "contradicting_evidence_ids": [],
                "verdict": "supported",
            }
        ]
        await session.flush()
        await session.refresh(inv)
        assert inv.hypotheses[0]["verdict"] == "supported"
        assert inv.hypotheses[0]["supporting_evidence_ids"]


class TestEvidence:
    async def test_requires_an_investigation(self, session: AsyncSession) -> None:
        """Orphan evidence would be uncitable, so the FK is NOT NULL."""
        t = await _tenant(session)
        session.add(
            Evidence(
                tenant_id=t.id,
                investigation_id=None,  # type: ignore[arg-type]
                tool_name="ga4",
                capability="get_sessions",
                payload={},
                payload_hash="0" * 64,
            )
        )
        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_payload_hash_pins_the_observation(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        inv = await _investigation(session, t)
        payload = {"sessions": 1200, "period": "2026-07-01/2026-07-07"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        ev = Evidence(
            tenant_id=t.id,
            investigation_id=inv.id,
            tool_name="ga4",
            capability="get_sessions",
            params={"days": 7},
            payload=payload,
            payload_hash=digest,
            source_ref="ga4://properties/12345/report/abc",
        )
        session.add(ev)
        await session.flush()
        assert ev.from_cache is False
        assert ev.observed_at is not None
        assert len(ev.payload_hash) == 64

    async def test_from_cache_marks_synced_data(self, session: AsyncSession) -> None:
        """Reports must be able to disclose staleness, so the flag has to persist."""
        t = await _tenant(session)
        inv = await _investigation(session, t)
        ev = Evidence(
            tenant_id=t.id,
            investigation_id=inv.id,
            tool_name="hubspot",
            capability="pipeline",
            payload={"deals": []},
            payload_hash="a" * 64,
            from_cache=True,
        )
        session.add(ev)
        await session.flush()
        assert ev.from_cache is True


class TestToolCall:
    async def test_failed_call_is_audited_without_evidence(self, session: AsyncSession) -> None:
        """A failed call produces no evidence but must still be auditable."""
        t = await _tenant(session)
        inv = await _investigation(session, t)
        tc = ToolCall(
            tenant_id=t.id,
            investigation_id=inv.id,
            evidence_id=None,
            tool_name="slack",
            capability="search_messages",
            params={"query": "onboarding"},
            succeeded=False,
            error="429 rate limited",
            duration_ms=812,
        )
        session.add(tc)
        await session.flush()
        assert tc.evidence_id is None
        assert tc.read_only is True

    async def test_defaults_to_read_only(self, session: AsyncSession) -> None:
        t = await _tenant(session)
        tc = ToolCall(tenant_id=t.id, tool_name="ga4", capability="get_funnel", succeeded=True)
        session.add(tc)
        await session.flush()
        assert tc.read_only is True


class TestReport:
    async def test_records_both_rejection_channels(self, session: AsyncSession) -> None:
        """Gate and verifier rejections are stored separately — the eval suite
        needs to distinguish a structural drop from a judged-unsupported claim."""
        t = await _tenant(session)
        inv = await _investigation(session, t)
        r = Report(
            tenant_id=t.id,
            investigation_id=inv.id,
            body={"executive_summary": "Signups fell 18%.", "findings": []},
            confidence=0.91,
            gate_rejections=[{"claim": "traffic doubled", "reason": "unknown evidence id"}],
            verifier_rejections=[{"claim": "caused by pricing", "verdict": "unsupported"}],
        )
        session.add(r)
        await session.flush()
        assert r.confidence == pytest.approx(0.91)
        assert len(r.gate_rejections) == 1
        assert len(r.verifier_rejections) == 1


class TestCascades:
    async def test_offboarding_a_tenant_removes_all_of_its_rows(
        self, session: AsyncSession
    ) -> None:
        """Deleting a tenant must leave nothing behind — the other half of
        offboarding, alongside dropping its graph."""
        t = await _tenant(session)
        inv = await _investigation(session, t)
        session.add_all(
            [
                User(tenant_id=t.id, email="a@example.com"),
                Credential(
                    tenant_id=t.id,
                    provider=CredentialProvider.GITHUB,
                    wrapped_data_key=b"w",
                    ciphertext=b"c",
                ),
                Evidence(
                    tenant_id=t.id,
                    investigation_id=inv.id,
                    tool_name="github",
                    capability="recent_prs",
                    payload={},
                    payload_hash="b" * 64,
                ),
                ToolCall(
                    tenant_id=t.id,
                    investigation_id=inv.id,
                    tool_name="github",
                    capability="recent_prs",
                    succeeded=True,
                ),
                Report(tenant_id=t.id, investigation_id=inv.id, body={}),
                AuditLog(tenant_id=t.id, action="credential.created"),
            ]
        )
        await session.flush()

        await session.delete(t)
        await session.flush()

        for model in (User, Credential, Evidence, ToolCall, Report, AuditLog, Investigation):
            remaining = await session.scalar(
                select(func.count()).select_from(model).where(model.tenant_id == t.id)
            )
            assert remaining == 0, f"{model.__name__} rows survived tenant deletion"

    async def test_deleting_an_investigation_keeps_the_tool_call_audit(
        self, session: AsyncSession
    ) -> None:
        """Evidence is scoped to its investigation, but the audit trail outlives it."""
        t = await _tenant(session)
        inv = await _investigation(session, t)
        session.add_all(
            [
                Evidence(
                    tenant_id=t.id,
                    investigation_id=inv.id,
                    tool_name="ga4",
                    capability="top_pages",
                    payload={},
                    payload_hash="c" * 64,
                ),
                ToolCall(
                    tenant_id=t.id,
                    investigation_id=inv.id,
                    tool_name="ga4",
                    capability="top_pages",
                    succeeded=True,
                ),
            ]
        )
        await session.flush()

        await session.delete(inv)
        await session.flush()

        assert (
            await session.scalar(
                select(func.count()).select_from(Evidence).where(Evidence.tenant_id == t.id)
            )
            == 0
        )
        surviving = await session.scalar(
            select(func.count()).select_from(ToolCall).where(ToolCall.tenant_id == t.id)
        )
        assert surviving == 1, "audit rows must survive investigation deletion"


class TestTenantScopedIndexes:
    def test_every_tenant_scoped_index_leads_with_tenant_id(self) -> None:
        """Scoping should be cheap and visible in query plans."""
        for model in (Evidence, ToolCall, Report, Investigation, AuditLog, Credential):
            for index in model.__table__.indexes:
                columns = [c.name for c in index.columns]
                assert columns[0] == "tenant_id", f"{model.__name__}.{index.name} -> {columns}"


class TestTheProviderEnumMatchesTheDatabase:
    """A connector needs a Python enum member *and* a Postgres enum label.

    Adding the member is free and adding the label needs a migration, so the two drift silently
    in the direction that matters: everything imports, the registry lists the new tool, the tests
    pass, and the first tenant to connect it fails at the vault write with an error about an enum
    type rather than about a connector.

    Found exactly that way when Mixpanel and Amplitude were added. The migration also has to name
    the type `credential_provider` -- SQLAlchemy takes the name from `sa.Enum(name=...)` and this
    schema is snake_case, so the plausible guess `credentialprovider` fails with "type does not
    exist" on an otherwise correct migration.
    """

    async def test_every_provider_exists_as_a_database_label(self, session: AsyncSession) -> None:
        # Compared by `.name`, which is what SQLAlchemy persists. Comparing values is the
        # mistake this class exists to catch, and it fails in the safe direction only by luck.
        labels = set(
            (
                await session.execute(
                    text("SELECT unnest(enum_range(NULL::credential_provider))::text")
                )
            )
            .scalars()
            .all()
        )
        missing = sorted(p.name for p in CredentialProvider if p.name not in labels)
        assert not missing, (
            f"{missing} exist in CredentialProvider but not in the credential_provider type; "
            "a tenant connecting one would fail at the vault write. Add a migration."
        )

    async def test_a_new_provider_can_actually_be_stored(self, session: AsyncSession) -> None:
        """The assertion above reads the type; this one exercises the write path that failed.

        A label present in `enum_range` but rejected on insert would be a stranger bug, and the
        whole point of the enum is what happens when a row uses it.
        """
        tenant = await _tenant(session)
        for provider in (CredentialProvider.MIXPANEL, CredentialProvider.AMPLITUDE):
            session.add(
                Credential(
                    tenant_id=tenant.id,
                    provider=provider,
                    label="default",
                    wrapped_data_key=b"wrapped",
                    ciphertext=b"sealed",
                )
            )
        await session.flush()
        stored = (
            (
                await session.execute(
                    select(Credential.provider).where(Credential.tenant_id == tenant.id)
                )
            )
            .scalars()
            .all()
        )
        assert CredentialProvider.MIXPANEL in stored
        assert CredentialProvider.AMPLITUDE in stored
