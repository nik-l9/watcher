"""Tool executor.

The executor is where "never hallucinate" stops being a prompt and becomes a
database constraint. Every guarantee it provides is asserted here against real
Postgres:

  - a successful call writes immutable Evidence and returns its id
  - a failed call writes an audit row and no evidence, so it is accountable but not
    citable
  - credentials are resolved per tenant and never leak into evidence or audit rows
  - parameters are validated before dispatch
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import (
    Credential,
    CredentialProvider,
    Evidence,
    Investigation,
    Tenant,
    ToolCall,
)
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.security.vault import encrypt_credential
from cortex.tenancy.context import TenantContext
from cortex.tools.base import (
    Capability,
    CredentialMissing,
    Freshness,
    InvalidParams,
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    UpstreamError,
)
from cortex.tools.executor import (
    InvestigationNotFound,
    ToolExecutor,
    canonical_hash,
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["days"],
    "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 90}},
}

SECRET = "pat-super-secret-token"


class _ProbeTool(Tool):
    """A tool with no credential requirement, for the general executor paths."""

    name = "probe"
    provider = None

    def __init__(self, behaviour: str = "ok") -> None:
        self.behaviour = behaviour
        self.seen: list[ToolContext] = []
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="observe",
                description="Return a structured observation for testing the executor.",
                params_schema=SCHEMA,
                handler=self.observe,
                result_key="rows",
            )
        ]

    async def observe(self, ctx: ToolContext, *, days: int) -> ToolResult:
        self.seen.append(ctx)
        if self.behaviour == "upstream":
            raise UpstreamError("probe: upstream exploded")
        if self.behaviour == "unexpected":
            raise ZeroDivisionError("an unexpected internal failure")
        if self.behaviour == "wrong_type":
            return {"not": "a ToolResult"}  # type: ignore[return-value]
        if self.behaviour == "non_finite":
            return ToolResult(
                payload={
                    "rate": float("nan"),
                    "ratio": float("inf"),
                    "negative": float("-inf"),
                    "finite": 0.5,
                }
            )
        if self.behaviour == "nested_non_finite":
            return ToolResult(payload={"rows": [{"rate": float("nan")}, {"rate": 0.1}]})
        if self.behaviour == "synced":
            return ToolResult(
                payload={"days": days}, freshness=Freshness.SYNCED, source_ref="cache://x"
            )
        return ToolResult(
            payload={"days": days, "sessions": 1200},
            source_ref="probe://observation",
            meta={"note": "test"},
        )


class _CredentialTool(Tool):
    """A tool that requires a credential, for the resolution paths."""

    name = "needs_credential"
    provider = CredentialProvider.HUBSPOT

    def __init__(self) -> None:
        self.observed_credential: str | None = None
        self.observed_metadata: dict | None = None
        super().__init__()

    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="fetch",
                description="Fetch something that requires a decrypted credential.",
                params_schema={"type": "object", "additionalProperties": False, "properties": {}},
                handler=self.fetch,
                result_key="rows",
            )
        ]

    async def fetch(self, ctx: ToolContext) -> ToolResult:
        self.observed_credential = ctx.credential
        self.observed_metadata = dict(ctx.credential_metadata)
        # Deliberately does NOT echo the credential into the payload.
        return ToolResult(payload={"ok": True})


async def _tenant(session: AsyncSession, slug: str = "exec-test") -> TenantContext:
    tenant_id = uuid.uuid4()
    session.add(
        Tenant(
            id=tenant_id,
            slug=slug,
            name=slug,
            graph_name=graph_name_for_new_tenant(slug, tenant_id),
        )
    )
    await session.flush()
    return TenantContext(
        tenant_id=tenant_id,
        tenant_slug=slug,
        graph_name=graph_name_for_new_tenant(slug, tenant_id),
    )


async def _investigation(session: AsyncSession, ctx: TenantContext) -> uuid.UUID:
    inv = Investigation(tenant_id=ctx.tenant_id, question="Why did signups fall?")
    session.add(inv)
    await session.flush()
    return inv.id


def _executor(tool: Tool) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry)


class TestSuccessfulCall:
    async def test_writes_evidence_and_returns_its_id(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        executor = _executor(_ProbeTool())

        executed = await executor.execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )

        evidence = await session.get(Evidence, executed.evidence_id)
        assert evidence is not None
        assert evidence.tenant_id == ctx.tenant_id
        assert evidence.investigation_id == investigation_id
        assert evidence.tool_name == "probe"
        assert evidence.capability == "observe"
        assert evidence.payload["sessions"] == 1200
        assert evidence.params == {"days": 7}
        assert evidence.source_ref == "probe://observation"

    async def test_hash_pins_the_observation(self, session: AsyncSession) -> None:
        """The verifier uses this to prove a citation refers to data actually
        observed and not since edited."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        executed = await _executor(_ProbeTool()).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )
        evidence = await session.get(Evidence, executed.evidence_id)
        assert evidence is not None
        assert evidence.payload_hash == canonical_hash(evidence.payload)
        assert executed.payload_hash == evidence.payload_hash

    async def test_writes_a_linked_audit_row(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        executed = await _executor(_ProbeTool()).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )

        call = (
            await session.execute(select(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id))
        ).scalar_one()
        assert call.succeeded is True
        assert call.evidence_id == executed.evidence_id
        assert call.read_only is True
        assert call.error is None
        assert call.duration_ms is not None

    async def test_records_synced_freshness(self, session: AsyncSession) -> None:
        """Reports must disclose staleness, so the flag has to reach the row."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        executed = await _executor(_ProbeTool("synced")).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )
        evidence = await session.get(Evidence, executed.evidence_id)
        assert evidence is not None
        assert evidence.from_cache is True
        assert executed.freshness is Freshness.SYNCED


class TestFailedCall:
    @pytest.mark.parametrize(
        ("behaviour", "expected"), [("upstream", UpstreamError), ("unexpected", ToolError)]
    )
    async def test_writes_audit_but_no_evidence(
        self, session: AsyncSession, behaviour: str, expected: type[Exception]
    ) -> None:
        """A failed call must be accountable but not citable."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        executor = _executor(_ProbeTool(behaviour))

        with pytest.raises(expected):
            await executor.execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params={"days": 7},
            )

        evidence_count = await session.scalar(
            select(func.count()).select_from(Evidence).where(Evidence.tenant_id == ctx.tenant_id)
        )
        assert evidence_count == 0

        call = (
            await session.execute(select(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id))
        ).scalar_one()
        assert call.succeeded is False
        assert call.evidence_id is None
        assert call.error

    async def test_unexpected_error_is_wrapped_without_leaking_internals(
        self, session: AsyncSession
    ) -> None:
        """The audit records the detail; the raised message stays generic so an
        upstream stack trace cannot masquerade as a Cortex bug."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        with pytest.raises(ToolError) as exc:
            await _executor(_ProbeTool("unexpected")).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params={"days": 7},
            )
        assert "ZeroDivisionError" in str(exc.value)

        call = (
            await session.execute(select(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id))
        ).scalar_one()
        assert "ZeroDivisionError" in (call.error or "")

    async def test_non_toolresult_return_is_rejected(self, session: AsyncSession) -> None:
        """A connector returning a bare dict would bypass the payload-shape check."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        with pytest.raises(ToolError, match="expected ToolResult"):
            await _executor(_ProbeTool("wrong_type")).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params={"days": 7},
            )


class TestParameterValidation:
    @pytest.mark.parametrize(
        "params",
        [
            {},  # missing required
            {"days": 0},  # below minimum
            {"days": 91},  # above maximum
            {"days": "seven"},  # wrong type
            {"days": 7, "hallucinated": True},  # unknown property
        ],
    )
    async def test_rejects_bad_params_before_dispatch(
        self, session: AsyncSession, params: dict
    ) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()

        with pytest.raises(InvalidParams):
            await _executor(tool).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params=params,
            )
        assert tool.seen == [], "the handler must not run on invalid params"

    async def test_validation_failure_is_still_audited(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        with pytest.raises(InvalidParams):
            await _executor(_ProbeTool()).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params={"days": 999},
            )

        call = (
            await session.execute(select(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id))
        ).scalar_one()
        assert call.succeeded is False
        assert "InvalidParams" in (call.error or "")

    async def test_error_names_the_offending_field(self, session: AsyncSession) -> None:
        """ "Invalid arguments" alone is not actionable for a model trying to correct
        its call."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        with pytest.raises(InvalidParams, match="days"):
            await _executor(_ProbeTool()).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params={"days": 999},
            )


class TestCredentialResolution:
    async def _seed_credential(self, session: AsyncSession, ctx: TenantContext) -> None:
        wrapped, ciphertext = encrypt_credential(
            ctx.tenant_id, CredentialProvider.HUBSPOT.value, SECRET
        )
        session.add(
            Credential(
                tenant_id=ctx.tenant_id,
                provider=CredentialProvider.HUBSPOT,
                wrapped_data_key=wrapped,
                ciphertext=ciphertext,
                metadata_={"portal_id": "12345"},
            )
        )
        await session.flush()

    async def test_decrypts_and_passes_the_credential(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        await self._seed_credential(session, ctx)

        tool = _CredentialTool()
        await _executor(tool).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="needs_credential__fetch",
        )
        assert tool.observed_credential == SECRET
        assert tool.observed_metadata == {"portal_id": "12345"}

    async def test_missing_credential_is_a_clear_error(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        with pytest.raises(CredentialMissing, match="hubspot"):
            await _executor(_CredentialTool()).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="needs_credential__fetch",
            )

    async def test_another_tenants_credential_is_not_used(self, session: AsyncSession) -> None:
        """The core cross-tenant case: tenant B has no credential and must not
        borrow tenant A's."""
        owner = await _tenant(session, "exec-owner")
        other = await _tenant(session, "exec-other")
        await self._seed_credential(session, owner)
        investigation_id = await _investigation(session, other)

        with pytest.raises(CredentialMissing):
            await _executor(_CredentialTool()).execute(
                session,
                other,
                investigation_id=investigation_id,
                qualified_name="needs_credential__fetch",
            )

    async def test_credential_never_reaches_evidence_or_audit(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        await self._seed_credential(session, ctx)

        await _executor(_CredentialTool()).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="needs_credential__fetch",
        )

        evidence = (
            await session.execute(select(Evidence).where(Evidence.tenant_id == ctx.tenant_id))
        ).scalar_one()
        call = (
            await session.execute(select(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id))
        ).scalar_one()
        assert SECRET not in str(evidence.payload)
        assert SECRET not in str(evidence.params)
        assert SECRET not in str(call.params)

    async def test_credential_free_tool_needs_no_credential(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()

        await _executor(tool).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )
        assert tool.seen[0].credential is None


class TestAuditDoesNotDuplicatePayloads:
    async def test_response_body_is_not_copied_into_the_audit_row(
        self, session: AsyncSession
    ) -> None:
        """Duplicating the payload would double the PII surface for no benefit."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        await _executor(_ProbeTool()).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )
        call = (
            await session.execute(select(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id))
        ).scalar_one()
        assert call.params == {"days": 7}
        assert "sessions" not in str(call.params)


class TestInvestigationScoping:
    """F-01, critical. The executor previously accepted any investigation_id, so a
    tenant could write evidence onto another tenant's investigation. Because the
    grounding gate resolves citations by investigation_id, that evidence would then
    be citable by the victim's report."""

    async def test_rejects_another_tenants_investigation(self, session: AsyncSession) -> None:
        victim = await _tenant(session, "scope-victim")
        attacker = await _tenant(session, "scope-attacker")
        victim_investigation = await _investigation(session, victim)

        with pytest.raises(InvestigationNotFound):
            await _executor(_ProbeTool()).execute(
                session,
                attacker,
                investigation_id=victim_investigation,
                qualified_name="probe__observe",
                params={"days": 7},
            )

    async def test_writes_no_evidence_on_rejection(self, session: AsyncSession) -> None:
        victim = await _tenant(session, "scope-victim")
        attacker = await _tenant(session, "scope-attacker")
        victim_investigation = await _investigation(session, victim)

        with pytest.raises(InvestigationNotFound):
            await _executor(_ProbeTool()).execute(
                session,
                attacker,
                investigation_id=victim_investigation,
                qualified_name="probe__observe",
                params={"days": 7},
            )

        planted = await session.scalar(
            select(func.count())
            .select_from(Evidence)
            .where(Evidence.investigation_id == victim_investigation)
        )
        assert planted == 0, "the victim's investigation must be untouched"

    async def test_the_attempt_is_audited(self, session: AsyncSession) -> None:
        """Probing for other tenants' investigation ids must leave a trail, recorded
        under the caller's own tenant."""
        victim = await _tenant(session, "scope-victim")
        attacker = await _tenant(session, "scope-attacker")
        victim_investigation = await _investigation(session, victim)

        with pytest.raises(InvestigationNotFound):
            await _executor(_ProbeTool()).execute(
                session,
                attacker,
                investigation_id=victim_investigation,
                qualified_name="probe__observe",
                params={"days": 7},
            )

        call = (
            await session.execute(select(ToolCall).where(ToolCall.tenant_id == attacker.tenant_id))
        ).scalar_one()
        assert call.succeeded is False
        assert "InvestigationNotFound" in (call.error or "")

    async def test_rejects_an_unknown_investigation(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        with pytest.raises(InvestigationNotFound):
            await _executor(_ProbeTool()).execute(
                session,
                ctx,
                investigation_id=uuid.uuid4(),
                qualified_name="probe__observe",
                params={"days": 7},
            )

    async def test_no_credential_is_decrypted_before_the_check(self, session: AsyncSession) -> None:
        """Scoping is verified before any credential is touched or upstream called,
        so a rejected caller cannot use the executor as a decryption oracle."""
        victim = await _tenant(session, "scope-victim")
        attacker = await _tenant(session, "scope-attacker")
        victim_investigation = await _investigation(session, victim)

        wrapped, ciphertext = encrypt_credential(
            attacker.tenant_id, CredentialProvider.HUBSPOT.value, SECRET
        )
        session.add(
            Credential(
                tenant_id=attacker.tenant_id,
                provider=CredentialProvider.HUBSPOT,
                wrapped_data_key=wrapped,
                ciphertext=ciphertext,
            )
        )
        await session.flush()

        tool = _CredentialTool()
        with pytest.raises(InvestigationNotFound):
            await _executor(tool).execute(
                session,
                attacker,
                investigation_id=victim_investigation,
                qualified_name="needs_credential__fetch",
            )
        assert tool.observed_credential is None


class TestPayloadSanitisation:
    """F-05. GA4 and BigQuery can return NaN or Infinity for a rate. Python's json
    accepts both; Postgres jsonb rejects them, and the write happened outside the
    guarded block so the investigation died with no audit row."""

    async def test_non_finite_floats_become_none(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        executed = await _executor(_ProbeTool("non_finite")).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )

        evidence = await session.get(Evidence, executed.evidence_id)
        assert evidence is not None
        # None, not 0: a fabricated zero would be a wrong number in a grounded report.
        assert evidence.payload["rate"] is None
        assert evidence.payload["ratio"] is None
        assert evidence.payload["negative"] is None
        assert evidence.payload["finite"] == 0.5

    async def test_returned_payload_matches_what_was_persisted(self, session: AsyncSession) -> None:
        """The caller must reason over exactly what was stored and hashed."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        executed = await _executor(_ProbeTool("non_finite")).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )
        evidence = await session.get(Evidence, executed.evidence_id)
        assert evidence is not None
        assert executed.payload == evidence.payload
        assert executed.payload_hash == canonical_hash(evidence.payload)

    async def test_nested_non_finite_values_are_handled(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)

        executed = await _executor(_ProbeTool("nested_non_finite")).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )
        evidence = await session.get(Evidence, executed.evidence_id)
        assert evidence is not None
        assert evidence.payload["rows"][0]["rate"] is None
        assert evidence.payload["rows"][1]["rate"] == 0.1


class TestCredentialLabels:
    """F-08. The lookup hardcoded label='default', so a tenant that connected a
    single non-default account got CredentialMissing even though the schema, the
    unique constraint and an existing test all advertise multiple accounts."""

    async def _seed(self, session: AsyncSession, ctx: TenantContext, label: str) -> None:
        wrapped, ciphertext = encrypt_credential(
            ctx.tenant_id, CredentialProvider.HUBSPOT.value, f"{SECRET}-{label}", label=label
        )
        session.add(
            Credential(
                tenant_id=ctx.tenant_id,
                provider=CredentialProvider.HUBSPOT,
                label=label,
                wrapped_data_key=wrapped,
                ciphertext=ciphertext,
            )
        )
        await session.flush()

    async def test_selects_the_requested_label(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        await self._seed(session, ctx, "marketing-site")
        await self._seed(session, ctx, "app")

        tool = _CredentialTool()
        await _executor(tool).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="needs_credential__fetch",
            credential_label="app",
        )
        assert tool.observed_credential == f"{SECRET}-app"

    async def test_labels_do_not_bleed_into_each_other(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        await self._seed(session, ctx, "marketing-site")
        await self._seed(session, ctx, "app")

        tool = _CredentialTool()
        await _executor(tool).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="needs_credential__fetch",
            credential_label="marketing-site",
        )
        assert tool.observed_credential == f"{SECRET}-marketing-site"

    async def test_unknown_label_names_the_available_ones(self, session: AsyncSession) -> None:
        """ "Credential missing" alone is not actionable when the tenant has one."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        await self._seed(session, ctx, "app")

        with pytest.raises(CredentialMissing, match="app"):
            await _executor(_CredentialTool()).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="needs_credential__fetch",
                credential_label="does-not-exist",
            )

    async def test_no_credential_at_all_says_so(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        with pytest.raises(CredentialMissing, match="has not connected"):
            await _executor(_CredentialTool()).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="needs_credential__fetch",
            )


class TestCanonicalHash:
    def test_is_order_independent(self) -> None:
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_distinguishes_different_payloads(self) -> None:
        assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})

    def test_object_does_not_collide_with_its_string_form(self) -> None:
        """F-07. default=str meant any unserialisable object hashed as its str(),
        so an object rendering as "same" collided with the literal string."""

        class Renders:
            def __str__(self) -> str:
                return "same"

        assert canonical_hash({"v": Renders()}) != canonical_hash({"v": "same"})

    def test_distinct_types_rendering_alike_stay_distinct(self) -> None:
        class A:
            def __str__(self) -> str:
                return "x"

        class B:
            def __str__(self) -> str:
                return "x"

        assert canonical_hash({"v": A()}) != canonical_hash({"v": B()})

    def test_non_finite_floats_hash_as_null(self) -> None:
        """Consistent with what gets persisted, so a citation's hash still matches
        the stored row."""
        assert canonical_hash({"v": float("nan")}) == canonical_hash({"v": None})

    def test_is_stable_across_calls(self) -> None:
        payload = {"rows": [{"x": 1}], "n": 2}
        assert canonical_hash(payload) == canonical_hash(payload)

    def test_handles_nested_structures(self) -> None:
        left = {"rows": [{"a": 1, "b": 2}]}
        right = {"rows": [{"b": 2, "a": 1}]}
        assert canonical_hash(left) == canonical_hash(right)

    def test_is_sha256_hex(self) -> None:
        digest = canonical_hash({"a": 1})
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestResolution:
    async def test_unknown_tool_is_refused(self, session: AsyncSession) -> None:
        from cortex.tools.base import ToolNotFound

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        with pytest.raises(ToolNotFound):
            await _executor(_ProbeTool()).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="ga4__get_sessions",
            )

    async def test_unknown_capability_is_refused(self, session: AsyncSession) -> None:
        from cortex.tools.base import CapabilityNotFound

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        with pytest.raises(CapabilityNotFound):
            await _executor(_ProbeTool()).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__does_not_exist",
            )


class TestPerTenantCeilings:
    """The per-investigation budget bounds one run. This bounds a tenant, which is what a
    runaway loop or a hammered button multiplies."""

    class _Limiter:
        def __init__(self, allowed: bool, *, limited: bool = True) -> None:
            from cortex.tenancy.limits import Decision

            self.calls: list[tuple[object, object]] = []
            self._decision = Decision(
                allowed=allowed,
                limited=limited,
                window_seconds=None if allowed else 60,
                limit=None if allowed else 120,
                retry_after_seconds=None if allowed else 60.0,
            )

        async def check(self, tenant_id: object, provider: object) -> object:
            self.calls.append((tenant_id, provider))
            return self._decision

    async def test_a_call_over_the_ceiling_is_refused_as_a_rate_limit(
        self, session: AsyncSession
    ) -> None:
        """Raised as a `RateLimited` subclass so the loop's existing handling applies — it
        already knows to route around a rate limit and report the gap."""
        from cortex.tools.base import RateLimited, TenantRateLimited

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        registry = ToolRegistry()
        registry.register(tool)
        limiter = self._Limiter(allowed=False)

        with pytest.raises(TenantRateLimited) as raised:
            await ToolExecutor(registry, limiter=limiter).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params={"days": 7},
            )

        # A subclass, so a caller catching the general case still catches this.
        assert isinstance(raised.value, RateLimited)
        assert raised.value.retry_after_seconds == 60.0
        # And it says whose limit it was, because "HubSpot throttled us" and "we throttled
        # ourselves" call for completely different responses.
        assert "Cortex's own per-tenant ceiling" in str(raised.value)
        assert tool.seen == [], "the upstream must not be called"

    async def test_the_refusal_is_audited(self, session: AsyncSession) -> None:
        """A rejected call is still an attempt, and an unaudited rejection is invisible when
        someone asks why an investigation came back thin."""
        from cortex.tools.base import TenantRateLimited

        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        registry = ToolRegistry()
        registry.register(_ProbeTool())

        with pytest.raises(TenantRateLimited):
            await ToolExecutor(registry, limiter=self._Limiter(allowed=False)).execute(
                session,
                ctx,
                investigation_id=investigation_id,
                qualified_name="probe__observe",
                params={"days": 7},
            )

        rows = (
            (await session.execute(select(ToolCall).where(ToolCall.tenant_id == ctx.tenant_id)))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].succeeded is False
        assert "rate limit" in (rows[0].error or "")

    async def test_the_limiter_is_consulted_after_scoping(self, session: AsyncSession) -> None:
        """A call this tenant may not make should not reach a rate limiter at all: charging a
        tenant's allowance for a request that was going to be refused anyway would let one
        tenant drain another's ceiling by naming their investigation."""
        from cortex.tools.executor import InvestigationNotFound

        ctx = await _tenant(session)
        foreign = await _tenant(session, slug="rl-other-tenant")
        foreign_investigation = await _investigation(session, foreign)
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        limiter = self._Limiter(allowed=True)

        with pytest.raises(InvestigationNotFound):
            await ToolExecutor(registry, limiter=limiter).execute(
                session,
                ctx,
                investigation_id=foreign_investigation,
                qualified_name="probe__observe",
                params={"days": 7},
            )

        assert limiter.calls == [], "scoping must be decided before the limiter is charged"

    async def test_a_permitted_call_runs_normally(self, session: AsyncSession) -> None:
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        tool = _ProbeTool()
        registry = ToolRegistry()
        registry.register(tool)
        limiter = self._Limiter(allowed=True)

        executed = await ToolExecutor(registry, limiter=limiter).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )

        assert executed.evidence_id is not None
        assert len(tool.seen) == 1
        # Charged against the tenant and the provider, not the investigation.
        assert limiter.calls == [(ctx.tenant_id, tool.provider)]

    async def test_no_limiter_permits_everything(self, session: AsyncSession) -> None:
        """The default, so a test or a script needs no Redis."""
        ctx = await _tenant(session)
        investigation_id = await _investigation(session, ctx)
        registry = ToolRegistry()
        registry.register(_ProbeTool())

        executed = await ToolExecutor(registry).execute(
            session,
            ctx,
            investigation_id=investigation_id,
            qualified_name="probe__observe",
            params={"days": 7},
        )
        assert executed.evidence_id is not None
