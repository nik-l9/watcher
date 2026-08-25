"""Tool executor — the single path through which every tool call runs.

Connectors implement only the call to their upstream API. Everything that must
happen on *every* invocation happens here, so no connector can omit it:

  - parameter validation against the declared schema
  - credential resolution and decryption, scoped to the tenant
  - timing
  - a ToolCall audit row, written whether the call succeeded or failed
  - an immutable Evidence row for successful calls, returning the evidence_id that
    a report must cite

A failed call produces an audit row but no evidence, which is why the two tables
are separate: an empty or errored call must still be accountable, but it must not
be citable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jsonschema
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Credential, Evidence, Investigation, ToolCall
from cortex.security.vault import decrypt_credential
from cortex.tenancy.context import TenantContext
from cortex.tenancy.limits import NullRateLimiter, RateLimiter
from cortex.tools.base import (
    Capability,
    CredentialMissing,
    Freshness,
    InvalidParams,
    TenantRateLimited,
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
)


class InvestigationNotFound(ToolError):
    """The investigation does not exist, or does not belong to this tenant.

    Deliberately one exception for both cases: distinguishing them would confirm
    the existence of another tenant's investigation to a caller probing ids.
    """


def sanitize_payload(value: Any) -> Any:
    """Make an observation safe to persist as JSONB and to hash.

    Two problems handled here, both of which previously produced a hard failure or
    a silent inaccuracy:

    - GA4 and BigQuery can return NaN or Infinity for a rate. Python's json accepts
      both; Postgres jsonb rejects them. They become None, which is what "no value"
      already means everywhere in Cortex — never 0, which would be a fabricated
      figure in a grounded report.
    - Unsupported types previously hashed as their str(), so an object rendering as
      "same" collided with the literal string "same". They are now tagged with their
      type, so distinct inputs stay distinct.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize_payload(v) for v in value]
    return f"<{type(value).__name__}:{value}>"


def canonical_hash(payload: dict[str, Any]) -> str:
    """Stable sha256 of an observation.

    Canonicalized — sorted keys, no insignificant whitespace, sanitised values — so
    the same observation hashes identically regardless of dict ordering. This is
    what lets the verifier prove a citation refers to data that was actually
    observed and has not since been edited, so a collision would weaken exactly the
    guarantee it exists to provide.
    """
    encoded = json.dumps(
        sanitize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutedTool:
    """The outcome of one tool invocation, as the investigation loop sees it."""

    evidence_id: uuid.UUID
    tool_name: str
    capability: str
    payload: dict[str, Any]
    source_ref: str | None
    freshness: Freshness
    payload_hash: str
    duration_ms: int

    #: True when the capability's declared result key holds nothing.
    #:
    #: Computed here rather than left for the loop to notice, because "the result was empty"
    #: is the fact four separate bugs have hidden. An empty observation reaches the analyst
    #: labelled as empty whether or not anyone remembered to check.
    is_empty: bool = False
    #: What the tool itself can say about the emptiness -- available environments, the
    #: project allowlist, an upstream total that disagrees with the rows returned.
    empty_hint: str = ""


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, limiter: RateLimiter | None = None) -> None:
        self._registry = registry
        # Defaults to permitting everything, so a test or a script needs no Redis. The
        # workers pass a real limiter.
        self._limiter: RateLimiter = limiter or NullRateLimiter()

    async def execute(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        qualified_name: str,
        params: dict[str, Any] | None = None,
        credential_label: str = "default",
    ) -> ExecutedTool:
        """Run one capability and record it.

        Raises ToolError subclasses on failure, after writing the audit row. The
        caller decides whether to retry, try a different tool, or record the
        failure as a gap in the investigation.
        """
        tool, capability = self._registry.resolve(qualified_name)
        params = params or {}
        started = time.perf_counter()

        # The audit row has a foreign key to investigations, so it can only carry an
        # id that exists. A rejected or unknown id is recorded in the error text
        # instead, which keeps the attempt auditable without a dangling reference.
        owned = await self._investigation_is_owned(session, tenant, investigation_id)
        audit_investigation_id = investigation_id if owned else None

        try:
            # Scoping is checked before any credential is decrypted or any upstream
            # call is made. Without it, evidence could be written onto another
            # tenant's investigation — and because the grounding gate resolves
            # citations by investigation_id, that evidence would become citable by
            # the victim's report. See docs/security-findings.md F-01.
            if not owned:
                raise InvestigationNotFound(
                    f"investigation {investigation_id} is not available to this tenant"
                )
            # Checked after scoping and before the credential is decrypted: a call this
            # tenant may not make should not reach a rate limiter, and a call the limiter
            # rejects should not decrypt a secret it will not use.
            decision = await self._limiter.check(tenant.tenant_id, tool.provider)
            if not decision.allowed:
                raise TenantRateLimited(
                    f"{tool.name}: {decision.reason}. This is Cortex's own per-tenant "
                    f"ceiling, not the upstream's — the same call will succeed later.",
                    retry_after_seconds=decision.retry_after_seconds,
                )
            self._validate(capability, params)
            context = await self._load_context(session, tenant, tool, credential_label)
            result = await capability.handler(context, **params)
            self._check_result(tool, capability, result)
            payload = sanitize_payload(result.payload)
            payload_hash = canonical_hash(result.payload)

            is_empty = capability.is_empty(payload)
            empty_hint = capability.empty_hint(payload) if is_empty else ""

            evidence = Evidence(
                tenant_id=tenant.tenant_id,
                investigation_id=investigation_id,
                tool_name=tool.name,
                capability=capability.name,
                params=params,
                payload=payload,
                payload_hash=payload_hash,
                source_ref=result.source_ref,
                from_cache=result.freshness is Freshness.SYNCED,
            )
            # Inside a savepoint, and inside the guarded block, so that a payload
            # Postgres rejects is audited rather than escaping as a raw database
            # error — and so the rollback discards only this write instead of the
            # caller's whole transaction (F-05).
            async with session.begin_nested():
                session.add(evidence)
                await session.flush()
        except ToolError as exc:
            await self._audit(
                session,
                tenant,
                investigation_id=audit_investigation_id,
                tool=tool,
                capability=capability,
                params=params,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(started),
                evidence_id=None,
            )
            raise
        except Exception as exc:
            # An unexpected failure is still auditable. The message is recorded but
            # the exception type leads, so a stack-trace-shaped upstream error
            # cannot masquerade as a Cortex bug in the audit log.
            await self._audit(
                session,
                tenant,
                investigation_id=audit_investigation_id,
                tool=tool,
                capability=capability,
                params=params,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(started),
                evidence_id=None,
            )
            raise ToolError(f"{tool.name}.{capability.name} failed: {type(exc).__name__}") from exc

        duration_ms = _elapsed_ms(started)

        await self._audit(
            session,
            tenant,
            investigation_id=investigation_id,
            tool=tool,
            capability=capability,
            params=params,
            succeeded=True,
            error=None,
            duration_ms=duration_ms,
            evidence_id=evidence.id,
        )

        return ExecutedTool(
            evidence_id=evidence.id,
            tool_name=tool.name,
            capability=capability.name,
            # The sanitised payload, not the raw one: what the caller reasons over
            # must be exactly what was persisted and hashed.
            payload=payload,
            source_ref=result.source_ref,
            freshness=result.freshness,
            payload_hash=payload_hash,
            duration_ms=duration_ms,
            is_empty=is_empty,
            empty_hint=empty_hint,
        )

    async def execute_many(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        calls: Sequence[tuple[str, dict[str, Any]]],
        credential_label: str = "default",
    ) -> list[ExecutedTool | ToolError]:
        """Run several capabilities, overlapping only their network time.

        Returns one entry per call, positionally, with failures as `ToolError` **values rather
        than raised** — a batch is several independent attempts, and one rate limit must not
        discard the observations that succeeded beside it.

        ## Why this is not `asyncio.gather` over `execute`

        `execute` interleaves database work with the network call: the F-01 ownership check
        before it, the F-05 savepoint that writes `Evidence` after it, and an audit row on
        either path. `AsyncSession` is not safe for concurrent use, and concurrent
        `begin_nested()` savepoints on one session would interleave and corrupt each other. So
        the phases are separated rather than the whole method wrapped:

          1. **Preflight, serial.** Ownership, rate limit, schema validation, credential load.
             Every check that guards the network call still happens before it, in the same order
             as `execute`, and a call that fails preflight never reaches the network.
          2. **Network, concurrent.** Only `capability.handler(...)`, which touches no database.
          3. **Record, serial.** Sanitise, hash, write `Evidence` in its own savepoint, write the
             audit row. Identical to `execute`, one call at a time.

        The measured reason: eight tool calls in one investigation took 23.7 seconds serially at
        a mean of 3.0 seconds each — 17% of a 143-second wall clock spent waiting on I/O one
        request at a time.

        ## What is deliberately *not* concurrent

        Rate-limit decisions are taken serially in preflight so a batch cannot slip past a
        per-tenant ceiling by asking eight questions at once. Evidence writes are serial so the
        savepoint semantics F-05 depends on are unchanged.
        """
        if not calls:
            return []

        # Checked once for the batch rather than per call: it is one investigation, and the
        # answer cannot differ between calls. Same query, same F-01 guarantee.
        owned = await self._investigation_is_owned(session, tenant, investigation_id)
        started = time.perf_counter()

        # ---------------------------------------------------------------- 1. preflight
        prepared: list[dict[str, Any]] = []
        # Credentials are decrypted once per provider for the batch. Two calls to the same
        # connector previously decrypted the same secret twice; the value is identical and
        # decryption is the expensive part.
        contexts: dict[tuple[str, str], ToolContext] = {}

        for qualified_name, raw_params in calls:
            params = raw_params or {}
            entry: dict[str, Any] = {"qualified_name": qualified_name, "params": params}
            try:
                tool, capability = self._registry.resolve(qualified_name)
            except ToolError as exc:
                # Unresolvable, so there is no tool or capability to audit against. Returned to
                # the caller, which feeds it back to the model as a correctable mistake.
                entry["error"] = exc
                prepared.append(entry)
                continue

            entry["tool"] = tool
            entry["capability"] = capability
            try:
                if not owned:
                    raise InvestigationNotFound(
                        f"investigation {investigation_id} is not available to this tenant"
                    )
                decision = await self._limiter.check(tenant.tenant_id, tool.provider)
                if not decision.allowed:
                    raise TenantRateLimited(
                        f"{tool.name}: {decision.reason}. This is Cortex's own per-tenant "
                        f"ceiling, not the upstream's — the same call will succeed later.",
                        retry_after_seconds=decision.retry_after_seconds,
                    )
                self._validate(capability, params)
                key = (tool.name, credential_label)
                if key not in contexts:
                    contexts[key] = await self._load_context(
                        session, tenant, tool, credential_label
                    )
                entry["context"] = contexts[key]
            except ToolError as exc:
                entry["error"] = exc
            prepared.append(entry)

        # ---------------------------------------------------------------- 2. network
        async def _invoke(entry: dict[str, Any]) -> Any:
            """Run one call, and time *that call*.

            Timed per entry rather than per batch. The batch's elapsed time was previously
            stamped on every row in it, so an audit trail of three parallel calls reported each
            of them taking the whole batch's wall clock -- on a real investigation, three calls
            each recorded at 8,003 ms when 8,003 ms was the total. That makes the slowest call
            in a batch indistinguishable from the fastest, which is precisely the thing a
            latency trace exists to show.
            """
            capability: Capability = entry["capability"]
            call_started = time.perf_counter()
            try:
                return await capability.handler(entry["context"], **entry["params"])
            finally:
                entry["duration_ms"] = _elapsed_ms(call_started)

        runnable = [entry for entry in prepared if "error" not in entry and "context" in entry]
        if runnable:
            # return_exceptions, because one connector's 503 must not cancel the others' calls
            # in flight -- which is exactly what an unguarded gather would do.
            outcomes = await asyncio.gather(
                *(_invoke(entry) for entry in runnable), return_exceptions=True
            )
            for entry, outcome in zip(runnable, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    entry["raised"] = outcome
                else:
                    entry["result"] = outcome

        # ---------------------------------------------------------------- 3. record
        # The batch's own elapsed time, kept for the entries that never reached the network --
        # a call rejected in preflight has no duration of its own, and reporting 0 would read
        # as "instant" rather than "never ran".
        batch_ms = _elapsed_ms(started)
        audit_investigation_id = investigation_id if owned else None
        results: list[ExecutedTool | ToolError] = []

        for entry in prepared:
            tool = entry.get("tool")
            capability = entry.get("capability")
            params = entry["params"]

            if tool is None or capability is None:
                results.append(entry["error"])
                continue

            duration_ms = entry.get("duration_ms", batch_ms)
            failure: BaseException | None = entry.get("error") or entry.get("raised")
            if failure is None:
                try:
                    self._check_result(tool, capability, entry["result"])
                except ToolError as exc:
                    failure = exc

            if failure is not None:
                await self._audit(
                    session,
                    tenant,
                    investigation_id=audit_investigation_id,
                    tool=tool,
                    capability=capability,
                    params=params,
                    succeeded=False,
                    error=f"{type(failure).__name__}: {failure}",
                    duration_ms=duration_ms,
                    evidence_id=None,
                )
                results.append(
                    failure
                    if isinstance(failure, ToolError)
                    # An unexpected exception becomes a ToolError, matching `execute`: the type
                    # leads so an upstream stack trace cannot masquerade as a Cortex bug.
                    else ToolError(
                        f"{tool.name}.{capability.name} failed: {type(failure).__name__}"
                    )
                )
                continue

            result = entry["result"]
            payload = sanitize_payload(result.payload)
            payload_hash = canonical_hash(result.payload)
            is_empty = capability.is_empty(payload)
            empty_hint = capability.empty_hint(payload) if is_empty else ""

            evidence = Evidence(
                tenant_id=tenant.tenant_id,
                investigation_id=investigation_id,
                tool_name=tool.name,
                capability=capability.name,
                params=params,
                payload=payload,
                payload_hash=payload_hash,
                source_ref=result.source_ref,
                from_cache=result.freshness is Freshness.SYNCED,
            )
            try:
                # Its own savepoint, exactly as in `execute`, so a payload Postgres rejects
                # discards only this write rather than the caller's whole transaction (F-05).
                async with session.begin_nested():
                    session.add(evidence)
                    await session.flush()
            except Exception as exc:  # noqa: BLE001 - audited, then returned as a failure
                await self._audit(
                    session,
                    tenant,
                    investigation_id=audit_investigation_id,
                    tool=tool,
                    capability=capability,
                    params=params,
                    succeeded=False,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                    evidence_id=None,
                )
                results.append(
                    ToolError(f"{tool.name}.{capability.name} failed: {type(exc).__name__}")
                )
                continue

            await self._audit(
                session,
                tenant,
                investigation_id=investigation_id,
                tool=tool,
                capability=capability,
                params=params,
                succeeded=True,
                error=None,
                duration_ms=duration_ms,
                evidence_id=evidence.id,
            )
            results.append(
                ExecutedTool(
                    evidence_id=evidence.id,
                    tool_name=tool.name,
                    capability=capability.name,
                    payload=payload,
                    source_ref=result.source_ref,
                    freshness=result.freshness,
                    payload_hash=payload_hash,
                    duration_ms=duration_ms,
                    is_empty=is_empty,
                    empty_hint=empty_hint,
                )
            )

        return results

    # ------------------------------------------------------------------ internals

    @staticmethod
    async def _investigation_is_owned(
        session: AsyncSession, tenant: TenantContext, investigation_id: uuid.UUID
    ) -> bool:
        """Whether the investigation exists AND belongs to this tenant.

        One boolean for both cases on purpose: distinguishing "does not exist" from
        "belongs to someone else" would confirm the existence of another tenant's
        investigation to a caller enumerating ids.
        """
        found = (
            await session.execute(
                select(Investigation.id).where(
                    Investigation.id == investigation_id,
                    Investigation.tenant_id == tenant.tenant_id,
                )
            )
        ).scalar_one_or_none()
        return found is not None

    @staticmethod
    def _validate(capability: Capability, params: dict[str, Any]) -> None:
        try:
            jsonschema.validate(params, capability.params_schema)
        except jsonschema.ValidationError as exc:
            # Surfaced back to the agent so it can correct the call. The path is
            # included because "invalid arguments" alone is not actionable.
            path = ".".join(str(p) for p in exc.absolute_path) or "(root)"
            raise InvalidParams(f"{capability.name}: {path}: {exc.message}") from None

    @staticmethod
    def _check_result(tool: Tool, capability: Capability, result: ToolResult) -> None:
        if not isinstance(result, ToolResult):
            raise ToolError(
                f"{tool.name}.{capability.name} returned {type(result).__name__}, "
                "expected ToolResult"
            )

    async def _load_context(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        tool: Tool,
        credential_label: str = "default",
    ) -> ToolContext:
        return await load_tool_context(session, tenant, tool, credential_label)

    @staticmethod
    async def _audit(
        session: AsyncSession,
        tenant: TenantContext,
        *,
        investigation_id: uuid.UUID,
        tool: Tool,
        capability: Capability,
        params: dict[str, Any],
        succeeded: bool,
        error: str | None,
        duration_ms: int,
        evidence_id: uuid.UUID | None,
    ) -> None:
        session.add(
            ToolCall(
                tenant_id=tenant.tenant_id,
                investigation_id=investigation_id,
                evidence_id=evidence_id,
                tool_name=tool.name,
                capability=capability.name,
                read_only=capability.read_only,
                # Params only. The response body lives in Evidence; duplicating it
                # here would double the PII surface for no benefit.
                params=params,
                succeeded=succeeded,
                error=error,
                duration_ms=duration_ms,
            )
        )
        await session.flush()


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def load_tool_context(
    session: AsyncSession,
    tenant: TenantContext,
    tool: Tool,
    credential_label: str = "default",
) -> ToolContext:
    """Decrypt one tenant's credential for one tool.

    Module-level and public because the nightly ingest needs exactly this and nothing
    around it. Ingest calls connector capabilities directly: it writes to the graph, to
    semantic memory and to the metric series, not to the evidence store, and minting an
    Evidence row per sync call would fill the citation space with rows no report will ever
    cite — while requiring an investigation id that does not exist.

    Shared rather than reimplemented, because the part worth not duplicating is the tenant
    scoping. A second decryption path is a second chance to resolve a credential across
    tenants, which is F-01.
    """
    if tool.provider is None:
        return ToolContext(tenant=tenant)

    row = (
        await session.execute(
            select(Credential).where(
                # Tenant-scoped: a credential is never resolvable across tenants.
                Credential.tenant_id == tenant.tenant_id,
                Credential.provider == tool.provider,
                Credential.label == credential_label,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        # The label is selectable because a tenant may connect several accounts
        # per provider — two GA4 properties, say. Naming the labels they do have
        # turns "credential missing" into an actionable message (F-08).
        available = (
            (
                await session.execute(
                    select(Credential.label).where(
                        Credential.tenant_id == tenant.tenant_id,
                        Credential.provider == tool.provider,
                    )
                )
            )
            .scalars()
            .all()
        )
        if available:
            raise CredentialMissing(
                f"tenant has no {tool.provider.value} credential labelled "
                f"{credential_label!r}; available labels: {', '.join(sorted(available))}"
            )
        raise CredentialMissing(
            f"tenant has not connected {tool.provider.value}; "
            f"{tool.name} is unavailable until it is"
        )

    secret = decrypt_credential(
        tenant.tenant_id,
        tool.provider.value,
        row.wrapped_data_key,
        row.ciphertext,
        label=row.label,
    )
    # Every other credential this tenant holds for the same provider, metadata only. One
    # provider can hold several credentials doing different jobs, and a capability sometimes
    # needs to know something about one it is not using -- see `ToolContext.peer_metadata`.
    peers = (
        (
            await session.execute(
                select(Credential.label, Credential.metadata_).where(
                    Credential.tenant_id == tenant.tenant_id,
                    Credential.provider == tool.provider,
                    Credential.label != row.label,
                )
            )
        )
        .tuples()
        .all()
    )
    return ToolContext(
        tenant=tenant,
        credential=secret,
        credential_metadata=dict(row.metadata_),
        peer_metadata={label: dict(metadata or {}) for label, metadata in peers},
    )
