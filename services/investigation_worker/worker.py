"""Investigation worker service.

Consumes only cortex.investigation. Runs the hypothesis-driven loop, writes evidence,
and produces the graded report. It is the only service that talks to an LLM, which is
why it is separately deployable: its scaling signal is token throughput, and its
failure mode (a hung provider) must not touch the HTTP tier.
"""

from __future__ import annotations

from typing import Any

from cortex.agents.investigator import InvestigationCancelled, InvestigationFailed
from cortex.agents.progress import QueueProgress
from cortex.agents.provider import build_llm
from cortex.agents.service import InvestigationService
from cortex.contracts.messages import RunInvestigation
from cortex.memory.recall import HybridRecall
from cortex.notify.deliver import deliver_report
from cortex.reports.gate import ReportRejected
from cortex.runtime.celery_app import make_celery
from cortex.runtime.resources import Resources, run_with_resources
from cortex.tenancy.context import TenantNotFound, load_tenant_context_by_id

SERVICE_NAME = "investigation-worker"

celery = make_celery(SERVICE_NAME)


@celery.task(name="cortex.investigation.run", bind=True, max_retries=2)
def run_investigation(self: Any, payload: dict) -> dict:  # noqa: ANN401 - Celery bind
    """Run one investigation.

    The message carries only ids. The worker re-reads the investigation row so a
    redelivered message cannot resurrect stale parameters, and re-resolves the tenant
    rather than trusting anything denormalized onto the wire.
    """
    message = RunInvestigation.model_validate(payload)
    # Reject an unrecognized version loudly rather than guessing at its meaning —
    # a half-deployed fleet should fail visibly, not produce subtly wrong answers.
    message.require_supported()

    async def _work(resources: Resources) -> dict:
        async with resources.session() as session:
            try:
                tenant = await load_tenant_context_by_id(session, message.tenant_id)
            except TenantNotFound as exc:
                # A suspended or deleted tenant. Not retryable, and not an error worth
                # alerting on: the queued message simply outlived its tenant.
                return {
                    "tenant_id": str(message.tenant_id),
                    "investigation_id": str(message.investigation_id),
                    # Correlated even when skipped: a message that goes nowhere is
                    # exactly the case someone will be tracing.
                    "correlation_id": str(message.correlation_id),
                    "status": "skipped",
                    "reason": f"tenant unavailable: {exc}",
                }

            # Memory is offered here because a worker always has the process resources
            # that back it. Recall failing is non-fatal inside the loop, so a tenant with
            # nothing ingested investigates exactly as before.
            service = InvestigationService(
                llm=build_llm(),
                recall=HybridRecall(resources.vectors, resources.graph),
                # Enables the per-step cancellation check, which needs its own session to see
                # a commit the gateway made after the loop's transaction began.
                sessionmaker=resources.sessionmaker,
                # The same events the CLI prints, published for the UI to stream. The
                # contract for these has existed since M0 and nothing ever sent one, so the
                # gateway's streaming endpoint had nothing to stream.
                limiter=resources.limiter,
                progress=QueueProgress(
                    producer=celery,
                    tenant_id=message.tenant_id,
                    investigation_id=message.investigation_id,
                    correlation_id=message.correlation_id,
                ),
            )
            try:
                completed = await service.run(
                    session, tenant, investigation_id=message.investigation_id
                )
            except InvestigationCancelled as exc:
                # A cancellation is a completed request, not a failed one: retrying it would
                # restart the very work a user asked to stop.
                return {
                    "tenant_id": str(message.tenant_id),
                    "investigation_id": str(message.investigation_id),
                    "correlation_id": str(message.correlation_id),
                    "status": "cancelled",
                    "reason": str(exc),
                }
            except (InvestigationFailed, ReportRejected) as exc:
                # The service already recorded the failure on the row and this session
                # commits it. Returned rather than raised so Celery does not retry: a
                # refused report will be refused again, and a retry would spend the
                # tokens twice to reach the same conclusion.
                return {
                    "tenant_id": str(message.tenant_id),
                    "investigation_id": str(message.investigation_id),
                    "correlation_id": str(message.correlation_id),
                    "status": "failed",
                    "reason": str(exc)[:500],
                }

            # Delivered after the report is stored, and never allowed to fail the investigation.
            # A question asked in Slack has to be answered in Slack; a question asked through the
            # API has nowhere to go and returns "nothing to deliver to". The reason is returned
            # rather than raised, so a revoked bot token is a log line beside a completed
            # investigation rather than a completed investigation marked failed.
            delivery = await deliver_report(
                session, tenant, investigation_id=message.investigation_id
            )

            return {
                "tenant_id": str(message.tenant_id),
                "investigation_id": str(completed.investigation_id),
                "report_id": str(completed.report_id),
                "correlation_id": str(message.correlation_id),
                "delivery": delivery,
                "status": "completed",
                "hallucinations": completed.hallucinations,
                "gate_rejections": completed.gate_rejections,
                "verifier_rejections": completed.verifier_rejections,
                "tokens": completed.tokens,
                "duration_ms": completed.duration_ms,
            }

    return run_with_resources(_work)
