"""Measuring how often the verifier deletes a claim that was actually supported.

`draft_reliability` has read between 0.62 and 1.00 and has been unreadable the whole time. A 0.62
means 38% of claims were removed, and nothing distinguished the drafter overreaching (the defences
working) from the verifier being wrong (silently making reports worse while every number says the
grounding machinery is fine).

The measurement works by making truth non-negotiable: each claim is generated mechanically from an
evidence row's own payload, so it is true by construction, cited to exactly the row it came from,
and needs no annotator. Any `UNSUPPORTED` verdict on one is a false positive.

These tests cover the generator and the accounting with a scripted verifier. The rate itself is a
live measurement — 0% of 24 claims on real evidence at the time of writing — and cannot be asserted
here without spending tokens on every test run.
"""

from __future__ import annotations

import types
import uuid

from cortex.eval.verifier_quality import (
    OverRejectionReport,
    claims_from_payload,
    measure_over_rejection,
)


def _row(payload: dict, *, tool: str = "posthog", capability: str = "event_trend"):
    return types.SimpleNamespace(
        id=uuid.uuid4(), tool_name=tool, capability=capability, payload=payload
    )


class TestTheGeneratedClaimsAreTrueByConstruction:
    def test_it_states_the_number_of_items(self) -> None:
        row = _row({"series": [1, 2, 3]})
        claims = claims_from_payload(row)
        assert any("returned 3 item(s) under 'series'" in claim for claim in claims)

    def test_it_quotes_a_scalar_exactly(self) -> None:
        """Copied verbatim. A paraphrase would need interpretation, and then a removal would stop
        being unambiguously a false positive."""
        row = _row({"rows": [], "count": 14})
        assert any("'count' is 14" in claim for claim in claims_from_payload(row))

    def test_it_quotes_a_short_string(self) -> None:
        row = _row({"rows": [], "event": "signup_completed"})
        assert any("'signup_completed'" in claim for claim in claims_from_payload(row))

    def test_it_skips_a_long_string(self) -> None:
        """A 400-character blob truncated into a claim would no longer be a restatement of the
        payload, so it must not be generated at all."""
        row = _row({"rows": [], "body": "x" * 400})
        assert not any("'body'" in claim for claim in claims_from_payload(row))

    def test_it_skips_booleans_and_nulls(self) -> None:
        """ "'ok' is True" is a claim about a flag, not an observation, and invites a defensible
        rejection."""
        row = _row({"ok": True, "missing": None})
        assert claims_from_payload(row) == []

    def test_it_is_bounded_per_row(self) -> None:
        from cortex.eval.verifier_quality import MAX_CLAIMS_PER_ROW

        row = _row({"a": [1], "b": 2, "c": "three", "d": 4, "e": [5, 6]})
        assert len(claims_from_payload(row)) <= MAX_CLAIMS_PER_ROW

    def test_an_empty_payload_generates_nothing(self) -> None:
        assert claims_from_payload(_row({})) == []

    def test_a_non_dict_payload_generates_nothing(self) -> None:
        assert claims_from_payload(_row([1, 2, 3])) == []  # type: ignore[arg-type]


class TestTheAccounting:
    def test_the_rate_is_removals_over_judged(self) -> None:
        outcome = OverRejectionReport(claims_judged=20, over_rejections=3)
        assert outcome.rate == 0.15

    def test_no_claims_is_zero_rather_than_a_division_error(self) -> None:
        assert OverRejectionReport().rate == 0.0

    def test_unjudged_claims_are_excluded_from_the_rate(self) -> None:
        """An unjudged claim is *kept* by the verifier, so counting it as a false positive would
        blame the verifier for a provider outage."""
        outcome = OverRejectionReport(claims_judged=10, over_rejections=1, unjudged=5)
        assert outcome.rate == 0.1

    def test_the_rendering_names_the_removed_claims(self) -> None:
        """A rate with no examples cannot be acted on: the reason is what says whether the
        verifier misread the payload or the generator wrote something ambiguous."""
        outcome = OverRejectionReport(
            claims_judged=4,
            over_rejections=1,
            examples=[("'count' is 14.", "the evidence does not state a count")],
        )
        rendered = outcome.render()
        assert "25%" in rendered
        assert "'count' is 14." in rendered
        assert "does not state a count" in rendered

    def test_it_says_so_when_nothing_could_be_generated(self) -> None:
        assert "no claims" in OverRejectionReport().render()


class TestAgainstAScriptedVerifier:
    async def test_a_verifier_that_rejects_everything_scores_one(self, session) -> None:  # type: ignore[no-untyped-def]
        """The measurement's own upper bound. If this did not read 100%, the accounting would be
        wrong in the direction that hides the failure."""
        from cortex.db.models import Evidence, Investigation, Tenant
        from cortex.memory.naming import graph_name_for_new_tenant

        # The verifier has its own Verdict enum (supported/overstated/unsupported), distinct
        # from the report schema's (supported/contradicted/inconclusive). Importing the wrong one
        # is an easy mistake and would make this test assert nothing.
        from cortex.reports.verifier import ClaimVerdict, Verdict, VerificationResult
        from cortex.tenancy.context import TenantContext

        tenant_id = uuid.uuid4()
        graph_name = graph_name_for_new_tenant("fp-all", tenant_id)
        session.add(Tenant(id=tenant_id, slug="fp-all", name="fp-all", graph_name=graph_name))
        await session.flush()
        investigation = Investigation(tenant_id=tenant_id, question="why?")
        session.add(investigation)
        await session.flush()
        session.add(
            Evidence(
                tenant_id=tenant_id,
                investigation_id=investigation.id,
                tool_name="posthog",
                capability="event_trend",
                params={},
                payload={"series": [1, 2], "count": 2},
                payload_hash="d" * 64,
            )
        )
        await session.flush()

        class _RejectAll:
            async def verify(self, session, tenant, *, investigation_id, report):  # type: ignore[no-untyped-def]
                verdicts = [
                    ClaimVerdict(
                        location="x",
                        claim_text=claim.text,
                        verdict=Verdict.UNSUPPORTED,
                        reason="rejected by the scripted verifier",
                    )
                    for claim in [
                        *report.executive_summary,
                        *(c for f in report.findings for c in f.claims),
                    ]
                ]
                return VerificationResult(report=report, verdicts=verdicts)

        tenant = TenantContext(tenant_id=tenant_id, tenant_slug="fp-all", graph_name=graph_name)
        outcome = await measure_over_rejection(
            session,
            tenant,
            investigation_id=investigation.id,
            verifier=_RejectAll(),  # type: ignore[arg-type]
        )
        assert outcome.claims_judged >= 1
        assert outcome.rate == 1.0
        assert outcome.examples

    async def test_no_evidence_means_no_measurement(self, session) -> None:  # type: ignore[no-untyped-def]
        """An investigation with nothing gathered cannot produce true-by-construction claims, and
        must report that rather than a flattering 0%."""
        from cortex.db.models import Investigation, Tenant
        from cortex.memory.naming import graph_name_for_new_tenant
        from cortex.tenancy.context import TenantContext

        tenant_id = uuid.uuid4()
        graph_name = graph_name_for_new_tenant("fp-empty", tenant_id)
        session.add(Tenant(id=tenant_id, slug="fp-empty", name="fp-empty", graph_name=graph_name))
        await session.flush()
        investigation = Investigation(tenant_id=tenant_id, question="why?")
        session.add(investigation)
        await session.flush()

        class _Unused:
            async def verify(self, *a, **k):  # type: ignore[no-untyped-def]
                raise AssertionError("must not be called when there is nothing to judge")

        outcome = await measure_over_rejection(
            session,
            TenantContext(tenant_id=tenant_id, tenant_slug="fp-empty", graph_name=graph_name),
            investigation_id=investigation.id,
            verifier=_Unused(),  # type: ignore[arg-type]
        )
        assert outcome.claims_judged == 0
        assert "no claims" in outcome.render()


class TestTheNegativeStrata:
    """Claims that are FALSE by construction, and the polarity the restatement stratum cannot see.

    The restatement stratum measures over-rejection — the verifier destroying true claims. It says
    nothing about the direction that maps to "never hallucinate": whether the verifier *accepts* a
    claim its own cited evidence contradicts.

    Two constructions cover that, both from arXiv 2604.09537:

      - **counterfactual** — the number or direction mutated, citation left alone. *"Semantically
        meaningful and often close to the correct evidence, but it supports the wrong state."*
      - **swapped** — a true claim repointed at an unrelated row. Their swap control drops a trained
        verifier from AUROC 97.43 to 55.62, so a verifier that still says supported here is not
        reading its evidence at all.

    Reported per stratum and never aggregated, following NEI-CAP (arXiv 2605.26663), which measured
    a verifier at NEI-F1 1.000 on its matched construction and 0.000 on a near miss: *"matched NEI
    performance does not imply transfer."*
    """

    def test_a_counterfactual_count_is_adjacent_not_absurd(self) -> None:
        """A wildly wrong claim would be rejected by a verifier that was barely reading, so it
        would measure nothing. The mutation has to require actually counting."""
        from cortex.eval.verifier_quality import counterfactual_claims

        row = _row({"series": [1] * 14})
        claims = counterfactual_claims(row)
        assert any("21 item(s)" in claim for claim in claims)
        assert not any("14 item(s)" in claim for claim in claims)

    def test_a_counterfactual_scalar_keeps_its_sign(self) -> None:
        """Same sign, wrong magnitude: plausible enough to need a real comparison rather than a
        sanity check."""
        from cortex.eval.verifier_quality import counterfactual_claims

        claims = counterfactual_claims(_row({"rows": [], "count": 14}))
        assert any("'count' is 140" in claim for claim in claims)

    def test_a_counterfactual_skips_zero(self) -> None:
        """Multiplying zero produces zero, so the "false" claim would be true."""
        from cortex.eval.verifier_quality import counterfactual_claims

        claims = counterfactual_claims(_row({"rows": [], "count": 0}))
        assert not any("'count' is 0" in claim for claim in claims)

    def test_the_strata_declare_which_direction_is_an_error(self) -> None:
        """The two directions are different failures with opposite difficulty, and conflating them
        under one word is the mistake this module already made once."""
        from cortex.eval.verifier_quality import Construction, StratumResult

        restatement = StratumResult(construction=Construction.RESTATEMENT, should_reject=False)
        swapped = StratumResult(construction=Construction.SWAPPED, should_reject=True)
        assert restatement.error_name == "wrongly rejected"
        assert swapped.error_name == "wrongly accepted"

    def test_an_overstated_verdict_counts_as_accepted_for_a_false_claim(self) -> None:
        """Deliberately strict: `rejected` is True only for UNSUPPORTED, so a counterfactual the
        verifier merely downgrades still reaches the reader. A downgraded false claim is a false
        claim."""
        from cortex.reports.verifier import ClaimVerdict, Verdict

        overstated = ClaimVerdict(
            location="x", claim_text="c", verdict=Verdict.OVERSTATED, reason="hedged"
        )
        assert overstated.rejected is False

    def test_the_rendering_never_aggregates(self) -> None:
        """A single number across strata is the reporting style NEI-CAP shows to be misleading."""
        from cortex.eval.verifier_quality import Construction, StratumResult, render_strata

        rendered = render_strata(
            {
                Construction.RESTATEMENT: StratumResult(
                    construction=Construction.RESTATEMENT,
                    should_reject=False,
                    judged=10,
                    errors=0,
                ),
                Construction.SWAPPED: StratumResult(
                    construction=Construction.SWAPPED,
                    should_reject=True,
                    judged=8,
                    errors=4,
                ),
            }
        )
        assert "restatement" in rendered and "swapped" in rendered
        assert "wrongly rejected" in rendered and "wrongly accepted" in rendered
        assert "50%" in rendered

    async def test_a_swapped_claim_cites_a_different_row(self, session) -> None:  # type: ignore[no-untyped-def]
        """The citation must still resolve and hash-match, because the point is that the gate
        cannot catch this — only the verifier stands between a swapped citation and the reader."""
        import uuid as _uuid

        from cortex.db.models import Evidence, Investigation, Tenant
        from cortex.eval.verifier_quality import Construction, _claims_for
        from cortex.memory.naming import graph_name_for_new_tenant

        tenant_id = _uuid.uuid4()
        graph_name = graph_name_for_new_tenant("fp-swap", tenant_id)
        session.add(Tenant(id=tenant_id, slug="fp-swap", name="fp-swap", graph_name=graph_name))
        await session.flush()
        investigation = Investigation(tenant_id=tenant_id, question="why?")
        session.add(investigation)
        await session.flush()

        rows = []
        for index in range(2):
            row = Evidence(
                tenant_id=tenant_id,
                investigation_id=investigation.id,
                tool_name="posthog",
                capability=f"cap_{index}",
                params={},
                payload={"series": [1] * (index + 3)},
                payload_hash="e" * 64,
            )
            session.add(row)
            rows.append(row)
        await session.flush()

        claims = _claims_for(Construction.SWAPPED, rows)
        assert claims
        # Each claim describes one row but cites the other.
        for claim in claims:
            cited = claim.evidence_ids[0]
            described = next(r for r in rows if str(len(r.payload["series"])) in claim.text)
            assert cited != described.id
