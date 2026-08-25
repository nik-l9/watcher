"""The report page's renderers, as pure functions.

The page had no test that did not need Postgres, so a rendering change could only be checked by
bringing the whole stack up. These take the models as plain objects — a `ToolCall` is
constructible without a session as long as nothing flushes it — which is enough to pin the two
things the page has to get right: that a reader can see what was ruled out and when, and that
nothing model-authored reaches the document unescaped.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from cortex.db.models import ToolCall
from services.gateway.views import _premise_html, _removed_html, _trace_html


def _call(
    tool: str = "posthog",
    capability: str = "event_trend",
    *,
    evidence_id: uuid.UUID | None = None,
    succeeded: bool = True,
    error: str | None = None,
    params: dict | None = None,
    minute: int = 0,
) -> ToolCall:
    call = ToolCall(
        tool_name=tool,
        capability=capability,
        succeeded=succeeded,
        error=error,
        params=params or {},
        evidence_id=evidence_id,
        duration_ms=1200,
    )
    call.created_at = datetime(2026, 8, 15, 14, minute, 0, tzinfo=UTC)
    return call


class TestTheTraceIsATimeline:
    async def test_every_call_becomes_a_step_in_order(self) -> None:
        html = _trace_html(
            [
                _call("posthog", "event_trend", evidence_id=uuid.uuid4(), minute=1),
                _call("github", "commits", evidence_id=uuid.uuid4(), minute=2),
            ],
            [],
        )
        assert html.index("posthog.event_trend") < html.index("github.commits")
        assert "class=tl" in html

    async def test_the_outcome_of_each_call_is_on_its_node(self) -> None:
        """An empty call and a failed call are different facts, and both differ from a call that
        returned evidence. The node's class is what carries that to the eye."""
        html = _trace_html(
            [
                _call(evidence_id=uuid.uuid4(), minute=1),
                _call(succeeded=True, evidence_id=None, minute=2),
                _call(succeeded=False, error="401 Unauthorized", minute=3),
            ],
            [],
        )
        assert "step-ok" in html and "step-empty" in html and "step-failed" in html
        assert "401 Unauthorized" in html

    async def test_an_empty_call_is_counted_and_explained(self) -> None:
        html = _trace_html([_call(succeeded=True, evidence_id=None)], [])
        assert "1 of 1 call(s) returned nothing" in html
        assert "not the same as what does not exist" in html

    async def test_no_calls_says_so_rather_than_rendering_an_empty_rail(self) -> None:
        assert "No tool calls yet" in _trace_html([], [])


class TestAVerdictAppearsWhereItWasDecided:
    """The feature that earns the page its keep.

    "Four hypotheses were tested" is a claim about diligence. "This one was contradicted here,
    on that call, by that evidence" is checkable, and the reader can follow the citation.
    """

    async def test_a_hypothesis_is_placed_at_the_step_that_settled_it(self) -> None:
        first, second = uuid.uuid4(), uuid.uuid4()
        html = _trace_html(
            [
                _call("posthog", "event_trend", evidence_id=first, minute=1),
                _call("posthog", "list_events", evidence_id=second, minute=2),
                _call("github", "commits", evidence_id=uuid.uuid4(), minute=3),
            ],
            [
                {
                    "statement": "The collapse reflects a real drop in demand.",
                    "verdict": "contradicted",
                    "contradicting_evidence_ids": [str(second)],
                    "reasoning": "Every product event ceased in the same hour.",
                }
            ],
        )
        # After the call whose evidence decided it, and before the next one.
        assert html.index("list_events") < html.index("real drop in demand")
        assert html.index("real drop in demand") < html.index("github.commits")
        assert "step-verdict" in html and "v-contradicted" in html

    async def test_it_lands_on_the_last_deciding_step_not_the_first(self) -> None:
        """A verdict was not available until the last piece of its evidence arrived. Placing it
        at the first would claim the analyst knew sooner than it did."""
        early, late = uuid.uuid4(), uuid.uuid4()
        html = _trace_html(
            [
                _call("posthog", "event_trend", evidence_id=early, minute=1),
                _call("github", "commits", evidence_id=uuid.uuid4(), minute=2),
                _call("posthog", "list_events", evidence_id=late, minute=3),
            ],
            [
                {
                    "statement": "Tracking broke.",
                    "verdict": "supported",
                    "supporting_evidence_ids": [str(early), str(late)],
                }
            ],
        )
        assert html.index("list_events") < html.index("Tracking broke")

    async def test_a_contradicted_statement_is_struck_through(self) -> None:
        evidence = uuid.uuid4()
        html = _trace_html(
            [_call(evidence_id=evidence)],
            [
                {
                    "statement": "Marketing spend fell.",
                    "verdict": "contradicted",
                    "contradicting_evidence_ids": [str(evidence)],
                }
            ],
        )
        assert "struck" in html

    async def test_an_undecided_hypothesis_is_listed_rather_than_dropped(self) -> None:
        """It has no step to sit on, which is exactly why it needs somewhere else to be. An
        investigation that could not rule a cause in or out has told the reader something real
        about the limits of the answer."""
        html = _trace_html(
            [_call(evidence_id=uuid.uuid4())],
            [
                {
                    "statement": "The June dip was the reverse-proxy change.",
                    "verdict": "inconclusive",
                    "reasoning": "Timing is close but nothing ties them together.",
                }
            ],
        )
        assert "Raised, not settled" in html
        assert "reverse-proxy change" in html
        assert "an answer's limits are part of the answer" in html

    async def test_a_verdict_citing_unknown_evidence_is_not_lost(self) -> None:
        """Evidence from a parent investigation resolves for the citation and has no step on
        *this* trace. Silently dropping the hypothesis would hide a real verdict."""
        html = _trace_html(
            [_call(evidence_id=uuid.uuid4())],
            [
                {
                    "statement": "Cited from the parent investigation.",
                    "verdict": "supported",
                    "supporting_evidence_ids": [str(uuid.uuid4())],
                }
            ],
        )
        assert "Cited from the parent investigation." in html


class TestRemovedClaimsAreShownNotCounted:
    """A bare count reads as an accusation or a boast depending on the reader's mood.
    The claims themselves read as a system that checks its work."""

    _GATE = [
        {
            "location": "executive_summary[1]",
            "reason": "unknown_evidence",
            "detail": "evidence_id not in this investigation",
            "text": "Signups fell 42% month over month.",
        }
    ]
    _VERIFIER = [
        {
            "location": "findings[0].claims[2]",
            "reason": "no_surviving_evidence",
            "detail": "verifier: unsupported: the cited series does not cover July",
            "text": "The July recovery was driven by the pricing page rewrite.",
        }
    ]

    async def test_the_text_of_each_removed_claim_is_shown(self) -> None:
        html = _removed_html(self._GATE, self._VERIFIER)
        assert "Signups fell 42% month over month." in html
        assert "The July recovery was driven by the pricing page rewrite." in html

    async def test_each_says_which_stage_removed_it_and_why(self) -> None:
        html = _removed_html(self._GATE, self._VERIFIER)
        assert "removed by citation gate" in html
        assert "removed by verification" in html
        assert "the cited series does not cover July" in html
        assert "executive_summary[1]" in html

    async def test_a_clean_report_takes_no_space(self) -> None:
        assert _removed_html([], []) == ""
        assert _removed_html(None, None) == ""

    async def test_it_is_collapsed_without_javascript(self) -> None:
        """A `<details>` is an element, not a script. The page still runs no JavaScript."""
        html = _removed_html(self._GATE, [])
        assert html.startswith("<details>") and "<summary>" in html
        assert "1 claim(s) were removed" in html

    async def test_a_rejection_with_no_recorded_text_still_appears(self) -> None:
        html = _removed_html([{"location": "findings[0]", "reason": "empty_citation"}], [])
        assert "no text recorded" in html


class TestNothingModelAuthoredReachesTheDocumentAsMarkup:
    """Claim text, hypothesis statements and tool errors are all model or vendor output."""

    async def test_a_hypothesis_statement_is_escaped(self) -> None:
        evidence = uuid.uuid4()
        html = _trace_html(
            [_call(evidence_id=evidence)],
            [
                {
                    "statement": "<img src=x onerror=alert(1)>",
                    "verdict": "contradicted",
                    "contradicting_evidence_ids": [str(evidence)],
                    "reasoning": "<script>alert(2)</script>",
                }
            ],
        )
        assert "<img" not in html and "<script>" not in html
        assert "&lt;img" in html and "&lt;script&gt;" in html

    async def test_a_tool_error_is_escaped(self) -> None:
        html = _trace_html([_call(succeeded=False, error="<b>boom</b>")], [])
        assert "<b>boom</b>" not in html and "&lt;b&gt;" in html

    async def test_removed_claim_text_is_escaped(self) -> None:
        html = _removed_html(
            [{"text": "<svg onload=alert(1)>", "reason": "r", "detail": "<i>d</i>"}], []
        )
        assert "<svg" not in html and "<i>" not in html


class TestAnEliminationReadsAsImpossibleNotContradicted:
    """The two are worth distinguishing on the page.

    "Contradicted" invites a reader to weigh the reasoning. "Impossible" says a cause post-dated
    its effect, which is arithmetic rather than a judgement anyone can disagree with -- and it is
    the strongest thing this system can say about a candidate, so filing it under the same label
    as every other rejection wastes it.
    """

    @staticmethod
    def _eliminated(evidence: uuid.UUID) -> dict:
        return {
            "statement": "The pageview collapse caused the signup cessation.",
            "verdict": "contradicted",
            "contradicting_evidence_ids": [str(evidence)],
            "cause_at": "2026-08-11",
            "effect_onset": "2026-08-04",
            "reasoning": "Pageviews ran at 14,197/day on 08-04.",
        }

    async def test_it_is_labelled_impossible(self) -> None:
        evidence = uuid.uuid4()
        html = _trace_html([_call(evidence_id=evidence)], [self._eliminated(evidence)])
        assert "impossible" in html

    async def test_the_two_dates_are_shown_with_the_rule(self) -> None:
        """A reader can check the arithmetic themselves, which is the point."""
        evidence = uuid.uuid4()
        html = _trace_html([_call(evidence_id=evidence)], [self._eliminated(evidence)])
        assert "Cause dated 2026-08-11" in html
        assert "the movement began 2026-08-04" in html
        assert "A cause cannot post-date its effect" in html

    async def test_an_ordinary_contradiction_keeps_its_own_label(self) -> None:
        evidence = uuid.uuid4()
        html = _trace_html(
            [_call(evidence_id=evidence)],
            [
                {
                    "statement": "Marketing spend fell.",
                    "verdict": "contradicted",
                    "contradicting_evidence_ids": [str(evidence)],
                }
            ],
        )
        assert "impossible" not in html
        assert "contradicted" in html

    async def test_a_valid_ordering_is_not_labelled_impossible(self) -> None:
        evidence = uuid.uuid4()
        html = _trace_html(
            [_call(evidence_id=evidence)],
            [
                {
                    "statement": "The reverse-proxy change broke tracking.",
                    "verdict": "supported",
                    "supporting_evidence_ids": [str(evidence)],
                    "cause_at": "2026-06-15",
                    "effect_onset": "2026-06-17",
                }
            ],
        )
        assert "impossible" not in html
        assert "A cause cannot post-date its effect" not in html


class TestTheReportLinkCanActuallyBeOpened:
    """A link posted into Slack exists to be clicked.

    Every report link Cortex had ever posted was unopenable: the view requires a tenant, a
    browser cannot send a header, and the response was 422. An ngrok outage had been masking it
    -- the URL failed before it reached the gateway -- so the 422 was only found by bringing the
    tunnel up and trying one.
    """

    def test_the_tenant_travels_in_the_url(self) -> None:
        from cortex.notify.slack import report_url

        investigation = uuid.uuid4()
        url = report_url("https://example.ngrok-free.app", investigation, "acme")
        assert url == (
            f"https://example.ngrok-free.app/ui/investigations/{investigation}?tenant=acme"
        )

    def test_a_slug_needing_escaping_is_escaped(self) -> None:
        from cortex.notify.slack import report_url

        url = report_url("https://example.test", uuid.uuid4(), "acme corp/eu")
        assert "tenant=acme%20corp/eu" in url or "tenant=acme%20corp%2Feu" in url

    def test_no_tenant_leaves_the_url_bare(self) -> None:
        """Callers that have no tenant to hand must not emit `?tenant=None`."""
        from cortex.notify.slack import report_url

        url = report_url("https://example.test", uuid.uuid4())
        assert url is not None and "?" not in url

    def test_no_base_url_still_yields_no_link(self) -> None:
        """A message telling a colleague to open localhost is worse than one with no link: it
        looks like a broken feature rather than an unconfigured one."""
        from cortex.notify.slack import report_url

        assert report_url(None, uuid.uuid4(), "acme") is None


class TestAFalsePremiseIsSaidBeforeTheAnswer:
    """The page's most important line, when it happens.

    The live failure this whole path exists for: asked whether signups had fallen, the analyst
    led with "619 events for Aug 1-12 compared to July's total of 4,849", refuted it in the same
    sentence, and reached the real answer in the fourth bullet. Every claim was accurate and a
    reader who stopped after one line had been misinformed.
    """

    def test_a_false_premise_is_announced(self) -> None:
        html = _premise_html({"premise": "false", "premise_checked": "that signups fell"})
        assert "assumes something the evidence contradicts" in html
        assert "corrects it rather than answering as asked" in html
        assert "class=notice" in html

    def test_what_was_checked_is_shown(self) -> None:
        """The premise as the report understood it, so a reader can see whether the report and
        the question were even talking about the same assertion."""
        html = _premise_html({"premise": "false", "premise_checked": "that signups fell"})
        assert "Checked: that signups fell" in html

    def test_unverifiable_is_quieter_than_false(self) -> None:
        """ "The evidence contradicts your assertion" and "the evidence cannot settle it" are
        different messages, and giving them the same weight would spend the reader's attention
        on the weaker one."""
        html = _premise_html({"premise": "unverifiable"})
        assert "neither confirm nor contradict" in html
        assert "class=notice" not in html

    def test_a_holding_premise_says_nothing(self) -> None:
        """A reader already assumes the question's premise holds, so announcing it would make
        the one case that matters harder to spot."""
        assert _premise_html({"premise": "holds"}) == ""

    def test_the_ordinary_report_says_nothing(self) -> None:
        assert _premise_html({}) == ""
        assert _premise_html({"premise": "none_asserted"}) == ""

    def test_it_is_escaped(self) -> None:
        html = _premise_html({"premise": "false", "premise_checked": "<script>alert(1)</script>"})
        assert "<script>" not in html and "&lt;script&gt;" in html

    def test_it_appears_before_the_answer(self) -> None:
        """Above, not below. Burying it is the failure the field exists to prevent."""
        from services.gateway.views import _report_html

        class _Row:
            body = {
                "premise": "false",
                "premise_checked": "that signups fell",
                "executive_summary": [{"text": "No, they did not.", "evidence_ids": []}],
                "sources": [],
            }
            gate_rejections: list = []
            verifier_rejections: list = []

        html = _report_html(_Row())
        assert html.index("assumes something the evidence contradicts") < html.index("<h2>Answer")
