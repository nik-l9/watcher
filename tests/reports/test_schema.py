"""Report schema.

The property this file defends: **an uncited claim cannot be represented.** Asking
a model to "cite your sources" in prose yields text that looks cited; requiring
`evidence_ids` on a validated object means the ungrounded case is unconstructable.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from pydantic import ValidationError

from cortex.reports.schema import (
    ChartAnnotation,
    ChartPoint,
    ChartSeries,
    ChartSpec,
    ChartType,
    Claim,
    Confidence,
    DataQualityNote,
    Finding,
    Hypothesis,
    InvestigationReport,
    PremiseVerdict,
    Recommendation,
    Risk,
    Verdict,
    llm_report_schema,
)


def _claim(text: str = "Signups fell 18%.", n: int = 1) -> Claim:
    return Claim(text=text, evidence_ids=[uuid.uuid4() for _ in range(n)])


def _report(**overrides: object) -> InvestigationReport:
    base: dict[str, object] = {
        "question": "Why did signups fall?",
        "executive_summary": [_claim()],
    }
    return InvestigationReport(**(base | overrides))  # type: ignore[arg-type]


class TestClaimsMustCiteEvidence:
    def test_claim_requires_at_least_one_evidence_id(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="Signups fell 18%.", evidence_ids=[])

    def test_claim_cannot_omit_evidence_ids(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="Signups fell 18%.")  # type: ignore[call-arg]

    def test_recommendation_requires_evidence(self) -> None:
        """An uncited recommendation is a guess wearing an imperative."""
        with pytest.raises(ValidationError):
            Recommendation(action="Roll back", rationale="It broke", evidence_ids=[])

    def test_chart_requires_evidence(self) -> None:
        """A chart is an assertion about data, so it obeys the same rule."""
        with pytest.raises(ValidationError):
            ChartSpec(
                type=ChartType.LINE,
                title="Signups",
                series=[ChartSeries(name="signups", points=[ChartPoint(x="2026-07-01", y=10.0)])],
                evidence_ids=[],
            )

    def test_report_requires_a_summary(self) -> None:
        with pytest.raises(ValidationError):
            InvestigationReport(question="Why?", executive_summary=[])

    def test_finding_requires_a_claim(self) -> None:
        with pytest.raises(ValidationError):
            Finding(title="Mobile conversion", claims=[])


class TestInlineCitationsRejected:
    """A model writing "[evidence: abc]" into the prose is trying to satisfy the
    citation requirement without populating the field the gate checks."""

    @pytest.mark.parametrize(
        "text",
        [
            "Signups fell 18% [evidence: 3f2a-11bb].",
            "Mobile dropped [source: ga4].",
            "Conversion fell (evidence 4).",
            "Traffic rose, evidence_id: abc-123",
            "Signups fell [EVIDENCE: X].",
        ],
    )
    def test_rejects_inline_citation_markers(self, text: str) -> None:
        with pytest.raises(ValidationError, match="inline citation"):
            Claim(text=text, evidence_ids=[uuid.uuid4()])

    def test_ordinary_prose_is_accepted(self) -> None:
        claim = Claim(
            text="Mobile conversion fell from 4.2% to 2.9% after deploy 91c3e.",
            evidence_ids=[uuid.uuid4()],
        )
        assert claim.text.endswith("91c3e.")


class TestStrictness:
    @pytest.mark.parametrize(
        ("model", "kwargs"),
        [
            (Claim, {"text": "x", "evidence_ids": [uuid.uuid4()]}),
            (Finding, {"title": "t", "claims": [_claim()]}),
            (Risk, {"description": "d"}),
            (DataQualityNote, {"note": "n"}),
        ],
    )
    def test_unknown_fields_are_refused(self, model: type, kwargs: dict) -> None:
        """A model inventing a field would otherwise have it silently dropped."""
        with pytest.raises(ValidationError):
            model(**kwargs, invented_field="x")

    def test_empty_claim_text_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="", evidence_ids=[uuid.uuid4()])


class TestHypothesisVerdicts:
    def test_supported_requires_supporting_evidence(self) -> None:
        with pytest.raises(ValidationError, match="must cite supporting evidence"):
            Hypothesis(statement="The deploy caused it", verdict=Verdict.SUPPORTED)

    def test_contradicted_requires_contradicting_evidence(self) -> None:
        with pytest.raises(ValidationError, match="must cite contradicting evidence"):
            Hypothesis(statement="Pricing caused it", verdict=Verdict.CONTRADICTED)

    def test_inconclusive_may_cite_nothing(self) -> None:
        """That is precisely what inconclusive means."""
        hypothesis = Hypothesis(statement="Seasonality", verdict=Verdict.INCONCLUSIVE)
        assert hypothesis.evidence_ids == set()

    def test_supported_with_evidence_is_accepted(self) -> None:
        evidence_id = uuid.uuid4()
        hypothesis = Hypothesis(
            statement="The deploy caused it",
            verdict=Verdict.SUPPORTED,
            supporting_evidence_ids=[evidence_id],
        )
        assert hypothesis.evidence_ids == {evidence_id}


class TestACauseCannotPostdateItsEffect:
    """Kepner-Tregoe's elimination test, as a type constraint.

    The failure this project spent a week on, in one sentence: a signup cessation dated
    2026-08-04 was attributed to a pageview collapse that began 2026-08-11, while pageviews ran
    at 14,197/day on 08-04. Every citation resolved, the verifier found nothing overstated, and
    the dates were never compared -- because nothing required them to be written down in a form
    that could be compared.
    """

    _EVIDENCE = staticmethod(lambda: [uuid.uuid4()])

    def test_the_real_wrong_answer_cannot_be_expressed(self) -> None:
        with pytest.raises(ValidationError, match="cannot post-date its effect"):
            Hypothesis(
                statement="The pageview collapse caused the signup cessation.",
                verdict=Verdict.SUPPORTED,
                supporting_evidence_ids=self._EVIDENCE(),
                cause_at=date(2026, 8, 11),
                effect_onset=date(2026, 8, 4),
            )

    def test_the_error_names_the_gap_and_the_fix(self) -> None:
        """Structured output retries on a validation error, so an error that explains itself
        produces a corrected report rather than the same one again."""
        with pytest.raises(ValidationError) as caught:
            Hypothesis(
                statement="A deploy caused it.",
                verdict=Verdict.SUPPORTED,
                supporting_evidence_ids=self._EVIDENCE(),
                cause_at=date(2026, 8, 11),
                effect_onset=date(2026, 8, 4),
            )
        message = str(caught.value)
        assert "7 day(s) earlier" in message
        assert "set verdict to 'contradicted'" in message
        # And the two escape hatches, so a redraft fixes the hypothesis rather than the field.
        assert "correct effect_onset from the evidence rather than removing it" in message
        assert "two findings and not one hypothesis" in message

    def test_inconclusive_is_refused_too(self) -> None:
        """An impossible cause is not undecided. Only `contradicted` is available."""
        with pytest.raises(ValidationError, match="cannot post-date its effect"):
            Hypothesis(
                statement="Maybe the deploy caused it.",
                verdict=Verdict.INCONCLUSIVE,
                cause_at=date(2026, 8, 11),
                effect_onset=date(2026, 8, 4),
            )

    def test_the_same_dates_are_accepted_as_contradicted(self) -> None:
        hypothesis = Hypothesis(
            statement="The pageview collapse caused the signup cessation.",
            verdict=Verdict.CONTRADICTED,
            contradicting_evidence_ids=self._EVIDENCE(),
            cause_at=date(2026, 8, 11),
            effect_onset=date(2026, 8, 4),
        )
        assert hypothesis.eliminated_by_onset

    def test_a_cause_preceding_its_effect_is_untouched(self) -> None:
        hypothesis = Hypothesis(
            statement="The reverse-proxy change broke tracking.",
            verdict=Verdict.SUPPORTED,
            supporting_evidence_ids=self._EVIDENCE(),
            cause_at=date(2026, 6, 15),
            effect_onset=date(2026, 6, 17),
        )
        assert not hypothesis.eliminated_by_onset

    def test_a_cause_on_the_same_day_is_allowed(self) -> None:
        """Same-day is not impossible -- a morning deploy explains an afternoon movement, and
        the separability question is a different one, handled by the identifiability gate."""
        hypothesis = Hypothesis(
            statement="The deploy caused it.",
            verdict=Verdict.SUPPORTED,
            supporting_evidence_ids=self._EVIDENCE(),
            cause_at=date(2026, 6, 17),
            effect_onset=date(2026, 6, 17),
        )
        assert not hypothesis.eliminated_by_onset

    def test_undated_hypotheses_behave_exactly_as_before(self) -> None:
        """The fields are optional, so a non-causal hypothesis is unaffected and no existing
        report becomes invalid."""
        hypothesis = Hypothesis(statement="Seasonality", verdict=Verdict.INCONCLUSIVE)
        assert hypothesis.cause_at is None
        assert hypothesis.effect_onset is None
        assert not hypothesis.eliminated_by_onset

    def test_one_date_alone_cannot_eliminate(self) -> None:
        """A comparison needs both. Half the information must not produce a verdict."""
        for kwargs in ({"cause_at": date(2026, 8, 11)}, {"effect_onset": date(2026, 8, 4)}):
            hypothesis = Hypothesis(
                statement="A deploy caused it.",
                verdict=Verdict.SUPPORTED,
                supporting_evidence_ids=self._EVIDENCE(),
                **kwargs,
            )
            assert not hypothesis.eliminated_by_onset


class TestCitationCollection:
    def test_collects_ids_from_every_section(self) -> None:
        """The gate resolves citations in one query, so nothing may be missed."""
        ids = {
            name: uuid.uuid4()
            for name in (
                "summary",
                "finding",
                "hypothesis",
                "chart",
                "annotation",
                "rec",
                "risk",
                "note",
            )
        }
        report = _report(
            executive_summary=[Claim(text="Summary claim.", evidence_ids=[ids["summary"]])],
            findings=[
                Finding(
                    title="Mobile",
                    claims=[Claim(text="Mobile fell.", evidence_ids=[ids["finding"]])],
                )
            ],
            hypotheses=[
                Hypothesis(
                    statement="Deploy",
                    verdict=Verdict.SUPPORTED,
                    supporting_evidence_ids=[ids["hypothesis"]],
                )
            ],
            charts=[
                ChartSpec(
                    type=ChartType.LINE,
                    title="Signups",
                    series=[ChartSeries(name="s", points=[ChartPoint(x="2026-07-01", y=1.0)])],
                    annotations=[
                        ChartAnnotation(
                            x="2026-07-20", label="deploy", evidence_id=ids["annotation"]
                        )
                    ],
                    evidence_ids=[ids["chart"]],
                )
            ],
            recommendations=[
                Recommendation(action="Roll back", rationale="It broke", evidence_ids=[ids["rec"]])
            ],
            risks=[Risk(description="Sampled data", evidence_ids=[ids["risk"]])],
            data_quality=[DataQualityNote(note="Stale sync", evidence_ids=[ids["note"]])],
        )
        assert report.cited_evidence_ids() == set(ids.values())

    def test_claim_count_covers_summary_and_findings(self) -> None:
        report = _report(
            executive_summary=[_claim(), _claim()],
            findings=[Finding(title="f", claims=[_claim(), _claim(), _claim()])],
        )
        assert report.claim_count() == 5


class TestChartSeries:
    def test_missing_point_is_none_not_zero(self) -> None:
        """A zero would draw a line to the floor and read as a collapse."""
        series = ChartSeries(name="signups", points=[ChartPoint(x="2026-07-01", y=None)])
        assert series.points[0].y is None

    def test_series_requires_a_point(self) -> None:
        with pytest.raises(ValidationError):
            ChartSeries(name="signups", points=[])


class TestStructuredOutputCompatibility:
    """The drafting schema must be one the API will accept.

    Structured outputs support only 0 or 1 as an array minimum, and a positional
    tuple field serializes to `prefixItems` with `minItems: 2`. Without this test
    that constraint is only discovered by a 400 at the moment a report is drafted --
    after the investigation has already spent its tokens.
    """

    #: A proxy tripwire, not the real limit. The limit is grammar complexity, which
    #: is not observable from here; serialized size is what correlates with it. The
    #: schema that failed was 9593 bytes and the same schema without charts -- which
    #: compiles -- is 5902, so this sits above the working size and below the broken
    #: one. It exists so growth is caught by a test rather than by a live run that
    #: fails after spending its tokens.
    MAX_SCHEMA_BYTES = 7000

    def test_the_schema_stays_within_the_grammar_budget(self) -> None:
        import json

        size = len(json.dumps(llm_report_schema()))
        assert size < self.MAX_SCHEMA_BYTES, (
            f"{size} bytes; structured outputs rejected 9593 with "
            f"'the compiled grammar is too large'"
        )

    def test_no_array_bound_the_api_rejects(self) -> None:
        offenders: list[str] = []

        def walk(node: object, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"minItems", "maxItems"} and value not in (0, 1):
                        offenders.append(f"{path}/{key}={value}")
                    if key == "prefixItems":
                        offenders.append(f"{path}/prefixItems")
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        walk(llm_report_schema(), "")
        assert offenders == [], offenders


class TestLLMSchema:
    def test_sources_are_not_model_authored(self) -> None:
        """A model-written source could carry an invented permalink into the one
        section a reader treats as independently verifiable."""
        assert "sources" not in llm_report_schema()["properties"]

    def test_charts_are_not_drafted_inline(self) -> None:
        """Two reasons that agree.

        The compiled grammar exceeds the structured-output limit with charts included,
        which failed every investigation at the drafting step. And a chart is data:
        asking for series inline invites invented points, where building them from
        stored evidence cannot produce a number that was never observed.
        """
        assert "charts" not in llm_report_schema()["properties"]

    def test_charts_remain_part_of_the_report(self) -> None:
        """Only the drafting call is affected -- the gate and view still handle them."""
        assert "charts" in InvestigationReport.model_json_schema()["properties"]

    def test_every_other_section_is_offered(self) -> None:
        properties = llm_report_schema()["properties"]
        for section in (
            "executive_summary",
            "findings",
            "hypotheses",
            "confidence",
            "risks",
            "recommendations",
            "data_quality",
        ):
            assert section in properties, section

    def test_report_still_accepts_derived_sources(self) -> None:
        """The gate populates them after drafting."""
        from cortex.reports.schema import Source

        report = _report(
            sources=[Source(evidence_id=uuid.uuid4(), tool_name="ga4", capability="get_sessions")]
        )
        assert report.sources[0].tool_name == "ga4"


class TestDefaults:
    def test_confidence_defaults_to_medium(self) -> None:
        """Not high: a default that overstates certainty is the wrong direction."""
        assert _report().confidence is Confidence.MEDIUM

    def test_optional_sections_default_empty(self) -> None:
        report = _report()
        assert report.findings == []
        assert report.charts == []
        assert report.sources == []


class TestDefPruning:
    """Definitions a removed property referenced still cost grammar."""

    def test_chart_definitions_are_gone_transitively(self) -> None:
        """ChartSpec is reachable only through the removed property, and ChartSeries
        only through ChartSpec. A single-pass prune would keep the latter."""
        defs = set(llm_report_schema()["$defs"])
        assert not {"ChartSpec", "ChartSeries", "ChartPoint", "ChartAnnotation", "ChartType"} & defs
        assert "Source" not in defs

    def test_definitions_still_referenced_are_kept(self) -> None:
        defs = set(llm_report_schema()["$defs"])
        assert {"Claim", "Finding", "Hypothesis", "Recommendation", "Risk"} <= defs

    def test_no_dangling_references_remain(self) -> None:
        """The failure a careless prune produces: "reference to non-existent
        definition", which is worse than the bytes it saves."""
        schema = llm_report_schema()
        defined = set(schema["$defs"])

        referenced: set[str] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "$ref" and isinstance(value, str):
                        referenced.add(value.rsplit("/", 1)[-1])
                    else:
                        walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(schema)
        assert referenced <= defined, referenced - defined


class TestTheQuestionsOwnAssertionIsAField:
    """A keyword list kept standing in for this, and got it wrong.

    A report whose first two words were "No -- ... not a real drop" scored as *never having
    refuted the premise*, because the twelve accepted denial phrasings included "did not drop"
    and "have not dropped" but not "not a real drop". At the same time the placement dimension,
    reading the same report, scored full marks for refuting it in the executive summary. Two
    dimensions contradicting each other about one sentence.

    So the report states its verdict. The same move that made Kepner-Tregoe's elimination rule
    structural rather than a matter of phrasing.
    """

    def test_a_report_asserts_nothing_about_the_premise_by_default(self) -> None:
        """Most questions assert nothing checkable, and a default of `false` would make every
        lookup look like a refutation."""
        report = InvestigationReport(
            question="Which plan is Acme on?",
            executive_summary=[Claim(text="Enterprise claim.", evidence_ids=[uuid.uuid4()])],
        )
        assert report.premise is PremiseVerdict.NONE_ASSERTED
        assert report.premise_checked == ""

    def test_a_refuted_premise_is_recorded_as_such(self) -> None:
        report = InvestigationReport(
            question="Why did signups fall last month?",
            executive_summary=[Claim(text="No -- not a real drop.", evidence_ids=[uuid.uuid4()])],
            premise=PremiseVerdict.FALSE,
            premise_checked="that signups fell in August",
        )
        assert report.premise is PremiseVerdict.FALSE
        assert report.premise_checked == "that signups fell in August"

    def test_unverifiable_is_distinct_from_false(self) -> None:
        """ "The evidence contradicts your assertion" and "the evidence cannot settle it" are
        different answers, and collapsing them would let a refusal read as a refutation."""
        assert PremiseVerdict.UNVERIFIABLE is not PremiseVerdict.FALSE
        assert {v.value for v in PremiseVerdict} == {
            "holds",
            "false",
            "unverifiable",
            "none_asserted",
        }

    def test_an_unknown_verdict_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            InvestigationReport(
                question="Why did signups fall?",
                executive_summary=[Claim(text="They did not.", evidence_ids=[uuid.uuid4()])],
                premise="probably-false",  # type: ignore[arg-type]
            )


class TestAFieldCapMustNotDiscardAnInvestigation:
    """`premise_checked` shipped with a 500-character cap and threw away a completed run.

    The model wrote "The question asserts a 3%... general, tenant-wide dip", the repair retry
    fired twice with an error naming the limit, and it returned an over-limit value both times.
    A constraint the model cannot satisfy on retry is not enforcing brevity — it is discarding
    investigations whose evidence has already been gathered and paid for.
    """

    def test_the_cap_matches_its_neighbours(self) -> None:
        """500 was out of line: a hypothesis statement gets 1,000 and its reasoning 2,000."""
        from annotated_types import MaxLen

        def cap(model, field: str) -> int:
            return next(
                m.max_length for m in model.model_fields[field].metadata if isinstance(m, MaxLen)
            )

        assert cap(InvestigationReport, "premise_checked") == 1000
        assert cap(Hypothesis, "statement") == 1000

    def test_a_realistic_premise_restatement_fits(self) -> None:
        """The length the model actually writes, not the length that seemed tidy."""
        stated = (
            "The question asserts a 3% week-over-week decline in enterprise signups for the "
            "week of 22 May 2026, which presupposes both that an enterprise segment is "
            "separable in the available data and that the decline is specific to it rather "
            "than a general, tenant-wide dip. "
        ) * 2
        assert 500 < len(stated) <= 1000
        report = InvestigationReport(
            question="Why did enterprise signups fall 3%?",
            executive_summary=[Claim(text="They did not.", evidence_ids=[uuid.uuid4()])],
            premise=PremiseVerdict.FALSE,
            premise_checked=stated,
        )
        assert report.premise_checked == stated


class TestAJudgementComesAfterItsInput:
    """Field order is behaviour, not style, and this is the third time it has bitten.

    Structured output is one left-to-right pass over the schema's properties, so a field placed
    above its own inputs has to be answered before they exist. The verifier's `VERDICT_SCHEMA`
    was reordered so `reason` precedes `verdict` after that discovery. The executive summary and
    its confidence were moved to the end for the same reason, after four live attempts wrote
    "placeholder" into a summary they could not yet write.

    `premise` had the verdict first. A measured run of five attempts at
    `partial_month_false_premise` showed what that produces: one emitted `premise: holds` and
    then wrote *"No — on the days for which we actually have data, signups did not fall"*. The
    prose refuted the premise and the field said it stood, because a one-token enum was decided
    before a word of reasoning about it existed — and `accuracy` reads the field, so a correct
    investigation was scored zero.

    Asserted here because it is free to assert and the failure is invisible otherwise: nothing
    about a wrongly-ordered schema fails until a live model fills it in.
    """

    #: Each pair is (input, judgement). The judgement must not be answerable before its input.
    ORDERED_PAIRS = (
        ("premise_checked", "premise"),
        ("findings", "executive_summary"),
        ("executive_summary", "confidence"),
    )

    def _properties(self) -> list[str]:
        return list(InvestigationReport.model_json_schema()["properties"])

    @pytest.mark.parametrize(("earlier", "later"), ORDERED_PAIRS)
    def test_the_input_is_declared_before_the_judgement(self, earlier: str, later: str) -> None:
        properties = self._properties()
        assert properties.index(earlier) < properties.index(later), (
            f"{later!r} is declared before {earlier!r}, so a single-pass decoder has to answer "
            f"it before {earlier!r} exists. This is how `premise` came to contradict the "
            "summary underneath it."
        )

    #: The only fields allowed after `confidence`, with the reason. `sources` is a list of the
    #: evidence already cited above -- a transcription, not a judgement -- so answering it last
    #: costs nothing. Anything else added here has to justify being answered after the summary.
    _AFTER_THE_JUDGEMENTS = ("sources",)

    def test_nothing_requiring_judgement_follows_the_confidence(self) -> None:
        """The summary and its confidence are judgements over everything above them, so a field
        inserted after them is a field answered after the report has been concluded."""
        properties = self._properties()
        summary = properties.index("executive_summary")
        assert properties[summary + 1] == "confidence", (
            "the confidence must be answered immediately after the summary it grades"
        )
        assert tuple(properties[summary + 2 :]) == self._AFTER_THE_JUDGEMENTS
