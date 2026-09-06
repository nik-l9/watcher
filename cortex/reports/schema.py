"""Investigation report schema.

The report format from the product doc — executive summary, evidence, charts,
confidence, risks, recommendations, sources — expressed as types rather than as a
prompt instruction.

The load-bearing design decision: a claim is a *structured object carrying evidence
ids*, not a sentence. Asking a model to "cite your sources" in prose produces text
that looks cited; requiring `evidence_ids` on a validated object means an
uncited claim cannot be represented at all. Everything the grounding gate does
downstream depends on that.

Validation here is deliberately shallow — it checks shape, not truth. Whether the
cited ids actually exist and belong to this investigation is
`cortex.reports.gate`'s job, because that requires the database.
"""

from __future__ import annotations

import datetime
import enum
import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Confidence(enum.StrEnum):
    """How much weight the analyst puts behind a conclusion.

    A coarse scale on purpose. A model asked for a percentage will produce one to
    two decimal places and mean nothing by it; four levels can be defined and held
    to.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Whether the question's own assertion survived checking.
#
# **Exists because a keyword list kept standing in for this, and got it wrong.** A report whose
# first two words were "No -- ... not a real drop" scored as *never having refuted the premise*,
# because the twelve accepted denial phrasings included "did not drop" and "have not dropped" but
# not "not a real drop". At the same time the placement dimension, reading the same report, scored
# full marks for refuting it in the executive summary -- two dimensions contradicting each other
# about one sentence. That is the fourth defect of this shape here: a scorer searching prose for
# something the report could simply have stated.
#
# So the report states it, as `Hypothesis.cause_at` did for the elimination rule.
#
# The reasoning is a comment rather than a docstring on purpose: pydantic copies a class
# docstring into the JSON schema, where it is paid for in compiled grammar and counted against
# the 7,000-byte tripwire in `TestStructuredOutputCompatibility`. This one cost 700 bytes and
# broke it.
#
#   holds          -- the assertion is supported by the evidence
#   false          -- the evidence contradicts it, including when an apparent movement is an
#                     artefact of an incomplete period or a changed definition
#   unverifiable   -- the evidence can neither confirm nor contradict it
#   none_asserted  -- the question asserted nothing checkable; the ordinary case and the default
class PremiseVerdict(enum.StrEnum):
    HOLDS = "holds"
    FALSE = "false"
    UNVERIFIABLE = "unverifiable"
    NONE_ASSERTED = "none_asserted"


class Verdict(enum.StrEnum):
    """The outcome of testing one hypothesis against evidence."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class ChartType(enum.StrEnum):
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    AREA = "area"
    SCATTER = "scatter"


class _Strict(BaseModel):
    """Reject unknown fields everywhere.

    A model that invents `sources_summary` alongside `sources` would otherwise have
    its invention silently dropped, and the report would render as though the field
    had never been asked for.
    """

    model_config = ConfigDict(extra="forbid")


# A claim must cite at least one piece of evidence. This is the type-level
# expression of "no claim renders without a resolvable evidence_id".
EvidenceIds = Annotated[list[uuid.UUID], Field(min_length=1)]


class Claim(_Strict):
    """One assertion, and the observations that support it.

    `text` is the sentence a reader sees. `evidence_ids` are what make it citable.
    The pairing is the whole point: prose and its grounding travel together, so the
    gate can drop one without having to parse the other.
    """

    text: str = Field(min_length=1, max_length=2000)
    evidence_ids: EvidenceIds

    @field_validator("text")
    @classmethod
    def _reject_a_stub(cls, value: str) -> str:
        """A claim has to assert something. Stubs were reaching readers.

        **The failure.** Four of eleven captured bundles contained a summary claim whose entire
        text was "placeholder", across two scenarios. Three of the four *passed*: the adversarial
        verifier deleted the stub as unsupported and the report shipped shorter, which is what
        several `draft_reliability` scores of 0.83-0.90 actually were -- not a strict verifier.
        The fourth had no other summary claim, so deleting it emptied the summary, and a report
        carrying two findings, seven claims, three judged hypotheses and a correct premise
        verdict was discarded whole.

        **Why not a character floor.** That was the first attempt and it does not work. Across
        261 captured claims the shortest legitimate one was 55 characters, which makes a floor
        of 25 look safe -- until a test asserted `"Signups fell 18%."`, 17 characters, exactly
        the terse factual claim a good analyst writes. Six characters separate that from
        `"placeholder"` at 11, so no length threshold separates an assertion from a stub
        without rejecting real claims.

        **Why the floor is two words and not three.** Three was the next attempt, on the
        reasoning that an assertion needs a subject, a verb and something asserted. That is
        wrong about English: intransitive verbs need no object, so `"Signups fell."` and
        `"Traffic doubled."` are complete assertions in two words, and the rule rejected 25
        real claims across the suite to catch stubs that were all one word.

        **What this does and does not catch.** Every stub observed in a live run was one word
        -- all four were literally "placeholder" -- and one word cannot assert anything, so
        this catches all of them with no false positives on real claims. It would not catch a
        two-word stub like "see findings". That is deliberate: the alternative is a list of
        forbidden phrases, and the real protection against a stub is that the model no longer
        has a reason to write one, now that the summary is generated after the findings it
        summarises. This is the backstop for that, not the mechanism.

        A validator rather than schema grammar because `minLength` is the only length
        constraint the grammar can express, and length is the constraint that does not work.
        So the model is not stopped up front; the repair retry catches it, is handed
        "a claim must assert something", and can fix it in one turn.
        """
        if len(value.split()) < 2:
            raise ValueError(f"a claim must assert something, and one word cannot: {value!r}")
        return value

    @field_validator("text")
    @classmethod
    def _reject_inline_citations(cls, value: str) -> str:
        """Citations belong in evidence_ids, not in the prose.

        A model that writes "[evidence: abc-123]" into the text is trying to satisfy
        the citation requirement without populating the field the gate checks —
        which would produce a report that looks grounded and is not.
        """
        lowered = value.lower()
        for marker in ("[evidence:", "[source:", "evidence_id:", "(evidence "):
            if marker in lowered:
                raise ValueError(
                    f"claim text contains an inline citation ({marker!r}); cite via "
                    "evidence_ids instead"
                )
        return value


class Finding(_Strict):
    """A substantive result of the investigation.

    Findings are the body of the report. Each carries its own claims, so the gate
    operates at claim granularity — one unsupported sentence is removed rather than
    the whole finding discarded.
    """

    title: str = Field(min_length=1, max_length=200)
    claims: list[Claim] = Field(min_length=1)
    confidence: Confidence = Confidence.MEDIUM

    @property
    def evidence_ids(self) -> set[uuid.UUID]:
        return {eid for claim in self.claims for eid in claim.evidence_ids}


class Hypothesis(_Strict):
    """A candidate explanation and how it fared.

    Recorded even when contradicted. A report that shows what was ruled out, and on
    what basis, is far more trustworthy than one that presents only the surviving
    story — and it stops the analyst from quietly discarding inconvenient evidence.
    """

    statement: str = Field(min_length=1, max_length=1000)
    verdict: Verdict
    supporting_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    contradicting_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=2000)

    #: When the proposed cause happened, where the hypothesis names one.
    cause_at: datetime.date | None = Field(
        default=None,
        description=(
            "The date the proposed cause happened -- the deploy, the campaign launch, the "
            "pricing change. Required whenever this hypothesis proposes a cause for a dated "
            "movement, because a cause that post-dates its effect is impossible and that is "
            "checked in code rather than left to judgement."
        ),
    )
    #: When the effect being explained began.
    effect_onset: datetime.date | None = Field(
        default=None,
        description=(
            "The date the movement being explained began. Take it from the evidence -- a "
            "series' `movement.level_shifts[].at`, or the last normal bucket before a gap -- "
            "never by estimating from a chart."
        ),
    )

    @model_validator(mode="after")
    def _verdict_requires_evidence(self) -> Hypothesis:
        """A verdict of supported or contradicted must rest on something.

        Only INCONCLUSIVE may cite nothing — that is precisely what it means.
        """
        if self.verdict is Verdict.SUPPORTED and not self.supporting_evidence_ids:
            raise ValueError("a supported hypothesis must cite supporting evidence")
        if self.verdict is Verdict.CONTRADICTED and not self.contradicting_evidence_ids:
            raise ValueError("a contradicted hypothesis must cite contradicting evidence")
        return self

    @model_validator(mode="after")
    def _a_cause_cannot_postdate_its_effect(self) -> Hypothesis:
        """Kepner-Tregoe's elimination test, as a type constraint.

        **A candidate cause whose own onset is later than the deviation's onset is eliminated,
        not ranked lower.** Enforced here rather than checked later because a report that has
        already been drafted around an impossible cause is a report whose every other claim was
        reasoned from it — correcting the verdict afterwards leaves the reasoning in place.

        This is the check that would have prevented the failure this project spent a week on:
        a signup cessation dated 2026-08-04 was attributed to a pageview collapse that began on
        2026-08-11, while pageviews ran at 14,197/day on 08-04. Every citation resolved. The
        verifier found nothing overstated. The dates were simply never compared, because nothing
        required them to be written down in a form that could be compared.

        Refused rather than downgraded, and the message says why, so a redraft can fix the
        hypothesis instead of the field: structured output retries on a validation error, and an
        error that explains itself produces a corrected report rather than the same one again.
        """
        if self.cause_at is None or self.effect_onset is None:
            return self
        if self.cause_at > self.effect_onset and self.verdict is not Verdict.CONTRADICTED:
            raise ValueError(
                f"this hypothesis proposes a cause on {self.cause_at.isoformat()} for a "
                f"movement that began on {self.effect_onset.isoformat()}, which is "
                f"{(self.cause_at - self.effect_onset).days} day(s) earlier. A cause cannot "
                "post-date its effect, so this candidate is eliminated: set verdict to "
                "'contradicted' citing the evidence that established the onset. If you believe "
                "the onset is wrong, correct effect_onset from the evidence rather than "
                "removing it. If the movement and the cause are genuinely different events, "
                "they are two findings and not one hypothesis."
            )
        return self

    @property
    def eliminated_by_onset(self) -> bool:
        """Whether this was ruled out because its cause post-dates its effect.

        Distinguished from other contradictions because it is the strongest kind: it says
        *impossible* rather than *unsupported*, and it was decided by arithmetic rather than by
        a judgement anyone can disagree with.
        """
        return (
            self.cause_at is not None
            and self.effect_onset is not None
            and self.cause_at > self.effect_onset
        )

    @property
    def evidence_ids(self) -> set[uuid.UUID]:
        return set(self.supporting_evidence_ids) | set(self.contradicting_evidence_ids)


class ChartPoint(_Strict):
    """One point on a series.

    Named fields rather than an (x, y) tuple. A positional pair serializes to a JSON
    schema with `prefixItems` and `minItems: 2`, which structured outputs reject —
    only 0 and 1 are supported minimums. Naming the axes also removes the ambiguity
    of which slot may be null.
    """

    x: str = Field(min_length=1, max_length=120)
    #: None for a genuine gap — never 0, which would draw a line to the floor and
    #: read as a collapse.
    y: float | None = None


class ChartSeries(_Strict):
    name: str = Field(min_length=1, max_length=120)
    points: list[ChartPoint] = Field(min_length=1)


class ChartAnnotation(_Strict):
    """A marker on the x axis — a deploy, a release, a campaign start.

    This is what makes a causal story visible rather than merely stated: the reader
    sees the line bend at the deploy instead of being told that it did.
    """

    x: str
    label: str = Field(min_length=1, max_length=120)
    evidence_id: uuid.UUID


class ChartSpec(_Strict):
    """A chart as data, rendered by the frontend.

    Carries evidence ids like any other claim: a chart is an assertion about the
    data, so it is subject to the same grounding rule as a sentence.
    """

    type: ChartType
    title: str = Field(min_length=1, max_length=200)
    x_label: str = Field(default="", max_length=120)
    y_label: str = Field(default="", max_length=120)
    series: list[ChartSeries] = Field(min_length=1)
    annotations: list[ChartAnnotation] = Field(default_factory=list)
    evidence_ids: EvidenceIds

    @property
    def all_evidence_ids(self) -> set[uuid.UUID]:
        return set(self.evidence_ids) | {a.evidence_id for a in self.annotations}


class Recommendation(_Strict):
    """What to do about it.

    Cites evidence for the same reason a finding does. "Roll back the onboarding
    modal" is a claim about cause, and an uncited recommendation is a guess wearing
    an imperative.
    """

    action: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1000)
    evidence_ids: EvidenceIds
    # Ordering hint for the reader; 1 is most urgent.
    priority: int = Field(default=2, ge=1, le=5)


class Risk(_Strict):
    """Something that could make the conclusion wrong.

    Distinct from low confidence: confidence is how sure the analyst is, a risk is
    the specific reason it might be mistaken. Stale connectors, sampled data and
    confounded time windows all belong here.
    """

    description: str = Field(min_length=1, max_length=1000)
    # Optional, because some risks are about absent data and so cite nothing.
    evidence_ids: list[uuid.UUID] = Field(default_factory=list)


class Source(_Strict):
    """A human-followable pointer to one observation.

    Derived from evidence rows rather than authored by the model, so a source cannot
    be invented. Rendered in the report's Sources section.
    """

    evidence_id: uuid.UUID
    tool_name: str
    capability: str
    source_ref: str | None = None
    observed_at: str | None = None
    from_cache: bool = False


class DataQualityNote(_Strict):
    """A disclosure about the data itself.

    Sampling, thresholding, and staleness from a nightly sync. Surfacing these is
    the difference between a real analyst and a confident one: a sampled figure
    presented as exact is a grounding failure even when the API call was real.
    """

    note: str = Field(min_length=1, max_length=500)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list)


class InvestigationReport(_Strict):
    """The complete answer.

    Mirrors the doc's required format. `sources` is populated from the evidence
    store rather than by the model — see cortex.reports.gate.
    """

    # ------------------------------------------------------------------ field order
    #
    # Ordered so that nothing is written before the thing it depends on, which is a
    # correctness property here rather than a style one. Structured output is generated in
    # one left-to-right pass over these fields, so a field placed above its own inputs has
    # to be answered before those inputs exist.
    #
    # `executive_summary` used to sit second, above `findings`, and the model was being asked
    # to summarise findings it had not written yet. On four of eleven captured attempts it did
    # what anyone would do with a form field it cannot yet fill: wrote "placeholder" and moved
    # on -- and single-pass generation gives it no way back. Every one of those four was on a
    # scenario whose conclusion is only knowable after enumerating what was checked
    # (`insufficient_evidence`, `tempting_coincidence`), which is exactly where writing the
    # summary first is impossible rather than merely awkward.
    #
    # So the summary and the confidence in it now come last: both are judgements over
    # everything above them.
    #
    # **And `premise_checked` now comes before `premise`**, for the third time this ordering
    # rule has had to be applied. The verifier's `VERDICT_SCHEMA` was reordered so `reason`
    # precedes `verdict` after the same discovery; this field had the verdict first, and a
    # measured run showed exactly what that produces. Across five attempts at
    # `partial_month_false_premise`, one emitted `premise: holds` and then wrote *"No — on the
    # days for which we actually have data, signups did not fall"*. The prose refuted the
    # premise and the field said it stood, because a one-token enum was decided before a word
    # of reasoning about it existed -- and `accuracy` reads the field, so a correct
    # investigation scored zero.
    #
    # The original argument for the old order was that "the method establishes the effect
    # before explaining it". That is the right rule for an *investigation* and the wrong one
    # for a *generation*: this field is not the effect, it is a judgement about the effect, and
    # its input is the sentence underneath it.
    #
    # Whether the provider's grammar honours this order strictly is not something this comment
    # can assert -- it is a claim about the constrained decoder, verifiable only by a live run.
    # The `Claim.text` floor is the protection that does not depend on it being true.
    question: str = Field(min_length=1, max_length=2000)
    #: The assertion the report tested, so a reader can see what was checked -- and so the
    #: verdict below is written after its own input rather than before it.
    #:
    #: 1,000 rather than the 500 this shipped with. A live attempt was **discarded** over the cap:
    #: the model wrote "The question asserts a 3%... general, tenant-wide dip", the repair retry
    #: fired twice with an error naming the limit, and it returned a value over the limit both
    #: times. A constraint the model cannot satisfy on retry is not enforcing brevity, it is
    #: throwing away completed investigations -- and 500 was out of line with its neighbours
    #: anyway, where a hypothesis statement gets 1,000 and its reasoning 2,000.
    premise_checked: str = Field(default="", max_length=1000)
    #: Whether the question's own assertion survived checking.
    #: Terse on purpose. The full instruction lives in `cortex.reports.shape`, which reaches the
    #: model as prose rather than as grammar -- a long description here is paid for twice, once
    #: in the compiled grammar and once against the 7,000-byte tripwire that structured outputs
    #: enforces from the other side. The first draft of this field cost 1,062 bytes and broke it.
    premise: PremiseVerdict = Field(
        default=PremiseVerdict.NONE_ASSERTED,
        description="Verdict on what the question itself asserted.",
    )
    findings: list[Finding] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    data_quality: list[DataQualityNote] = Field(default_factory=list)
    #: Last on purpose, and last of the model-written fields: a summary of findings cannot be
    #: written before the findings. See the field-order note at the top of this class.
    executive_summary: list[Claim] = Field(min_length=1)
    confidence: Confidence = Confidence.MEDIUM
    sources: list[Source] = Field(default_factory=list)

    def cited_evidence_ids(self) -> set[uuid.UUID]:
        """Every evidence id the report references, anywhere.

        The gate uses this to resolve citations in one query rather than walking the
        tree repeatedly.
        """
        ids: set[uuid.UUID] = set()
        for claim in self.executive_summary:
            ids |= set(claim.evidence_ids)
        for finding in self.findings:
            ids |= finding.evidence_ids
        for hypothesis in self.hypotheses:
            ids |= hypothesis.evidence_ids
        for chart in self.charts:
            ids |= chart.all_evidence_ids
        for recommendation in self.recommendations:
            ids |= set(recommendation.evidence_ids)
        for risk in self.risks:
            ids |= set(risk.evidence_ids)
        for note in self.data_quality:
            ids |= set(note.evidence_ids)
        return ids

    def claim_count(self) -> int:
        """Total claims, for the eval suite's grounding score."""
        return len(self.executive_summary) + sum(len(f.claims) for f in self.findings)


def llm_report_schema() -> dict:
    """JSON schema handed to the model when it drafts a report.

    Two sections are withheld from the model.

    **Sources** are derived from the evidence store. Letting a model author them would
    allow an invented permalink to appear in the one section a reader treats as
    verifiable.

    **Charts** are withheld for two reasons that happen to agree. Structured outputs
    compile the schema into a grammar, and the full report with charts exceeds the
    limit — "the compiled grammar is too large" — which failed every investigation at
    the drafting step. Charts alone account for it: without them the same schema
    compiles. And a chart is data, not prose. Asking for series inline invites invented
    points, whereas building them from stored evidence cannot produce a number that was
    never observed. M5 therefore generates charts in a dedicated pass over the
    evidence, which keeps this grammar small as a side effect.

    `ChartSpec` stays in the model: the gate and the report view already handle charts,
    and only the drafting call is affected.
    """
    schema = InvestigationReport.model_json_schema()
    properties = schema.get("properties", {})
    for withheld in ("sources", "charts"):
        properties.pop(withheld, None)
    # Removing a property does not remove the definitions it referenced, and those
    # still count against the grammar — the chart definitions are the largest group in
    # the schema. Reachability has to be transitive: ChartSpec is only reachable
    # through the removed property, but ChartSeries is reachable only through
    # ChartSpec.
    return _prune_unreachable_defs(schema)


def _prune_unreachable_defs(schema: dict) -> dict:
    """Drop `$defs` nothing reaches from the root.

    Structured outputs compile the whole schema, so a definition left behind by a
    removed property still costs grammar. Reachability is computed transitively to a
    fixed point: a wrong pass produces "reference to non-existent definition", which
    is a worse failure than the bytes it saves, so definitions are only dropped once
    nothing reachable mentions them.
    """
    defs = schema.get("$defs")
    if not defs:
        return schema

    def refs_in(node: object) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref" and isinstance(value, str):
                    found.add(value.rsplit("/", 1)[-1])
                else:
                    found |= refs_in(value)
        elif isinstance(node, list):
            for value in node:
                found |= refs_in(value)
        return found

    reachable = refs_in({key: value for key, value in schema.items() if key != "$defs"})
    frontier = set(reachable)
    while frontier:
        name = frontier.pop()
        for ref in refs_in(defs.get(name, {})):
            if ref not in reachable:
                reachable.add(ref)
                frontier.add(ref)

    schema["$defs"] = {name: body for name, body in defs.items() if name in reachable}
    return schema
