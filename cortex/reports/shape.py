"""How much report a question deserves.

Asked *"check if we have integrated the Apollo company deanonymiser in our website"* — a
yes/no lookup — the analyst returned an executive summary, four findings, three tested
hypotheses, recommendations, risks and data-quality notes. Every claim was grounded and the
answer was correct. It was still the wrong answer, because the reader wanted one sentence
and had to find it inside a page.

The report schema makes almost every section optional (`findings`, `hypotheses`,
`recommendations`, `risks` all default to empty), so nothing forced that shape. The model
filled every field it was offered, which is what a model does when a schema offers a field
and nothing says when to leave it out.

**Why this is decided in code rather than asked of the model.** "Be brief when the question
is simple" is a judgement call made in the same breath as the whole draft, and the failure
mode is asymmetric: a padded answer to a lookup is annoying, whereas a thin answer to "why
did signups fall" is the product not working. Computing the shape first means the drafting
instruction *states* which one applies, and the classification is testable on its own — and
when it is unsure it says CAUSAL, so the failure it can make is the harmless one.

**Why this is not a correctness mechanism.** Report length is a shape preference, not a
grounding property, so there is deliberately nothing here that strips or rejects. The
report a factual question gets is a full `InvestigationReport` with fewer sections filled,
still subject to the citation gate, the verifier, and the completeness judge — and it is
that judge which catches the one real risk of this change: a report that came back short
because it under-answered rather than because the question was simple.
"""

from __future__ import annotations

import enum
import re


class Shape(enum.StrEnum):
    """What kind of answer the question is asking for."""

    #: A lookup. "Do we have X", "which plan is Y on", "when did Z ship". There is one
    #: fact, it is either found or not found, and hypotheses about why it is that way are
    #: not what was asked.
    FACTUAL = "factual"

    #: An investigation. "Why did X change", "what caused Y", "explain the drop". A cause
    #: has to be argued for, alternatives have to be ruled out, and the reader needs to
    #: see what was tested — which is the entire report.
    CAUSAL = "causal"


#: Words that ask for a **cause**. Any of these makes the question CAUSAL, whatever it opens with.
#:
#: "what changed" and "what happened" are here because they request the explanatory event in
#: disguise: they are how a person asks "why" without saying it.
_CAUSAL_REQUEST = re.compile(
    r"\b("
    r"why|cause|caused|causes|causing|reason|reasons|because|"
    r"explain|explanation|diagnose|investigate|"
    r"driv(?:e|es|ing)|attribut(?:e|ed|ion)|"
    r"blame|responsible|"
    r"what changed|what happened|what drove"
    r")\b",
    re.I,
)

#: Words naming a **movement** in a metric. These describe the subject, not the request.
#:
#: **Separated from the causal words after a live failure, and the separation is the point.** They
#: were one set, so `"did our signups fall from last month?"` classified as CAUSAL on the strength
#: of "fall" — and the analyst was told to find a cause, record hypotheses and test explanations for
#: a question that asked *whether something happened*. What came back was four paragraphs about
#: mid-June, deploy searches and Slack incident hunts, and it never said yes or no.
#:
#: They still need to be here, because a bare statement of a movement is a request to explain it:
#: "signups are down 12%" with no question mark wants a cause. What they must not do is override a
#: question that opens as a check.
_MOVEMENT = re.compile(
    r"\b("
    r"drop(?:ped)?|fall|fell|falling|declin(?:e|ed|ing)|"
    r"spike|spiked|surge|surged|jump(?:ed)?|"
    r"chang(?:e|ed)|shift(?:ed)?|mov(?:e|ed)|trend(?:ing)?|"
    r"regress(?:ion|ed)|broke|broken"
    r")\b",
    re.I,
)

#: Openers that mean a lookup. Checked against a causal *request* rather than against a movement:
#: "did signups fall?" is a check, "did signups fall, and why?" is not.
_LOOKUP = re.compile(
    r"^\s*(?:can you |could you |please |tell me |help me )*"
    r"("
    r"do we|do |does|did we|did |are we|are |is there|is |are there|have we|have |has |had |"
    r"was |were |"
    r"which|who|when|where|what is|what are|what's|how many|how much|"
    r"list|show|find|check|confirm|verify|look up"
    r")\b",
    re.I,
)


def shape_for(question: str) -> Shape:
    """Which shape of report this question is asking for.

    Defaults to `CAUSAL`. An over-full answer to a simple question wastes a reader's
    time; a thin answer to a causal question is the product failing, so an unrecognised
    question gets the full treatment.
    """
    text = question or ""
    # An explicit request for a cause wins over everything, including a lookup opener: "did
    # signups fall, and why?" opens like a check and is not one.
    if _CAUSAL_REQUEST.search(text):
        return Shape.CAUSAL
    # A question that opens as a check is a check, even when it names a movement. This is the case
    # a single combined pattern got wrong.
    if _LOOKUP.match(text):
        return Shape.FACTUAL
    # No causal request and no lookup opener. A bare statement naming a movement -- "signups are
    # down 12%" -- is a request to explain it, and so is anything else unrecognised: an over-full
    # answer wastes a reader's time, a thin answer to a causal question is the product failing.
    return Shape.CAUSAL


#: The premise check, shared by both shapes.
#:
#: **It belongs to both, and lived on only one.** It was written for FACTUAL, where a yes/no
#: question wears its premise openly, and CAUSAL never received it — even though the paragraph's
#: own second example ("why is checkout slower on mobile") is a causal question. So the check was
#: attached to the shape that *asks* whether something happened and missing from the shape that
#: *assumes* it did, which is exactly backwards: "why did signups fall" cannot be answered at all
#: until the fall is established, and a question opening with "why" has already granted it.
#:
#: This is not hypothetical. Asked "did our signups fall from last month", the analyst routed
#: FACTUAL, checked the premise, and correctly answered that demand had not fallen. Asked to
#: compare three months and correlate the movement with product changes, it routed CAUSAL, was
#: handed no premise instruction, and produced a causal story linking two incidents a week apart.
_PREMISE_CHECK = (
    "**If the question assumes something that is not true, say that first.** A question "
    "can carry a premise — 'did signups fall last month' assumes they fell, 'why is "
    "checkout slower on mobile' assumes it is. When the evidence says the premise is "
    "wrong, the answer is 'no, and here is what the numbers actually show', in the first "
    "sentence. Do not lead with a figure that appears to confirm the premise and correct "
    "it afterwards: a reader who stops after one line has then been misinformed by a "
    "report that was technically accurate throughout. If a comparison looks alarming only "
    "because the periods are different lengths, or the data is incomplete, or the metric "
    "changed definition, that *is* the answer rather than a caveat on it.\n"
    "**And set `premise` and `premise_checked`.** State the assertion you tested and your "
    "verdict on it -- `false` when the evidence contradicts it, `unverifiable` when the evidence "
    "cannot settle it, `holds` when it is supported. Putting the verdict in a field rather than "
    "only in prose is what lets a reader see at a glance whether the question itself survived, "
    "instead of inferring it from how the first sentence happens to be phrased."
)


#: Anything that names a comparison the reader can check the answer against.
#:
#: Two kinds, and both count. A *period* -- "in August", "last week", "since June" -- pins when.
#: A *baseline* -- "versus July", "compared to last month", "week over week" -- pins against what.
#: Either makes the question answerable as asked; neither leaves the answer's meaning up to the
#: analyst.
_BASELINE = re.compile(
    r"\b("
    r"versus|vs\.?|compared (?:to|with)|against|relative to|"
    r"week[- ]over[- ]week|month[- ]over[- ]month|year[- ]over[- ]year|wow|mom|yoy|"
    r"last (?:week|month|quarter|year|\d+ days?)|previous (?:week|month|quarter|year)|"
    r"this (?:week|month|quarter|year)|since|between|from \d|"
    r"in (?:january|february|march|april|may|june|july|august|september|october|november"
    r"|december)|"
    r"q[1-4]|\d{4}-\d{2}|\b\d{4}\b|yesterday|today"
    r")\b",
    re.I,
)


class Ambiguity(enum.StrEnum):
    """A structural gap in the question that the answer's meaning depends on."""

    #: A movement is named with nothing to measure it against.
    UNSTATED_BASELINE = "unstated_baseline"


def ambiguities(question: str) -> tuple[Ambiguity, ...]:
    """Structural gaps in the question, decided in code.

    **Deliberately only what a predicate can settle.** The literature is unambiguous that a
    model cannot estimate its own need to clarify -- six independent
    measurements, including a clarification-need F1 of 0.33-0.37 and one benchmark whose R-squared
    is *negative*, worse than a constant. So this returns facts about the sentence, never a sense
    that it is vague.

    One gap today, because it is the one that is both common and computable: a question naming a
    movement with no period and no comparison. "Why did signups fall" is unanswerable as asked --
    fall against last week, last month, last year? Each is a different question with a different
    answer, and the analyst picks one silently.

    The research's other ambiguity types need a tenant's own catalogue to detect: how many
    entities match a name, how many definitions a metric has. Those are lookups this function has
    no access to, and guessing at them from prose is the keyword-list mistake this codebase has
    made four times.
    """
    found: list[Ambiguity] = []
    if _MOVEMENT.search(question) and not _BASELINE.search(question):
        found.append(Ambiguity.UNSTATED_BASELINE)
    return tuple(found)


#: What the analyst must state for each gap, and why stating it is the whole intervention.
#:
#: Not a question back to the user. The research prices asking expensively for us -- a question in
#: a public Slack channel is latency-visible and breaks the one promise the product makes, and our
#: users are by construction in a hurry, which *lowers* the threshold for acting. Every term in
#: Horvitz's expected-utility calculation pushes the ask band narrow.
#:
#: What it prices cheaply is *stating the assumption*: the gap between a wrong reading delivered
#: silently and a wrong reading delivered with its assumption named is enormous, and entirely
#: under our control. A reader who sees "read as against the prior 30 days" and meant year over
#: year re-asks in one line. A reader who sees neither is misinformed and does not know it.
AMBIGUITY_GUIDANCE: dict[Ambiguity, str] = {
    Ambiguity.UNSTATED_BASELINE: (
        "**This question names a movement without saying what to measure it against.** Pick the "
        "most useful comparison, say which you picked in the first sentence, and put it in the "
        "risks as an assumption -- 'read as against the prior 30 days; against the same month "
        "last year the answer differs'. Do not ask which was meant: state one and let the reader "
        "correct you in a line if you chose wrong."
    ),
}


#: Shape-specific drafting guidance, keyed by shape.
#:
#: Written as *when to leave a section out* rather than as "be brief". A length instruction
#: gets applied to the sentences, which makes the prose terse and the section count
#: unchanged; naming the sections is what actually changes the shape.
GUIDANCE: dict[Shape, str] = {
    Shape.FACTUAL: (
        "This is a factual question, not a causal investigation. The reader wants the "
        "answer, not a case for it.\n"
        "  - Lead with the answer in the executive summary, in one or two sentences. If "
        "it is a yes/no question, the first word should settle it.\n"
        "  - Add a finding only for something the answer does not already contain — a "
        "detail the reader would otherwise have to ask a follow-up to get.\n"
        "  - Leave hypotheses empty. Nothing needs explaining: you were asked what is "
        "true, not why it is true.\n"
        "  - Leave recommendations empty unless the finding itself is a problem someone "
        "has to act on. 'Consider monitoring this' is not a recommendation.\n"
        "  - Keep risks to things that could make the answer itself wrong — you looked in "
        "the wrong place, or the source could not confirm a negative.\n"
        "If you could not determine the answer, say so plainly and say what you looked "
        "at. A confident-sounding page around an unanswered question is worse than a "
        "short 'I could not confirm this, here is where I looked'.\n\n" + _PREMISE_CHECK
    ),
    Shape.CAUSAL: (
        "This is a causal question. A cause has to be argued for, so the reader needs the "
        "full report: what you found, which explanations you tested and how each fared, "
        "what you would do about it, and what could still make you wrong. Record "
        "hypotheses you ruled out as well as the one that survived — a report showing "
        "only the surviving story is the one a reader cannot check.\n\n"
        "**Establish the effect before explaining it.** A question asking why something "
        "happened has already granted that it happened, and that grant is the first thing to "
        "check rather than the one thing taken for granted. If the movement is not there, or "
        "is an artefact of an incomplete period, a changed definition, or collection that "
        "stopped, then saying so *is* the answer — a cause correctly argued for an effect that "
        "did not occur is wrong in a way no amount of evidence behind it can fix.\n"
        "**A series marked `data_trust: broken` cannot answer the question.** When an observation "
        "carries that state, the answer is the data incident -- what stopped, when, on which "
        "emitter, and what is still flowing -- not a cause for a movement that was never measured. "
        "Do not attribute it to product, marketing or demand: there is no movement there to "
        "attribute, only an absence of measurement. A `degraded` series answers a narrowed "
        "question, and the narrowing belongs in the answer rather than in a caveat.\n"
        "**Two things going wrong are not one thing going wrong.** Before attributing a shared "
        "cause to two movements, check they moved at the same time. Metrics that turn down on "
        "different dates are separate incidents until something ties them together, and the "
        "second is not corroboration for the first.\n"
        "**Date every hypothesis that names a cause.** Fill `cause_at` with the date the "
        "proposed cause happened and `effect_onset` with the date the movement began, taking "
        "the onset from the evidence -- a series' `movement.level_shifts[].at`, or the last "
        "normal bucket before a gap -- and never by estimating from a chart. A cause dated "
        "after its effect is refused outright rather than ranked lower, so writing the dates "
        "down is what lets an impossible explanation be eliminated instead of argued about.\n"
        "**Name the change, not its category.** When the cause is a specific artefact -- a pull "
        "request, a commit, a release, a flag flip -- name it the way the evidence names it: the "
        "number, the sha, the tag. An evaluated report described its cause as 'a mobile-only "
        "onboarding modal rework merged on 14 July', having read PR 913 and the sha of the commit "
        "in it. That identifies the right change and still leaves the reader searching for it, "
        "where the number is one click. Finding the cause and then describing it rather than "
        "naming it is doing the hard part and withholding the useful part."
        "\n\n" + _PREMISE_CHECK
    ),
}
