"""Before the loop is allowed to finish: is anything missing that we could still fetch?

## The hole this fills

The loop ends when the analyst makes no tool call, which means it believes it has enough.
Everything after that point -- drafting, the grounding gate, the verifier, the sufficiency
gate -- can only *subtract*: delete a claim, lower a confidence, withhold a cause. None of
them can send work back, because none of them has tools.

That is not a loop, it is a pipeline with a loop in the middle, and it shows. A live
investigation drafted this into its own risks:

    No GitHub, PostHog, or Slack evidence was gathered to explain why the January
    bulk-import deals exist or why they were never closed or re-dated -- the CRM-only
    view cannot distinguish a forgotten deal from a legitimately paused one.

The analyst identified the gap itself, named the sources that would close it, and could do
nothing about it: that sentence is written during `_draft`, a `structured()` call with no
`tools` argument at all. It found the question after losing the ability to answer it. Across
sixteen recorded investigations, roughly one in five leaves a gap of that shape on the table.

## Why this is not the reflection turn that failed

A reflection turn was tried in this loop before and measured worse -- 5/6 instead of 6/6,
with one attempt lost entirely to a 432-second overrun (docs/eval-results.md run 15). It
asked the analyst, unconditionally and on every run, to reconsider its own observations
before concluding. That is *intrinsic* self-correction, and the literature is consistent
about it: models cannot reliably self-correct reasoning without external signal, and
intrinsic self-critique is close to useless on factual tasks (arXiv:2406.01297; the
control-theoretic framing in arXiv:2604.22273 makes the same point as error dynamics that
drift without an exogenous term).

This is the other shape, the one that does work: a separate assessment of the *evidence*,
producing named gaps, which are then closed by fetching -- external signal, not more
thinking. It is the structure FAIR-RAG calls a structured evidence assessment feeding a
query-refinement module (arXiv:2510.22344), and what EfficientGraph-RAG describes as
"targeted re-planning after evidence verification fails" rather than merely more budget
(arXiv:2605.25379).

Three properties follow from that, and each is load-bearing:

  - **It fires conditionally.** Only when the analyst chose to stop, and only when it holds
    evidence. A run that ran out of time or tokens is not short of ideas, it is short of
    budget, and asking it for more work is the failure mode above.
  - **It carries content.** The re-entry states the gap, not "think again".
  - **It can only name what is reachable.** The capabilities this tenant actually has are in
    the prompt, and the instruction is explicit that a gap no listed capability can close is
    not a gap worth naming. Half the gaps found in recorded runs were unreachable -- "a
    Google Ads change history log" for a tenant with no ad platform connected -- and
    re-entering on those is pure cost.

## Why the evidence and not the report

The sufficiency gate answers a neighbouring question and cannot be reused here: it takes a
drafted report, because it needs the report's causal claims to know whether there is
anything to veto. Running it early would mean drafting twice, and drafting is the most
expensive phase in the system -- 42 seconds and 40k input tokens on a live run, against a
90-second budget the suite is already failing. This call sees an inventory of what was
fetched, costs one short round trip, and leaves drafting where it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cortex.agents.llm import LLM, Message, Usage

#: How many gaps a single re-entry may carry.
#:
#: Three, because the instruction is a prompt and not a work queue: an analyst handed eight
#: missing observations will spread one step across all of them rather than closing any. The
#: cap is on what is *asked for*, not on what the loop may then do.
MAX_GAPS = 3

#: Cap on the inventory shown to the assessor, in characters.
_INVENTORY_CHARS = 6000

SYSTEM = """You review an investigation that is about to stop, and decide whether it has \
gathered enough to answer the question it was given.

You are not writing the answer, judging the answer, or checking anyone's reasoning. You see \
only what was fetched and what could still be fetched.

Name a gap only when all three hold:

1. It is a specific observation, not a topic. "Deal stage history for the three open deals" \
is a gap. "More context about the pipeline" is not.
2. Answering the question is materially worse without it. A detail that would be nice to \
have is not a gap.
3. One of the capabilities listed below could actually return it. This is the one that \
matters most. A gap no listed capability can close is not a gap -- it is a limit of what \
this tenant has connected, and naming it wastes the only step available. If the evidence \
that would settle the question lives in a system that is not listed, say the investigation \
is sufficient and let the report disclose the limit.

An investigation that looked and found nothing has not left a gap. Repeating a query that \
returned no rows will return no rows again; say it is sufficient.

Prefer sufficient. The cost of a wrong "sufficient" is a report that discloses its own \
limits, which is the normal and honest outcome. The cost of a wrong gap is a step spent \
fetching something that does not help, on a budget that is already tight."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sufficient", "gaps"],
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": "True when nothing reachable and material is missing.",
        },
        "gaps": {
            "type": "array",
            "maxItems": MAX_GAPS,
            "items": {"type": "string", "maxLength": 300},
            "description": (
                "Specific observations that are missing and that a listed capability could "
                "return. Empty when sufficient."
            ),
        },
    },
}


@dataclass(frozen=True, slots=True)
class GapVerdict:
    """What the check decided, and what it cost."""

    #: False only when the check ran and named at least one reachable gap.
    sufficient: bool = True
    gaps: tuple[str, ...] = ()
    #: False when the check could not be obtained. `sufficient` is then not a judgement, and
    #: a caller must not read it as one -- it is the safe default, which is to carry on.
    ran: bool = False
    usage: Usage = field(default_factory=Usage)

    @property
    def reopens(self) -> bool:
        return self.ran and not self.sufficient and bool(self.gaps)


def _inventory(gathered: Sequence[str]) -> str:
    """What has been fetched, oldest first, truncated from the end.

    Oldest first and cut from the end, because the early calls establish what the
    investigation is *about* and the late ones are refinements. A tail-first truncation
    would hand the assessor the follow-ups without the subject.
    """
    text = "\n".join(f"- {line}" for line in gathered)
    if len(text) <= _INVENTORY_CHARS:
        return text
    return text[:_INVENTORY_CHARS] + "\n- ... (earlier calls omitted)"


def instruction(gaps: Sequence[str]) -> str:
    """The message handed back to the loop.

    Phrased as observations to fetch rather than as criticism. The analyst is not being told
    it was wrong -- it was not -- it is being told the loop is not over and what to spend the
    remaining steps on.
    """
    listed = "\n".join(f"{n}. {gap}" for n, gap in enumerate(gaps[:MAX_GAPS], start=1))
    return (
        "Before you conclude: a separate check of the observations you have gathered found "
        "these still missing, and reachable with the capabilities you already have.\n\n"
        f"{listed}\n\n"
        "Fetch what is worth fetching, then conclude. Keep everything you have already "
        "established -- this does not ask you to revisit it. If one of these turns out to be "
        "unreachable or returns nothing, that is itself an observation: note it and stop "
        "rather than trying variations of the same call."
    )


class GapCheck:
    """One short call, made only when the analyst decided it was finished."""

    def __init__(self, llm: LLM, *, timeout: float | None = 30.0) -> None:
        self._llm = llm
        self._timeout = timeout

    async def assess(
        self, *, question: str, gathered: Sequence[str], capabilities: Sequence[str]
    ) -> GapVerdict:
        """Never raises. An unavailable check must not end an investigation that worked."""
        if not gathered:
            # Nothing was fetched, so there is nothing to assess and no basis for naming what
            # is missing. The loop's own failure handling covers this case.
            return GapVerdict(sufficient=True, ran=False)

        content = (
            f"QUESTION\n{question}\n\n"
            f"OBSERVATIONS ALREADY GATHERED\n{_inventory(gathered)}\n\n"
            f"CAPABILITIES STILL AVAILABLE\n" + "\n".join(f"- {name}" for name in capabilities)
        )
        try:
            payload, usage = await self._llm.structured(
                system=SYSTEM,
                messages=[Message(role="user", content=content)],
                schema=SCHEMA,
                max_tokens=1024,
                timeout=self._timeout,
            )
        except Exception:  # noqa: BLE001 -- see the docstring: this must never end a run
            return GapVerdict(sufficient=True, ran=False)

        gaps = tuple(str(gap).strip() for gap in (payload.get("gaps") or []) if str(gap).strip())
        sufficient = bool(payload.get("sufficient", True))
        return GapVerdict(
            # A verdict naming no gap is sufficient whatever the boolean says: there is
            # nothing to re-enter the loop *for*, and re-entering with an empty instruction
            # is the contentless reflection turn that already measured worse.
            sufficient=sufficient or not gaps,
            gaps=() if sufficient else gaps[:MAX_GAPS],
            ran=True,
            usage=usage,
        )
