# ADR 0002 — Harness mechanisms: adopted, queued, declined

**Date:** 2026-07-31
**Status:** Accepted; two mechanisms adopted, three queued with triggers, three declined
**Supersedes nothing.** [ADR 0001](0001-own-investigation-loop.md) decided *whether* to own the
investigation loop. This decides *which harness mechanisms the loop needs*, which is a different
question and the one that was actually worth asking.

## Why this is a separate decision

ADR 0001 compared **tools** — bash and file editing against GTM connectors — and concluded that
a general-purpose agent runtime's maturity does not transfer. That comparison is beside the point.
Grep and read are tools; the *harness* around them — how state is persisted, how context is kept
valid, how a loop knows it is stuck, how secrets stay out of a transcript — is domain-independent.
First-principles structure does not change because the tools do.

So the question here is not "is this agent our agent" but "what does any long-running tool loop
have to solve, and has this one solved it?" Two answers were no.

---

## Adopted

### 1. Multi-pattern stall detection → `cortex/agents/spin.py`

**One signal, with a hole in it.** `barren_streak` counted consecutive steps that called tools and
recorded no evidence. An analyst calling the *same capability with the same parameters* four times
writes a new evidence row every time — so the streak reset on each one and the loop ran to its
step limit. The budget eventually stopped it, which means the symptom was a slow expensive
investigation rather than an error. **That is the worst way to fail: it looks like work.**

Four patterns now, shaped by evidence gathering rather than by editing code:

| Pattern | Signal | Why this one |
|---|---|---|
| Same call repeated | `(capability, sorted params)` | The hole above, and the most common stall |
| Same observation returned | `payload_hash` — already computed and indexed for the gate | Different questions reaching identical data means circling |
| Same failure repeated | capability + first line of the error | A credential is wrong, not the timing |
| No new evidence | the original streak | Kept: catches a loop whose every call errors |

**Deliberately not watched:** monologue and alternating-pattern detection. An analyst thinking for
two turns without a tool call is reasoning, and this loop already treats a turn with no tool
request as a decision to conclude.

The recorded reason is the most *specific* pattern that fired, not the first: "stopped early" and
"stopped early because it asked the same question three times" call for different fixes.

### 2. Transcript invariants → `cortex/agents/transcript.py`

A transcript is not a list of events. It is a list that satisfies **properties** the provider's
API requires, and stating those as code is what makes them hold — maintained by care, they hold
until the first refactor. Each property carries two mechanisms: `enforce`, which removes offending
events, and *manipulation indices*, which mark where the list may be cut without breaking the
property. A condenser that respects the indices cannot emit an invalid payload.

The requirement that forced this: every `tool_use` must have one corresponding `tool_result` in
the immediately following user message.

**That is F-12.** It surfaced as a provider 400 mid-run — `unexpected tool_use_id found in
tool_result blocks` — after the tokens were spent, naming no turn. The instance was fixed. The
invariant was never stated, so the next change to transcript construction could reintroduce it.

Now stated, and checked before every provider call: unanswered calls, orphaned results,
positionally-misplaced results, and results or calls on the wrong role. The error names the turn.

**And `safe_cut_points` exists before condensation does.** The transcript grows unbounded; memory
recall and several connectors push toward needing to drop turns. The first thing a naive
truncation does is cut a tool call away from its result — reintroducing F-12 in a form that is
*harder* to attribute, because the transcript was valid when it was built.

---

## Queued, with the trigger that should pull them in

### 3. Event-sourced loop state — when investigations need to resume

Loop state is a local list. A worker killed mid-run loses it and the Celery redelivery starts
over. The evidence rows survive, so nothing is *corrupted* — the tokens are paid twice.

**Not adopted now**, because at 60–110 seconds per investigation a restart is cheap and the change
is invasive: it would move the loop from "list of messages" to "log of events plus a projection",
touching the loop, the drafting call and the audit trail. **Trigger:** when a single investigation
reliably exceeds ~5 minutes, or when a UI needs to replay one.

### 4. A summarising condenser — when a transcript stops fitting

A rolling condenser that summarises older events while respecting the manipulation indices. The
indices exist; the condenser is not yet needed. **Trigger:** the first investigation that hits a
context limit, which the per-call `max_tokens` currently prevents by truncating output instead.

### 5. A secret registry — when a tool argument can contain a secret

Such a mechanism scans a command for secret keys and injects values at execution, keeping
plaintext out of the transcript. This tool surface has a stronger property today *by
construction*: no capability takes a secret as a parameter, credentials are decrypted inside the
executor, and none of them enter the model's context at all. The mechanism exists for agents that
can be *asked* to run `curl -H "Authorization: ..."`. **Trigger:** the first capability whose
parameters could carry a credential — which V1's read-only, typed-parameter design deliberately
avoids, and which `tests/unit/test_borrow_triggers.py` asserts mechanically rather than leaving
the trigger written down where nobody checks it.

---

## Declined, with reasons

**A richer tool abstraction.** The obvious one is more expressive than this project's `Tool`, but
this one carries something those do not: `result_key`, which makes "what does empty mean for this
capability" a declaration a capability cannot omit (F-24). Adopting a richer abstraction would
lose that.

**Confirmation policies and security analyzers.** These gate *actions* before execution. V1 ships
no write capability, so "humans approve" holds by absence of capability rather than by policy — a
stronger guarantee than any analyzer. Revisit only if a write tool is ever added.

**A multi-provider routing layer.** ADR 0001 concluded that writing `AnthropicLLM` by hand cost
four transport defects (F-15, F-16, F-18, F-19) that a mature layer would have handled. That
conclusion stands and this ADR does not change it: rewriting now reintroduces paid-for risk
against a suite that includes regressions for all four. **Trigger unchanged:** the second
provider.

---

## What this says about ADR 0001

ADR 0001's conclusion — keep the loop, because the grounding guarantee depends on this codebase
controlling what enters context — survives. Nothing here undermines it: there is still no
off-the-shelf equivalent of the evidence store, the citation gate or the adversarial verifier.

But its *reasoning* was too quick in one place. It said there was nothing to inherit, and that was
wrong. There was: a stall detector with a hole in the existing one, and an invariant mechanism for
precisely the bug class that cost F-12 and would have cost it again at the first truncation. Both
are harness concerns, both are domain-independent, and both were invisible from the comparison ADR
0001 made.

The method is the lesson. Comparing *tool surfaces* answers "is that agent our agent" — no.
Comparing *harness structure* answers "what has been solved that this has not", which is the
question worth asking of any mature system.

## Round two: how live state reaches a screen

The same question, asked of the report view: **how does live agent state reach a screen without
lying to the reader?** Answering it wrong is not a cosmetic bug — a progress view that silently
drops events shows an investigation as stalled when it is working.

**History is REST-driven; the socket carries only new events.** One authoritative backlog, one
live tail, and no attempt to reconstruct history from a stream — because a transport that
guarantees neither ordering nor completeness cannot be asked to reconstruct the past.

This project goes one step further. That backlog is normally a persisted event log: a second
record of what happened, which can disagree with the first. Every tool call here already writes an
immutable audit row, so the backlog *is* the audit trail. It cannot drift from what happened, it
survives a worker restart, and it needed no new table. See
[ADR 0003](0003-server-rendered-report-view.md).

**Sequence numbers on progress frames.** A terminal prints each line as it arrives and never
reconnects, so an unnumbered stream is fine. A UI drops out, comes back, and receives some frames
twice while missing others. Without a sequence, two identical "step 5 calling `github__commits`"
frames are indistinguishable from one call reported twice, and a missing frame is invisible.
