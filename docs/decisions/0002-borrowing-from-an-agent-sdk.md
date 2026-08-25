# ADR 0002 — What Cortex borrows from a mature agent SDK

**Date:** 2026-07-31
**Status:** Accepted; two mechanisms adopted, three queued, three declined
**Supersedes nothing.** ADR 0001 decided *whether* to use their loop. This decides *what to
take from their engine*, which is a different question and the one that was actually asked.

## How this was done, and why the method matters

ADR 0001 was written from reading a few files and reasoning about domain fit. That produced a
defensible conclusion and a blind spot: it compared **tools** (bash and file editing versus
GTM connectors) and concluded the maturity does not transfer. That comparison is beside the
point. Grep and read are tools; the *harness* around them — how state is persisted, how
context is kept valid, how a loop knows it is stuck, how secrets stay out of a transcript — is
domain-independent. First-principles structure does not change because the tools do.

So this ADR was produced differently: both repositories were cloned and the SDK was indexed as
a graph (29,068 nodes, 61,991 edges, 1,065 communities), and the graph was asked structural
questions rather than grepped for keywords.

The most connected nodes name the architecture immediately:

| Node | Degree | What it tells you |
|---|---|---|
| `LLM` | 1218 | The provider boundary is one object, used everywhere |
| `Agent` | 684 | The agent is data, not a function |
| `ConversationState` | 491 | **State is a first-class persisted object** |
| `LocalConversation` / `RemoteConversation` | 480 / — | Local and remote execution behind one interface |
| `ActionEvent` / `ObservationEvent` | 272 | **The loop is an event log, not a message list** |
| `ConversationService` | 277 | The server layer owns lifecycle, not the agent |

That is a different shape from ours in one specific way: **their loop's state is an event log
projected into an LLM view; ours is a Python list held in memory.** Everything below follows
from that difference.

---

## Adopted

### 1. Multi-pattern stall detection → `cortex/agents/spin.py`

**Theirs:** `StuckDetector` watches five patterns against configurable thresholds — repeated
action-observation cycles, repeated action-error cycles, agent monologue, alternating patterns,
and context-window errors.

**Ours had one signal, and it had a hole.** `barren_streak` counted consecutive steps that
called tools and recorded no evidence. An analyst calling the *same capability with the same
parameters* four times writes a new evidence row every time — so the streak reset on each one
and the loop ran to its step limit. The budget eventually stopped it, which means the symptom
was a slow expensive investigation rather than an error. **That is the worst way to fail: it
looks like work.**

Four patterns now, shaped by evidence gathering rather than by coding:

| Pattern | Signal | Why it is ours and not theirs |
|---|---|---|
| Same call repeated | `(capability, sorted params)` | Their `action_observation`, and the hole above |
| Same observation returned | `payload_hash` — already computed and indexed for the gate | Different questions reaching identical data means circling |
| Same failure repeated | capability + first line of the error | Their `action_error`. A credential is wrong, not the timing |
| No new evidence | the original streak | Kept: catches a loop whose every call errors |

**Declined from theirs:** monologue and alternating-pattern detection. A GTM analyst thinking
for two turns without a tool call is reasoning, and our loop already treats a turn with no tool
request as a decision to conclude.

The recorded reason is the most *specific* pattern that fired, not the first: "stopped early"
and "stopped early because it asked the same question three times" call for different fixes.

### 2. Transcript invariants → `cortex/agents/transcript.py`

**Theirs:** `View` is not a list of events — it is a list that satisfies **properties** the
provider's API requires, stated as code. Four of them, each with two mechanisms: `enforce`
(remove offending events) and *manipulation indices* (where the list may be cut without
breaking the property). A condenser that respects the indices cannot emit an invalid payload.

Their `ToolCallMatchingProperty` docstring names the exact requirement I discovered by hitting
it: *"some providers (for example Anthropic tool use) require every `tool_use` to have one
corresponding `tool_result` in the immediately following user message."*

**That is F-12.** We found it as a provider 400 mid-run — `unexpected tool_use_id found in
tool_result blocks` — after the tokens were spent, naming no turn. We fixed the instance. We
never stated the invariant, so the next change to transcript construction could reintroduce it.

Now stated, and checked before every provider call: unanswered calls, orphaned results,
positionally-misplaced results, and results or calls on the wrong role. The error names the
turn.

**And `safe_cut_points` exists before condensation does.** Our transcript grows unbounded;
memory recall and four connectors push toward needing to drop turns. The first thing a naive
truncation does is cut a tool call away from its result — reintroducing F-12 in a form that is
*harder* to attribute, because the transcript was valid when it was built.

---

## Queued, with the trigger that should pull them in

### 3. Event-sourced conversation state — when investigations need to resume

Their `ConversationState` persists to a pluggable `FileStore`, with `.snapshot()` and
reconstruction from an `EventLog`. A killed conversation resumes.

Ours does not. An investigation's loop state is a local list; a worker killed mid-run loses it
and the Celery redelivery starts over. The evidence rows survive, so nothing is *corrupted* —
we pay for the tokens twice.

**Not adopted now**, because at 60–110 seconds per investigation a restart is cheap and the
change is invasive: it would move the loop from "list of messages" to "log of events plus a
projection", touching the loop, the drafting call and the audit trail. **Trigger:** when a
single investigation reliably exceeds ~5 minutes, or when a UI needs to replay one.

### 4. `LLMSummarizingCondenser` — when a transcript stops fitting

A rolling condenser that summarises older events while respecting the manipulation indices.
We now have the indices; we do not yet need the condenser. **Trigger:** the first investigation
that hits a context limit, which our per-call `max_tokens` currently prevents by truncating
output instead.

### 5. `SecretRegistry` — when a tool argument can contain a secret

Theirs scans a command for secret keys and injects values at execution, keeping plaintext out
of the transcript, and redacts or encrypts on serialization depending on context.

Ours has a stronger property today *by construction*: no tool takes a secret as a parameter,
credentials are decrypted inside the executor and never enter the model's context at all. Their
mechanism exists because a bash agent can be *asked* to run `curl -H "Authorization: ..."`.
**Trigger:** the first capability whose parameters could carry a credential — which V1's
read-only, typed-parameter design deliberately avoids.

---

## Declined, with reasons

**Their tool abstraction.** `ClientToolSpec`/`Observation` is richer than ours, but ours carries
something theirs does not: `result_key`, which makes "what does empty mean for this capability"
a declaration a capability cannot omit (F-24). Adopting theirs would lose that.

**`ConfirmationPolicy` / `SecurityAnalyzerBase`.** These gate *actions* before execution. V1
ships no write capability, so "humans approve" holds by absence of capability rather than by
policy — a stronger guarantee than any analyzer. Revisit only if a write tool is ever added.

**litellm as the provider layer.** ADR 0001 already concluded that building `AnthropicLLM` by
hand cost us four transport defects (F-15, F-16, F-18, F-19) that a mature layer would have
handled. That conclusion stands and this ADR does not change it: rewriting now reintroduces
paid-for risk against 54 tests including regressions for all four. **Trigger unchanged:** the
second provider.

---

## What this exercise says about ADR 0001

ADR 0001's conclusion — keep the loop, because the grounding guarantee depends on Cortex
controlling what enters context — survives. Nothing found here undermines it: there is still no
OpenHands equivalent of the evidence store, the citation gate or the adversarial verifier.

But ADR 0001's *reasoning* was too quick in one place. It said "there is nothing to inherit",
and that was wrong. There was: a stall detector with a hole in ours, and an invariant mechanism
for precisely the bug class that cost us F-12 and would have cost us again at the first
truncation. Both are harness concerns, both are domain-independent, and both were invisible
from the comparison ADR 0001 made.

The method is the lesson. Comparing *tool surfaces* answers "is their agent our agent" — no.
Comparing *harness structure* answers "what have they solved that we have not" — which is the
question worth asking of any mature system, and the one that required indexing their repository
rather than reasoning about it.

---

## Round two: the platform frontend

The same method applied to `openhands/platform` — the React app, indexed the same way
(1,080 files, 4,763 nodes, 16,154 edges). The question this time was structural: **how does
live agent state reach a screen without lying to the reader?** M7 has to answer it, and
answering it wrong is not a cosmetic bug — a progress view that silently drops events shows
an investigation as stalled when it is working.

The graph's highest-degree nodes point straight at the answer:
`contexts/conversation-websocket-context` (89), `types/agent-server/type-guards` (85),
`stores/conversation-store` (71), `components/conversation/events/chat-event-message` (62).
Reading those four files rather than guessing produced three findings, and the third is a
defect in code we have already shipped.

**1. History is REST-driven; the socket carries only new events.** Their comment says it
outright: *"History loading for the main conversation is REST-driven now; every WS message is
a new event we add to the store."* One authoritative backlog, one live tail, and no attempt to
reconstruct history from a stream. Worth adopting because the alternative — a socket that
replays everything on connect — has to be correct about ordering and completeness in a
transport that guarantees neither.

**2. Events are validated by a type guard before they enter the store**
(`isAgentServerEvent`). A malformed event is rejected at the boundary rather than rendered as
a blank row. Same principle as our contracts module, applied at the UI edge.

**3. A reconnect replays a backlog from a stale anchor, so the store dedups by event id —
and skips side-effects for events it has already seen.** Their comment cites their own issue
#1656. This is the finding that matters, because it names a bug we have:

> `QueueProgress` publishes `InvestigationProgress` to `QUEUE_EVENTS` with a 60-second TTL.
> The message carries `investigation_id`, `status`, `step`, `note`, `at` — **no event id and
> no sequence number.** A client that reconnects cannot tell what it missed, and cannot
> deduplicate what it receives twice. Two identical "step 5 calling github__commits" lines
> are indistinguishable from one retried call.

Nothing consumes these events yet, which is the only reason this has not bitten: the
progress contract was designed for a terminal that prints each line as it arrives and never
reconnects. A UI is a different consumer, and it is the consumer M7 adds.

So the borrow is a precondition of M7 rather than part of it: progress events need a
monotonic sequence per investigation and a stable id before anything renders them.
`InvestigationProgress`'s own docstring already says progress is advisory and the Postgres
row is authoritative — which is exactly the REST-plus-tail split above, arrived at
independently and then not implemented.

**Declined from the frontend:** their store-per-concern layout (24 Zustand stores), their
dual-socket arrangement for a planning agent, and their optimistic-user-message reconciliation.
All are answers to interaction problems a report view does not have.

---

## The queued borrows, checked against measurements (2026-07-31)

All three triggers above were checked rather than left as notes. **None has fired**, and the
numbers are recorded so the next check is a comparison rather than a fresh judgement:

| Borrow | Trigger | Measured | |
|---|---|---|---|
| Event-sourced resumable state | an investigation over 5 minutes, or a UI replaying one | longest completed run **123s**; replay answered by `/investigations/{id}/trace` over the audit rows | not fired |
| `LLMSummarizingCondenser` | first context-limit hit | peak context **129,605** of a **1,000,000** window (13%) | not fired |
| `SecretRegistry` | a tool parameter that could carry a credential | **0** of 101 parameters across 28 capabilities | not fired |

The third is a property of the tool surface rather than a measurement, so it is now
`tests/unit/test_borrow_triggers.py` instead of a sentence here. It fails on the commit that
first adds a capability accepting a token — which is exactly the commit at which the borrow
becomes worth making. A trigger written in a document is a trigger nobody checks.

Its failure message says why it matters beyond the borrow: a credential in a tool argument is
written to the `tool_calls` audit row, rendered on the report page, and — because of prompt
caching — kept in the model's context for the rest of the investigation.

**One real defect came out of taking the measurement**, which is the argument for measuring
rather than reasoning. Reading the investigation rows to find the longest run showed that
**eleven of twelve sat at `queued` permanently**: `POST /investigations` enqueues work, no
worker consumed it, and nothing ever revisits the row. On a page "queued" and "queued since
Tuesday" render identically, so the new list read as a broken product rather than one whose
worker was not running. The page now discloses a non-terminal row older than twenty minutes as
likely abandoned, and stops refreshing it. Disclosed, never corrected — marking the row FAILED
during a page render would be a GET that mutates state, and the gateway does not get to decide
that a worker is dead.
