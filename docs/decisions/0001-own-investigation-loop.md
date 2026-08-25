# ADR 0001 — Cortex owns the investigation loop

**Date:** 2026-07-30
**Status:** Accepted. Its *conclusion* stands; its reasoning was corrected by
[ADR 0002](0002-borrowing-from-an-agent-sdk.md), which indexed a mature agent SDK as a graph and
found two harness mechanisms this ADR wrongly concluded there was "nothing to inherit" from.

## Context

The implementation plan recorded a decision to build on an existing agent SDK
([`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk), MIT) **as a
library**, plus its Agent Server for per-investigation Docker sandboxes.

What was built diverges from that. `cortex/agents/investigator.py` drives its own loop against
this project's own `LLM` interface, and no agent SDK is a dependency.

This ADR exists because that divergence happened during implementation without being
written down. It was raised as a fair challenge: OpenHands is a mature, research-backed
system built by a dedicated team, so why is a hand-rolled loop preferable?

## Decision

Keep the investigation loop in Cortex. Adopt OpenHands for the sandbox, not for the
agent.

## Why the loop stays ours

OpenHands models a **software-engineering** agent. Its tools are bash, file editing and a
browser; the sandbox is the point; the stop condition is "the task is done"; its context
management is tuned for long coding sessions. Its research results are reported on coding
benchmarks, and benchmark maturity is task-specific — it does not transfer to causal
analysis of GTM metrics, because the thing being optimised is different.

Cortex's loop is not a specialisation of that. It needs:

- a hypothesis ledger tracking support **and contradiction** per hypothesis, since the
  product's value is rejecting the plausible-but-wrong answer;
- one immutable `Evidence` row per tool call, with the model seeing evidence **ids** and
  nothing else;
- a gate that removes any claim whose ids do not resolve, in code, before render;
- a separate drafting call under a constrained schema.

The grounding guarantee — the actual product — depends on Cortex controlling what enters
context. If a framework owns the loop, the guarantee weakens from *code-enforced* to *the
framework did not drop the ids*, which is not a guarantee. A coding agent establishes
correctness by running tests; Cortex establishes it by citation. There is no OpenHands
equivalent of the evidence store or the gate, so there is nothing to inherit.

The same argument covers read-only-ness. V1 ships no write tools, so "humans approve"
holds by absence of capability rather than by prompt. Adopting a bash-in-a-sandbox agent
would end that property.

Adapting OpenHands to this use case would mean replacing its tools, its prompts, its stop
condition and its context strategy. That keeps the scaffolding and discards precisely the
parts that its maturity applies to.

## What this cost us, and where the reasoning was wrong

OpenHands routes providers through litellm, which has retry, fallback and timeout policy
already written and exercised at scale. Four of the nine live-path defects recorded in
`security-findings.md` are provider-transport plumbing:

| Finding | Defect |
|---|---|
| F-15 | streaming request with no deadline |
| F-16 | timeout escaping as an httpx exception, killing the process |
| F-18 | one per-request ceiling for calls of very different length |
| F-19 | capacity error delivered inside a 200, so nothing retried it |

A mature provider layer would likely have handled or exposed all four. Building that layer
by hand is the part of the divergence that was a mistake — not keeping the loop. Each of
those was discovered by a separate multi-minute live run, one at a time.

The remaining five (F-11 schema subset, F-12 tool-result pairing, F-13 error opacity,
F-14 key resolution, F-17 grammar size) are API contract issues that any caller hits,
framework or not.

## Consequences

- `AnthropicLLM` stays. Rewriting it onto litellm now would reintroduce risk for a cost
  already paid — it is covered by 54 tests, including regressions for all nine findings.
- Revisit a provider abstraction layer at the **second** provider, which is where one
  earns its keep.
- Use a hosted agent sandbox where it is genuinely better than anything we would write:
  the matplotlib chart escape hatch was the candidate. **Superseded** — that escape hatch was
  declined outright (see `cortex/reports/png.py`), so no sandbox container is declared any
  more. Also worth borrowing later: context condensation, if investigations outgrow a window.

## Evidence available so far

Run 5 of the eval, the first to score real analyst behaviour rather than fail on
infrastructure:

- 2 of 3 scenarios passed, **hallucinations = 0**
- grounding 1.00 and accuracy 1.00 on both passing scenarios
- `insufficient_evidence` correctly declined to name a cause — the hardest behaviour in
  the suite, and the one the evidence-and-gate machinery exists to produce
- weaknesses measured, not guessed: decoy rejection 0.33 and 0.75, latency 193s and 204s
  against a 90s budget

## Update, same day: the spike ran, and the open question below was under-specified

Three arms were scored on `campaign_traffic_drop`, everything but the loop held constant:

| Arm | decoy_rejection | draft_reliability | wall clock |
|---|---|---|---|
| our loop | 0.33 | — | 193–245s |
| bare OpenHands loop | 0.33 | 0.93 (14/15 claims) | 174s |
| OpenHands + critic + condenser | 0.33 | 1.00 (23/23 claims) | 224s |

**The identical 0.33 turned out to be a bug in our scorer, not a property of any loop** —
see `eval-results.md`. So this table's headline column measured the measuring instrument.
Recorded rather than deleted, because it is the reason the scorer bug was found at all.

What still stands from the spike:

- **Arm 2 tested almost nothing.** `condenser` and `critic` both default to `None`, so
  swapping their loop in with default settings measures their bare tool-calling loop. The
  original open question below proposed exactly that, which was the wrong experiment for
  the hypothesis — the hypothesis is about the *harness mechanisms*, and those ship off.
- **Two mechanisms are worth porting.** `LLMSummarizingCondenser` addresses a real
  limitation: our loop stops at `TOKEN_LIMIT` where theirs compresses and continues.
  Scored refinement is worth having for genuinely weak trajectories.
- **Refinement never fired**, so its effect is unmeasured. Our critic scored the trajectory
  0.91, above the 0.6 threshold, and it was right to — `tool_selection` was 1.00 and the
  transcript shows the agent pursuing a competing explanation. Testing the effect needs a
  scenario where the method is genuinely poor.
- **Reading their source paid for itself.** The default `get_followup_prompt` discards
  `critic_result.message`, so a refinement iteration would have retried blind. Found by
  reading 339 lines, not by a graph.
- **The machinery is simple.** `AgentFinishedCritic` is 49 lines and scores "non-empty git
  patch AND FinishAction → 1.0". `IterativeRefinementConfig` is two numbers. There is no
  hidden research in this code; the value is that the pattern has been validated at scale,
  which supports porting rather than adopting.
- **Adoption cost, measured.** Five undocumented shapes had to be probed to wire it up:
  `ToolDefinition`'s abstract/ClassVar contract, `include_default_tools` being a list of
  names, `kind` leaking into tool params, `Observation.content` being a block list, and
  `ExecutedTool` having no `.result`. Plus 88 packages including telemetry, and a permanent
  thread bridge because `Conversation.run()` has no async variant.

## Open question

This ADR argues from architecture. The honest way to settle it is measurement: wire the
OpenHands SDK behind the same tool registry on a branch and score it on the same three
scenarios. Not yet done. Until it is, the reasoning above is a considered position rather
than a demonstrated one.
