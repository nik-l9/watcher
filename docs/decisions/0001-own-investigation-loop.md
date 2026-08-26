# ADR 0001 — Cortex owns the investigation loop

**Date:** 2026-07-30
**Status:** Accepted. Its *conclusion* stands; its reasoning was corrected by
[ADR 0002](0002-harness-mechanisms.md), which found two harness mechanisms this ADR wrongly
concluded there was nothing to inherit from.

## Context

The implementation plan recorded a decision to build on an existing general-purpose agent SDK as
a library, plus a hosted agent server for per-investigation Docker sandboxes.

What was built diverges from that. `cortex/agents/investigator.py` drives its own loop against
this project's own `LLM` interface, and no agent SDK is a dependency.

This ADR exists because that divergence happened during implementation without being written
down. It was raised as a fair challenge: mature, research-backed agent runtimes exist and are
built by dedicated teams, so why is a hand-rolled loop preferable?

## Decision

Keep the investigation loop here. Adopt a hosted runtime for sandboxing if sandboxing is ever
needed, not for the agent.

## Why the loop stays

A general-purpose agent SDK models a **software-engineering** agent. Its tools are bash, file
editing and a browser; the sandbox is the point; the stop condition is "the task is done"; its
context management is tuned for long coding sessions. Its research results are reported on coding
benchmarks, and benchmark maturity is task-specific — it does not transfer to causal analysis of
business metrics, because the thing being optimised is different.

This loop is not a specialisation of that. It needs:

- a hypothesis ledger tracking support **and contradiction** per hypothesis, since the product's
  value is rejecting the plausible-but-wrong answer;
- one immutable `Evidence` row per tool call, with the model seeing evidence **ids** and nothing
  else;
- a gate that removes any claim whose ids do not resolve, in code, before render;
- a separate drafting call under a constrained schema.

The grounding guarantee — the actual product — depends on this codebase controlling what enters
context. If a framework owns the loop, the guarantee weakens from *code-enforced* to *the
framework did not drop the ids*, which is not a guarantee. A coding agent establishes correctness
by running tests; this one establishes it by citation.

The same argument covers read-only-ness. V1 ships no write tools, so "humans approve" holds by
absence of capability rather than by prompt. Adopting a bash-in-a-sandbox agent would end that
property.

Adapting such an SDK to this use case would mean replacing its tools, its prompts, its stop
condition and its context strategy. That keeps the scaffolding and discards precisely the parts
its maturity applies to.

## What this cost, and where the reasoning was wrong

A mature provider layer has retry, fallback and timeout policy already written and exercised at
scale. Four of the live-path defects in [`findings.md`](../findings.md) are provider-transport
plumbing:

| Finding | Defect |
|---|---|
| F-15 | streaming request with no deadline |
| F-16 | timeout escaping as a transport exception, killing the process |
| F-18 | one per-request ceiling for calls of very different length |
| F-19 | capacity error delivered inside a 200, so nothing retried it |

A mature provider layer would likely have handled or exposed all four. Building that layer by
hand is the part of the divergence that was a mistake — not keeping the loop. Each of those was
discovered by a separate multi-minute live run, one at a time.

The remaining findings in that group (F-11 schema subset, F-12 tool-result pairing, F-13 error
opacity, F-14 key resolution, F-17 grammar size) are API contract issues that any caller hits,
framework or not.

## Consequences

- `AnthropicLLM` stays. Rewriting it onto a multi-provider layer now would reintroduce risk for a
  cost already paid — it is covered by tests including regressions for every finding above.
- Revisit a provider abstraction layer at the **second** provider, which is where one earns its
  keep.
- Use a hosted agent sandbox only where it is genuinely better than anything written here. The
  matplotlib chart escape hatch was the candidate; it was **declined outright** (see
  `cortex/reports/png.py`), so no sandbox container is declared any more.

## Evidence available at the time

The first eval run to score analyst behaviour rather than fail on infrastructure:

- 2 of 3 scenarios passed, **hallucinations = 0**
- grounding 1.00 and accuracy 1.00 on both passing scenarios
- `insufficient_evidence` correctly declined to name a cause — the hardest behaviour in the
  suite, and the one the evidence-and-gate machinery exists to produce
- weaknesses measured, not guessed: decoy rejection 0.33 and 0.75, latency 193s and 204s against
  a 90s budget

That last line is why this ADR is not the end of the argument. It records a considered position
supported by early measurement, and the loop has been measured continuously since.
