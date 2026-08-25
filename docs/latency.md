# Latency: where it goes, and what the literature actually offers

The north star is **90 seconds**. A real investigation now takes 143, and `insufficient_evidence`
reached 206 in eval 15. This document measures where the time goes, then reads three 2026 papers
against that measurement — because the failure mode with performance work is optimising the wrong
phase, which costs a day and buys nothing.

Every paper here was read in full, not from its abstract. That rule exists because an earlier
round of this work put an abstract-derived figure into a code comment and it was wrong by an order
of magnitude.

## The measurement first

One real investigation, `--real --tenant acme`, "Why did conversation volume change in the
last two weeks?", with the survey and per-tenant capability scoping in place:

```
phase breakdown (143s wall, 134s measured)
  phase        calls  seconds  share    mean  tokens
  loop_model       3     57.0   40%    19.0  in 6 out 4,433 cached 37,343
  loop_tools       8     23.7   17%     3.0  in 0 out 0
  draft            1     40.3   28%    40.3  in 1 out 3,451
  gate             1      0.0    0%     0.0  in 0 out 0
  verify           1     12.6    9%    12.6  in 876 out 1,083
  unmeasured              9.5    7%
  cache hit rate: 98%
```

Two facts follow, and they decide everything below.

**1. Output generation is 68% of wall clock.** `loop_model` plus `draft` is 97 of 143 seconds, for
about 7,900 output tokens — roughly 80 tokens per second, which is simply the model's generation
rate. Input is free by comparison: 98% cache hit, 37,343 cached tokens read for 6 fresh ones.
**We are output-bound, and have been since the first time this was measured.** No amount of
prompt trimming helps; only generating fewer tokens, or generating them in parallel, does.

**2. The tool phases were serial; the verifier already was not.** `loop_tools` is 8 calls at a
mean of 3.0 seconds for 23.7 seconds total — a serial sum, and the loop confirmed it:
`for request in response.tool_requests:` awaited each call in turn. The survey's four discovery
calls were serial too.

**Correction to an earlier draft of this document, which claimed the verifier was also serial and
worth ~9 seconds.** It is not: `AdversarialVerifier` has judged claims concurrently since it was
written, bounded by `_VERDICT_CONCURRENCY = 4` — deliberately bounded, because twenty simultaneous
requests reliably provoke the provider capacity errors that then surface as unverified claims. The
claim came from a `grep` whose output was truncated before it reached the verifier's matches, and
the 12.6 seconds it costs is *already* the concurrent figure. There is no win there to take.

## Paper 1 — PASTE: Parallelizing Tool Execution and LLM Generation

[arXiv 2603.18897](https://arxiv.org/html/2603.18897v3). Headline: **43.5% lower task completion
time**, tool latency down 1.8×.

Their mechanism is genuinely clever. A Pattern Analyzer mines recurring control-flow patterns from
historical traces, splitting events into stable *signatures* and volatile *payloads* — so
"search → visit" predicts that a web-fetch is coming while the concrete URL is copied from the
current search output. Predicted calls execute concurrently with generation on an isolated
speculative path; on a hit the result is reused or the in-flight job is promoted, on a miss it is
discarded without touching session state. Side-effecting tools need an explicit safe-speculation
policy; their ablation caught 602 potentially-mutating speculative actions out of 20,000+ and
prevented all of them from committing.

**It does not apply to us, for three independent reasons.**

- **It requires control of the inference server.** The implementation is a middleware on the
  tool-dispatch path *plus a Python startup hook inside a vLLM server*, and the LLM-side pacing is
  half the gain. We call a hosted API. Their own ablation (PASTE-Tool-Only) shows that accelerating
  tools *without* LLM-side pacing can make end-to-end latency worse by increasing queueing — which
  is precisely the half we would be limited to.
- **Their decomposition is the inverse of ours.** They measure tool execution at **45–57%** of
  agent latency; ours is **17%**. Their gains are "largest for tool-heavy tasks". We are the other
  kind, and the ceiling on the whole idea here is 17% before any speculation misses.
- **Speculation costs vendor quota we do not own.** Top-1 accuracy is 27.8%. In their setting a
  wasted speculation costs local compute; in ours it costs a real GitHub, Slack or PostHog call
  against a per-tenant rate limit we added deliberately, for data a customer is paying for.

What we do take: their load finding. Under 192 concurrent sessions, LLM generation time grew
**17×** while tool time changed modestly. If Cortex is ever loaded, the output-bound share gets
worse, not better. That argues for reducing generated tokens rather than for chasing I/O.

## Paper 2 — W&D: Scaling Parallel Tool Calling for Efficient Deep Research Agents

[arXiv 2602.07359](https://arxiv.org/html/2602.07359v1), Salesforce AI Research. **This one
applies, and it needs no infrastructure we do not have.**

They scale an agent along two axes: *depth* (more sequential steps, the usual approach) and
*width* (more tool calls per step). Width is implemented **by prompting alone, with no
fine-tuning** — a per-step user instruction of the form *"you MUST make at least m but not more
than m+1 function calls in a single response"*.

The sweep over 1, 2, 3, 5 and 8 calls per turn on BrowseComp, HLE and GAIA found three the best,
with performance plateauing or declining beyond it. Against single-tool calling on BrowseComp:

| | accuracy | cost | wall clock | turns |
|---|---|---|---|---|
| 1 tool per turn | 66% | $102.50 | 1522.6s | 45.7 |
| 3 tools per turn | **68%** | **$65.70** (−35.9%) | **904.2s** (−40.6%) | **23.8** |

Accuracy went *up* slightly while wall clock and cost fell by a third or more. Their explanation is
that width buys three things: a broader search scope, redundancy that survives a tool failure, and
natural decomposition of a compound question into simpler ones.

**Why it fits our measurement.** Our `loop_model` is 40% of wall clock across 3 calls at 19
seconds each. Fewer turns means fewer of those 19-second calls, and turns are exactly what width
reduces — 45.7 to 23.8 in their data.

**The condition they state and we must respect.** Their limitation section is explicit that the
paper does not address *tools whose arguments depend on a previous tool's output*; every example is
independent parallel calls. Our investigations contain exactly those chains —
`list_repositories` → `recent_prs(repo=…)` — so width applies to breadth (several events, several
periods, several repositories) and not to dependency chains. Our environment survey helps here by
accident: it resolves the commonest chain (*what exists* → *measure it*) before the loop starts,
which leaves more of the remaining work genuinely parallel.

**And a prerequisite our own data exposes.** Width is only a latency win if the width is
*executed* in parallel. We execute tool calls serially inside a step, so asking for three per turn
today would trade fewer 19-second model calls for longer serial tool phases, with an unknown net
effect. The two changes are one change: parallel execution first, then width.

## Paper 3 — When Agents Go Quiet: Output Generation Capacity and Format-Cost Separation

[arXiv 2604.16736](https://arxiv.org/html/2604.16736). Headline: **48–72% fewer generated
tokens**, and 2.5× faster wall clock, via deferred template rendering.

Their Format-Cost Separation Theorem separates *content cost* from *format cost*: the model emits
content as structured JSON, and a pre-registered template renders the target format afterwards at
zero LLM cost. Measured format multipliers: raw text 1.00, Markdown 1.05, **JSON 1.15**, HTML 1.20,
LaTeX 1.30, python-docx 1.40. Asymptotic saving is ρ → 1 − μ_JSON/μ_f.

**We are already at the floor of this technique, and the theorem says so exactly.** Our drafting
call emits JSON, so μ_f = μ_JSON = 1.15 and ρ → 1 − 1.15/1.15 = **0**. There is no deferred
rendering to do, because our target format *is* their intermediate representation. We also already
apply the underlying idea beyond format: charts and sources are withheld from the drafting schema
and derived deterministically from evidence — which is content-cost separation, not just
format-cost separation, and was done for grounding reasons rather than latency ones.

Their second contribution does not reach us either. Output Generation Capacity degrades as a
sigmoid in context occupancy o/C, dropping below 50% of raw headroom at o/C ≈ 0.55 (k ∈ [8.2, 9.1],
r₀ ∈ [0.52, 0.58]). Our drafting call carries roughly 38,000 tokens against a 1,000,000-token
window: **o/C ≈ 0.04**. We are nowhere near the knee.

Worth recording anyway, because it *nearly* explained something real: eval 15's drafting timeout at
300 seconds looked like their output stalling. It was not — stalling presents as an empty response,
and we got a slow complete one. The cause was simply more report to generate.

## What this leaves

Ranked by expected saving against risk, from our own numbers rather than from any paper's headline:

| # | Change | Evidence | Expected | Risk |
|---|---|---|---|---|
| 1 | Execute a step's tool calls concurrently | ours: 8 calls, 23.7s serial, mean 3.0s | ~15s (10%) | Touches the executor's F-01 scoping and F-05 savepoints. `AsyncSession` is not concurrency-safe, so the network phase must be separated from the DB phase rather than simply wrapped in `gather` |
| 2 | Survey the four discovery capabilities concurrently | ours: 1.8s serial warm | ~1.3s (1%) | Low — one call site, all four independent |
| ~~3~~ | ~~Verify claims concurrently~~ | **already done** — `_VERDICT_CONCURRENCY = 4` | **0** | — |
| 4 | Ask for ~3 tool calls per turn | W&D: −40.6% wall clock, accuracy +2pp, prompting only | large, unquantified for us | Behavioural. Requires (1) first, and an eval run — the last three prompt changes each cost something the suite caught |
| 5 | Cache the survey per tenant | ours: survey is per-investigation and its content changes daily at most | ~2s plus 4 vendor calls | Low, but needs an invalidation story |

Deliberately **not** on the list: speculative tool execution (needs inference-server control we do
not have, and spends customer rate limit on 27.8% Top-1 accuracy), deferred template rendering
(ρ = 0 for a JSON target), and raising `effort` (measured worse: 6/6 at low against 4/6 at high,
and thirty times the cost).

The uncomfortable honest note: none of these attacks the 68% directly. Output-bound means the
floor is set by how many tokens the report and the reasoning need, and the only real levers there
are generating a shorter report — already done for factual questions, and the suite punished going
further — or generating in parallel, which for a single coherent report is not available. Getting
from 143 seconds to 90 will take items 1 through 4 *and* an accepted reduction in how much the
analyst says.


## Result of items 1, 2 and 4 (2026-08-12)

Implemented: concurrent in-step tool execution (`ToolExecutor.execute_many`), the survey's four
discovery calls overlapped, and width prompting at ~3 calls per turn. Item 3 turned out to be
already done. Item 5 (caching the survey per tenant) is not done — at 1.0s measured it is no
longer worth an invalidation story.

**Live, same question as the baseline:**

```
                 baseline        after
survey           (unmeasured)     1.0s   1%
recall           (unmeasured)     4.0s   3%
loop_model    57.0s  40%         64.1s  50%   3 calls
loop_tools    23.7s  17%         11.6s   9%   7 calls in 2 batches
draft         40.3s  28%         35.6s  28%
verify        12.6s   9%          7.8s   6%
unmeasured     9.5s   7%          3.1s   2%
wall         143s              127s
```

Four calls issued together in step 1, three in step 2; steps fell from 5 to 3. `loop_tools` went
from 8 serial calls at 23.7s to 7 calls in 2 overlapped batches at 11.6s. Instrumenting `survey`
and `recall` cut the unmeasured remainder from 25.5s (on an intervening run) to 3.1s.

**The eval is the honest measurement**, because single live runs cannot separate a change from
variance — the same question took 143s, then 215s, then 127s. Over 6 eval attempts (run 16 vs
14): mean latency 93,424ms → 85,489ms, −8.5%; the worst scenario −16.0%; attempts inside the
90-second budget 3/6 → 4/6. Scores slipped 0.98/1.00/0.97 → 0.97/0.98/0.96, inside the noise band
but not nothing.

**Where the remaining time is, unchanged in character.** `loop_model` plus `draft` is ~78% of the
post-change run, all serial token generation at roughly 80 tokens per second. Concurrency has now
taken what concurrency can take. The next real lever is generating fewer tokens — and the two
attempts we made at that (a shorter report shape, and a reflection turn) cost accuracy and
latency respectively, both caught by the suite. That is the wall, and it is an output-rate wall
rather than an architectural one.
