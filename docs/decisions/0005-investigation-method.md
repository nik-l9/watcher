# ADR 0005 — Cortex investigates by a method, not by improvisation

**Status:** accepted 2026-08-23; every decision below has working code except the ask
lifecycle in decision 9, whose free half is shipped and whose expensive half is deliberately
deferred — see `docs/rollout-plan.md` for what each cost against its estimate.
**Question it answers:** we have fixed the same class of bug four times, each time by adding
another disclosure to a connector. What is the actual root cause, and what replaces the fifth patch?

## The pattern, stated plainly

| # | fix | what it disclosed | how it was found |
|---|---|---|---|
| 1-4 | `result_key` | an empty result is empty, not absent | four wrong answers in Slack |
| 5 | `partial_buckets` | this month is not over yet | a wrong month-over-month rate |
| 6 | `series_ends_early` | the series stops before the range does | "signups fell" when tracking had stopped |
| 7 | `blast_radius` | which other events stopped with it | a cause dated a week after its effect |

Seven fixes. One root cause, and it is not in any of the connectors:

> **A connector discloses a problem and does not supply its resolution. The model resolves it
> with whatever is nearest to hand.**

`series_ends_early` said "12 days are missing; this could be three things; do not guess." That
is an unanswerable warning, and an unanswerable warning gets answered by the nearest falling
line on the screen. `blast_radius` is the first of the seven that resolves rather than warns —
which is why it is the last one of this shape we should need.

But the deeper failure is one of **ordering**. Every one of these bugs is the analyst answering
a business question when the answer was a data question. Nothing in the loop decides which kind
of question it is facing, so the business question always wins by default.

## What the literature says, before we design anything

Full detail with citations in `docs/research/` (five documents, ~7,700 lines). The four findings
that change the design:

**1. This failure has a name and a measured base rate.** Frontier models emit a wrong answer
13-25% of the time when the correct answer *is* in their context (14-16% on easy QA). Two named
mechanisms: **presence bias** — reading absence of activity as normality — and
**context-induced confidence**, where supplying any context collapses abstention (one model:
84.1% → 52%). We hit both: thirteen series went silent, "no rows" read as "nothing to see," and
a pile of unrelated tool results licensed a confident answer. See `llm-rca.md` §1.5, §4.1.

**2. Everything measured to work adds structure *outside* the model.** A second independent
call, a fresh context, a vote, a narrower tool. Everything measured to fail asks the same model,
in the same context, to try harder. Intrinsic self-correction is **measurably negative** (GPT-4
on GSM8K 95.5 → 89.0); vanilla reflection in RCA specifically is −2%; multi-agent debate loses
to plain majority voting at equal budget (83.0 vs 88.2). All three were on the list of things to
try. See `llm-rca.md` §5.6.

**3. Kepner-Tregoe's IS-NOT column is the check we were missing, and it is codeable.** Of seven
RCA frameworks surveyed, KT-PA scores HIGH on codeability because its specification matrix is a
fixed schema and its elimination step is a set predicate. Filled in honestly for our bug:

| | IS | IS NOT |
|---|---|---|
| **What** | server-side signup events at zero | browser autocapture (still flowing) |
| **When** | 2026-08-04, inside a 72-minute window | pageviews normal 08-04 through 08-10 |
| **Where** | server-side ingest path | client-side ingest unaffected |
| **Extent** | 13 event types, all server-side | 0 client-side event types |

The candidate "pageview collapse" is tested against the WHEN row and dies: it cannot explain
signups being at zero on 08-04 while pageviews were healthy. See `rca-frameworks.md` §1.5.

**4. Asking the user is worth far less than it feels like it should be.** ConDABench (1,420
problems) reports verbatim that "Longer Conversations are not always better"; heavy clarifiers
show elevated unnecessary-response rates. And a model cannot estimate its own need to clarify —
six independent measurements agree (clarification-need F1 0.33-0.37; one benchmark's R² is
*negative*, worse than a constant; ambiguity-detection 54%). See `clarifying-questions.md` §1.4,
§4.1.

## Decision

### 1. A data-trust gate runs before the business question, and can refuse it

Not a fifth disclosure. An ordering. Eleven checks, ordered cheap-before-expensive and
metadata-before-data, each resolving to exactly one of three actions — and defining the actions
precisely matters more than the thresholds, because an ambiguous action is what turns a gate
back into a warning nobody reads:

- **BLOCK** — the business question is not answered. The response is the data verdict: what
  stopped, when, on which emitter, and what would have to be true to answer it. Reserved for
  cases where the requested computation is *arithmetically void*, not merely uncertain.
- **DEGRADE** — answered, with a stated limitation that changes the answer's scope, and naming
  what was cut: a narrowed range, an excluded series, "this covers 6 of 19 event streams."
- **ANNOTATE** — answer unchanged, a note travels with it.

**At most two checks may ever BLOCK.** Not a preference. Microsoft's experimentation platform
sets its blocking threshold at p < 0.0005, explicitly conservative to hold down false positives,
and Airbnb reported that a single-tier framework gets disabled by its own false positives. A
third BLOCK candidate goes to DEGRADE until it earns promotion with recorded evidence.

Every BLOCK is overridable by a human with a recorded reason stored on the answer — and
**an override must never widen a threshold.** Borrowed from Monte Carlo's `Expected` status,
along with the constraint that ships with it: "Alert statuses do not provide feedback to the
models that generate thresholds." Otherwise every dismissal quietly disarms the check that
would have caught the next break.

The two BLOCK checks are the range overlapping an open un-backfilled incident, and
emitter-partitioned correlated cessation: ≥3 qualifying series ceasing within a 6-hour window on
one emitter **while at least one sibling emitter stays healthy**. Below that bar — 2 series, or
no surviving sibling — it degrades rather than blocks, because with no healthy sibling a real
product outage cannot be ruled out. A single series ceasing never blocks; it is genuinely
ambiguous.

Read-time gating is precedented but **rare**, and the ADR should not overstate it. Two confirmed
cases: Microsoft ExP withholds the scorecard until the sample-ratio check passes, and
Booking.com hides comparative statistics. Airbnb's "Wall" and Netflix's write-audit-publish are
**write-time** gates, which is a different thing. Uber warns and lets the human proceed. Spotify
checks but does not gate.

**And the composition is ours, not the industry's.** The search for prior art on the specific
signature came back empty, and that is a finding rather than a gap in the search: grouping
correlated signals by shared upstream identity is established and defaulted (Alertmanager
`group_wait=30s`), but always to *suppress noise*, never to conclude anything; data-observability
vendors group by schema or lineage — the *destination*, never the *emitter*. Nobody publishes
grouping metric series by upstream emitter, using a healthy sibling emitter as corroborating
evidence, treating correlated cessation as evidence about the cause class, or letting any of it
gate an answer. No thresholds exist to inherit for it.

So the thresholds carried here are **our choices with reasoning, not citations**, and the ADR
must not write "industry standard" beside them. What is citable is that every component is
independently established, and that the tools' own docs admit the gap in exactly the metric
class where our failure lived — Monte Carlo's row-count alerts offer "only 'all rows' since
there is no differentiation", Segment ships no volume default and recommends one chart per
event, and all fourteen of PostHog's ingestion warnings are arrival-conditioned, so none of them
can see an event that stopped arriving.

One consolation: the same search found no published argument *against* it either.

### 2. Kepner-Tregoe is the backbone, but it is layer 2 — and the ordering is the finding

The framework survey (eight frameworks, 3,378 lines) picks KT-PA as the backbone on four
grounds: it eliminates rather than ranks, it requires onset as a field where no other framework
does, its IS-NOT column is enumerable as a `GROUP BY`, and it is independently named as one of
six methods in DOE-NE-STD-1004-92.

**But KT has no data-trust gate, and on our bug it would have gone wrong a second way.** KT's
spec matrix takes facts as given. Fill in "IS: server-side signup events at zero from
2026-08-04" and KT correctly eliminates the pageview candidate — then keeps hunting for a
*world*-cause of an *instrument* failure. It would have asked "what changed on 08-04 that
stopped people signing up?" when the question was "what changed on 08-04 that stopped us
recording?" — and found a deploy, or a pricing change, and reported it as a demand-side cause of
a recording failure.

So:

> **Elimination fixes the answer. Only the gate fixes the question.**

That sentence is the reason decision 1 comes first, and it is a better reason than the one this
ADR originally gave. DOE states the principle as a rule rather than an ordering preference: *"The
smoke detector and alarm functioned as intended; the problem to be solved is the dust in the air,
not the false fire alarm."* Three things must be separated — **the reading, the instrument's
state, and the world-state** — and the instrument must be resolved *before the problem is
stated*, because which of the other two is "the problem" depends on the answer.

The onset test remains, and remains one line:

> **A candidate cause whose own onset is later than the deviation's onset is eliminated, not
> ranked lower.**

Eliminated in code, before the model weighs anything, and it requires every candidate to carry a
machine-readable onset — a schema constraint on the generator, not a research problem. It is the
most *informative* of the checks, because it returns "impossible" rather than "unproven".

### 3. Two independent lines of evidence, or the tree does not narrow

From DOE-NE-STD-1004-92, and found in no other framework surveyed. A cause needs two independent
lines of evidence; with one, the investigation **stays broad** rather than narrowing.

This is the cheapest catch for our bug and the most robust, because it fires on **evidence
cardinality before any content reasoning** — it needs nobody to notice the dates at all. The
wrong answer had exactly one line: a correlation over a non-overlapping window. It is also the
least informative: it says "not proven", never "impossible".

Two other DOE rules come with it. **Actionability, not depth, is the stopping rule** — which
retires "five whys" as a stopping criterion, since five is not a number Toyota defends. And the
output is a typed triple: **one direct cause, one root cause, at most three contributing** — a
schema that can be *unfillable*, which is a fourth independent way to refuse.

Counting them: the gate, an empty elimination set, the two-evidence rule, and an unfillable
output schema. **Four independent ways to return no answer. The system that produced our bug had
zero** — and a system that must produce a cause always will.

### 4. What we are not using, including one that would have made it worse

**Pareto would have ranked our wrong answer first.** It is admitted only for ranking *already
verified* contributors, never for selecting among candidates. Juran's own formulation contains no
numbers at all; the "80/20" is later folklore, and the Juran Institute's licence to re-slice until
a vital few appears is benign at human speed and a defect generator at machine speed.

**5 Whys is discarded as a method** — no elimination step — though its point-of-cause
localisation survives, already subsumed by KT's spec step. **Ishikawa generates and never
rejects**, so it is demoted to the candidate *generator*. **A3's gate is a person**, which is
not a gate we can run.

### 5. A second gate, for identifiability — can *any* cause be established?

Decision 1's gate asks whether the instrument worked. This one asks a different question that
also has to be answered before a cause is named: **given what we can observe, is a causal claim
available at all?** Both are needed, and neither substitutes for the other.

`metric-attribution.md` derives a gate of eight predicates (G0-G7) a program can evaluate, two of
which run **on the shapes alone, before any values are read**. Applied to our own signup series it
produces the finding that matters most for this ADR:

> **We have zero admissible control series.** Abadie's placebo p-value is
> `p = 1/(J+1) · Σ I₊(rⱼ − r₁)`, the `j=1` term is always 1, so `p ≥ 1/(J+1)` — and with `J = 0`
> donors, `p = 1` identically.

That is arithmetic, not a modelling limitation. **No synthetic-control claim is available to us at
any effect size**, and none of DiD, synthetic control or CausalImpact is blocked on being hard to
implement — all three are ~20-90 lines of numpy. They are blocked on an input we do not have.

Going from `J = 0` to `J = 1` moves `p_min` from 1.0 to 0.5; **`J = 19` is what buys `p ≤ 0.05`**.
So the infrastructure requirement is quotable: roughly twenty comparable, independently-unaffected
series — per-region, per-plan, per-acquisition-channel signup series would do — plus someone
willing to attest their exposure.

**What is available, and it is not a consolation prize.** The Chernozhukov-Wüthrich-Zhu conformal
permutation test yields a valid significance statement with a single treated unit and *no control
series*, which is exactly our case. Verified against a rebuilt replica of our real data: the
2026-06-17 break gives `p = 0.0143`, while two in-time placebos give `0.881` and `1.000`. Its
resolution floor is `1/T`, and disclosing that floor is part of the claim.

**So we can now answer the question that started this.** The June movement is real: signups fell
197/day → 82/day on 2026-06-17, a 58% drop at 16× the minimum detectable effect, `p = 0.014`. The
break is established. **Its cause is not, and cannot be from this data** — there is no series the
change did not touch, so no counterfactual can be constructed. That is a stronger and more useful
statement than either "signups fell because of the reverse-proxy change" or "inconclusive".

**A refusal must carry the thing that would lift it.** The failure mode to design against is not
saying "I don't know" — it is saying it uninformatively. Each refusing gate emits what is missing:
G3/G4 says "no unaffected comparison series exists, so `p ≥ 1` and a causal estimate is unavailable
by construction," and the fix is one admissible control series. This matters more than usual,
because the alternative to an informative refusal is not silence — **it is a human inventing a
cause.**

Two received beliefs this document falsified, both worth recording because we would have relied on
them:

- **"A contaminated control gives a conservative lower bound" is false in additive DiD.** Verified:
  the estimate crosses zero at `c* ≈ 0.72` and *flips sign*. It holds exactly in logs, which makes
  the specification choice load-bearing rather than stylistic.
- **A plug-in-MLE BSTS-lite drives `σ²_level` to zero**, so its intervals do not widen with
  horizon (±32/±32/±32 against a correct ±34/±41/±61) — losing precisely the property Brodersen
  et al. name as a key characteristic of the method. A cheap approximation of CausalImpact is not
  a cheap CausalImpact.

**Implementation order and size**, since all three are small: optimal partitioning with a `min_len`
constraint over a calendar-reindexed three-state series (~40 lines, and the reindex is the only
guard against a documented silent failure); then the conformal test (~25 lines); then G0-G7 (~120
lines of predicates). Everything else consumes the first one's output — sections 1 and 2 of that
document both assume someone has already chosen the windows.

### 6. A sufficiency gate vetoes causal claims the evidence cannot carry

A separate call, asking only "does the evidence I hold support a definitive answer?", with veto
power. +2-10% correct-among-answered, 93%-accurate autorater. This is the best-evidenced
intervention aimed at our exact failure, and it is the same shape as the data-trust gate: a
question about the evidence, asked separately from the question about the business.

### 7. Verification of a causal claim stops showing the verifier its conclusion

Our adversarial verifier is closer to the literature than expected: each claim is judged in its
own call, against only its cited evidence, instructed not to reason from the wider
investigation. What it still does is **show the model the claim and ask whether the evidence
supports it**, which anchors the judgement on the conclusion. The measured form asks the
evidence an open question in a context that does not contain the proposed answer (precision 0.17
→ 0.36). For causal claims specifically, that is a change to a prompt we already ship.

### 8. Retrieved-but-unread becomes explicit state

Our bug in one sentence: `list_events` was fetched, contained the answer, and was never read.
An agent that cannot enumerate what it has fetched cannot notice what it has ignored. Bulk tool
results go behind a handle with a head shown, and *reading* the full result becomes a deliberate,
logged action.

### 9. Asking the user: one gate, at intake, narrow by construction

Every term in the expected-utility calculation pushes the same way for us — a question in a
public Slack channel is latency-visible and breaks the one promise the product makes, and our
users are by construction in a hurry, which *lowers* the threshold for acting. So the ask band
is narrow. Asking is on the table only when all of these hold:

1. The ambiguity is **structural** — *n* entities match, *k* metric definitions exist, no
   baseline named — never a model's sense of vagueness, because that number cannot be estimated.
2. The candidate interpretations have been **evaluated** and disagree on the *conclusion*, not
   the number. Our answer space is enumerable and each candidate is a sub-second query, so this
   is measured rather than guessed — and it replaces most asking.
3. No substitute, reported range, or reframing makes the disagreement moot.
4. A human is actually waiting. A scheduled run or a public channel prices asking at infinity.
5. It is **intake, before the first tool call**. Goal-clarification value decays to baseline
   after ~10% of execution and is net-negative past the midpoint.

At most one question, ever, and it always carries a default it decays to.

**The highest-leverage move here is not the ask decision at all.** The gap between a wrong
assumption stated and a wrong assumption hidden is enormous and entirely under our control. So:
always state the assumption. That shrinks the band we have to get right.

A missing source is **never** a question — that is a capability fact, and the honest response is
to reframe to the answerable neighbouring question or to state the gap and its sensitivity.

## What this costs, stated before it is built rather than after

**The gate's real cost is infrastructure, not tokens.** It is cheap to run and expensive to
supply. It needs per-series ingest health, schema-version history by series, a collection-outage
log, partial-bucket flags, and a mapping from each series to its collection path — so that
"siblings on the same path stopped together while siblings on other paths did not" is a *query*
rather than an inference. We have some of this for PostHog and almost none of it generally.

That matters more than any threshold in decision 1, because:

> **A gate whose checks all return `unknown` passes everything.**

So the honest description of decision 1 is a metadata-plumbing project with a gate on the end,
not a prompt change. Sequenced the other way round it is worse than nothing: it looks like a
safeguard and behaves like a pass-through.

**The gate will read as a regression.** It produces non-answers — "the data does not support
answering this" — to questions a human would have taken a swing at. Anyone measuring the system
on answer rate will see the fix as a degradation, and they will be reading the number correctly.
The trade is still right: a confident wrong causal story is worse than a refusal, because the
refusal is correctable and the story gets acted on. But that argument has to be made and accepted
*before* this ships, not produced defensively afterwards.

**And there is a class of question we have no method for at all.** KT is scoped to special-cause
deviations. A large share of "why did metric X change" is drift, seasonality, mix shift, or a
definition change — none of which have a crisp onset, and none of which the eight frameworks
surveyed handle. DOE names the gap precisely ("not recognizing the introduction of gradual change
as compared with immediate change") and then does not close it either. The design's answer is
that step 1 classifies the deviation and **terminates on drift, saying we do not have a method**.
That is honest and it is not sufficient, and it should not be discovered later as a surprise.

## What we are explicitly not building

- **A self-critique pass.** Measurably negative, twice, once in RCA specifically.
- **A debate architecture.** Loses to majority voting at equal token budget.
- **A BLEU/ROUGE eval.** Metric-human correlation ranged −0.42 to +0.62 and *inverted* the
  model ranking.
- **An unreferenced LLM judge.** 14 false accepts out of 20 in the source study.

Each of these was a plausible next step. The research's main practical value is the four things
it stopped.

## Consequences, including one for the eval suite

Trained human analysts, given the same incident report, agree on the free-text root cause **54%
of the time** (κ=0.46), and no paper in the LLM-RCA literature reports an inter-rater κ for its
own human grading. That caps what our `accuracy` dimension can mean, since it is scored against a
single labelled cause.

`_accuracy` already avoids the worst version of this: it string-matches required signals rather
than asking a model whether two prose explanations agree, and its docstring gives the right reason
— "the headline accuracy number would depend on a model's opinion." The signal is structured. The
**search space is not**, and checking that turned up a live defect:

`_report_text` flattens every piece of prose in the report into one string — including
`hypothesis.statement` for hypotheses the report **contradicted**, plus risks and data-quality
notes. So a report that names the planted cause in order to *reject* it matches the required
signal and scores as having found it. Verified: a report whose summary blames a pricing change,
carrying one `CONTRADICTED` hypothesis mentioning deploy `91c3e`, matches the signal `91c3e`.

The eval has therefore been crediting the precise failure it exists to punish, and every accuracy
figure this project has quoted is an upper bound rather than a measurement. Fixing it means the
search space is the report's **asserted** surface — summary, findings, supported hypotheses,
recommendations — and not statements the report itself says are false. This invalidates the
existing accuracy baselines, which is a cost worth paying to know the number means something.

## Honest caveats

Three of the six interventions above are measured **outside** RCA — they come from QA, biography
generation and grade-school maths. Forced hypothesis enumeration, the intervention this project
would most like to believe in, has **no controlled RCA measurement** anywhere we could find; its
support is one voting ablation and one benchmark naming premature commitment, not an A/B.

And the single most relevant experiment for our bug does not exist in the literature: **nobody
has measured whether an agent notices an anomaly it was not told to look for.** Every
needle-in-a-haystack result assumes the model knows a needle is being requested. That experiment
is ours to run, and it is the one this ADR most wants to see before its middle sections are
treated as settled.

All five research documents are complete — ~14,000 lines, primary sources, with unverifiable
claims marked rather than smoothed over. Three sourcing holes are recorded in them rather than
papered over: ISO 13053-1 clause 12.2 is paywalled, DOE Figure E-1's step labels are an image, and
Kepner-Tregoe is encoded from the Virginia Tech and AHRQ reproductions of the matrix rather than
from *The New Rational Manager* — so what this ADR proposes is a defensible reconstruction of KT,
not "we implemented KT".

Decision 3 is weaker than it reads, and decision 5 says why. Two independent lines of evidence is
the right rule, but in a single-warehouse architecture "two ways you know it" is usually two views
of one source — and we now know `J = 0`, so there is no second, unaffected series to be the other
line. The two decisions constrain each other, and the honest reading is that decision 3 will
usually resolve to "stay broad" until the control-series infrastructure exists.

Two methodological cautions from the framework survey, both worth keeping:

**The codeability ranking biased its own answer.** The survey scored each framework on how
encodable it was, and KT won. But the single most important thing it found — the data-trust gate
— is one clause inside DMAIC, a framework that scores *poorly* overall because most of it is
belts, training and maturity levels. A table of per-framework averages will always select KT and
always miss the gate. Worth remembering the next time this project ranks options by an aggregate.

**The best source was the least famous one.** DOE-NE-STD-1004-92, a free 1992 US government PDF,
contains more directly encodable rules than any of the named frameworks — including the
two-evidence rule, which appears nowhere else in the survey.

And what the survey concedes a program cannot decide alone: whether an IS-NOT is genuinely
*comparable*, and whether a decomposition is really mutually exclusive. Both need schema-level
configuration — a declaration of which of our dimensions are comparable to which — rather than a
cleverer prompt. Also unresolved: real evidential independence, since in a single-warehouse
architecture "two ways you know it" is usually two views of one source. Decision 3 is weaker than
it looks for that reason.

`data-trust-gate.md` is complete, and decision 1's thresholds above are taken from it. Its
severity model is worth noting because it inverts the obvious design: **severity comes from the
asset's declared consumers, not from which check tripped.** A freshness failure on a series
nothing reads is not an incident. Blast radius is *reported and never thresholded* — the
published mechanism everywhere is declared consumers plus a graph walk, and nobody publishes a
number for it, so inventing one would be the kind of false precision this ADR is trying to
avoid.

One check in that gate is **not implementable today** and is recorded as an annotated gap rather
than quietly dropped: source-versus-destination reconciliation needs an emitter-side count, and
we have no way to ask an emitter how many events it believes it sent. A per-emitter heartbeat
canary is the top follow-up, and until it exists the gate cannot distinguish "the emitter sent
nothing" from "the emitter sent it and it was lost in transit".
