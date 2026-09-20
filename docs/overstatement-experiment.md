# Overstatement factorial — predictions registered before the run

Three candidate fixes for a measured defect: on a live investigation, 8 of ~15 drafted claims
failed verification (7 `overstated`, 1 `unsupported`), and two of the eight were factually
wrong rather than merely strong -- sixteen HubSpot records with `lifecycle_stage="lead"`
described as "prospect and customer companies", retained and delivered.

## The arms

Each fix is a prompt fragment behind an environment toggle, so an arm differs by exactly one
thing and combinations compose. Eight arms: baseline, 1, 2, 3, 1+2, 1+3, 2+3, 1+2+3.

| fix | where | what it says |
|---|---|---|
| **1** | verifier system prompt | the wrong-figure exemption is about NUMBERS only; a contradicted category, status or label is `unsupported`, not `overstated` |
| **2** | drafting guidance | an absence is only as wide as the query that looked for it |
| **3** | drafting guidance | match the strength of the claim to the strength of the warrant |

## Metrics

Taken from the verifier's own verdicts in each captured bundle, which do not change shape
between arms.

- **M1** `overstated` per report — retained, so this is what reaches the reader shaky
- **M2** `unsupported` per report — removed
- **M3** = M1 + M2, total non-supported
- **M4** total claims drafted

## Predictions, written before the first run

**Fix 1 shifts M1 into M2 and leaves M3 roughly flat.** It is a reclassification, not a
drafting change: the same claims should be caught, and more of them removed rather than
softened. If M3 moves much, something other than the intended mechanism is acting.

**Fix 2 reduces M3**, and the reduction should be concentrated in absence claims. It targets
3 of the 8 observed failures, so the effect should be visible but partial.

**Fix 3 reduces M3 broadly**, and is the arm most at risk of a false win: arXiv:2604.19768
measured that a general instruction to engage with uncertainty raises genuine hedging *and
leaves miscalibration elevated*, producing text that reads calibrated and is not. Our
verifier would likely score that as `supported`.

**M4 is the guard against exactly that.** If M3 falls because M4 fell, the fix suppressed
claims rather than improving them, and ResearchLoop (arXiv:2605.28282) shows a structured
protocol doing precisely this -- fewer claims, lower token count, better score. A win must
show M3/M4 falling, not just M3.

## What would count as a failure

Any arm where M3 falls and M4 falls by a similar proportion. Any arm where M1+M2 falls but
the reports get visibly vaguer on reading. Fix 3 scoring well while fix 2 does not, given
fix 2 targets a failure mode we actually observed and fix 3 is extrapolated from a
post-hoc evaluator benchmark whose authors do not claim it transfers to drafting.

---

# Result: all three fail. None shipped.

Eight arms, nine hand-written scenarios each, 72 investigations. Analysed as a 2^3 factorial:
each fix is ON in four arms and OFF in four, contrasted within scenario, so the nine
scenario-level contrasts are the independent units.

## The control validated the design

`M4` (claims drafted) cannot be affected by fix 1, which only changes the verifier's prompt.
On the partial four-arm data it read t = −2.83 — a false positive, because the design was not
yet balanced. At eight arms it collapses to **t = +0.67**. The contrasts are clean.

## Main effects on M3 (non-supported claims; lower is better)

| fix | estimate | SE | t |
|---|---|---|---|
| 1 — verifier, numeric-only exemption | +0.38 | 0.33 | 1.13 |
| 2 — absence is as wide as the query | +0.19 | 0.40 | 0.47 |
| 3 — warrant strength | +0.31 | 0.24 | 1.30 |

None significant (|t| < 2.31, df=8). All three point the wrong way. No two-way interaction
reaches significance either (|t| ≤ 1.04).

## Two results worse than "no effect"

**Fix 1's mechanism ran backwards.** It exists to move claims from M1 (overstated, *retained*)
into M2 (unsupported, *removed*). Observed: M1 **+0.62** — the largest single main effect — and
M2 **−0.25**. It retained more, not fewer.

**Fix 3 produced the suppression signature registered as its risk.** M4 **−0.91**, the largest
effect on claims drafted, with M3 unchanged: roughly one fewer claim per report and no fewer
errors. That is the pattern ResearchLoop shows and the reason M4 was tracked at all.

## What this does not say

The defect is still real. Sixteen records marked `lifecycle_stage="lead"` were described as
"prospect and customer companies", judged `overstated`, retained, and delivered — that was
verified directly against the stored evidence. What failed is a *prompt* remedy for it.

Baseline has the lowest error rate of all eight arms (0.192 against 0.277–0.353). It is
tempting to read that as "any addition to the prompt hurts", which would match ForceBench's
dummy-axis control scoring worse than no rubric at all. **The data does not support that
claim**: there is one baseline run, and the measured noise floor is ~1 claim per report
(paired sd 1.7–2.15), which covers the gap. Under the null, a baseline that happens to be
lowest of eight arms occurs one time in eight.

## What the failure points at

Yesterday's premise fix worked because it was **structural** — the four-way enum was split
into three binaries, and the guidance change was secondary. Here there was no structural
change: three prompt fragments against an unchanged verdict taxonomy, and none moved the
number. The next attempt should change what is *asked*, not how it is *worded* — for example
requiring a claim to name the evidence span it rests on, so scope inflation has nowhere to
hide, rather than instructing against it in prose.

---

# What the failure bought: a taxonomy from 212 verdicts

The eight arms produced 72 reports and **212 non-supported verdicts** — 26× the sample the
fixes were designed from. Reading them changes the diagnosis.

## My original diagnosis was overfit to one report

From the Boston investigation I concluded that absence-scope inflation was the dominant
failure: 3 of its 8 flagged claims, 38%. Across 212 verdicts it is **11%**. Fix 2 was aimed
at a problem an order of magnitude smaller than I thought, which on its own explains why it
did nothing. One report is not a sample.

## A fifth of the flags are not drafting failures

43 of 212 (**20%**) begin *"asked as an open question, with this claim withheld…"* — that is
`_reconcile`, our own causal fresh-reading disagreeing with an anchored verdict. That is the
system working as designed, not a drafting miss, and counting it as one inflated the apparent
rate of causal overreach.

Genuine drafting misses: **169**. Of those, **133 are `overstated`**, which the pipeline
retains with reduced confidence — so they reach the reader.

## Half of the genuine misses are one mechanism

85 of 169 (**50%**) cite an extent or quantifier problem: the claim asserts something over a
set, period or alternative wider than the cited evidence spans.

> "only two channels are shown so **'all other traffic'** overstates"
> "only covers through 21 July, so claiming it **'has persisted since'** extends beyond"
> "the issues query only checked since 2026-06-10, not all-time, so **'at all'** overstates"
> "the evidence does not **isolate** the decline to conversion behaviour versus traffic"

Scope, time and absence looked like three categories in a regex and are one bug underneath:
**the claim's quantifier exceeds the evidence's coverage.** The first taxonomy split it four
ways and made each piece look too small to chase.

## Why this points away from prompting

An extent claim is checkable *mechanically* if the evidence carries its own extent — this
query covered these rows, this window, truncated at twenty of seventy-two. Our observations
already know all of that; the claim just never has to state which extent it relies on. That
is a change to what is asked for, not to how it is worded, which is the category of change
that worked for the premise verdict and the category these three fixes were not.

---

# Why the prompt fixes failed: three findings from the follow-up

## 1. Fix A's backwards result is a named effect, and our wording triggered it

*Semantic Gravity Wells: Why Negative Constraints Backfire* (arXiv:2601.08070) measures that
an instruction's explicit mention of a concept **activates** rather than suppresses it —
87.5% priming failure rate, with 2-3x attention amplification on the prohibited term.

Fix A's text: *"a contradicted category, status or label is `unsupported`, **not**
`overstated`"* — which names `overstated` inside a negative clause. Predicted outcome:
`overstated` usage rises. Observed: M1 **+0.62**, the largest single main effect, with M2
**−0.25**. Direction, magnitude ranking and mechanism all match.

The paper evaluates *generation*, not classification, so this is a strong hypothesis rather
than a demonstrated result for a judge. It is also cheap to test: restate the rule as an
action and never name the competing verdict.

## 2. Our null result replicates a published one

*Generalization Bias in LLM Summarization of Scientific Research* (arXiv:2504.00025, Royal
Society Open Science, 10 models, 4,900 summaries) finds LLMs "omit details that limit the
scope of research conclusions" — our defect, named — and that summaries produced under an
**accuracy prompt were about twice as likely** to contain generalized conclusions than under
a simple prompt: OR 1.90, 95% CI [1.11, 3.26], p = 0.02. Newer models were worse than older.

Fixes B and C are that intervention class. Their nulls are consistent with published work at
n=4,900, not an artefact of our design.

## 3. Prompting cannot fix this class, for a mechanical reason

Narrowing a claim's extent is a **downward-entailing** inference, and LLMs are documented as
near-chance on downward monotonicity — harder than upward across models, because it runs
against the generalization heuristic they learn. Our 50% extent share is the predicted
signature of that bias. Instructing a model to perform the inference class it is worst at is
not a promising lever.

# What is mechanically decidable: measured, 22%

The extent data is **already visible to the judge** — `render_evidence` writes
`parameters: {...}` into every evidence block, so windows, filters and limits are in the
prompt today and 50% of misses are still extent errors. Visibility is not the gap;
*comparability* is.

A read-only audit over the 169 genuine misses, comparing each claim's asserted extent against
the bounds recorded in its cited `Evidence.params`:

| | n | share |
|---|---|---|
| genuine drafting misses | 169 | |
| cited evidence resolvable in the bundle | 121 | 72% |
| claim asserts an extent word | 44 | 36% of linked |
| **and cited params carry a bound to compare** | **38** | **22% of all, 31% of linked** |

Worked example, decidable with no judgement:

> claim: "No code deploys occurred in **any of the four tracked repositories** during this window"
> params: `{"repo": "acme/company-website", "since": "2026-05-15", "until": "2026-05-28"}`
> — the claim spans four repositories; the evidence covers one.

So a deterministic extent gate has a measured ceiling of roughly a fifth to a third of genuine
misses. That is worth building and is not a silver bullet, and knowing the ceiling before
building it is the point of running the audit first.

## The deterministic gate does not work over prose — measured, at zero cost

Built as a pure function and scored against all 562 claims in the 72 captured reports whose
cited params resolve. No generation, no API calls.

| rule | true pos | false pos | precision | recall |
|---|---|---|---|---|
| loose: extent word + any bound present | 22 | 52 | **0.30** | 0.18 |
| strict: claim names a value comparable to the bound | 1 | 0 | **1.00** | **0.01** |

Neither ships. The loose rule flags more supported claims than unsupported ones, because
extent words and bounded params are just as common in claims that are *correct* -- "since",
"no ...", "all" alongside an `end_date` describes most legitimate trend claims. The
discriminating fact is whether the asserted extent *exceeds* the bound, and that is a value
comparison, not a word match.

**This corrects the 22% figure above.** That number measured "claim contains an extent word
AND cited params contain a bound", which is a test of *surface features*, not of
comparability. Requiring the two to actually be comparable takes it to roughly 1%: "the four
tracked repositories" is catchable because it says *four*, and almost nothing else names a
value a check can use.

The conclusion is not that extent errors are unreachable. It is that they are unreachable
**from prose**. A mechanical check needs the claim to carry its extent as a field, which is a
change to what the model is asked to emit -- the claim-locked direction -- and not a parser
bolted onto the text it emits today. CAMS's own ordering ablation is the warning attached to
that: requiring a span first buys the guarantee that every claim has one, and the accuracy
has to come from the field being checked or rendered from, not from its position.

---

# What shipped: name the correction, not the count

The one intervention from this whole investigation that is both cheap and verifiable, and it
required no model change at all.

## The finding

Across 72 captured reports, **17% of flagged claims (32 of 193) had the correction already
written in the verifier's own reason**, and **29 of those 32 were retained and delivered**:

| the evidence says | the claim said |
|---|---|
| `26-28` | `26-29` |
| `after=2026-05-15` | `1 May 2026` |
| `June 2026` | `the entire visible history` |
| `only two empty Slack searches` | `a full check across annotations/flags` |

The right answer was computed, persisted in `verifier_rejections`, and then replaced in the
report by a number: *"7 claim(s) go beyond their evidence."* A reader given a count cannot
act on it -- they do not know which sentence to distrust or what the evidence showed. The
last row is the extent defect with its own correction attached.

## The change

The overstated risk now lists each flagged claim's location and what the check found,
bounded per entry and truncated to the schema's 1,000-character limit. Rendered from a real
capture:

> 5 claim(s) go beyond their evidence and are retained with reduced confidence ... What the
> check found:
> - `executive_summary[3]`: The evidence shows only two empty Slack searches, not a full
>   check across annotations/flags, but it does support the core claim that no Slack record
>   was found

**Zero marginal cost.** The judgement already happened; this stops discarding it.

## Why this one and not the others

Every intervention that tried to make the *model* better failed: three prompt fragments in a
pre-registered factorial, and a deterministic gate that scored precision 0.30 over prose.
This one does not ask the model for anything new. It surfaces a judgement already made, and
it was verifiable offline against captures that already existed -- which is why it was worth
trying before anything requiring generation.

## Fix 1b: directionally right, not established, removed

Restating fix 1's rule without naming the verdict to avoid moved every number the predicted
way -- `overstated` 22 → 17, `unsupported` 4 → 7, removal ratio 0.15 → 0.29 against a 0.18
baseline. None of it reaches significance (|t| = 1.00), and resolving an effect that size
against a noise floor of ~2 claims needs roughly 72 scenario-pairs, or four hours of runs for
one bit. The toggle is removed rather than left switched off: an unproven flag is a second
code path nobody reads. The hypothesis is recorded here if it is worth revisiting.
