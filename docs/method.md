# The method, and where its numbers come from

This project replaces a human data analyst's *procedure*, not just their output. That procedure is
written down here, along with the measurements the thresholds in the code were chosen from.

It is distilled from a longer body of research — root-cause frameworks used by analysts and
incident responders, the causal-inference literature on single-unit interventions, and papers on
grounding and sufficient context. What survives here is the part the code actually depends on. A
finding that changed no threshold and no control flow is not in this file.

[ADR 0005](decisions/0005-investigation-method.md) is the decision record; this is the reasoning
underneath it.

---

## 1. Establish the effect before explaining it

The single most expensive failure this project has recorded was not a wrong cause. It was a
confident explanation of a movement that had not happened.

A question asserts something — "why did signups fall 18%?" — and that assertion is a claim about
the data, not a given. So the first move is to check it, and the report carries the verdict as a
field (`premise`, `premise_checked`) rather than burying it in prose.

**Why it is a field and not an instruction.** A report that reaches the right answer in its fourth
bullet, after opening with a figure that appears to confirm the premise, has misinformed every
reader who stopped early. Position is scored, not just presence.

## 2. Elimination, not ranking

From Kepner-Tregoe's IS/IS-NOT analysis, one rule is precise enough to encode:

> A candidate cause whose onset is **later** than the deviation's onset is eliminated, not ranked
> lower.

A deploy that shipped after the drop began cannot have caused it. Ranking it below other causes
still leaves it in the answer; eliminating it removes it. This is enforced by a schema validator —
a hypothesis carrying `cause_at` later than `effect_onset` fails validation — rather than by asking
the model to be careful.

**What KT does not have, and this needs.** IS/IS-NOT assumes the data describes reality. It has no
step at which you ask whether the series can be trusted at all. Elimination fixes the *answer*;
only a data-trust gate fixes the *question*.

## 3. Two independent lines, or the tree does not narrow

From DOE-NE-STD-1004-92. One line of evidence establishes only itself.

Concretely: noticing that a sessions series stops collecting on the 3rd establishes only that it
stops on the 3rd — it cannot distinguish a broken measurement tag from a site that genuinely went
dark. A second, independent source showing signups still arriving is what makes "the measurement
broke" a conclusion rather than the more comfortable of two guesses.

Implemented in `cortex/reports/corroboration.py`, with independence defined by *tool*: two
observations from the same connector are one line, not two.

## 4. A series that cannot be trusted cannot answer a business question

The gate has eleven rows in the research. Three are computable from what the connectors return,
and the rest report as **not evaluated** rather than as passing.

That distinction is the whole design. The research's warning is blunt: *a gate whose checks all
return `unknown` passes everything.* Sequenced wrongly it looks like a safeguard and behaves like a
pass-through — so a check with no input says so, and `unknown` never reads as `ok`.

| Verdict | Meaning |
|---|---|
| `ok` | Answer the business question |
| `degraded` | Answer a narrowed question, with the limitation stated in the answer itself |
| `broken` | The business question is the wrong question. The answer is the data incident |

**At most one check may block.** A correlated cessation — a group of series falling silent together
while a sibling emitter keeps recording — is not something changing user behaviour can produce.
Everything else degrades. That ceiling is deliberate: one published framework was reported as
having been disabled by its own false positives, and a gate nobody trusts is worse than no gate.

A source is graded only for the checks it can actually compute (`cortex/analysis/trust.py`,
`COMPUTED_CHECKS`). Granting a source the whole gate reports passes for checks that never ran.

## 5. Significance, on one treated unit with no controls

Attribution methods mostly assume control units. A single tenant asking why its own metric moved
has none, which rules out most of the standard toolkit.

**Abadie's placebo test gives `p ≥ 1/(J+1)`** for `J` control units. At `J=0` that is `p = 1`
identically — not "weak evidence", but *no attainable p-value at all*. Reaching `p ≤ 0.05` needs 19
controls.

**The conformal permutation test** (Chernozhukov–Wüthrich–Zhu) is valid with one treated unit and no
controls, with a floor of `1/T` for `T` periods. That is what is implemented.

### The power cliff — measured here, and it decides the API

Simulation over this project's own series shapes found the test has **zero power unless the
post-period is shorter than the pre-period**: power 1.000 versus 0.000 across that boundary, with
Type I error unaffected (1.7–3.3% against a nominal 5%).

So `ConformalResult.resolvable` reports an *inability* rather than a negative. A test that cannot
resolve a question must not return "not significant" — that is a false reassurance, and it is the
difference between "we found no effect" and "this method cannot see one".

An AR(K) proxy was tried to work around it and made things worse: the p-value degenerated to
roughly `n_post/T`, which is a restatement of the window, not evidence.

## 6. Changepoints: `min_len = 14`

Optimal partitioning with a minimum segment length. The length was chosen by measurement, not
convention: at `min_len=14` the true changepoint was recovered in 189 of 200 simulated series,
against 109 of 200 at `min_len=7`.

The failure mode matters as much as the score. A too-large `min_len` **misdates** a shorter dip
rather than missing it — so the disclosure says a movement shorter than the minimum is an incident
rather than a level shift, and that this check cannot see it.

The noise estimator returns 0 when it cannot estimate — a degenerate series gets a refusal, not a
guess. An earlier fallback averaged the non-zero differences and so estimated the *jumps* as noise,
producing a noise scale of 9,800 on a series whose real variation was two orders of magnitude
smaller.

## 7. Sufficient context

From Joren et al. (ICLR 2025): context is sufficient when it **plausibly supports** a definitive
answer — not when it proves one. The distinction is load-bearing. A gate demanding proof rejects
answers that are correct and well-evidenced, and a gate demanding nothing passes everything.

## 8. Asking the user a question

Mixed-initiative interaction (Horvitz) says the decision to interrupt should weigh the cost of
asking against the value of the answer. Applied here, the band in which asking beats proceeding
turns out to be **narrow**: most ambiguity is either resolvable from the data or not resolvable by
the user either.

So the leverage is not in asking. It is in **stating the assumption** — an answer that says which
reading of the question it took can be corrected by a reader in seconds, while a question blocks
the whole investigation on a reply that may never come. `cortex/reports/shape.py` detects unstated
baselines and requires the report to name the one it used.

## 9. What the eval can and cannot tell you

**Grounding is scored mechanically**, against the evidence store, never by asking a model whether
a citation holds. The two grounding mechanisms — the code-enforced citation gate and the
adversarial verifier — are scored separately, because merging them makes it impossible to say which
one removed what.

**Paired MDE** is `(z_{0.975} + z_{power}) · σ_att · √(2/n)`. Two consequences worth knowing before
reading any score difference:

- Halving the minimum detectable effect requires **four times** the scenarios.
- A single attempt is a sample, not a measurement. The same scenario has passed at 0.85 and then
  failed on unchanged code.

**Scores hide bugs that artifacts reveal.** Two defects here were found by reading captured
payloads, not by reading scorecards: a summary field that was silently a placeholder in 4 of 11
runs, and fixtures describing repositories their own discovery survey denied existed. Both showed
up as a dimension reading slightly low — 0.77 to 0.90 — which reads exactly like a working
defence. Every run therefore captures replayable bundles, and failed attempts are captured too.

---

## What is deliberately not claimed

- **This is not causal identification.** `cortex/analysis/identifiability.py` enumerates the
  assumptions (G0–G7) an attribution rests on and discloses which are unverifiable. A movement
  that coincides with a change is a coincidence until two independent lines say otherwise, and the
  report is required to say so.
- **The trust gate covers three connectors**, and grades each only for the checks it can compute.
  Others report as ungraded rather than as sound.
- **Latency is bounded by serial token generation**, around 78% of wall clock. It is an output-rate
  wall, not a tuning problem. See [`latency.md`](latency.md).
