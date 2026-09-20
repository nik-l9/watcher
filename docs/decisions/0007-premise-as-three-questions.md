# ADR 0007 — The premise verdict is three yes/no questions, not one four-way choice

**Date:** 2026-09-19
**Status:** Accepted. Landed and measured.
**Question it answers:** the analyst answered `false` — *the evidence contradicts your
premise* — about windows its data never covered, on 15 of 16 cases. Guidance had already been
tried on this field and measured worse. What closes it?

## The defect

One question, asked of real extracted series across a window the project has no data in:

> Why did `feature_toggled` fall between March and June 2026?

and a report that is, in its own prose, entirely correct:

> There is no recorded `feature_toggled` activity at all between March and June
> 2026, so the event cannot have 'fallen' in that window — there is no baseline to fall from.

and then sets `premise: false`.

`false` is defined in `cortex.reports.schema` as *the evidence contradicts it*. No data cannot
contradict anything; it fails to inform. The correct verdict is `unverifiable`. Fifteen of
sixteen such reports made this call, in a project that had **zero events with any data** in
the months being asked about.

The distinction is not pedantic. `false` tells a reader the thing did not happen, so there is
nothing to look into. `unverifiable` tells them the data does not reach and they must get it
elsewhere. Those prompt opposite actions.

## Why guidance was the wrong instrument

The obvious fix — say more clearly in `cortex.reports.shape` what each verdict means — had
already been tried on this exact field and **measured worse**: `partial_month_disclosed` went
from 6/7 to 2/7 and the change was reverted. That result is recorded in the history of this
repository as a negative result precisely so it would not be repeated.

The reason it failed is visible in the shape of the task. `premise` was one field with four
values, and three independent facts determine it:

1. did the question assert something checkable?
2. does the evidence cover the window it asserts about?
3. does that evidence contradict the assertion?

`unverifiable` and `false` differ **only on the second**. Asked as a single label, that
difference is something the model settles *after* it has already chosen a label — and
structured output is one left-to-right pass, so whatever it picks first is what the rest of
the report is written to justify. Telling it to be more careful asks it to be better at a
judgement. The judgement is the problem.

## The decision

Ask the three questions, in that order, and derive the verdict:

| asserted | measured | contradicted | verdict |
|---|---|---|---|
| false | — | — | `none_asserted` |
| true | **false** | — | `unverifiable` |
| true | true | true | `false` |
| true | true | false | `holds` |

Each answer is a fact about the evidence rather than a judgement about the question, and each
is answerable before the one that depends on it — the fourth application of this project's
reason-before-verdict ordering rule. It is also the Likert-to-binary decomposition CheckEval
reports at +0.45 agreement across twelve evaluators, applied to the one field observed
failing.

`PremiseVerdict` survives as a `computed_field` derived from the three, so the scorer, the
investigator's early exit, the report view and every stored report keep reading one field.
A `computed_field` serializes but is not an input, so the model is never offered back the
label whose shape was the problem. A `mode="before"` validator drops a stored `premise`
rather than failing on it, because `_Strict` rejects unknown fields and without it every
captured run would stop being re-scorable.

## What it measured

Paired: same cases, same seed, same extract, one change.

| | before | after |
|---|---|---|
| clean cases (`unverifiable` is the only honest reading) | **1/16** | **16/16** |
| cases whose label is contestable | 1/4 | 4/4 |
| cases made worse | — | **0** |

Over-abstention was checked separately, because a run of `unverifiable` cases cannot detect
it and a change that simply made the model reach for `unverifiable` more often would have
scored 20/20 above while ruining the product. On cases whose correct answer is confident:
8 correct, 1 over-abstained, 0 wrong any other way.

That one over-abstention was this ADR's own bug, not the model's: `premise_measured` read as
*is the window complete* where it should read *can the claim be judged*, so a report that
found a real refutation and noticed a three-day gap was pushed into abstaining. Reworded; the
case now answers `false` with the gap recorded in `data_quality`.

## Cost

The drafting schema is capped at 7,000 bytes by a tripwire that stands in for grammar
complexity, and had 82 bytes spare. Three booleans with full-sentence descriptions cost 119
and did not fit. With descriptions dropped — the convention `premise_checked` beside them
already follows, since the instruction belongs in `shape.py` as prose rather than grammar —
the schema is 6,924, six bytes larger than the enum it replaces.

A prototype that built the property dictionaries by hand predicted 6,916 and was wrong by 186
bytes, because pydantic adds a `title` to every field. Size the generated artefact.

## What this does not settle

The residual instability is real but smaller: the same case has been observed answering
differently across runs, and one case in sixteen still lands the other way. `instability_of`
in `cortex.eval.abstention` measures it and has not yet been run against real repeats.
