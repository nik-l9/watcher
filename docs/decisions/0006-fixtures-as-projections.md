# ADR 0006 — Eval fixtures project a planted world, rather than returning canned payloads

**Date:** 2026-09-05
**Status:** Proposed. Not started.
**Question it answers:** six defects of one family turned up in the fixtures in a single day, each
patched separately. What is the root cause, and what closes the class rather than the instance?

## The defects

All six are one sentence: **a fixture answered a question it was not asked.**

| # | Asked | Returned |
|---|---|---|
| 1 | commits for `acme/product` | a payload labelled `acme/web` |
| 2 | a range ending 26 Aug | a fixed series ending 12 Aug, read as a collection failure |
| 3 | the event `user signed up` | the one series planted, whatever the name |
| 4 | one event among several | the same series for every event |
| 5 | `interval="day"` | monthly buckets |
| 6 | `interval="day"` on another scenario | weekly buckets |

Three more sit alongside them, the same shape one level up: a capability planted in a newer field
was invisible to `events_described` (twice) and to `as_of`, and then to `connected_tools` — where
it left a connector **unreachable** while the scenario still scored 1.00 from a different source.
That last one is the reading that should worry a reader most. The score was not wrong about the
answer; it was measuring something else.

## The cause, stated precisely

`ScenarioTool` is faithful. It mirrors the real capability's name, description, `params_schema`,
`result_key` and `discovery` flag, so the model sees production's tool surface; `ToolExecutor`
validates params against that schema before the handler runs; the disclosure and grading passes
are production code, unmodified.

Exactly one line is not faithful:

```python
payload = dict(self._scenario.response_for(qualified, params))
```

`response_for` dispatches over hand-written dictionaries. A dictionary cannot be a function of the
request, so honouring a parameter is something each fixture must *remember* to do — and each new
planting field (`period_responses`, `subject_responses`, `daily_truth`) added a branch here and a
blind spot in three derivations elsewhere.

**The fixtures do not implement the connectors' contract. They impersonate their output.**

## The decision

A scenario plants **a world**: dated series, dated records, dated messages, entities. Each
capability is served by a **projection** from that world, given its params. Honouring the request
stops being a thing a fixture author remembers and becomes the only way to produce a response at
all.

The precedent already exists and works. `DailyTruth` + `_resample` is this design for one
capability: plant daily counts once, and any interval, any window, any event derives from it. The
monthly trap in `partial_month_false_premise` still traps, because it is *computed* from the same
counts rather than typed beside them — which is also how a payload that declared `total: 9300`
beside a series summing to 11,405 stopped being possible.

## Scope: 18 capabilities, but five shapes

The 18 planted capabilities do not need 18 implementations. Their parameters cluster:

| Shape | Params | Capabilities | Status |
|---|---|---|---|
| Series over time | range, interval, subject | `posthog__event_trend`, `ga4__get_sessions` | **Done** — `DailyTruth` |
| Dated records | repo, window, path/state | `github__commits`, `__deployment_history`, `__recent_prs`, `__issues`, `__pull_request_activity` | To build |
| Dated messages | query, window | `slack__search_messages`, `__find_decision`, `posthog__annotations` | To build |
| Two-window comparison | two ranges, metrics, dimensions | `ga4__compare_periods` | Derivable from the series |
| Windowed aggregate | range, dimension | `ga4__get_funnel`, `__top_pages`, `hubspot__contacts`, `__closed_won` | To build |
| Listing | none | `posthog__list_events`, `__list_projects`, `github__list_repositories` | **Done** — derived |

Four projections to write, one to extend. Not eighteen.

`ga4__compare_periods` is the one worth singling out: deriving it from the same daily series that
serves `get_sessions` removes the largest remaining place where two hand-written numbers can
disagree about one fact.

## Estimate

Three focused days, in this order, each landing green:

1. **Dated records and dated messages** (8 of 18 capabilities, one projection used twice) — 1 day.
2. **`compare_periods` from the series, and the windowed aggregates** — 1 day.
3. **Migrating the eight scenarios' data into planted form**, deleting `response_for`'s dispatch
   chain, and keeping every existing test green — 1 day.

## What this costs, and it is not nothing

**Every scenario's numbers will move once.** A projection computes what a hand-written payload
asserted, and the two will not agree everywhere. Across the rewrite, improvement is
indistinguishable from regression.

So it is done in one pass rather than incrementally, with a recorded baseline on the current
suite before and after, and the first run afterwards is a **re-baseline, not a result**. Any
scenario whose score moves materially gets its bundle read by hand before the number is believed.

## What it does not fix

The product code is untouched. This buys nothing a user sees — it buys eval numbers that can be
trusted, which is what the last week of grounding work has been spending.

It also does not address whether the analyst declines *for the right reason*, or the abstention
cost of a stricter gate. Those are what the decline twins measure, and they are orthogonal.

## The alternative, and why it is rejected

Keep patching, and keep the guard tests that now exist (`planted_capabilities`, the
canned-payload guard, the four-derivation check). That is genuinely cheaper today and it is what
produced six defects in one day: each guard closes the instance it was written for, and the
seventh arrives in a field nobody has thought of yet. The guards are worth keeping either way —
they are what would catch a projection that stops projecting.

---

## Landing log

Written as each landing goes green, because the plan above is an estimate and the record of what
actually happened is worth more than the estimate.

### Landing one — dated records, dated messages, and the GA4 session series

`DatedRecords` + `_project_records` for the eight record-shaped capabilities, `_auto_records`
inferring the same projection from a canned payload's own shape so a capability nobody has
migrated by hand still honours `repo`, `since` and `until`, and `_project_metric_rows` filtering
GA4 rows by window and **computing** the totals rather than reading a declared one.

The declared totals were all wrong. Every scenario's `ga4__get_sessions` carried a
`totals.sessions` its own rows contradicted — 6,101 out on `onboarding_regression` (13%), 1,485
on `measurement_stopped` (30%), 2,610 on `campaign_traffic_drop`. An analyst reading the total
and an analyst summing the rows got different answers from one payload.

Two tests failed on that change and both were the same bug — an undated row compared against a
date before it was checked for `None`. Neither was a signal about the fixture.

### Landing two — one GA4 world per scenario

`MetricSeries`: a scenario states its traffic once, as `(day, sessions)`, plus how that traffic
divides (`Segment`, as *shares* and *rates*, never counts) and which pages it lands on
(`PageShare`, as views per session). `get_sessions`, `compare_periods`, `get_funnel` and
`top_pages` are all projections of it.

**What that removed.** Four capabilities were each declaring their own session count for the same
window, and they disagreed:

| Scenario | Window | Session series | Comparison | Funnel |
|---|---|---|---|---|
| `campaign_traffic_drop` | 15–30 Jun | 13,239 | 11,300 | 11,300 |
| `onboarding_regression` | 15–21 Jul | 13,767 | — | 13,900 |

An analyst citing the series and the comparison in one paragraph cited a 15% contradiction. One
did, and was marked down for it. `insufficient_evidence` and `partial_month_false_premise` were
worse in kind rather than degree: their comparisons declared no periods **at all**, so whatever
window was asked about, the answer was the same pair of numbers — and one of those scenarios
exists specifically to test whether the analyst notices a truncated month.

That state is now unrepresentable rather than merely fixed. There is one place a session count
exists; a funnel that disagreed with the series would have to be a different series.

**The numbers moved, as predicted, and by less than expected.** Deriving the onboarding funnel
from the daily series gives mobile 281 conversions where the canned payload said 284, and 412 in
the prior week against a declared 412. The story survives intact because it was always a story
about *rates*: mobile's conversion rate falls 0.035 → 0.029 on the day of the deploy and
desktop's does not move. What changes is that desktop's conversion *count* now falls too, with
the traffic — so an analyst reading counts sees both segments drop and only an analyst reading
rates can name mobile. That is a better test of the same skill.

**Four properties, all mutation-checked** (`tests/db/test_fixture_metamorphic.py`):

- the funnel and the session series report the same traffic for any window
- the period comparison and the session series report the same traffic
- total conversions do not depend on which dimension was asked for — a breakdown divides a
  total, it does not restate one
- a narrower window returns a subset of the wider one's rows, and nothing outside itself

Each was confirmed to fail against a deliberate mutation before being kept. The last one is the
row-shaped twin of the existing bucket-shaped subset property, which could not see GA4 at all:
those payloads carry `rows`, not `series`, and every one of them answered a narrow window with
the same thirty days it answered a wide one.

**Six tests had to be rewritten**, all of which read `scenario.responses[...]` directly — asserting
what the fixture *stores* rather than what it *serves*. That is the fourth time that pattern has
turned up in this repository. A test that reads the store keeps passing while the served payload
says whatever it likes.

**Rounding turned out to matter.** Conversions were first rounded per (day × segment) and then
summed, which put a window's conversion rate 2.4% off the rate that produced it — enough to make
`campaign_traffic_drop`, whose entire claim is *the rate held flat*, report a rate that moved.
Fixed by summing the exact parts and rounding the aggregate. Session splits round the other way
round, per day, with the last segment taking the remainder, so a day's breakdown adds up to the
day exactly.

Still canned after this landing: `github__pull_request_activity` (two payloads, genuinely
ambiguous — `reviews` against `comments`), `posthog__list_projects` (a catalogue, no window), and
`posthog__event_trend` on the onboarding pair (a segmented series `DailyTruth` cannot express;
tracked as an open violation rather than an exemption).

### Landing three — the last three canned payloads

Three left after landing two, and each turned out to be hiding a defect rather than waiting for
one.

**The segmented event trend.** `onboarding_regression` planted eighteen rows by hand covering
nine days, and it was the only entry left on the open-violations list: asked for weekly buckets
it returned daily ones, because a canned payload answers with the granularity it was typed at.
It had been *exempted* from the canned-payload guard on the reasoning that `DailyTruth` had no
shape for a breakdown. `DailyTruth` has the shape now — `breakdown_property` and `segments`, as
shares of the day the series already states — so the exemption is deleted and the rule is
unconditional. The series also extends from nine days to the same twenty-one the session series
covers, which removes a second bug: an analyst asking any wider range was told the event had
stopped firing, and a data-incident verdict outranks the real answer.

**The pull-request lookup, which was the worst of the three.** `github__pull_request_activity`
returned pull request 913 — the mobile onboarding modal, with the review thread that *is* the
scenario's answer — whatever number was asked for. So an analyst that opened the pricing decoy
was handed the cause's evidence under the decoy's number. On a required capability. The scenario
was rewarding the wrong lookup with the right answer, and nothing in the scorecard could show it.

Now keyed by number, with the decoy pull request described honestly on its own terms: a copy
change nobody raised anything about. The decoy became *disprovable* rather than indistinguishable,
which is a better fixture as well as a correct one. A number that matches nothing returns the
envelope with every field about the other record nulled and the request's own subject echoed —
"I looked and there is nothing here" rather than "I could not look", a distinction this layer
keeps having to preserve.

**The projects catalogue.** Kept canned — it is a catalogue, with no window and no subject — but
reading it turned up the hazard beside it. `onboarding_regression` advertises two PostHog
projects, `web-app` and `oss-client`, and every capability answered both with the same series.
An analyst that queried the wrong project scored exactly as well as one that read the listing
and chose, so the listing was a decoration rather than a step. The project is part of the
question now, derived from the listing rather than declared beside it — the same invariant the
repository and event listings each had to learn separately.

Two more properties, both mutation-checked: the tenant's own project answers as if no project had
been named, and another project answers empty.

**Where the canned payloads went.** Eighteen capabilities across eight scenarios, at the start of
ADR 0006, every one of them a payload that answered whatever it was typed with. One is left —
`posthog__list_projects`, a catalogue that takes no parameters and has nothing to project.

### The re-baseline

Run 36, immediately after the three landings, then run 37 for the two scenarios run 36 could
not score. Together: **8/8 pass, `delivered_hallucinations` 0.**

| Scenario | Overall | Note |
|---|---|---|
| `onboarding_regression` | 1.00 | 11/11 drafted claims survived review |
| `campaign_traffic_drop` | 1.00 | |
| `insufficient_evidence` | 0.99 | |
| `partial_month_false_premise` | 0.97 | `summary_placement` 0.50 — refuted, mechanism not in the summary |
| `tempting_coincidence` | 0.98 | |
| `measurement_stopped` | 0.98 | |
| `onboarding_regression_undecidable` | 0.91 | |
| `campaign_traffic_drop_undecidable` | 0.99 | |

**Two things the re-baseline itself taught, both worth more than the numbers.**

Run 36 failed `onboarding_regression` with *"no response within 300s"* — an LLM timeout, from
running the eval alongside the full test suite on one laptop. It passed at 1.00 on its own. An
eval run competing for the machine is not a measurement.

Run 36 failed `partial_month_false_premise` on accuracy: the report hedged the premise to
`unverifiable` rather than refuting it, having compared a truncated twelve-day August against a
complete thirty-one-day July and found conversions down 62% — which is 12/31 almost exactly, so
the number it reported *was* the truncation. The precise mistake the scenario is named for. Run
37 refuted the premise correctly on the same code.

So: variance, not a regression. But the bundle is worth reading anyway, because the reason that
failure was newly *possible* is the point of this ADR. The canned `compare_periods` this landing
deleted answered every request with a pre-computed equal-length comparison — 1,884 against 1,871,
+0.7% — whatever windows were asked for. It was handing the analyst the refutation it was
supposed to have to construct. The scenario has been harder since the payload became a
projection, and its pass rate should be expected to sit below 1.0 until the analyst learns to
normalise window lengths before comparing them. That is a real product gap, now measured instead
of concealed.
