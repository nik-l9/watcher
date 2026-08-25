# Ingest — the nightly sync, and why its failure modes drove the design

Cortex reads live during an investigation *and* stores a nightly history. The live half is
obvious; the stored half is what makes the product more than a query tool:

- **Baselines.** "Signups fell 12%" is not a finding without "the last eight weeks averaged
  4% variation". A live call cannot supply the comparison.
- **A walkable chain.** A live call finds a deploy, and another finds a PR. Only a graph
  turns them into *metric → deploy → PR → the reviewer who warned about it*.
- **Anomaly detection and the weekly brief.** Both are impossible without stored series;
  both are Phase 3, and both are the reason this exists now.

`make ingest` / `python -m cortex.ingest --tenant X --provider Y` runs it by hand. Celery
beat runs `cortex.ingest.dispatch_nightly` at 02:00.

---

## Every design decision here is about a silent failure

An ingest bug does not throw. It produces a graph that looks populated, or a series that
looks complete, and the investigation reading it reports something false with perfect
confidence. So:

**A watermark advances only on success.** A failed pull leaves the cursor where it was and
the next run re-reads that window. Re-reading is cheap and every write path absorbs it;
skipping a window loses data with nothing left to point at it.

**Data first, watermark second, one transaction.** If the watermark committed first and the
write failed, the next run would skip a window nothing ingested. The other ordering costs a
round trip. One failure mode loses data; the other wastes a little work.

**The watermark and the health record are the same row.** Separated, they can disagree — a
watermark that advanced while the health row reads `failed` describes a sync that both did
and did not happen, and afterwards nothing can tell you which is true.

**One watermark per stream, not per provider.** GitHub pulls commits, deployments, issues
and links, and they advance independently. With one cursor, a failed issues pull would
rewind commits, or a successful commits pull would mark issues current when it never ran.

**The window overlaps by six hours.** Upstreams attribute records to a timestamp *before*
they became visible — a deployment status backfills, an analytics pipeline processes late
events. A window starting exactly at the previous watermark misses whatever landed behind
it. The uniqueness constraints make the re-read a no-op.

**A stream failing does not fail its siblings.** "GitHub is down" and "GitHub's issues
endpoint is down" need different responses, and collapsing them throws away the commits
that did arrive. Only a provider whose *every* stream failed is retried.

---

## What is stored where

| | Postgres | FalkorDB | Qdrant |
|---|---|---|---|
| Metric series | `metric_points` | — | — |
| Entities and relationships | — | graph per tenant | — |
| Prose (issue bodies, commit subjects, annotations) | — | — | collection per tenant |
| Watermarks and health | `sync_state` | — | — |

Metric points are keyed on (tenant, provider, metric, `segment_key`, timestamp), and the
segment is a **sorted flattened string** rather than JSONB. Two JSONB objects holding the
same pairs in a different order are not equal, so the same segment would insert twice and
every total over the series would double-count it.

Conflicts `DO UPDATE`, not `DO NOTHING`: the second arrival is often the better one, because
an analytics upstream revises a figure as late events land. Keeping the first read would
freeze a number the source itself no longer agrees with.

---

## The syncers

| Provider | Streams | What it produces |
|---|---|---|
| `github` | `commits`, `deployments`, `issues`, `links` | PR / Deploy / Person / SupportTicket nodes, AUTHORED and RAISED edges, commit subjects and issue bodies as text |
| `posthog` | `events`, `annotations` | daily metric points per event per project; Decision nodes from the change log |

GA4, HubSpot and Slack have **no syncer yet** and are read live during an investigation.
That absence is a row in `SYNCERS` rather than something inferred from what does not crash.

**`links` is the stream that earns the graph.** A deployment record carries a sha and knows
nothing about pull requests; a commit carries a sha and knows nothing about whether it ever
deployed. The edge between them exists only once both have landed, so it is derived by
reading the graph back — which is why a syncer receives the graph read-only.

**Repositories and projects are allowlists.** `--meta repos=owner/name,owner/other`.
Listing what the token can see would ingest a tenant's forks, dependencies and personal
repositories: noise, and data nobody asked us to hold.

**A commit with no PR reference adds no node.** The label vocabulary is a closed enum so
traversal cannot silently stop matching, and there is no `Commit` label — a dependency bump
with no PR would be a node nothing ever walks to.

---

## Verified against real data

Two GitHub repositories, three PostHog projects, 30 July 2026.

```
github (default) — ok
   commits        nodes=302  edges=151  docs=151   [capped at 100 per repo]
   deployments    nodes=100  edges=0    docs=0     [capped at 100]
   issues         nodes=80   edges=40   docs=0     [documents not embedded: 429]
   links          nodes=0    edges=22   docs=0

posthog (default) — ok
   events         points=2221
   annotations    points=0                          [no annotations in any project]
```

The graph: **406 nodes, 258 relationships**, and the walk works —

```
DEPLOY 000433a env=Production state=success
   <- PR 608: New home page tagline (#608)
   <- authored by ['jpelletier1']
```

The series: 2,221 points over 90 days — 6.09M `agent_task_completed`, 1.00M
`user_activated`, 325k `conversation finished`, all segmented by project.

**The second run wrote 35 points, not 2,221.** That is the watermark working, and it is the
only way to see that it does.

---

## Staleness is enforced, not requested

The plan requires a stale connector to appear in a report's confidence section.
`ingest/health.py` turns `sync_state` into `DataQualityNote`s and the **grounding gate
appends them**, rather than the drafting prompt asking for them.

The reason is the failure this codebase keeps meeting: **absence of access is
indistinguishable from absence of evidence.** An analyst cannot know its inputs are three
days stale. Reading a failing sync, it would honestly report "no deploys that week" from
data that was never fetched — and every claim would be properly cited to a genuinely empty
result, so the grounding machinery would be powerless against it.

Three distinct disclosures, phrased differently on purpose:

- **failing** — the data has stopped arriving, and an absence here is not evidence that
  nothing happened;
- **stale** — how old what we have is;
- **incomplete** — current but truncated, so counts are a floor rather than a total.

A healthy sync says nothing. A report disclosing every connector's status on every question
buries the one disclosure that matters.

---

## Five things the first real runs exposed

Each was found by running it, not by a test.

**A Qdrant transport error aborted an entire sync** — *after* the graph writes had
succeeded. Semantic memory is additive; the graph is the core. An unreachable vector store
now degrades the run and records the gap.

**The deployments cap was silent while the commits cap was disclosed.** I shipped the two
inconsistently, and the first run hit exactly 100 deployments. A history cut at 100 that
reads as complete lets a report reason about "the deploys that week" from a window that may
not contain them.

**`$set`, `$identify` and `$groupidentify` took three of twelve event slots per project.**
Person-property bookkeeping, crowding out real metrics under the cap. The first fix was a
hand-written exclusion list, which is always one PostHog release behind; the actual rule is
that PostHog reserves the `$` prefix and a custom event never uses it.

**"No annotations" was reported as incomplete data.** All five real projects have zero
annotations. Telling a reader their change log might be missing entries when nobody writes
any is how disclosures stop being read. `partial` (a gap in our collection) and `note` (a
real emptiness) are now separate fields, and only the first sets `PARTIAL`.

**Voyage's rate-limit body was on course for a customer-facing report** — including a link
to its billing dashboard and an explanation of its free-tier token allowance, inside
someone's answer about their own funnel. `sync_state.detail` keeps the full text for
diagnosis; what reaches a report is cut at the first brace or URL.

---

## Known limits

| Limit | Consequence | Status |
|---|---|---|
| 100 items per stream per window | A busy repository's history is truncated | Disclosed as `partial`, so a count reads as a floor |
| Voyage free tier: 3 requests/minute | Documents often go unembedded on a large sync | Disclosed; needs a paid tier |
| 22 of 100 deploys link to a PR | The chain is not walkable for the rest | Disclosed; most are CI or dependency deploys with no PR, and the commits cap hides some |
| No GA4 / HubSpot / Slack syncer | No stored history for those sources | Read live; a row in `SYNCERS` says so |
| Recall is not wired into the loop | Stored memory is not yet used during an investigation | Next step, now that memory is non-empty |
