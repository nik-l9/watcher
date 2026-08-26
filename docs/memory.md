# Memory — the graph, the vectors, and what recall is allowed to be

Cortex has two memories, and they answer different questions.

The **knowledge graph** (FalkorDB, one graph per tenant) holds entities and the edges
between them: which PR delivered a feature, which deploy carried it, which metric it
affected. It can walk from a campaign to revenue. It cannot find a sentence.

**Semantic memory** (Qdrant, one collection per tenant per kind) holds prose the graph
cannot represent: Slack threads, meeting notes, ticket bodies, documents. It can find
*"we're pausing the spring campaign today"* from the question "why did paid traffic
fall". It knows nothing about what that campaign promoted.

`cortex/memory/recall.py` joins them: the search supplies an entry point, the traversal
supplies the structure around it. That is the whole of M3.

---

## Isolation is structural in both stores

The graph argument, restated for vectors because it is the same argument.

Qdrant supports one shared collection filtered by a payload field. That was rejected on
the same grounds as a `tenant_id` predicate in Cypher: its failure mode is one forgotten
filter returning another tenant's Slack messages, and no test catches the day somebody
adds a query without it. So a tenant's semantic memory is a **different collection**, and
the name comes from `naming.collection_name` — the same single resolver that names graphs.
No caller constructs either.

`tests/tenancy/test_vector_isolation.py` gates it, on the same terms as the graph suite:
it must pass before a real credential is entered. It searches tenant A for the exact text
tenant B stored, and asserts A sees only its own.

**Offboarding is total.** `drop()` removes every kind, and is idempotent so an interrupted
offboarding can be retried. A leftover collection is retained customer data.

---

## Four decisions worth the words

**Point ids are derived, not random.** A point id is `uuid5(namespace, kind:source_id)`,
so re-ingesting an edited Slack thread *replaces* its point. Random ids would let a
nightly sync accumulate near-duplicates of one conversation, each competing for the same
recall slot — so the analyst would be shown the same thread three times and read it as
three pieces of corroborating context. The namespace is fixed forever: changing it would
orphan every point already written.

**A kind per collection, not a `kind` field.** Four collections (`slack`, `notes`,
`tickets`, `docs`) rather than one with a filter. A recall for meeting notes cannot
accidentally rank support tickets, and a retention policy on ticket bodies can drop one
collection without touching anything else.

**The vector width is declared and checked.** `Embeddings.dimensions` is part of the
interface, and a provider returning a different width fails at the boundary. A collection
built for 1024 dimensions and written with 256 does not error — it becomes an index that
returns plausible nonsense, which is the worst possible failure for a system whose claim
is groundedness.

**Entity linking is conservative to the point of being unhelpful, deliberately.** A hit is
connected to a graph node only when the ingest recorded which node it was about. Matching
on prose was considered and rejected: a message mentioning "onboarding" matches every
onboarding PR ever merged, and pulling all of them in presents several unrelated changes
as equally implicated. A missing link costs recall. A wrong link *manufactures evidence*,
and the analyst has no way to tell an invented association from a real one.

---

## Recall is context, never evidence

The report layer requires a resolvable `evidence_id` behind every claim. A recalled Slack
message has none — it is something the system read last month, not something it just
observed.

So `Recall.render()` says so, in the text itself:

```
MEMORY (prior context, NOT evidence). Every item here is a lead to confirm with a
tool call. Do not cite it: only tool results carry evidence ids.
```

The framing is attached where the text is produced rather than left to whoever writes the
prompt. An analyst that cites memory produces a claim the gate then strips, and the reader
sees a thinner report for no visible reason.

**An empty recall says it is empty.** `"nothing recalled for this question. Treat it as no
prior context, not as an absence of history."` This is the third place in the codebase
where that distinction has had to be made explicit — after Slack's over-narrow query and
GitHub's non-existent `production` environment — because absence of access reads exactly
like absence of evidence, and the grounding machinery is powerless against it: every claim
is properly cited to a genuinely empty result.

---

## Verified live

Voyage `voyage-3` (1024 dimensions), the managed Qdrant cluster, local FalkorDB. Three
documents ingested, one question asked:

```
Q: why are mobile signups down?

  (slack,   0.70) [slack://growth/2]    mobile signups look off since yesterday's
                                        onboarding modal change
  (tickets, 0.33) [help://9]            cannot tap the continue button on iphone
                                        during signup
  (slack,   0.22) [slack://marketing/1] pausing the spring campaign today, budget
                                        is exhausted

Related entities in the knowledge graph:
  - PR 913     (linked directly) author=dwhitfield, title=Rework mobile onboarding modal
  - Deploy 91c3e4a (1 hop away)  created_at=2026-07-14T11:04:00Z, environment=prod-web
```

The ranking is right, the campaign decoy is last, and the two graph entities are ones no
search could have produced — nothing in the Slack message names the PR or the deploy.

---

## Three things real use exposed

**Voyage's free tier is 3 requests per minute.** Found by hitting it: a smoke test
embedding four short strings got a 429 on the fourth. The code raised immediately, which
is the wrong behaviour for an ingest — aborting part way leaves memory half-populated with
no record of which half, and a partly embedded corpus produces confidently incomplete
recall. Now retried with backoff, honouring `Retry-After`. A 4xx other than 429 still
fails fast, because it will fail identically however long we wait.

*Standing item:* a nightly sync of a few thousand messages needs a paid tier. Batching
helps (128 texts per request) but 3 RPM is 384 texts a minute at best.

**`VoyageEmbeddings(api_key="")` silently used the ambient key.** `api_key or
settings.voyage_api_key` conflated "take it from settings" with "there is no key", so a
caller deliberately constructing an unconfigured provider picked up the real one. `None`
now means settings; an explicit empty string means empty.

**The Qdrant client is bound to its event loop.** A cached async client carries connection
state tied to the loop that created it, so using it from a second loop fails with *"is
bound to a different event loop"*. This is the failure `runtime/resources.py` was written
to document, arriving through a different door — and a Celery worker running one loop per
task would have hit it in production. The client is now rebound per loop, against the same
cluster it was originally created for.

---

## The isolation gate runs against local Qdrant

Not against whatever `.env` names. Pointed at the managed cluster the suite took **110
seconds** and failed on a different test each run with an empty
`ResponseHandlingException` — a round trip to another continent, not a logic error. It is
1.3 seconds locally.

A gate that is slow and flaky stops being run, and a gate nobody runs is worse than a
smaller one that everybody does. Real engine, local instance: the same choice the rest of
the test stack makes for Postgres and FalkorDB.

---

## Not yet done

- **Nothing ingests into semantic memory yet.** That is M4. Until then a tenant's
  collections are provisioned and empty, and recall correctly reports nothing.
- **Recall is not wired into the investigation loop.** Deliberate: wiring it before there
  is anything to recall would add a step to every investigation that always returns
  "nothing recalled". It is one injection point when M4 lands.
- **No age-based expiry.** Tenant deletion drops the collections; there is no retention
  policy, and it is a standing item rather than a closed one.
