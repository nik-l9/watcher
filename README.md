# Cortex — AI GTM Workforce

An AI workforce where every specialist has a defined role, tools, memory, KPIs and
deliverables. V1 ships one employee: the **GTM Data Analyst**.

It does not return numbers. It investigates, cites, and recommends:

> Signups dropped 18%. The largest contributor was mobile traffic, whose conversion
> fell from 4.2% to 2.9%. This began after deploy `91c3e`. Recommendation: roll back
> the onboarding modal.

Every claim in that answer resolves to a stored piece of evidence from a real tool
call, or it does not render.

Apache-2.0. Runs locally under Docker Compose, and on GCP.

## What you need

The analyst is only offered connectors the tenant has credentials for, so **you can run this
with whatever subset you have** — one connector is enough to ask a question. Nothing is
required beyond an LLM key and the datastores that `docker compose` brings up.

| | |
|---|---|
| Required | An Anthropic API key |
| Connectors, any subset | PostHog, Mixpanel, GA4, BigQuery, HubSpot, GitHub, Slack |
| Optional | A Slack app (to ask questions by `@mention`), an embedding key (for recall) |

See [`docs/running-it.md`](docs/running-it.md) to get it up, and
[`.env.example`](.env.example) for every setting.

## Status

**Every milestone M0–M8 has shipped.** Two were met differently from the original plan and one
is limited by what a given deployment has connected; all three are noted below rather than
smoothed over.

| Milestone | State |
|---|---|
| M0 Foundation: tenancy, vault, schema, graph layer | done |
| M1 Tool framework and connectors | done — GA4, BigQuery, HubSpot, GitHub, Slack, PostHog |
| M2 Investigation engine, grounding gate, verifier | done |
| M3 Memory: graph + vectors | done — FalkorDB graph-per-tenant, per-tenant Qdrant collections, hybrid recall |
| M4 Hybrid ingest | done — nightly Celery beat sync, watermarks, connector health surfaced in reports |
| M5 Charts | done — validated `ChartSpec` with SVG and PNG renderers. **The sandboxed-matplotlib escape hatch was declined**, with reasons, in [cortex/reports/png.py](cortex/reports/png.py) |
| M6 Eval suite — gates the first real investigation | done |
| M7 API + minimal UI | done — **server-rendered rather than Next.js**, see [ADR 0003](docs/decisions/0003-server-rendered-report-view.md) |
| M8 Real data cutover | done — PostHog, GitHub and Slack answer live questions in Slack. A connector with no credentials is never offered to the analyst, so an unconfigured one costs nothing |

Phase 3 is not started: scheduled investigations, anomaly detection, the weekly exec brief,
alerting, OAuth in place of BYO credentials, and any specialist beyond the analyst.

## Services

Cortex is a set of independently deployable services in one repo. They share the
`cortex` library and communicate only through the message contracts in
[cortex/contracts/messages.py](cortex/contracts/messages.py) — never by importing
one another. That boundary is enforced by
[tests/unit/test_service_boundaries.py](tests/unit/test_service_boundaries.py); a
direct sibling import means the two can no longer deploy independently, which is
how microservices decay into a distributed monolith.

| Service | Role | Scaling signal |
|---|---|---|
| [gateway](services/gateway/) | Only externally-reachable service. Authenticates, resolves tenancy, enqueues work. Never runs an investigation — a long LLM loop in the request path would couple availability to the slowest tool call. | HTTP concurrency |
| [investigation_worker](services/investigation_worker/) | Consumes `cortex.investigation`. Runs the hypothesis loop, writes evidence, produces the graded report. The only service that talks to an LLM. | Token throughput |
| [ingest_worker](services/ingest_worker/) | Consumes `cortex.ingest`. Nightly metric and entity sync so the analyst has baselines. | Connector volume |
| scheduler | Celery beat. Exactly one replica — two would double-fire every nightly sync. | n/a |

Separate queues per worker are what make the load independent: a HubSpot backfill
cannot occupy the slots a waiting user needs.

## Quickstart

Two ways in. Both need the four datastores running, because this is a stateful system: an
evidence store a claim can be checked against, a graph, a vector store, and a queue.

### Fork it and run it

The path to take if you want to change anything, and the one CI runs.

```bash
make setup      # venv + deps from the lockfile + .env with a generated vault key
make up         # infrastructure only: Postgres, FalkorDB, Qdrant, Redis
make migrate    # apply schema
make test       # full suite
make up-all     # + gateway, both workers, scheduler → http://localhost:8000/ready
make eval       # score the analyst against labeled fixtures (spends tokens)
```

Then read one investigation, before spending anything on a suite:

```bash
make ask DATASET=campaign_traffic_drop   # ~100s, one LLM key, no connector credentials
```

That runs the production path — real investigator, real evidence rows, real grounding gate, real
adversarial verifier — over a dataset whose true cause is known, and prints the planted cause
afterwards so the answer can be checked rather than admired. It is the cheapest way to find out
whether this works before wiring up a connector.

Infrastructure and application services are split by compose profile so test-driven work does
not wait on image builds. `make setup` installs from `uv.lock`, so you get the versions this
project was tested against rather than whatever resolves today.

### Install it as a package

The path to take if you want to use the analyst rather than work on it.

```bash
pip install watcher-gtm

cp .env.example .env    # then fill in your keys — see "What you need" above
watcher migrate         # create the schema (migrations ship inside the package)
watcher ask --dataset campaign_traffic_drop --show-truth
```

That last line needs no connector credentials and no real data: it investigates a labelled
dataset whose true cause is known, and prints the planted cause afterwards so the answer can be
checked rather than admired. Point it at your own data once you believe it.

```bash
watcher connect --tenant acme --provider posthog   # secret from the environment, never argv
watcher ask --real --tenant acme "Why did signups fall last week?"
```

You still supply the datastores. The compose file in this repository is the quickest way, and
`.env.example` documents the four connection settings if you already run them elsewhere.

| Command | What it does |
|---|---|
| `watcher migrate` | Apply the schema. Run this first; the rest need it |
| `watcher ask` | Ask a question and print the report |
| `watcher connect` | Store a tenant's connector credentials in the vault |
| `watcher ingest` | Sync a connector now, rather than waiting for the nightly run |
| `watcher spend` | What each tenant is costing |
| `watcher eval` | Score the analyst against labelled fixtures |

`watcher --help` lists them; `watcher <command> --help` gives one command's own options.

**On the names.** The distribution is `watcher-gtm` because PyPI's `watcher` is an active
file-watching utility. The import name is still `cortex` — as `beautifulsoup4` gives you `bs4` —
because renaming it touches 208 files and the collection prefix already written into live vector
stores, which deserves its own change rather than riding along with a naming decision.

The API and the Slack transport are packaged too. The gateway is an app *factory* rather than a
module-level instance, so that tests can build isolated apps — which means uvicorn needs
`--factory`:

```bash
uvicorn --factory services.gateway.app:create_app --port 8000
```

### Local ports

Cortex avoids the defaults because a developer machine usually already has
Postgres on 5432 and Redis on 6379. FalkorDB speaks the Redis protocol, so a
shadowed port surfaces as a baffling `unknown command 'GRAPH.QUERY'` rather than
a connection error.

| Service | Host port |
|---|---|
| Postgres | 5433 |
| FalkorDB | 6381 |
| FalkorDB Browser | 3001 |
| Qdrant | 6333 |
| Redis (Celery) | 6380 |

## Architecture notes

**Runtime.** Python + FastAPI, with a hand-written investigation loop. An
off-the-shelf agent framework was evaluated first and rejected — see
[ADR 0001](docs/decisions/0001-own-investigation-loop.md) — because the loop is
where every grounding guarantee is enforced, and a framework's loop is the one
part you cannot reach into. Rust was evaluated and rejected too: its real
advantages, cold start and memory and throughput, do not apply to a 90-second
LLM-bound investigation.

**Resources are per-process, never module globals.** An async client caches a
connection bound to the event loop that created it, so a process running more than
one loop — a Celery worker, a test suite — reuses it against a dead loop and fails
with `Event loop is closed`. Every service therefore builds its resources through
[open_resources()](cortex/runtime/resources.py) and receives them by injection.
Enforced by an AST check in the service-boundary tests.

**Evaluation gates the first real investigation.** `make eval` runs the real loop,
gate, verifier and executor against [labeled fixtures](cortex/eval/fixtures.py) —
synthetic GTM histories with a planted causal chain and decoy correlations that
correlate just as strongly. Grounding and accuracy are computed mechanically and both
gate the run; a hallucination count above zero fails it. Neither consults a model,
because the product's central claim must not be scored by the same kind of component
it exists to constrain. See [docs/testing.md](docs/testing.md).

**Tenant isolation is structural, not disciplinary.** Each tenant gets its own
FalkorDB graph and its own Qdrant collections. Isolation is a property of *which
graph you open*, not of a predicate someone remembered to add. Enforced by:

- `GraphStore` (`cortex/memory/graph_store.py`) exposes no method that takes a
  graph name, crosses tenants, or accepts caller-authored Cypher.
- Graph names are constructed in exactly one place, `cortex/memory/naming.py`.
- Cypher may only appear inside `cortex/memory/`. Asserted by
  `tests/tenancy/test_cypher_containment.py`, which runs as an ordinary test so
  there is no separate CI step to forget.
- `tests/tenancy/test_graph_isolation.py` seeds two live tenants and proves no
  read, scan, traversal, or path query crosses between them.

FalkorDB was chosen over Neo4j because Neo4j Community supports exactly one
database, which would have forced either an Enterprise license or `tenant_id`
filtering in application code — whose failure mode is a cross-tenant leak. A Neo4j
Enterprise implementation (one database per tenant) remains a drop-in behind
`GraphStore` if the Startup Program is taken up later.

**Credentials.** Envelope encrypted, AES-256-GCM, fresh data key per credential,
with tenant and provider bound in as additional authenticated data — so a
ciphertext lifted into another tenant's row fails to decrypt rather than leaking.
V1 is bring-your-own credentials; OAuth is Phase 2.

**Grounding.** Two independent mechanisms, per the plan: a structural gate that
drops any claim whose evidence ids do not resolve, and an adversarial verifier
pass that re-reads each claim against only its cited evidence. Neither is a prompt
instruction.

**Tools.** Five read-only connectors, 20 capabilities, in [cortex/tools/](cortex/tools/).
Each capability declares a closed JSON schema and returns typed JSON — never
markdown. Prose is generated once, in the report layer, from evidence; a tool
returning prose would leave nothing structured to cite.

Everything that must happen on *every* call happens in
[ToolExecutor](cortex/tools/executor.py), not in the connectors, so none can omit
it: schema validation, per-tenant credential decryption, timing, a `tool_calls`
audit row, and an immutable `Evidence` row whose id a report must cite. A failed
call writes an audit row and no evidence — accountable, but not citable.

Three properties are enforced structurally rather than by convention:

- **Read-only.** `Capability(read_only=False)` raises at construction. V1 has no
  write path, so "humans approve" holds because destructive tools cannot be
  declared — not because a prompt asks nicely.
- **No agent-authored SQL.** [BigQuery](cortex/tools/bigquery.py) exposes only
  reviewed named queries with bound parameters. Generated SQL is unreviewable and
  can scan terabytes of a production warehouse. Every job also caps
  `maximumBytesBilled`, so a runaway scan fails instead of arriving as an invoice.
- **Closed schemas.** `additionalProperties: false` everywhere, so a hallucinated
  argument surfaces as a correctable error rather than being silently ignored.

Slack message text is third-party user-generated content. It is returned as payload
data flagged `content_is_user_generated` — to cite, never to follow.

## Non-negotiables

- All integrations read-only. "Humans approve" is enforced by the absence of
  destructive tools, not by a prompt.
- No claim renders without a resolvable `evidence_id`.
- Every tool call and generated report is audit-logged with tenant, user, timestamp.
- No customer data in model training; no credentials or PII in logs or fixtures.
- Tools return JSON. Prose is generated only in the report layer.
- `make test-tenancy` must pass before any real credential is entered.

## How it decides things

The investigation method is not improvised per question. It follows a documented procedure —
establish the effect before explaining it, eliminate a candidate cause whose onset postdates the
deviation rather than ranking it lower, refuse to attribute a movement in a series that cannot be
trusted — and the parts that are enforced are enforced in code rather than asked for in a prompt.

- [`docs/method.md`](docs/method.md) — the procedure, and the measured findings the thresholds
  come from.
- [`docs/decisions/`](docs/decisions/) — five ADRs covering the loop, prior art, the report view,
  connector breadth, and the investigation method itself.
- `make eval` scores the analyst against labelled fixtures where the true cause is planted and
  the decoys are chosen to punish the obvious-but-wrong answer.

## Contributing

Two things matter more than style here:

1. **A claim must not be able to reach a reader without evidence behind it.** If a change makes
   the citation gate or the verifier optional, bypassable, or advisory, it is the wrong change.
2. **Findings belong next to the code.** Most comments in this repository record a specific
   failure and what it cost. If you fix something that went wrong, write down what it was —
   that is the part a future reader cannot reconstruct.

`make test` and `make lint` before a PR. CI runs both against the real datastores.

Found a security issue? [`SECURITY.md`](SECURITY.md) — please not a public issue.
