# Architecture

How watcher is put together, and why. The README covers what you need to run it; this is
the reasoning behind the parts you will meet once you are in the code.

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
