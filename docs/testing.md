# Testing

What is tested, how, and what is deliberately not. Kept current because a coverage
number nobody reads is worth less than a short honest record of where the gaps are.

## Current state

| | |
|---|---|
| Tests | 1148 |
| Coverage (`cortex` + `services`) | 83% |
| Runtime | ~58s |
| Command | `make test` (needs `make up`) |

Measure it yourself:

```bash
.venv/bin/python -m pytest tests/ -q --cov=cortex --cov=services --cov-report=term-missing
```

## Layers

Layers are named for what they exercise, not for a taxonomy. The dividing line is
what would have to be wrong for the test to fail.

| Directory | What it exercises | Backing services | Count |
|---|---|---|---|
| `tests/unit/` | Pure logic, schemas, config, boundary guards, error paths | none | 285 |
| `tests/tools/` | Connectors and ingest syncers against recorded upstream payloads | none | 367 |
| `tests/reports/` | Report schema validation | none | 37 |
| `tests/memory/` | Embedding providers and their failure handling | none | 13 |
| `tests/tenancy/` | Vault crypto, Cypher containment, graph and vector isolation, recall | FalkorDB + Qdrant | 56 |
| `tests/db/` | Executor, gate, verifier, loop, ingest, workers, gateway wiring | Postgres + FalkorDB | 309 |

### Why so much runs against real services

Tenant isolation, grounding, and the audit trail are all properties *of the
database and the graph engine*, not of our code in isolation. A mocked evidence
store would happily confirm a citation the real one rejects, and a fake graph would
"prove" an isolation guarantee FalkorDB does not provide. Three findings in
[`findings.md`](findings.md) were only reachable against a live engine — F-05 (`NaN`
rejected by `jsonb`), the FalkorDB "empty key" phrasing, and F-01's cross-tenant
evidence injection.

Isolation between DB tests comes from truncating every table before each one.
Rollback alone was tried and was insufficient: any test that legitimately commits
— anything exercising a real request path — leaked rows into later tests and
produced failures that only appeared in a full run.

### Why the LLM is never called

Every agent test drives `RecordedLLM`, which implements the real `LLM` interface
and records what it was asked. That makes the loop, the gate and the verifier
deterministic and free, and lets tests assert on the *prompt* as well as the
outcome — an adversarial verifier prompt that silently became agreeable would still
pass an outcome-only test.

The Anthropic provider is the exception: it is tested against an `httpx` mock
transport so the SDK's own request construction and response parsing run for real.
A hand-rolled fake of the SDK would encode our assumptions rather than the SDK's
behaviour, which is precisely the thing worth checking.

**What this cannot catch, and what we do about it.** A recorded provider accepts any
request, so it cannot tell us whether the API would. Two defects (F-11, F-12) lived
through 800 green tests for exactly that reason: an unusable output schema and a
transcript whose tool results referenced calls that were not in it. Both broke every
investigation, and both surfaced only on the first live run.

The response is not "call the model in tests" — that would be slow, costly and
non-deterministic. It is to assert the *provider's contract* against our own output,
without a network call:

| Assertion | Defect it would have caught |
|---|---|
| `TestStructuredSchemaTransform` walks the schema actually sent and rejects keywords structured outputs do not accept | F-11 |
| `TestTranscriptPairing` asserts every `tool_result` id has a matching `tool_use` in the previous turn | F-12 |
| `TestStructuredOutputCompatibility` checks the report schema before any provider sees it | F-11, at the schema layer |

That is the general shape: where a fake is permissive, encode the real constraint as
a test over the bytes we would have sent.

## Guard tests

Tests that keep an architectural property true as the codebase grows, rather than
checking a behaviour. Each was verified to actually fire by planting a violation
and watching it fail.

| Guard | Protects |
|---|---|
| `test_cypher_containment.py` | Cypher only in `cortex/memory/`, so tenant graph selection cannot be bypassed |
| `test_service_boundaries.py` | Services never import siblings; `cortex/` imports no web framework; no module-level resource singletons |
| `test_framework.py::TestSchemaMatchesHandler` | Capability schemas match handler signatures in both directions (F-06's whole class) |
| `test_worker_tasks.py::TestWorkerQueueBindings` | Each worker consumes exactly one queue; only the scheduler runs beat |
| `test_framework.py::TestReadOnlyEnforcement` | Every shipped capability is read-only |
| `test_bigquery.py::TestNoAgentAuthoredSql` | No capability exposes a SQL-shaped parameter |
| `test_schema.py::TestStructuredOutputCompatibility` | The report schema stays within the subset structured outputs accept, so drafting cannot 400 after the tokens are spent |
| `test_anthropic_llm.py::TestTranscriptPairing` | Every tool result pairs with a request in the preceding turn |
| `test_api_health.py::TestSurface` | The public route list is an explicit inventory, so an unreviewed endpoint cannot appear quietly |

The schema guard was proven the same way: restoring the positional-tuple chart point
produced three offenders (`maxItems=2`, `minItems=2`, `prefixItems`), and removing it
returned to green.

**The containment guard has been proven twice.** Once when introduced, and again
after the F-06 false-positive fix — a planted file with uppercase Cypher, lowercase
Cypher, `select_graph()` and a `falkordb` import produced 9 findings, and the file's
removal returned it to green. A guard that has never been seen to fail is
decoration.

## Regression tests

Every finding in [`findings.md`](findings.md) has a test named for it. The critical one
also has a standalone attack script that was run before and after the fix:
`ToolExecutor` accepted any `investigation_id`, the probe wrote attacker evidence
onto a victim's investigation, and after the fix the same unmodified probe returns
`InvestigationNotFound`.

| Finding | Test |
|---|---|
| F-01 cross-tenant evidence injection | `test_tool_executor.py::TestInvestigationScoping` |
| F-02 unverified identity headers | `test_gateway_tenant_dep.py::TestFailClosedOutsideLocal` |
| F-03 vault AAD omitted label | `test_vault.py::TestLabelBinding`, `::TestLegacyCompatibility` |
| F-04 unbounded response size | `test_http.py::TestResponseSizeLimit` |
| F-05 non-finite floats | `test_tool_executor.py::TestPayloadSanitisation` |
| F-06 schema/handler drift | `test_framework.py::TestSchemaMatchesHandler` |
| F-07 hash collisions | `test_tool_executor.py::TestCanonicalHash` |
| F-08 credential labels | `test_tool_executor.py::TestCredentialLabels` |

## Coverage gaps, and whether they matter

99% overall — 19 uncovered statements. Each is listed with why it is not covered,
rather than left to be discovered.

| Module | Cover | Uncovered | Why |
|---|---|---|---|
| `services/gateway/deps.py` | 84% | Tenancy error translation | Covered end to end by `test_gateway_tenant_dep.py`; the four statements are `raise HTTPException` lines whose effect is asserted through the HTTP status |
| `agents/anthropic_llm.py` | 90% | Non-`APIStatusError` transport branch, absent-`usage` fallback | Requires the SDK to raise an `APIError` that is not a status error, or to return a message with no usage block. Its own types make both unreachable from a mock transport; they are defensive |
| `reports/verifier.py` | 97% | Finding-claim verifier-outage branch | The outage path is covered for summary claims; the finding-level branch is the same three lines reached through a second loop |
| `tools/google_auth.py` | 95% | Non-JSON and absent-token replies on the token endpoint | Both covered for the exchange itself; these two are the same checks on a second call site |
| `tools/hubspot.py` | 99% | One optional filter branch | Reached only when `pipeline_id` and `min_amount` are supplied together |

None of these hide a behaviour a reader would assume is tested. The two that
previously did — a verifier and a provider at 0% — are described below.

### Two modules shipped untested, and were caught by measuring

`reports/verifier.py` (138 statements) and `agents/anthropic_llm.py` (71) were
written, committed, and described as done at **0% coverage**. Neither had a single
test. The verifier is one of the two grounding mechanisms the product's central
claim rests on.

Nothing detected this except running `--cov`. The suite was green throughout,
because a module with no tests cannot fail. Recorded because the failure mode is
general: a growing green suite reads as increasing confidence while coverage of new
code silently drops.

Both are now covered (97% and 90%). Writing the provider tests also corrected a
wrong assumption of mine — that a 429 raises. The SDK retries 429 and 5xx with
backoff, so a transient rate limit self-heals; the tests now document that, and the
reason the loop bounds wall clock rather than only step count is that those retries
spend it.

## What is not tested, on purpose

| Not tested | Why |
|---|---|
| Live connector calls | Would need real credentials, be non-deterministic, and could not exercise the 429/401/malformed-body paths that matter most |
| Real LLM calls in `tests/` | Non-deterministic and metered. Model behaviour is scored by the eval suite instead — see below |
| Report rendering | No UI yet (M7) |
| Nightly sync | Not built (M4) |
| Alembic migration content | `alembic check` confirms no drift against the models; `create_all` builds the test schema |

## The eval suite is not part of `make test`

`make test` measures the code. `make eval` measures the **analyst** — it runs the real
loop, gate, verifier and executor against labeled fixtures with a live model, and
fails when a gating dimension regresses.

They are separate commands because they answer different questions and have different
costs. `make test` is deterministic, free, and runs in seconds. `make eval` spends
tokens and its result depends on a model, so putting it in the unit suite would make
every run slow, metered, and occasionally red for reasons unrelated to the commit.

`make eval-harness` exercises the harness itself deterministically — the scripted
analysts in `tests/db/test_eval.py` prove the suite fails on a fabricated citation, a
decoy named as the cause, a right answer reached without the discriminating tool call,
and an invented conclusion on unanswerable data. That last one caught a real defect in
the scorer (F-10): grounding alone gated, so a fabricated *conclusion* with real
citations passed.

## Conventions

- **A test names the property, not the method.** `test_another_tenants_credential_is_not_used`, not `test_load_context_2`.
- **Docstrings say why it matters** when the reason is not obvious from the name — usually the consequence if the property broke.
- **Hostile input is parametrised.** Slugs, repo names, identifiers and SQL values all have adversarial cases beside the happy path.
- **A guard must be seen to fail.** Plant a violation, watch it fire, remove it.
- **Inject a seam rather than sleep.** The loop's wall-clock budget takes a `clock` callable. A test that reached the limit by sleeping would be slow and flaky, which in practice means the limit goes untested — and it is the budget most likely to matter under a hung upstream.
- **Fix the test when the test is wrong.** Several expectations of mine were wrong, not the code — float rounding, a hardcoded epoch, the retry contract above. Those are corrected in the test, and the reasoning is recorded.

## Coverage fell from 99% to 82%, and where it went

Worth stating rather than quietly reporting the new number: the drop is not a regression in
discipline on tested code, it is a batch of modules that shipped without tests while the
product was moving fast — benchmark adapters and two CLIs.

Two of them were genuinely risky and are now covered:

| Module | Was | Now | Why it mattered |
|---|---|---|---|
| `cortex/tools/posthog.py` | 26% | 94% | Enforces the per-tenant project allowlist and composes HogQL. An access-control boundary and a SQL-injection surface, both untested. |
| `cortex/tools/github.py` | 66% | 93% | The three new capabilities were untested — and `commits`/`issues` are exactly where the `request_json` array-wrapping bug would have silently returned "nothing shipped". |
| `cortex/connect.py` | 0% | 54% | The one place a real credential is stored. The isolation gate, the environment-only secret rule and the `--meta` parsing are covered; the database write path is not. |

The PostHog file is the one that should have been written first. A connector at 26% whose
job includes *refusing to read a project the tenant did not declare* is an untested
security control, and it was untested for the whole day it was in use against live data.

What is still at 0%, and the honest reason:

| Module | Lines | Why not yet |
|---|---|---|
| `cortex/bench/{bird,dabstep,__main__}.py` | 258 | Development tooling. A wrong benchmark number is embarrassing, not dangerous, and every number it produces is published with its denominator. Worth testing before any figure is quoted externally. |
| `cortex/ask.py` | 186 | A rendering CLI. Its output is read by a human who would notice it being wrong. |

None of these are on the request path for an investigation. That is the line being drawn —
not "coverage is fine", but "the untested code cannot mislead a tenant".

## The ingest tests are mostly about failure

`tests/db/test_ingest.py` and `tests/tools/test_ingest_syncers.py` are shaped differently
from the rest of the suite, deliberately. **Every interesting bug in a sync is silent:**

| Bug | What it looks like | Test |
|---|---|---|
| Watermark advances past a window nothing wrote | A gap in history with nothing pointing at it | `test_a_watermark_advances_only_on_success` |
| Watermark rewinds on a late retry | The next run re-reads weeks | `test_a_watermark_never_moves_backwards` |
| A metric point written twice | Every baseline over the series is wrong | `test_re_reading_the_same_window_does_not_duplicate` |
| Edges written before nodes | A graph that looks populated and cannot be walked | `test_it_writes_nodes_before_edges` |
| One stream taking its siblings down | The data that did arrive is lost | `test_one_failing_stream_does_not_fail_its_siblings` |
| A payload shape that maps to nothing | An empty graph reading as "nothing happened" | the syncer mapping tests |

None of those throw. All of them would pass a test that only asserted "data moved", which is
why the suite asserts the failure behaviour instead.

Two of these tests exist because the failure happened in a real run first: a vector-store
outage losing a sync whose graph writes had succeeded, and a vendor's rate-limit body — with
a link to its billing dashboard — on course for a customer-facing report.

## Two mechanisms that replaced a habit of fixing instances

Both were added after the same realisation: naming a bug class in a commit message does not
prevent the next instance of it.

**Empty results.** Four bugs in two days handed the analyst an empty result that was
indistinguishable from a real absence. `tests/db/test_empty_results.py` asserts the three
layers that now close it, and its last test asserts the four original payload shapes would
each be caught — the class, not the instances. The layer that matters most is the first:
`Capability` cannot be constructed without declaring `result_key`, so a new connector cannot
reintroduce the bug by omission.

**Charts.** Pydantic proves a chart's shape; it cannot prove the picture and the claim agree.
`tests/reports/test_charts.py` covers the three ways a well-formed chart still misleads — a
deploy marker at a position the data does not cover, an empty panel under a confident title,
a trend drawn from one point — plus the renderer, which is part of the honesty surface rather
than cosmetics. One test exists because the first renderer drew the chart's *lowest value* as
blank, and blank is what a gap draws as: a flat series at the minimum appeared as no data at
all.

## The one gap 1,325 passing tests had

A commit shipped `from cortex.db.titles import Tenant, title_for`. `Tenant` lives in
`cortex.db.models`, so `python -m cortex.ask` — the way every real investigation in this
project has ever been run — died on an `ImportError` before it parsed an argument. The full
suite passed over that commit, and would have kept passing.

The reason is structural rather than an oversight in one commit: **a module whose only job is
to be run as `__main__` is imported by nothing.** `cortex/ask.py`, `cortex/connect.py`,
`cortex/spend.py` and the three `__main__.py` files have their argument parsing and their
output tested through functions, but nothing in the suite imports the module itself, and a
module's imports only execute when it is imported.

`tests/unit/test_entrypoints_import.py` closes it for the cost of one import per entrypoint.
It asserts nothing about behaviour — that is tested elsewhere — only that the door opens. Its
second test walks the tree and fails if a new CLI is added without a line in the list, so the
guard cannot be outgrown quietly.

`runpy` is deliberately not used: it would execute `main()`, which for these modules means
spending money.
