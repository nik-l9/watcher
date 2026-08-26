# Findings

Most comments in this repository record a specific failure and what it cost, and many cite a
finding by id — `F-12`, `F-17`. This is the index those ids resolve against.

It is deliberately an index rather than the audit it came from. Each row is the defect and where
its regression test lives; the reasoning is next to the code that carries the fix, which is where
it is useful. Ids are never reused, and a gap in the sequence means a finding that was withdrawn
or folded into another rather than one held back.

Severity is as assessed at the time, against this project's own bar: **Critical** means a tenant
could read another tenant's data, **High** means the product produced a wrong answer or could not
produce one at all.

## Isolation and credentials

| Id | Severity | Defect |
|---|---|---|
| F-01 | Critical | Cross-tenant evidence injection: an evidence id from another tenant resolved, so a claim could cite data it had no right to |
| F-02 | High | The gateway trusted unverified identity headers. Closed by real token verification; the suite now asserts the attacks it refuses |
| F-03 | Medium | Vault additional authenticated data omitted the credential label, so a ciphertext was not bound to its exact row |
| F-08 | Low | The executor could only reach the `default` credential label, making a second account for one provider unreachable |
| F-14 | Medium | The API key in `.env` was ignored in favour of the ambient environment |
| F-22 | Medium | An explicitly empty credential fell back to the ambient one, so "configured as blank" and "not configured" behaved differently than they read |

## Evidence, grounding and the report

| Id | Severity | Defect |
|---|---|---|
| F-05 | Medium | Non-finite floats crashed the evidence write, and the failure was not audited |
| F-07 | Low | `canonical_hash` collided distinct payloads, so two different observations could hash alike |
| F-10 | High | The eval suite passed a fabricated conclusion — the grading, not the analyst, was at fault |
| F-11 | High | Every investigation would have failed at the drafting step: the schema the model was given was a subset of the one it was parsed against |
| F-17 | High | The report schema was too complex to compile into a structured-output grammar |
| F-24 | High | An empty result was indistinguishable from a real absence, in four capabilities. Fixed by making each declare what empty means for it |

## The provider transport

Four of these are the cost of writing a provider layer by hand rather than adopting one. Each was
found by a separate multi-minute live run, one at a time.

| Id | Severity | Defect |
|---|---|---|
| F-12 | High | The transcript orphaned every tool result, sending a `tool_result` with no matching `tool_use` |
| F-13 | Medium | Provider errors were unactionable — the message said something failed and not what |
| F-15 | High | A single provider call could outlive the whole investigation budget: a streaming request with no deadline |
| F-16 | High | A mid-stream transport failure killed the process instead of raising something catchable |
| F-18 | High | One per-request ceiling for calls of very different length, so the ceiling fitted neither |
| F-19 | High | A capacity error delivered inside an HTTP 200, so nothing retried it |
| F-23 | Medium | A cached async client survived its event loop and failed on the next one |

## Coverage and schemas

| Id | Severity | Defect |
|---|---|---|
| F-04 | Medium | Unbounded upstream response size — a large payload had no ceiling before it reached memory |
| F-06 | Low | Three capability schemas omitted required parameters, so a call could be built that the API would refuse |
| F-09 | Medium | Two modules shipped at zero coverage |

## Where the tests are

Every finding above has a test named for it. The transport ones are in
[`tests/unit/test_anthropic_llm.py`](../tests/unit/test_anthropic_llm.py), isolation in
[`tests/tenancy/`](../tests/tenancy/), and the rest sit beside the module they constrain. Grepping
the id finds both the test and the comment explaining why the code is shaped the way it is:

```bash
grep -rn "F-12" cortex/ tests/
```
