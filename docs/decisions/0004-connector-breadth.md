# ADR 0004 — How Cortex gets breadth: MCP for live sources, own connectors for the investigation path

**Status:** accepted, 2026-08-15
**Question it answers:** rather than writing a connector per tool, can we adopt an existing
project that already integrates hundreds — Airbyte, Nango, Meltano, Steampipe, dlt?

## Context

Cortex ships eight connectors. Every one was hand-written, and each cost a day or more. Airbyte
advertises 600+. The obvious question is why we are not simply using it, and the obvious
answer — "the licence" — turned out to be **partly wrong when checked**, which is why this
document quotes rather than summarises.

## What the licences actually say, verified 2026-08-15

Every claim below was read from the licence file itself, not from a comparison blog.

| project | licence | how it was verified |
|---|---|---|
| **Airbyte** monorepo, including `airbyte-integrations/connectors/*` | **Elastic License 2.0** | repo-root `LICENSE`; no `LICENSE` file exists in the connector directories; connector sources carry `Copyright (c) 2024 Airbyte, Inc., all rights reserved.` |
| **airbyte-python-cdk** (separate repo) | **MIT** | `LICENSE.txt` is the verbatim MIT text |
| **airbyte-protocol** (separate repo) | **MIT** | `LICENSE` names MIT |
| **Nango** | **Elastic License 2.0** | repo-root `LICENSE` |
| **Steampipe** | **AGPL-3.0** | repo-root `LICENSE` |
| **Meltano** | **MIT** | repo-root `LICENSE` |
| **dlt** | **Apache-2.0** | `LICENSE.txt` |

### ELv2 restricts less than its reputation suggests

The operative sentence, quoted in full:

> You may not provide the software to third parties as a **hosted or managed service**, where the
> service provides users with access to any substantial set of the features or functionality of
> the software.

Plus: do not circumvent licence-key functionality, do not strip notices. That is the whole of the
limitations section. The copyright grant immediately above it permits *"use, copy, distribute,
make available, and prepare derivative works."*

Three consequences, and the first two correct an earlier overstatement in this project's notes:

1. **For the open-source goal there is no restriction at all.** Someone clones Cortex and runs
   Airbyte locally against their own API keys. That is ordinary use and is expressly granted.
   ELv2 says nothing about what you run for yourself.
2. **For a hosted commercial Cortex the question is narrow**, and it is not the SSPL question. It
   turns on whether *our* users get access to a substantial set of *Airbyte's* features. Airbyte's
   own FAQ reads its licence as "you cannot host and sell Airbyte itself as a service or expose its
   UI/API directly to customers". Cortex using it as an internal pipeline component, with no
   Airbyte surface exposed, is not obviously that. **This is a lawyer's judgement, not an
   engineer's**, and nothing in this ADR should be read as legal advice.
3. **Bundling it would make Cortex mixed-licence.** ELv2 requires notices to travel with the
   software, so the repo could no longer be described plainly as Apache-2.0. That is a real cost
   for a project whose licence is part of its pitch.

### The connector licences are genuinely ambiguous, and that is the finding

Airbyte publishes an ELv2 page and an MIT page and **never states which components each covers.**
The FAQ says only "our own connectors remain open-source", naming no licence. Meanwhile the
connector directories contain no `LICENSE` file, so the repo-root ELv2 is the only licence in their
ancestry, and the sources carry an "all rights reserved" copyright header.

So the safe engineering assumption is that **the connectors are ELv2**, and anyone wishing to rely
on them being MIT should get that in writing rather than inferring it. This is a stronger reason to
avoid depending on connector *code* than the vague licence worry we started with.

The CDK being MIT is the interesting part: **the machinery for running a declarative connector is
MIT, while the connector definitions live in an ELv2 repo.**

## The architectural objection, which decides it regardless of licence

Airbyte, Meltano/Singer and dlt are **ELT**: they move records into a warehouse. Cortex has two
data paths and they want opposite things.

- **Nightly sync (M4).** Batch, high volume, schema-shaped. ELT tools fit well.
- **Live drill-down.** One tool call, one API response, hashed, stored as an `Evidence` row with a
  `source_ref` a human can follow. A claim renders only if its `evidence_ids` resolve.

An Airbyte connector is not a function returning JSON; it is a Docker image emitting a *stream* of
records for a destination. Routing investigations through one would leave every claim citing "a row
in a warehouse table" rather than "this API call at this moment", which quietly breaks the citation
gate. **600 connectors is not 600 tools we can use** — it is 600 pipelines into storage.

It would also cost the disclosures this project keeps discovering it needs: `partial_buckets`,
`series_ends_early`, "counts are a floor", connector-health surfaced in the report. Those are
per-call semantics that a generic pipeline has nowhere to put.

## Decision

1. **Keep hand-written connectors for the investigation path.** The per-call evidence contract is
   the product; nothing surveyed preserves it.
2. **Build an MCP client adapter for live breadth.** One adapter exposes any MCP server's tools as
   Cortex capabilities, writing evidence rows exactly as native connectors do. MCP is a protocol,
   so each server carries its own licence and none of it entangles ours. It is request/response,
   which is the shape our evidence model needs.
3. **Reconsider dlt (Apache-2.0) for bulk ingest breadth if and when M4 needs it.** A library, not
   a platform: it runs inside the existing ingest worker rather than becoming a tenth service.
   Meltano/Singer is the fallback if the tap ecosystem matters more than the ergonomics.
4. **Do not adopt Airbyte or Nango**, on the combination of mixed-licence cost, ambiguous connector
   licensing, and — decisively — the wrong shape for the path that matters.

## What the MCP adapter has to solve, recorded before building it

These are the reasons it is a real piece of work rather than a wrapper:

- **Tools return JSON, never prose.** MCP tools return content blocks that are frequently text. A
  tool whose result is a paragraph cannot be cited the way a row can, and would smuggle an
  unverifiable claim into the evidence store. The adapter must require structured content or
  refuse the tool.
- **Read-only is enforced by absence, not by prompting.** V1 ships no destructive capability
  anywhere. MCP servers routinely expose writes, so the adapter must admit only tools it can
  establish are read-only, and default to refusing.
- **`result_key` has no MCP equivalent.** It is how the executor marks an empty observation as
  empty — the defence against "no rows" being read as "no such thing", a bug this codebase has
  shipped four times. It must be declared per server or inferred, and inferring it wrongly
  reintroduces exactly that bug.
- **Credentials stay per tenant.** An MCP server is configured per tenant with its own secret in
  the vault, and the tenant filter must apply to MCP-sourced tools exactly as it does to native
  ones.

## Consequences

Breadth arrives through MCP rather than through an ELT platform, which means it arrives one server
at a time rather than 600 at once — slower, and it keeps the evidence contract intact. Cortex stays
Apache-2.0 with no mixed-licence notices to carry. If the hosted product later wants warehouse-scale
ingest, dlt is the pre-cleared option and this ADR is the record of why.

The earlier claim in this project's notes that Airbyte was "licence-disqualified" is **withdrawn**.
It is not disqualified; it is unsuitable, for architectural reasons, with a licence cost attached.
The distinction matters because it changes what a future reader is allowed to reconsider.
