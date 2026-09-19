# Licensing — what can be open-sourced, and what the datastores cost you

Written because two of the four datastores are **source-available rather than open source**, and
that fact lands on the commercial plan rather than the open-source one. Getting the direction of
that wrong in either direction is expensive: refusing to publish over a constraint that does not
apply, or publishing a hosted service that violates one that does.

Audited 2026-08-12. **This is engineering analysis, not legal advice** — before any hosted
commercial launch, the SSPL §13 question below needs a lawyer, not a README.

**Correction, 2026-08-15.** An earlier version of this file, and of the notes around it, treated
Elastic License 2.0 dependencies as disqualified. That was too strong, and
[ADR 0004](decisions/0004-connector-breadth.md) quotes ELv2's limitations in full: it restricts
offering *that software* as a hosted service to third parties, and nothing else. It does not
restrict internal use, and it does not reach a whole service the way SSPL section 13 does. The two
are different problems and conflating them makes the smaller one look unfixable.

## watcher's own code: AGPL-3.0, dual-licensed

`LICENSE` is the GNU Affero General Public License v3.

**Changed from Apache-2.0 on 2026-09-19, while the repository was still private** — which is the
window this file's earlier version flagged: "swapping it is a one-file change while nothing is
published; after a public push it is effectively permanent, because every fork keeps the terms
it received." The decision was taken deliberately inside that window rather than inherited.

The reasoning. Apache-2.0 was chosen for its express patent grant in §3, and AGPL-3.0 carries an
equivalent patent grant in §11, so nothing is lost there. What Apache-2.0 also permitted was a
competitor or a cloud vendor taking this, hosting it, charging for it, and owing nothing — and
the differentiated work here is exactly the part worth hosting: the grounding stack, the
evaluation harness, the tool framework.

AGPL's §13 closes that and nothing else. It reaches someone who **offers the program to third
parties over a network**; it does not reach a user who forks it, modifies it, and runs it for
themselves or their company, which is the intended user in every case. So the "fork it, add your
own API keys, run it locally" story is untouched.

**Dual-licensed**, which is what makes it a commercial basis rather than only a restriction: a
commercial licence is sold to anyone who wants to host watcher as a service without the AGPL's
reciprocity, or whose legal department refuses AGPL outright. This is the GitLab / Grafana /
pre-SSPL MongoDB shape, and it works only while the copyright stays with one holder — see the
CLA note below.

**Contributor licensing.** Dual-licensing requires the right to relicense contributions. Today
the copyright is single-holder and that right exists. A contributor licence agreement should be
in place *before* the first outside contribution is merged, not after: retroactive relicensing
needs every contributor's individual consent, and that is how dual-licensed projects get stuck
unable to sell the licence they planned on.

**The one cost, stated plainly.** Some companies ban AGPL by policy, so a fraction of potential
users self-select out before evaluating. That was accepted knowingly: those are, by and large,
the companies most likely to have wanted the hosted product anyway.

## Python dependencies: all permissive

Every runtime dependency in `pyproject.toml` is MIT, Apache-2.0, BSD-3-Clause, or PSF. Nothing
here constrains redistribution, including `falkordb` — but note that package is the **client**,
which is MIT even though the server is not. Only the client is imported.

## Datastores: two of four are source-available

| Service | Version | License | OSI open source? |
|---|---|---|---|
| Postgres | 16 | PostgreSQL License | yes |
| Qdrant | latest | Apache-2.0 | yes |
| Redis | **7.4.7** | RSALv2 / SSPLv1 (dual) | **no** |
| FalkorDB | **latest** (Redis 8.6.3 inside) | **SSPLv1** | **no** |

Redis relicensed at 7.4; the image `redis:7-alpine` resolves to 7.4.7 and therefore to the
non-OSI terms. FalkorDB is SSPLv1 and is itself built on Redis.

### Why this does not block the open-source release

The distinction that decides it: **Cortex does not distribute either database.** It talks to
them over a network protocol, and `docker-compose.yml` names public images that a user pulls
themselves. watcher's own source can be AGPL-3.0 regardless of what those images are licensed
under — a license binds the thing it covers.

And for the intended user — fork it, add their own API keys, run it locally — SSPL imposes
nothing at all. Its §13 obligation triggers on *offering the program's functionality to third
parties as a service*. Running a database for yourself, your team, or your company is not that.
This is the ordinary case for every self-hosted SSPL deployment in existence.

So the "anyone can fork it and run it locally" goal is unaffected. That is the honest answer and
it is worth stating plainly, because SSPL has a reputation that outruns what it actually
restricts.

### Where it does bite: a hosted commercial Cortex

If Cortex is later offered as a service, SSPL §13 asks for the source of **the entire service**
— management, orchestration, provisioning, monitoring, the lot — released under SSPL. RSALv2 on
Redis is blunter still: it prohibits offering the software as a commercial database service
outright.

Neither is a problem to solve at open-source time. Both are problems to solve before a hosted
launch, and there are three routes:

1. **Redis → Valkey.** BSD-3-Clause, Linux-Foundation-governed fork of Redis 7.2.4, wire- and
   command-compatible. Cortex uses Redis only as a Celery broker and cache, so this is a
   compose-file change and a test run. The cheapest of the three by a wide margin, and worth
   doing early simply to stop the constraint spreading.
2. **FalkorDB → another graph.** This is the one the architecture already anticipated.
   `cortex/memory/graph_store.py` exists as an ABC with a graph-per-tenant contract precisely so
   the backend can be swapped, and the M0 notes record Neo4j Enterprise as the intended
   alternative. openCypher is shared, so the port is mechanical rather than a redesign. A
   decision made for tenancy reasons turns out to also be the licensing escape hatch.
3. **Buy a licence.** FalkorDB sells commercial terms, and FalkorDB Cloud makes you a customer of
   a hosted database rather than a re-offerer of one. Often the cheapest answer measured in
   engineering time.

None of this is urgent. All of it is cheaper to decide before a hosted product exists than after.

## What stays private when the core is published

The split is not "core versus tenancy". Multi-tenancy is *structural* here — graph-per-tenant,
per-tenant Qdrant collections, per-tenant vault keys, a `tenant_id` predicate on every query —
and removing it would leave a weaker single-tenant system plus a private fork to keep merging
forever. A solo user simply runs one tenant, which is what "run it locally" means in practice.

What is genuinely commercial, and genuinely separable:

- **Real business data.** A private working copy of this repository carried a real-data findings
  file and a security analysis of a running deployment. Neither belongs in a public repo, and
  neither is here — deleting such a file at HEAD does not remove it from history, so this
  repository starts from a fresh one.
- **The workspace's own configuration** — the Slack app manifest, `slack_team_id`, tenant seeds.
- **Anything hosted**: billing, plan enforcement, org/SSO management, operational runbooks.

The reusable, differentiated core is the grounding stack (evidence store, structural citation
gate, adversarial verifier, a report schema whose claims cannot render without a resolvable
`evidence_id`), the evaluation harness, and the tool framework. Those are domain-agnostic and are
the part worth having other people use.
