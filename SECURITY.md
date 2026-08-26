# Security policy

## Reporting a vulnerability

Email **nklkumar0321@gmail.com**. Please do not open a public issue for anything that could be
exploited before a fix exists.

Useful in a report, in rough order of value: what an attacker gets, the smallest reproduction you
have, the commit or version you tested, and whether it needs valid credentials for a tenant. A
one-paragraph description of a real problem is worth more than a polished report of a theoretical
one — send what you have.

Expect an acknowledgement within a few days. This is a small project without a paid security team,
so there is no bounty and no guaranteed remediation window. What is guaranteed is that a real
report gets a straight answer about whether it is being fixed.

## What this project treats as a vulnerability

The two guarantees worth attacking, because everything else is a bug:

**1. Tenant isolation.** Data belonging to one tenant must be unreachable from another. Isolation
is structural rather than filtered — one FalkorDB graph per tenant, one Qdrant collection per
tenant, a per-tenant vault key, a tenant predicate on every Postgres query — so a way to read
across tenants is the most serious class of report this project can receive. `tests/tenancy/`
exists to prove it, and a case that suite does not cover is itself worth reporting.

**2. Grounding.** No claim reaches a reader without a resolvable evidence id. A way to get an
unevidenced claim past the citation gate or the adversarial verifier is a vulnerability in the
product's central promise, not a quality issue. Fabricating a citation that resolves is the
strongest version of this.

Also in scope: credential disclosure (the vault, connector tokens, anything reaching a log or a
model's context), authentication and tenant-binding on the gateway, and any path that lets a
read-only deployment write.

## What is out of scope

- **The datastores' own configuration.** `docker-compose.yml` is a development stack: no
  passwords worth the name, no TLS, ports bound to localhost. Hardening it for exposure is
  deployment work, and reports that it is insecure as shipped will be closed as intended.
- **Denial of service through your own API keys.** An investigation spends tokens by design. Cost
  controls are budgets and step limits, not a security boundary.
- **Prompt injection through connector data** — a Slack message or a PR title that argues with the
  analyst. Real, and treated as a correctness problem rather than a vulnerability: the mitigation
  is that a claim must cite an evidence row, so injected text cannot manufacture a citation. A
  case where it *can* is in scope, and interesting.
- Anything requiring the attacker to already hold that tenant's credentials.

## Handling your own deployment

Two notes that have bitten in practice, since a public repository invites people to run this:

- **Every `.env` variant is gitignored, including `.env.bak.<timestamp>`.** A backup written by a
  tool once sat untracked with live credentials in it, matched by neither `.env` nor `.env.local`.
  Check what your editor and your tooling leave behind.
- **`docker compose` bakes environment values at container *create* time.** A rotated key needs
  `up -d` to recreate the containers, not `restart` — a `restart` leaves half the services on the
  old value with nothing to indicate it.
