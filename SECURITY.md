# Security

## Reporting a vulnerability

Open a [private security advisory](https://github.com/nik-l9/watcher/security/advisories/new).
Please do not open a public issue for anything exploitable.

I will acknowledge within 72 hours and tell you honestly whether I can fix it quickly, slowly,
or not at all.

## The threat model, stated plainly

You run this yourself, with your own API keys, against your own data. So the thing you are
trusting is the code in this repository, and you deserve to know what it does with those keys
and where it can be pushed around. This section is what I would want to read before running
someone else's agent against my company's Slack.

### What the keys do, and where they go

- **Your LLM key** is read from the environment and used for exactly one thing: calls to the
  model provider's API. It is never written to disk, never logged, and never sent anywhere else.
- **Connector credentials** are envelope-encrypted with `CORTEX_VAULT_MASTER_KEY` and stored as
  ciphertext in Postgres. They are decrypted per call, after checking the tool being invoked
  belongs to the tenant asking. `watcher connect` reads secrets from the **environment, never
  from a command-line argument** — an argument lands in shell history and in the process table,
  where any other user on the machine can read it.
- **Nothing phones home.** There is no telemetry, no usage reporting, no update check. The only
  outbound connections are to your model provider and to the connectors you configured.

### The main risk: prompt injection through your own data

This is an agent that reads Slack messages, GitHub issue bodies, pull-request review comments,
event names and CRM fields, and feeds them to a model. **Anyone who can write into those systems
can write into the model's context.** A colleague, a contractor, an external user filing an issue
on a public repository — all of them can author text the agent will read.

What that cannot do, structurally:

- **It cannot make the agent change anything.** Every capability is read-only, and that is
  enforced at construction rather than by convention: a capability declaring `read_only=False`
  raises at import, so a write path cannot exist to be reached. See `cortex/tools/base.py`.
- **It cannot reach another tenant.** Graph-per-tenant, per-tenant vector collections, and a
  tenant predicate on every query, with the isolation suite in `tests/tenancy/` gating any
  release.
- **It cannot exfiltrate to an attacker-chosen destination.** Connectors call fixed API hosts;
  no URL from model output or tool output is fetched.

What it can still do, and what to watch for:

- **It can influence a report.** An attacker's Slack message becomes an *observation*, so a claim
  citing it is technically grounded — in the attacker's sentence. The grounding gate and the
  adversarial verifier check that a claim matches its evidence; neither can check that the
  evidence itself is honest. **Read the sources on a report before acting on it**, particularly
  where the evidence is free text rather than a metric.
- **It can be steered into which questions it asks.** Read-only and same-tenant, so the cost is
  wasted steps rather than disclosure.

If your Slack or issue tracker is open to people you would not let read your analytics, that is
the risk to weigh.

### Multi-tenancy

Present and structural, but the isolation boundary has only been exercised by this repository's
own test suite. It has not been through third-party review or a pentest. **Do not treat it as a
security boundary between mutually hostile tenants** without doing that work yourself.

## What is checked in CI

- **The whole git history is scanned for secrets** on every push, not just the working tree — a
  key that was committed and later deleted is still in the pack and still exploitable.
- The tenancy isolation suite runs against real datastores rather than mocks. A mocked graph
  would happily prove a guarantee the real engine does not provide.
- GitHub Actions are pinned to full commit SHAs. Tags are mutable, and in March 2025
  `tj-actions/changed-files` was compromised by repointing existing tags at a malicious commit.
- Workflows declare `permissions: contents: read`, so a compromised step cannot use
  `GITHUB_TOKEN` to write to the repository.

## Running it safely

- Give each connector credential **the narrowest scope that works**. The analyst only ever reads;
  a token with write scope grants nothing it uses and everything an attacker would want.
- Keep `CORTEX_VAULT_MASTER_KEY` out of the repository and out of your shell history. `make
  setup` generates one into `.env`, which is gitignored.
- The compose file is for **local development**. It binds datastores to localhost with
  development passwords; it is not a production deployment.
- If a secret has ever been typed into a terminal you share, a chat, or a screenshot, treat it as
  compromised and rotate it. Removing it from a file does not remove it from history.
