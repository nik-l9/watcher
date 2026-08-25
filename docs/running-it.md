# Running it

Four ways in, in the order you are likely to want them. The Slack one is the only one a colleague
can use without being taught anything, which is why it exists.

Every example below uses `acme` as the tenant slug. Substitute your own.

## 0. What has to be running

```bash
cp .env.example .env                        # then fill in the keys you have
docker compose up -d                        # datastores only
docker compose --profile app up -d --build  # + gateway, both workers, scheduler, slack-socket
```

Eight services. `slack-socket` exits cleanly when `CORTEX_SLACK_APP_TOKEN` is unset, so a webhook
deployment can leave it in the file unused.

**Without the investigation worker nothing consumes the queue**, so a question asked through the
API or Slack sits at `queued` forever. The report view discloses a row stuck there for over twenty
minutes as likely abandoned, which is how that failure was found rather than guessed at.

The CLI does not need the worker: it runs the investigation in its own process.

Note that `docker compose` bakes environment values at container *create* time. A new `.env`
value needs `up -d` to recreate the containers, not `restart` — a `restart` leaves half the
services on the old value with nothing to indicate it.

## 1. Slack

An `@mention` in any channel the app is in:

> **@cortex** why did conversation volume change in the last two weeks?

The answer arrives in that thread in roughly two minutes: the summary, its citations, the
confidence, one chart as a PNG, and a link to the full report.

### Setting it up

A workspace is bound to a tenant by `tenants.slack_team_id`. **That mapping is the only thing
deciding whose data an inbound event reads**, so set it before the app is used.

1. **Create the app** — api.slack.com/apps → *From an app manifest* → your workspace, pasting
   [`infra/slack/cortex-app-manifest.yaml`](../infra/slack/cortex-app-manifest.yaml).
2. **Install it**, then copy the Bot User OAuth Token and the Signing Secret.
3. **Store them** as `CORTEX_SLACK_BOT_TOKEN` and `CORTEX_SLACK_SIGNING_SECRET`. For Socket Mode
   also set `CORTEX_SLACK_APP_TOKEN` (an app-level token, `connections:write`).
4. `CORTEX_PUBLIC_BASE_URL=https://…` — only used to build the "full report" link in a delivered
   message. Unset means no link rather than a broken one.
5. **Recreate the services**, then `/invite @cortex` in a channel and ask it something.
6. `./scripts/setup_slack.sh acme` registers the workspace against a tenant.

### Socket Mode or a webhook

Socket Mode needs nothing inbound exposed, which is the main reason to prefer it. It also
*disables* the HTTP Request URL, so one Slack app cannot serve both transports.

For a webhook a tunnel is enough — `cloudflared tunnel --url http://localhost:8000` or
`ngrok http 8000` — with the URL in both the Slack event subscription and
`CORTEX_PUBLIC_BASE_URL`. **A tunnel URL changes when it restarts and Slack does not follow it**,
so answers stop arriving with no error anywhere. That is why Socket Mode is implemented here.

## 2. The web report view

```
http://localhost:8000/ui/investigations
```

A list of investigations by title, each opening to the full report: the answer, findings with every
citation clickable through to its source row, charts, hypotheses tested, risks, data-quality notes,
and the trace of every tool call with its parameters and duration. A running investigation shows
its trace and refreshes itself.

Locally it needs an `X-Cortex-Tenant: acme` header, which browsers do not send. Outside `local` or
`test` a Clerk token is the only accepted identity and the header is ignored — see
`services/gateway/deps.py`. Making this usable by other people is deploy work, not build work.

## 3. The HTTP API

```bash
# ask
curl -X POST localhost:8000/investigations \
  -H 'X-Cortex-Tenant: acme' -H 'Content-Type: application/json' \
  -d '{"question":"Why did conversation volume change in the last two weeks?"}'

# poll — status, steps, tokens, and report_id when it is ready
curl -H 'X-Cortex-Tenant: acme' localhost:8000/investigations/<id>

# what it did, while it is still working
curl -H 'X-Cortex-Tenant: acme' localhost:8000/investigations/<id>/trace

# the report
curl -H 'X-Cortex-Tenant: acme' localhost:8000/reports/<report_id>

# a follow-up that may cite the parent's evidence instead of re-fetching it
curl -X POST localhost:8000/investigations \
  -H 'X-Cortex-Tenant: acme' -H 'Content-Type: application/json' \
  -d '{"question":"And which event drove that?","parent_id":"<id>"}'

# this tenant's own spend
curl -H 'X-Cortex-Tenant: acme' 'localhost:8000/spend?days=30'
```

## 4. The CLI

For development, and for checking answers against a known truth.

```bash
# real data
python -m cortex.ask --real --tenant acme "Why did signups fall last week?"

# a follow-up
python -m cortex.ask --real --tenant acme --follow-up <investigation-id> "And on mobile?"

# against a labelled fixture, where the true cause is known and the answer can be checked
python -m cortex.ask --dataset onboarding_regression --show-truth
```

Both print progress to stderr and a phase breakdown at the end, so stdout stays the report and
stays pipeable.

## Operational commands

```bash
make test                                 # the full suite
make eval                                 # score the analyst against labelled fixtures
python -m cortex.spend                    # cross-tenant spend, read-only transaction
python -m cortex.ingest --tenant acme --provider github   # nightly sync, on demand
make ingest-status TENANT=acme            # watermarks, health and staleness per stream
```

## Known limits

- **The 90-second target is met on most but not all runs.** The remaining cost is serial token
  generation — around 78% of wall clock — so it is an output-rate wall rather than something to
  tune. See [`latency.md`](latency.md).
- **A connector with no credentials is not offered to the analyst at all.** That is deliberate:
  offering tools it cannot reach cost real time and produced reports apologising at length for
  an absence that was never relevant. So an unconfigured connector costs nothing at runtime, and
  you can run this with only the credentials you have.
- **Embedding rate limits show up as a slow sync rather than an error.** The embedder waits out a
  429 instead of failing on it, so a free-tier key finishes — slowly.
