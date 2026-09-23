# Case studies

Seventeen recorded investigations: six against a live HubSpot account, eleven against
labelled datasets whose true answer is known.

Each `.cast` is an [asciicast v3](https://docs.asciinema.org/manual/asciicast/v3/) recording
of the production path — the real investigator loop, real tool calls writing real evidence
rows, the real grounding gate, the real adversarial verifier. Nothing is staged. The `.txt`
beside it is the same session as plain text, so a reviewer can read one in a diff without a
player.

```
asciinema play docs/case-studies/won_accounts_not_activated.cast
```

## Live data, every figure and name masked

Recorded against a real CRM with `watcher ask --real`. The masking is mechanical, not
eyeballed: `scripts/mask_cast.py` blanks every digit in the report's prose — carving out
only dates, evidence ids and citation brackets — then blanks an identifying-term list that
`scripts/sensitive_terms.py` derives from the investigation's *own stored evidence*. Deal
names, owners, repositories, the individual words inside them, and all-caps acronyms down to
two characters. Both passes end in an assertion, and the recorder deletes its own output if
the tenant slug survives.

| recording | what this run did |
|---|---|
| `real_pipeline_health` | gave the total, then weighted it by HubSpot's own probability field and found a cluster of zero-contact, no-source deals all created on one January date |
| `real_where_deals_die` | answered, then said the question cannot be answered as asked: closed-lost records do not preserve which stage a deal was lost from. Recommends instrumenting it |
| `real_crm_trust` | found systematic duplicate company records inflating pipeline counts |
| `real_source_quality` | **declined** — source attribution is missing on most closed-won deals, so channels cannot be compared. Recommends fixing capture before measuring |
| `real_stalled_deals` | answered **no** — every open deal has been touched inside the window. A clean negative, and no list invented to fill the space |
| `real_hubspot_win_rate` | gave the rate two ways, by deal count and by dollar value, because those are different numbers |

Two of the six end without the answer that was asked for. That is the point of them.

## Labelled datasets, with the answer key printed

Each ends by printing what was actually planted in the data, underneath what it just claimed,
so a reader can mark the run rather than admire it.

| recording | correct verdict | what this run did |
|---|---|---|
| `onboarding_regression` | name the cause | found the mobile onboarding regression |
| `campaign_traffic_drop` | name the cause | traced the 42% fall to the campaign ending on 2026-06-15 |
| `measurement_stopped` | name the cause | *"sessions did not collapse — GA4's own series simply stops recording"* |
| `won_accounts_not_activated` | name the cause | only three of nine accounts closed last quarter show product usage |
| `forecast_ignores_the_decision` | name the cause | $900K of $2.4M pipeline is dead — found in Slack, recorded nowhere in the CRM |
| `partial_month_false_premise` | refuse the premise | *"No, signups have not fallen: the daily rate was about 157/day"* |
| `partial_month_disclosed` | refuse the premise | weaker — said it could not confirm, rather than refuting the premise outright |
| `insufficient_evidence` | no cause | declined: the enterprise segment the question asks about cannot be isolated |
| `tempting_coincidence` | no cause | established a real 55% drop and declined to explain it |
| `onboarding_regression_undecidable` | no cause | named the segment, declined the cause |
| `campaign_traffic_drop_undecidable` | no cause | named the channel, declined the cause |

## These are single runs, not a scorecard

The analyst is stochastic, and the variance is large: two runs of the same question can
differ by seven cited claims. In this set `partial_month_disclosed` refused weakly while
`partial_month_false_premise` refused cleanly — in an earlier recording of the same pair it
was the other way round. Both are in the table as they happened.

For how often the system gets these right, run the eval suite, which scores many runs. What
a recording shows that a score cannot is the shape of the reasoning: which hypotheses were
raised, which were contradicted, what was cited, and what the verifier removed.

## Videos

`video/<name>.mp4`, rebuilt from the recordings in seconds with no API call, so they are not
committed:

```
scripts/make_case_study_videos.sh
```

They are re-timed before rendering. Played back as recorded, a run is unwatchable: the
analyst works for ninety seconds printing four progress lines, then prints its whole report
in a burst that scrolls past in about a second. `scripts/pace_cast.py` rewrites the timing
and nothing else — it asserts the byte stream is unchanged before writing.

## Known defects visible in these recordings

Recorded honestly rather than edited out:

- **`onboarding_regression_undecidable` contains a literal `—`** where an em-dash
  belongs. The sufficiency gate's own prompt contains em-dashes, and the model echoes them
  back into its JSON as an escape rather than a character. It reproduces across runs. Fixing
  it needs escape decoding on model output, which is not yet done.
- **`tempting_coincidence` does not test what it is named for.** Its bait — a copy-only PR
  merged the day before the drop — is reachable through exactly one call, and the analyst
  usually takes a different and equally reasonable route, finds an empty world, and declines.
  The verdict is right; the scenario is not measuring the temptation.
