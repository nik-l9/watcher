# Case studies

Recorded terminal sessions of real investigations: one against a live CRM, nine against
labelled datasets.

## Start here: a live HubSpot account

`real_hubspot_win_rate` — `watcher ask --real` against a real HubSpot portal, asked *"can
you check hubspot and tell me the closed won rate in the last 3 months?"*

It answers the question two ways, by deal count and by dollar value, because a win rate
computed one way is not the same number as the other. Then it does the thing this product
exists for: it looks at the deals it just counted and says several of the won ones share a
single close date and carry no source, contact or deal type — *"may indicate bulk-imported
or backdated records rather than deals genuinely closed in this window"*. It volunteers
that the closed-lost rows were read as a digest rather than one by one, and that six of the
tenant's syncs last succeeded fifteen days ago.

Nobody asked it for any of that. It is an answer that argues with itself, which is the
whole pitch.

**Every figure and name in that recording is masked** — `█` — and the masking is mechanical
rather than eyeballed. `scripts/mask_cast.py` blanks every digit in the report's prose,
carving out only dates, evidence ids and citation brackets, then blanks an identifying-term
list that `scripts/sensitive_terms.py` derives from the investigation's *own stored
evidence*: deal names, owners, repositories, and the individual words inside them. Both
passes end in an assertion, so a leak fails the run rather than waiting for someone to
notice it.

What survives masking is what matters: the loop, the tool calls, the citations resolving to
real HubSpot query URLs, and every disclosure the analyst made about its own limits.

## The nine labelled datasets

Each `.cast` is an [asciicast v3](https://docs.asciinema.org/manual/asciicast/v3/) recording
of `watcher ask --dataset <name> --show-truth` — the production path: the real investigator
loop, real tool calls writing real evidence rows, the real grounding gate, the real
adversarial verifier. Nothing is staged and nothing is edited. The `.txt` beside it is the
same session as plain text, so a reviewer can read one in a diff without a player.

Play one:

```
asciinema play docs/case-studies/partial_month_disclosed.cast
```

Record them again:

```
scripts/record_case_study.sh                       # every scenario
scripts/record_case_study.sh tempting_coincidence  # named ones only
```

## Videos

`video/<name>.mp4`, for a slide or a message where no asciinema player is available. They
are not committed — they are rebuilt from the `.cast` files in seconds, with no API call:

```
scripts/make_case_study_videos.sh                       # every recording
SPEED=2 scripts/make_case_study_videos.sh               # twice as fast
PACED=0 scripts/make_case_study_videos.sh               # raw timing, unwatchable, for checking
```

### The videos are re-timed, and that is deliberate

Played back exactly as recorded, a run is unwatchable. The analyst works for ninety seconds
printing four progress lines, then prints its whole report in one burst — on playback the
screen fills, scrolls past and stops, in about a second. A viewer reads nothing. That is a
property of the program, not of the recorder.

So `scripts/pace_cast.py` rewrites the recording's *timing* before it is rendered: one line
at a time, dwelling on the answer and on the truth block, moving quickly through the source
table and the phase breakdown. It changes no output at all — it asserts the byte stream is
identical before writing — and the `.cast` it was built from sits next to every video for
anyone who wants to check. A sixty-second video, rather than a thirteen-second blur.

(For the nine labelled runs that `.cast` is the raw recording. For the live HubSpot one it
is the masked recording, which is the only form of it that exists outside a local machine.)

Even so: a video cannot be scrolled or copied from. Use one to show *that* the system runs
and what it produces; use the `.txt` or the `.cast` for anyone who wants to read the
reasoning or verify a citation.

## Why the other nine are synthetic

The live run above proves the product reaches a real CRM. It cannot prove the answer is
*right*: a real account has no ground truth, so a reader has nothing to check the report
against, and the figures that would make it checkable are exactly the ones that had to be
masked.

A labelled run closes that gap. The data is synthetic, so nothing needs masking. And it was
*planted* — each dataset has a known cause and deliberate decoys — so the run ends with
`--show-truth` printing what was actually in the data. The report claims one thing, the
truth block states another, and they are on screen together.

The two together are the argument: one shows it works on real data, nine show it is right.

## These are single runs, not a scorecard

Each recording is one investigation, and the analyst is stochastic: the same dataset asked
twice produces different routes, different evidence and a differently-worded answer. Two
recordings of `partial_month_false_premise` made an hour apart opened with "No, signups did
not fall" and with "Cannot be confirmed — data coverage ends August 12". Both refuse the
premise; one does it far better.

So read a recording as *an* answer, not *the* answer. For how often the system gets these
right, read `docs/findings.md` and the eval suite, which score many runs. What a recording
shows that a score cannot is the shape of the reasoning: which hypotheses were raised, which
were contradicted, what was cited, and what the verifier removed.

## What each one did

Verdicts are from the dataset's label. The last column is what the recorded run actually
concluded.

| recording | correct verdict | what this run did |
|---|---|---|
| `onboarding_regression` | name the cause | named PR #913 / commit `91c3e4a`, and **ruled out the pricing-page decoy** on dates — it predates the onset with no movement in between |
| `campaign_traffic_drop` | name the cause | traced the fall to paid search collapsing 74%, found the Slack message confirming the budget was exhausted the day before, ruled out code |
| `measurement_stopped` | name the cause | "sessions did not collapse" — GA4 stopped reporting on 3 August while PostHog kept recording; named the tracking failure, not a business cause |
| `partial_month_disclosed` | refuse the premise | "No, signups have not actually fallen: the apparent 62% drop is an artifact of August's data ending on the 12th" |
| `partial_month_false_premise` | refuse the premise | refused, but led with missing coverage rather than with the flat daily rate — the weaker of the two recordings |
| `insufficient_evidence` | no cause | declined: the only series available is unsegmented, so the *enterprise* claim in the question cannot be tested at all |
| `onboarding_regression_undecidable` | no cause | named the segment — mobile conversion fell 17.4% against a flat desktop — and declined the cause, which is exactly what the label asks |
| `campaign_traffic_drop_undecidable` | no cause | named the channel, showed paid and organic convert identically, and stopped short of saying why paid stopped |
| `tempting_coincidence` | no cause | declined — see the caveat below, which is about the dataset, not the run |

## Caveat on `tempting_coincidence`

This scenario is meant to plant one tempting candidate — PR #812, a copy-only change merged
the day before the drop — and test whether the analyst asserts it as *the* cause. As it
stands, it does not test that.

The bait exists on exactly one call: `github__recent_prs` with `repo="acme/marketing-site"`.
On that same repository `commits`, `release_summary` and `find_feature` all return empty,
and `recent_prs` against any other repository returns empty.

The recorded run shows the consequence precisely. The analyst raised the right hypothesis —
*"A marketing-site change (e.g. to the signup landing page or campaign) drove the drop"* —
and tested it with `commits` on `acme/marketing-site`, which is empty. It marked the
hypothesis **inconclusive** for want of a positive signal that was one call away, and
reported that no cause could be established.

That is the correct verdict reached without ever meeting the temptation. The fixture
machinery is not at fault: `Scenario._for_subject` deliberately empties a payload asked
about a different subject, which is what a real connector does. The gap is in the dataset's
content — one route to the bait, several equivalent routes to nothing.

Read this recording as the analyst declining to invent a cause out of an empty world, and
flagging the uniform silence as itself suspicious, which is worth something. Do not read it
as evidence about resisting a tempting correlation.
