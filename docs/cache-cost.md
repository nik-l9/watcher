# Where the $1.06 goes, measured per request

Per investigation, averaged over 44 real runs at this repository's own rates
(`cortex/db/spend.py`: read 0.10x input, write 1.25x): **$1.0638**, of which **$0.7784 --
73% -- is prompt-cache writes**. Cache reads were 107k against 124k writes.

That aggregate cannot say *which call* writes, and a healthy loop and a broken one produce
the same total. So `_log_cache` emits one line per request. The first run settled it.

## The loop is healthy. It was never the problem.

```
read=0       write=12,482   share=0.00   <- writes the prefix once
read=12,482  write=3,643    share=0.77
read=16,125  write=2,063    share=0.89
read=18,188  write=1,304    share=0.93
read=19,492  write=891      share=0.96
```

Writes shrink from 12,482 to 891 while the read share climbs to 0.96. That is delta billing
working exactly as the rolling top-level breakpoint intends.

## The `structured` calls are where the money goes

```
read=0       write=16,755   share=0.00   <- the report draft
read=16,755  write=4,162    share=0.80   <- its repair attempt reads it back
read=0       write=1,726    share=0.00   <- verifier, one claim
read=0       write=1,818    share=0.00   <- another claim
read=0       write=1,786    share=0.00   <- another
   ... seven of these, each writing, none reading
```

About 32k of the ~52k written per scenario comes from calls that read almost nothing back.

## Why the drafting call cannot hit the loop's cache

The obvious explanation was that `structured()` omits `tools`, which render ahead of system
and messages -- so the drafting request would differ from the loop's at byte zero. **Tested,
and it is wrong.** Passing the identical tool array made it worse: the write grew from 16,755
to 30,731 and the read stayed at zero. A bigger uncacheable prefix.

Both calls use the same `system_prompt`, and with tools equalised the only remaining
difference is `output_config.format`. The corroborating observation is in the trace above:
the draft's *repair* call reads the draft's 16,755 back. **Structured-output calls cache
against each other and never against tool-use calls.** So this is a property of the request
shape, not a missing field, and there is nothing to fix by threading arguments around.

## What is actually left

The verifier: seven calls per report, each writing ~1.8k and reading none. What they share is
the system prompt and whatever evidence rows two claims cite in common. Whether that is worth
a breakpoint depends on how much overlaps, which is measurable from the captured bundles
before any API call is spent -- and that is the next thing to check rather than guess.

## Two failed fixes, recorded

1. **Fixed breakpoint replacing the rolling one.** Predicted 60% off; measured 15x the billed
   input (8,488 -> 127,450 tokens) and doubled the non-supported claims. The documented
   pattern is a fixed breakpoint *and* the rolling tail; substituting one for the other stops
   the tail being cached at all.
2. **Passing `tools` on the drafting call.** Predicted the largest single win; measured a
   larger write and no read.

Both were reverted. The diagnostic that found them stays.

## What actually shipped: stop paying to cache what nothing reads

The verifier's overlap was measured from captured bundles before spending anything: 8.3
claims per report over 10.7 distinct evidence rows, 47% of rows cited by more than one
claim -- but a **median evidence payload of 199 characters, about 50 tokens**. Even with that
overlap the shareable prefix is the system prompt plus a few hundred tokens of evidence, and
caching it across seven calls saves under a cent. **A breakpoint there is not worth building**,
which is the useful half of the measurement.

The same numbers show the real thing. Those seven calls each *write* ~1.8k and read none, and
a cache write bills at 1.25x fresh input against a read at 0.1x. The premium is only repaid
when something reads the entry. Nothing ever does: each verdict judges a different claim, so
no later call shares its prefix. Caching them is a 25% surcharge for storage nobody collects.

`structured(cacheable=False)` on the verdict call, measured on the same scenario:

```
before   cache_write=1,726   fresh_in=2       read=0
after    cache_write=0       fresh_in=2,210   read=0
```

| | |
|---|---|
| verifier input per investigation | 16,892 tokens |
| billed as cache writes (1.25x) | $0.1056 |
| billed as fresh input (1.0x) | $0.0845 |
| **saving** | **$0.0211 — 2.0% of the bill** |

Scenario still passes, zero delivered hallucinations. No prompt changed, no output changed;
the same request is simply not written to a cache that never serves it.

**Two percent, not the forty the arithmetic promised.** The large projections assumed the
drafting call could be made to share the loop's cache -- it cannot, structured-output and
tool-use requests do not share prefixes -- and that a fixed breakpoint could be added
alongside the rolling one, which broke the loop when tried. What survives contact is small,
free and certain.

## The drafting call, measured in the suite

The verifier measurement above was one scenario. The same argument applies to the drafting
call -- it writes a large prefix and only the repair path would read it back -- but the
break-even is different, because a repair does sometimes happen. With a write at 1.25x and a
read at 0.1x, caching pays once the repair rate clears `N* = (w-r)/(1-r)`, about 28% of
drafts. The measured rate over the suite was 0/9.

Opting it out and running the whole suite against a matched baseline -- same nine scenarios,
dates pinned in both, nothing else changed:

| | write | read | fresh | out | $/investigation |
|---|---|---|---|---|---|
| draft cached | 450,728 | 833,489 | 231,537 | 78,931 | 0.7072 |
| draft uncached | 254,125 | 934,498 | 433,590 | 76,632 | 0.6821 |

**$0.0251 per investigation, 3.5%**, against $0.0280 predicted offline. Both runs: 9/9
scenarios passed, `delivered_hallucinations=0`, `caught_before_delivery=5`, zero non-supported
claims. Read tokens *rise* because the loop is unchanged and slightly longer; what falls is
the write premium on a prefix nothing collected.

**Where this leaves the bill.** $1.0638 measured at the start of the work, $0.6821 now -- 36%
off, and almost all of it from the two opt-outs plus the earlier effort setting. The levers
left are each worth a cent or two: output shaping tops out near $0.02 because only ~2,910 of
9,105 output tokens are report JSON, and the large multipliers were spent before this
session. Anything claiming more than that here should be measured before it is believed.
