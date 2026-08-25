"""Where an investigation's wall clock goes.

An investigation takes 170–440 seconds against a 90-second target, and until now the only
thing measured was the total. That is not enough to optimise against: the plausible
culprits — sequential model calls in the loop, one very large drafting call, per-claim
verification, tool round trips — have wildly different fixes, and picking wrong costs a
day and buys nothing.

So each phase is timed and each phase's tokens are attributed to it. The point is to make
the next decision evidence-based rather than architectural taste:

  - if the **loop** dominates, the answer is fewer steps and prompt caching, because the
    cost is a dependency chain of full-price requests;
  - if **drafting** dominates, the answer is splitting or streaming one 32k-token call;
  - if **verification** dominates, the answer is concurrency (already partly done) or
    fewer claims;
  - if **tools** dominate, the answer is precomputed data or speculative execution.

Cache tokens are recorded alongside, because "we turned caching on" is a claim and
`cache_read_input_tokens` rising while `input_tokens` falls is the measurement.

Deliberately not a tracing framework. A dict of counters costs nothing, needs no
collector running, and survives being read out of a log file six hours later — which is
how these numbers actually get looked at.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from cortex.agents.llm import Usage

#: Phase names. Fixed rather than free-form so a typo cannot silently create a second
#: bucket that makes one phase look cheap.
#: Establishing what exists, before the first step. Timed separately because it is the one
#: phase that is pure overhead on the critical path: it happens before any work, so if it ever
#: grows it delays every investigation without contributing to any answer.
SURVEY = "survey"
#: Memory recall. Also pre-loop, also pure overhead, and also non-fatal -- so if it is slow it
#: should be visible as its own line rather than folded into the unmeasured remainder.
RECALL = "recall"
LOOP_MODEL = "loop_model"
LOOP_TOOLS = "loop_tools"
DRAFT = "draft"
GATE = "gate"
VERIFY = "verify"

#: The sufficiency gate (ADR 0005 decision 6), added after these were first named.
SUFFICIENCY = "sufficiency"

PHASES = (SURVEY, RECALL, LOOP_MODEL, LOOP_TOOLS, DRAFT, GATE, VERIFY, SUFFICIENCY)


@dataclass(slots=True)
class Phase:
    calls: int = 0
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)

    @property
    def mean_seconds(self) -> float:
        return self.seconds / self.calls if self.calls else 0.0


@dataclass(slots=True)
class Timings:
    """Per-phase wall clock and tokens for one investigation."""

    phases: dict[str, Phase] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)

    @contextmanager
    def measure(self, phase: str, usage: Usage | None = None) -> Iterator[None]:
        """Time a block and attribute it to `phase`.

        Usage is often not known until the block finishes, so `record` exists separately;
        this is for the timing half.
        """
        start = time.monotonic()
        try:
            yield
        finally:
            # Recorded in `finally` so a failed phase still shows its cost. A timeout that
            # burned 120 seconds is exactly the measurement wanted, and skipping it on the
            # error path would make a slow failure look free.
            self.add(phase, time.monotonic() - start, usage)

    def add(self, phase: str, seconds: float, usage: Usage | None = None) -> None:
        entry = self.phases.setdefault(phase, Phase())
        entry.calls += 1
        entry.seconds += seconds
        if usage is not None:
            entry.usage = entry.usage + usage

    def record_usage(self, phase: str, usage: Usage) -> None:
        """Attribute tokens to a phase whose timing was already recorded."""
        entry = self.phases.setdefault(phase, Phase())
        entry.usage = entry.usage + usage

    @property
    def measured_seconds(self) -> float:
        return sum(p.seconds for p in self.phases.values())

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for phase in self.phases.values():
            total = total + phase.usage
        return total

    def render(self, wall_seconds: float | None = None) -> str:
        """A breakdown a human reads without opening a tracing UI."""
        wall = wall_seconds if wall_seconds is not None else time.monotonic() - self.started
        measured = self.measured_seconds
        lines = [
            f"phase breakdown ({wall:.0f}s wall, {measured:.0f}s measured)",
            f"  {'phase':<12} {'calls':>5} {'seconds':>8} {'share':>6} {'mean':>7}  tokens",
        ]
        for name in PHASES:
            phase = self.phases.get(name)
            if phase is None or not phase.calls:
                continue
            share = phase.seconds / wall if wall else 0.0
            usage = phase.usage
            detail = f"in {usage.input_tokens:,} out {usage.output_tokens:,}"
            if usage.cache_read_input_tokens:
                detail += f" cached {usage.cache_read_input_tokens:,}"
            lines.append(
                f"  {name:<12} {phase.calls:>5} {phase.seconds:>8.1f} "
                f"{share:>5.0%} {phase.mean_seconds:>7.1f}  {detail}"
            )

        unmeasured = wall - measured
        if unmeasured > 1.0:
            # Named rather than hidden: a large gap means the instrumentation is missing a
            # phase, which is worth knowing before drawing a conclusion from the rest.
            lines.append(
                f"  {'unmeasured':<12} {'':>5} {unmeasured:>8.1f} {unmeasured / wall:>5.0%}"
            )

        total = self.total_usage
        if total.cache_read_input_tokens:
            lines.append(f"  cache hit rate: {total.cache_hit_rate:.0%}")
        else:
            lines.append("  cache hit rate: 0% — prompt caching is not enabled")
        return "\n".join(lines)
