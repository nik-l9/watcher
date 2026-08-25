"""Running an external benchmark against Cortex, honestly.

Our own eval is three scenarios scored by code I wrote, and it produced four wrong
numbers in one afternoon. External benchmarks exist here as the yardstick nobody on this
side can bend. What they measure is *components* — reasoning over data, SQL authoring,
tool-use reliability — never the product's own claim, which no public benchmark tests.

Three rules are built into the harness rather than left to whoever writes the report:

  - **The subset is always stated.** A benchmark run under a spending ceiling is a
    sample. `n=50 of 450` is a result; `16%` with the denominator omitted is a
    misrepresentation, and the difference is one line of formatting.
  - **The subset is deterministic.** Tasks are chosen by seeded shuffle, so two runs
    compare like with like and a score change means the agent changed. Picking "the
    first 50" instead would silently sort by whatever order the dataset ships in, which
    is often by difficulty.
  - **Stopping early is a result, not a failure.** When the budget is exhausted the run
    reports what it completed. Silently spending triple the ceiling is the only real
    failure available here.

An adapter supplies the tasks and grades an answer. It never grades by asking a model
whether the answer looks right: every benchmark worth running ships an objective key,
and substituting a judge for it would reintroduce exactly the softness these runs are
meant to remove.
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from cortex.agents.llm import Usage
from cortex.bench.budget import Budget, BudgetExhausted


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark item."""

    task_id: str
    question: str
    #: Whatever the adapter needs to set up and grade: a database path, a CSV directory,
    #: the expected answer. Opaque to the harness on purpose — a harness that understood
    #: task payloads would need changing for every new benchmark.
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TaskResult:
    task_id: str
    correct: bool
    answer: str = ""
    expected: str = ""
    error: str | None = None
    #: The task could not be graded through no fault of the agent — a corrupt database, a
    #: gold query that does not run against the data we have. Excluded from the
    #: denominator rather than counted as a failure.
    #:
    #: The distinction matters and I got it wrong first: "errors count against the score"
    #: is right for an agent that crashed and wrong for a harness that cannot grade. A
    #: contaminated benchmark that reports a single number understates the agent by an
    #: unknown amount, which is worse than reporting a smaller clean sample.
    unscoreable: bool = False
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    cost_usd: float = 0.0


class Adapter(Protocol):
    """What a benchmark must provide.

    Deliberately small. Everything benchmark-specific — obtaining the data, giving the
    agent the right tools, comparing an answer to a key — lives behind these two methods,
    so adding a benchmark never touches the runner.
    """

    name: str

    def load(self) -> Sequence[Task]:
        """Every task in the benchmark, before subsetting."""

    async def solve(self, task: Task) -> TaskResult:
        """Run one task and grade it against the benchmark's own key."""


@dataclass(slots=True)
class BenchmarkRun:
    benchmark: str
    model: str
    total_available: int
    results: list[TaskResult] = field(default_factory=list)
    stopped_early: str | None = None
    seed: int = 0

    @property
    def attempted(self) -> int:
        """Tasks that could actually be graded."""
        return sum(1 for r in self.results if not r.unscoreable)

    @property
    def run(self) -> int:
        """Tasks the agent was given, including ones the harness could not grade."""
        return len(self.results)

    @property
    def unscoreable(self) -> int:
        return sum(1 for r in self.results if r.unscoreable)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.error and not r.unscoreable)

    @property
    def accuracy(self) -> float:
        """Solved over *attempted*.

        Errors count against it. A task that crashed the agent is a task the agent did
        not solve, and excluding failures would let a fragile agent post a strong score
        by falling over on everything hard.
        """
        return self.solved / self.attempted if self.attempted else 0.0

    @property
    def total_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def render(self) -> str:
        lines = [
            "",
            f"{self.benchmark} — {self.model}",
            "=" * 72,
            # The denominator is not optional. This is the line that keeps a subset score
            # from being read as a full-benchmark score.
            f"score: {self.solved}/{self.attempted} = {self.accuracy:.1%}   "
            f"(subset of {self.total_available} available, seed {self.seed})",
            f"cost:  ${self.total_usd:.2f}   errors: {self.errored}",
        ]
        if self.results:
            times = [r.seconds for r in self.results]
            lines.append(f"time:  median {statistics.median(times):.0f}s   max {max(times):.0f}s")
        if self.unscoreable:
            # Named prominently. A benchmark the harness cannot grade is a harness
            # problem, and burying it would let a depressed score read as an agent
            # weakness.
            lines.append(
                f"note:  {self.unscoreable} of {self.run} task(s) were UNSCOREABLE and "
                f"excluded — the harness could not grade them (corrupt data, or a gold "
                f"query that does not run against the data available)"
            )
        if self.stopped_early:
            lines += ["", f"stopped early: {self.stopped_early}"]

        failures = [r for r in self.results if not r.correct and not r.unscoreable]
        if failures:
            lines += ["", f"first {min(5, len(failures))} of {len(failures)} not solved:"]
            for result in failures[:5]:
                detail = result.error or f"answered {result.answer[:60]!r}"
                lines.append(f"  {result.task_id}: {detail}")
                if not result.error and result.expected:
                    lines.append(f"    expected {result.expected[:60]!r}")
        return "\n".join(lines) + "\n"


async def run_benchmark(
    adapter: Adapter,
    *,
    model: str,
    budget: Budget,
    limit: int | None = None,
    seed: int = 20260730,
) -> BenchmarkRun:
    """Run up to `limit` tasks, stopping when the budget will not cover another.

    The seed is fixed by default so a subset is reproducible across runs. Changing it
    changes which tasks are sampled, which is occasionally what you want and never what
    you want silently — hence it is recorded on the result.
    """
    tasks = list(adapter.load())
    run = BenchmarkRun(benchmark=adapter.name, model=model, total_available=len(tasks), seed=seed)

    # Seeded shuffle rather than a slice: datasets frequently ship in difficulty or
    # source order, and taking a prefix would sample one end of it.
    random.Random(seed).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]

    for index, task in enumerate(tasks, start=1):
        # Progress to stderr, leaving stdout as the scorecard alone so it stays pipeable.
        # A fifty-task run takes half an hour, and a run that prints nothing until the end
        # is indistinguishable from a hung one -- the same lesson the eval harness learned
        # when an unbounded provider call went unnoticed for minutes.
        _progress(f"[{index}/{len(tasks)}] {task.task_id}: solving")
        try:
            budget.require_task()
        except BudgetExhausted as exc:
            run.stopped_early = str(exc)
            break

        started = time.monotonic()
        try:
            result = await adapter.solve(task)
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the run
            # Recorded as an unsolved task rather than raised: a benchmark that aborts on
            # its first awkward item reports nothing at all, and a crash *is* a failure to
            # solve, so it belongs in the denominator.
            result = TaskResult(
                task_id=task.task_id, correct=False, error=f"{type(exc).__name__}: {exc}"
            )

        result.seconds = time.monotonic() - started
        result.cost_usd = budget.record(model, result.usage)
        budget.task_finished()
        run.results.append(result)

        verdict = "solved" if result.correct else (result.error or "not solved")
        _progress(
            f"[{index}/{len(tasks)}] {task.task_id}: {verdict} "
            f"in {result.seconds:.0f}s (${budget.spent_usd:.2f} of "
            f"${budget.ceiling_usd:.2f}, {run.solved}/{run.attempted} so far)"
        )

    return run


def _progress(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
