"""The benchmark harness.

What matters here is honesty rather than throughput: a subset score that omits its
denominator, a run that quietly exceeds its ceiling, or an accuracy figure that excludes
the tasks which crashed would each turn a benchmark from a check into a decoration.
"""

from __future__ import annotations

import pytest

from cortex.agents.llm import Usage
from cortex.bench.budget import Budget
from cortex.bench.harness import Task, TaskResult, run_benchmark


class _Adapter:
    """A benchmark whose answers are fully determined by the task id."""

    name = "test-bench"

    def __init__(self, count: int = 100, *, raise_on: set[str] | None = None) -> None:
        self.count = count
        self.raise_on = raise_on or set()
        self.solved_ids: list[str] = []

    def load(self) -> list[Task]:
        return [Task(task_id=f"t{i}", question=f"q{i}") for i in range(self.count)]

    async def solve(self, task: Task) -> TaskResult:
        self.solved_ids.append(task.task_id)
        if task.task_id in self.raise_on:
            raise RuntimeError("adapter failed")
        n = int(task.task_id[1:])
        return TaskResult(
            task_id=task.task_id,
            correct=n % 2 == 0,
            answer=str(n),
            expected=str(n if n % 2 == 0 else n + 1),
            usage=Usage(1_000, 100),
        )


def _budget(ceiling: float = 100.0) -> Budget:
    return Budget(ceiling_usd=ceiling, reserve_usd=0.001)


class TestSubsetting:
    async def test_the_subset_size_is_respected(self) -> None:
        run = await run_benchmark(
            _Adapter(100), model="claude-sonnet-5", budget=_budget(), limit=10
        )
        assert run.attempted == 10
        assert run.total_available == 100

    async def test_the_same_seed_selects_the_same_tasks(self) -> None:
        """Two runs must compare like with like, or a score change cannot be attributed
        to the agent."""
        first, second = _Adapter(100), _Adapter(100)
        await run_benchmark(first, model="m", budget=_budget(), limit=15, seed=7)
        await run_benchmark(second, model="m", budget=_budget(), limit=15, seed=7)
        assert first.solved_ids == second.solved_ids

    async def test_a_different_seed_selects_different_tasks(self) -> None:
        first, second = _Adapter(100), _Adapter(100)
        await run_benchmark(first, model="m", budget=_budget(), limit=15, seed=1)
        await run_benchmark(second, model="m", budget=_budget(), limit=15, seed=2)
        assert first.solved_ids != second.solved_ids

    async def test_tasks_are_shuffled_not_taken_as_a_prefix(self) -> None:
        """Datasets often ship in difficulty or source order, so a prefix samples one end
        of the distribution and reports it as representative."""
        adapter = _Adapter(100)
        await run_benchmark(adapter, model="m", budget=_budget(), limit=20)
        assert adapter.solved_ids != [f"t{i}" for i in range(20)]

    async def test_the_denominator_is_always_printed(self) -> None:
        """A subset presented without its denominator reads as a full-benchmark score."""
        run = await run_benchmark(_Adapter(450), model="claude-sonnet-5", budget=_budget(), limit=5)
        rendered = run.render()
        assert "subset of 450 available" in rendered
        assert "seed" in rendered


class TestBudgetIsRespected:
    async def test_the_run_stops_when_the_ceiling_is_reached(self) -> None:
        # 1,000 input + 100 output on Sonnet 5 is $0.0045 per task, so this affords a
        # handful and no more.
        run = await run_benchmark(
            _Adapter(500), model="claude-sonnet-5", budget=Budget(ceiling_usd=0.05), limit=500
        )
        assert run.attempted < 500
        assert run.stopped_early is not None
        assert "ceiling" in run.stopped_early or "spent" in run.stopped_early

    async def test_stopping_early_is_reported_not_hidden(self) -> None:
        run = await run_benchmark(
            _Adapter(500), model="claude-sonnet-5", budget=Budget(ceiling_usd=0.05), limit=500
        )
        assert "stopped early" in run.render()

    async def test_spend_stays_under_the_ceiling(self) -> None:
        """The rule is a maximum, so exceeding it and reporting it afterwards is still a
        breach."""
        budget = Budget(ceiling_usd=0.10)
        await run_benchmark(_Adapter(500), model="claude-sonnet-5", budget=budget, limit=500)
        assert budget.spent_usd <= budget.ceiling_usd


class TestFailuresCountAgainstTheScore:
    async def test_an_adapter_crash_becomes_an_unsolved_task(self) -> None:
        """A benchmark that aborted on its first awkward item would report nothing."""
        run = await run_benchmark(
            _Adapter(20, raise_on={"t3", "t11"}),
            model="m",
            budget=_budget(),
            limit=20,
        )
        assert run.attempted == 20
        assert run.errored == 2
        assert all(not r.correct for r in run.results if r.error)

    async def test_errors_are_in_the_denominator(self) -> None:
        """Excluding failures would let a fragile agent post a strong score by falling
        over on everything hard."""
        every = {f"t{i}" for i in range(20)}
        run = await run_benchmark(
            _Adapter(20, raise_on=every), model="m", budget=_budget(), limit=20
        )
        assert run.attempted == 20
        assert run.accuracy == 0.0

    async def test_the_error_reason_survives(self) -> None:
        run = await run_benchmark(
            _Adapter(5, raise_on={"t1"}), model="m", budget=_budget(), limit=5
        )
        failed = next(r for r in run.results if r.error)
        assert "adapter failed" in failed.error


class TestScoring:
    async def test_accuracy_is_solved_over_attempted(self) -> None:
        run = await run_benchmark(_Adapter(100), model="m", budget=_budget(), limit=10)
        assert run.accuracy == pytest.approx(run.solved / run.attempted)

    async def test_an_empty_run_does_not_divide_by_zero(self) -> None:
        run = await run_benchmark(_Adapter(0), model="m", budget=_budget(), limit=10)
        assert run.attempted == 0
        assert run.accuracy == 0.0
        assert run.render()


class TestUnscoreableTasksAreExcluded:
    """A task the harness cannot grade is a harness fault, not an agent failure.

    BIRD's DuckDB conversion silently dropped tables named `order` and `trans` — both
    reserved words — leaving 17% of the dev set with gold queries that do not run. Counting
    those as failures understated the agent by an unknown amount, which is worse than
    reporting a smaller clean sample.
    """

    class _MixedAdapter:
        name = "mixed"

        def load(self) -> list[Task]:
            return [Task(task_id=f"t{i}", question="q") for i in range(10)]

        async def solve(self, task: Task) -> TaskResult:
            n = int(task.task_id[1:])
            if n < 4:
                # Ungradeable: the key itself is broken.
                return TaskResult(
                    task_id=task.task_id,
                    correct=False,
                    unscoreable=True,
                    expected="GOLD QUERY FAILED: no such table",
                    usage=Usage(100, 10),
                )
            return TaskResult(task_id=task.task_id, correct=n % 2 == 0, usage=Usage(100, 10))

    async def test_they_leave_the_denominator(self) -> None:
        run = await run_benchmark(self._MixedAdapter(), model="m", budget=_budget(), limit=10)
        assert run.run == 10, "every task was still attempted"
        assert run.unscoreable == 4
        assert run.attempted == 6, "only gradeable tasks count"
        assert run.accuracy == pytest.approx(run.solved / 6)

    async def test_they_are_named_prominently(self) -> None:
        """Burying it would let a depressed score read as an agent weakness."""
        run = await run_benchmark(self._MixedAdapter(), model="m", budget=_budget(), limit=10)
        rendered = run.render()
        assert "UNSCOREABLE" in rendered
        assert "4 of 10" in rendered

    async def test_they_are_not_listed_as_failures(self) -> None:
        run = await run_benchmark(self._MixedAdapter(), model="m", budget=_budget(), limit=10)
        assert "no such table" not in run.render()

    async def test_an_agent_error_still_counts_against_the_score(self) -> None:
        """The distinction must not become an excuse: a crash is still a failure."""
        run = await run_benchmark(
            _Adapter(6, raise_on={"t1", "t2"}), model="m", budget=_budget(), limit=6
        )
        assert run.unscoreable == 0
        assert run.attempted == 6
        assert run.errored == 2
