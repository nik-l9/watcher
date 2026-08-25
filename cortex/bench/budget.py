"""A hard spending ceiling for benchmark runs.

The rule is a maximum spend, not a target, so it is enforced in code rather than
estimated in advance. An estimate is how a budget gets exceeded: per-task cost varies by
an order of magnitude across benchmarks — a text-to-SQL task is one call with a schema in
the prompt, a multi-step data-analysis task is a dozen calls over real data — and a
projection built from the cheap ones runs out of money halfway through the expensive ones.

So the meter is authoritative. Every call's usage is recorded as it happens, and the
runner asks permission before starting each task. When the ceiling is reached the run
**stops and says so**, reporting the subset it actually completed. A partial run with
`n` disclosed is a real result; a run that silently spent triple its budget is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cortex.agents.llm import Usage

#: USD per million tokens, input and output, by model id.
#:
#: Hard-coded deliberately rather than fetched: a benchmark's cost has to be computable
#: offline and reproducible from the record, and a price that changes under us would make
#: two runs incomparable. Stale entries are safer than absent ones — an unknown model
#: falls back to the most expensive rate below, so a typo overestimates rather than
#: quietly undercounting.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

_FALLBACK = (10.00, 50.00)


def cost_usd(model: str, usage: Usage) -> float:
    """What one call cost, in dollars."""
    # Provider prefixes such as "anthropic:" or "anthropic/" appear depending on which
    # layer reports the model, and an unrecognised key would silently price at the
    # fallback rate.
    key = model.split(":")[-1].split("/")[-1]
    input_rate, output_rate = PRICING.get(key, _FALLBACK)
    return (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000


class BudgetExhausted(Exception):
    """The ceiling was reached. Carries what was completed, so the run can report it."""


@dataclass(slots=True)
class Budget:
    """A spend meter with a hard ceiling.

    `reserve` is the headroom kept back for the last task. Without it the meter approves a
    task while $0.02 remains, that task costs $0.60, and the ceiling is passed after the
    fact — which is exactly the failure mode a hard rule is meant to prevent. The check is
    therefore "is there room for a task of the size we have been seeing", not "is there
    any money left".
    """

    ceiling_usd: float
    reserve_usd: float = 0.50

    spent_usd: float = 0.0
    calls: int = 0
    tasks_completed: int = 0
    by_model: dict[str, float] = field(default_factory=dict)

    def record(self, model: str, usage: Usage) -> float:
        """Charge one call to the meter and return its cost."""
        amount = cost_usd(model, usage)
        self.spent_usd += amount
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0.0) + amount
        return amount

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd)

    @property
    def average_task_usd(self) -> float:
        """Observed cost per completed task, which is what the next task will cost.

        Before any task completes there is nothing to average, so the reserve stands in.
        """
        if not self.tasks_completed:
            return self.reserve_usd
        return self.spent_usd / self.tasks_completed

    def may_start_task(self) -> bool:
        """Whether another task fits, judged on what tasks have actually cost."""
        needed = max(self.average_task_usd, self.reserve_usd)
        return self.remaining_usd >= needed

    def require_task(self) -> None:
        if not self.may_start_task():
            raise BudgetExhausted(
                f"stopping: ${self.spent_usd:.2f} of ${self.ceiling_usd:.2f} spent over "
                f"{self.tasks_completed} task(s); the next is expected to cost about "
                f"${self.average_task_usd:.2f} and would risk the ceiling"
            )

    def task_finished(self) -> None:
        self.tasks_completed += 1

    def summary(self) -> str:
        lines = [
            f"spend: ${self.spent_usd:.2f} of ${self.ceiling_usd:.2f} ceiling "
            f"({self.calls} calls, {self.tasks_completed} tasks)"
        ]
        if self.tasks_completed:
            lines.append(f"  ${self.average_task_usd:.3f} per task")
        for model, amount in sorted(self.by_model.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {model:<28} ${amount:.2f}")
        return "\n".join(lines)
