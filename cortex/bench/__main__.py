"""`python -m cortex.bench` — run an external benchmark under a hard spending ceiling.

Every score printed here is a **component** score. No public benchmark tests the product's
own claim, so these measure the analyst's reasoning over unfamiliar data and its SQL, and
they are never to be presented as a Cortex product score. The product score remains the
labelled GTM suite plus judgement on real data.

The ceiling is enforced rather than estimated: the runner asks the meter before each task
and stops when the next one would risk it, reporting the subset it completed. A partial run
with the denominator printed is a result; silently spending triple the budget is not.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from cortex.agents.anthropic_llm import DEFAULT_EFFORT, DEFAULT_MODEL
from cortex.bench.budget import Budget
from cortex.bench.harness import run_benchmark
from cortex.runtime.resources import open_resources

#: Default ceiling for one benchmark invocation.
#:
#: Deliberately per-benchmark rather than global. A single shared budget means whichever
#: benchmark runs first consumes all of it, and the last one silently reports n=0.
DEFAULT_CEILING_USD = 8.0


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cortex.bench", description=__doc__)
    parser.add_argument(
        "benchmark",
        choices=["dabstep", "bird"],
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--data",
        default="/tmp/dabstep",
        help="Directory holding the benchmark's data files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum tasks to attempt. The budget may stop the run sooner.",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=DEFAULT_CEILING_USD,
        help=f"Hard spending ceiling in USD for this run (default {DEFAULT_CEILING_USD}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260730,
        help="Subset seed. Fixed by default so two runs compare like with like.",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    if args.benchmark == "bird" and args.data == "/tmp/dabstep":
        # Each benchmark has its own data layout, so a shared default would silently
        # point one at the other's files.
        args.data = "/tmp/bird"

    workspace = Path(args.data)
    context = workspace / "context"
    if context.is_dir():
        # The downloader puts data files in a `context/` subdirectory, which is also what
        # the SQL tool should treat as the workspace.
        workspace = context
    if not workspace.is_dir():
        sys.stderr.write(
            f"No benchmark data at {workspace}. Fetch it first — see docs/benchmarks.md.\n"
        )
        return 2

    budget = Budget(ceiling_usd=args.ceiling)

    async with open_resources() as resources:
        if args.benchmark == "dabstep":
            from cortex.bench.dabstep import DABstepAdapter

            adapter = DABstepAdapter(resources, workspace)
        elif args.benchmark == "bird":
            from cortex.bench.bird import BirdAdapter

            adapter = BirdAdapter(resources, workspace)
        else:  # pragma: no cover - argparse restricts this
            raise SystemExit(f"unknown benchmark {args.benchmark}")

        print(
            f"Running {adapter.name} on {DEFAULT_MODEL} (effort {DEFAULT_EFFORT}), "
            f"ceiling ${args.ceiling:.2f}",
            file=sys.stderr,
        )
        run = await run_benchmark(
            adapter,
            model=DEFAULT_MODEL,
            budget=budget,
            limit=args.limit,
            seed=args.seed,
        )

    sys.stdout.write(run.render())
    sys.stdout.write("\n" + budget.summary() + "\n")
    # Exit zero regardless of score: a benchmark result is a measurement, not a build
    # gate. Only an inability to run one is a failure.
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
