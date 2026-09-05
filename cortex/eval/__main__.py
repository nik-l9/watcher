"""`python -m cortex.eval` — run the evaluation suite and print the scorecard.

Exits non-zero when a gating dimension fails or any hallucination is recorded, so
`make eval` can fail a build.

Default provider is the real Anthropic one, because the point of the suite is to
measure the shipped analyst. `--recorded` swaps in a scripted provider so the harness
itself can be exercised without spending tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from cortex.agents.anthropic_llm import DEFAULT_EFFORT, DEFAULT_MODEL
from cortex.agents.llm import LLM
from cortex.agents.provider import build_llm
from cortex.eval.fixtures import SCENARIOS, by_name
from cortex.eval.replay import FAILURES_DIRNAME
from cortex.eval.runner import EvalHarness
from cortex.runtime.resources import open_resources


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cortex.eval", description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Run only this scenario. Repeatable. Defaults to all.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Thinking depth. Worth sweeping: a cheaper setting that still scores "
        "well is a real cost saving.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the adversarial verifier. Faster, but stops scoring the second "
        "grounding mechanism — only useful when iterating on the loop itself.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Attempts per scenario. The same scenario has passed at 0.85 and then "
        "failed on unchanged code, so one attempt is a sample, not a measurement. "
        "Use more than one before concluding anything from a score difference.",
    )
    parser.add_argument(
        "--dump-dir",
        default=None,
        metavar="DIR",
        help="Write the delivered report of every failing attempt here, as JSON. A run "
        "rolls its tenants and evidence back, so without this a failure leaves only its "
        "one-line reason -- and 'missing required signal: 91c3e4a' does not say whether "
        "the analyst named the change some other way or missed it, which are opposite "
        "bugs. Failures only: a passing report is not what needs reading.",
    )
    parser.add_argument(
        "--recorded",
        action="store_true",
        help="Use a scripted provider instead of a real model. Exercises the harness "
        "without spending tokens; does not measure the analyst.",
    )
    parser.add_argument(
        "--variance",
        default=None,
        metavar="DIR",
        help="Measure how much each dimension varies across repeated attempts in a captured "
        "run, and the smallest difference a paired comparison could detect. Calls no provider. "
        "Needs a directory captured with --repeat 2 or more.",
    )
    parser.add_argument(
        "--no-sufficiency",
        action="store_true",
        help="Skip the sufficiency gate. On by default, matching both delivery paths -- an "
        "eval running a different pipeline from the one that ships measures something nobody "
        "uses.",
    )
    parser.add_argument(
        "--capture-to",
        default=None,
        metavar="DIR",
        help="Where to write a replayable bundle per attempt. Defaults to a timestamped "
        "directory under eval-runs/. The run's transaction is rolled back, so without a "
        "capture nothing about an attempt survives it and every later change to a scoring "
        "dimension costs another paid run.",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Do not write bundles. Only worth it for a throwaway run: the reports are gone "
        "afterwards and the run cannot be re-scored.",
    )
    parser.add_argument(
        "--rescore",
        default=None,
        metavar="DIR",
        help="Score bundles from a captured run, calling no provider. Accepts `latest`. Use this "
        "after changing a scoring dimension: the numbers are comparable with the original run "
        "because the same Scorer runs over the same rows.",
    )
    return parser.parse_args(argv)


#: Where bundles go when nobody says otherwise.
#:
#: Capturing is the default rather than an opt-in, because the one time it mattered it was
#: forgotten: a fifteen-attempt variance run was written to a session scratchpad, the scratchpad
#: was cleared, and the run could not be re-scored -- which is the entire purpose of capturing it.
#: The fix is not remembering a flag, it is not needing to.
CAPTURE_ROOT = Path("eval-runs")


def _capture_directory(args: argparse.Namespace) -> Path | None:
    """The directory to capture into, or None when explicitly disabled."""
    if args.no_capture:
        return None
    if args.capture_to:
        return Path(args.capture_to)
    # Timestamped rather than a fixed "latest", so a run cannot quietly overwrite the baseline
    # someone is about to compare against.
    return CAPTURE_ROOT / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _resolve_run(given: str) -> Path:
    """A capture directory, accepting `latest` for the most recent one under `eval-runs/`.

    Typing a timestamp is how a comparison gets pointed at the wrong run.
    """
    if given != "latest":
        return Path(given)
    runs = sorted((path for path in CAPTURE_ROOT.glob("*") if path.is_dir()), reverse=True)
    if not runs:
        raise SystemExit(f"no captured runs under {CAPTURE_ROOT}/")
    return runs[0]


def _provider(args: argparse.Namespace) -> LLM:
    if args.recorded:
        from cortex.agents.llm import RecordedLLM

        # Deliberately empty: the harness will report every scenario as errored,
        # which is the correct outcome for "no analyst was available" and proves the
        # error path rather than fabricating a passing score.
        return RecordedLLM()
    return build_llm(model=args.model, effort=args.effort)


def _investigator_factory(args: argparse.Namespace) -> Any:
    """Which loop investigates.

    A function rather than the class itself because the harness takes a factory: a scenario
    builds one investigator per attempt, and an alternative loop can be substituted here to
    compare it against this one over identical tools, evidence store, drafting call, gate and
    verifier. See ADR 0001 for why this project runs its own loop rather than a framework's.
    """
    del args
    from cortex.agents.investigator import Investigator

    return Investigator


async def _rescore(directory: Path) -> Any:
    """Re-score captured bundles, calling no provider.

    Replays each bundle's rows into a scratch tenant and runs the **unmodified** `Scorer`. Not a
    second scoring implementation over a file: a separate path would drift from the real one and
    then the numbers would not be comparable, which is the problem this exists to solve rather
    than a new one to introduce.
    """
    from cortex.eval.replay import load_bundle, replay_bundle
    from cortex.eval.runner import EvalRun, ScenarioOutcome
    from cortex.eval.scorer import Scorer

    bundles = sorted(directory.glob("*.json"))
    if not bundles:
        raise SystemExit(f"no bundles in {directory}")

    scorer = Scorer()
    run = EvalRun()
    async with open_resources() as resources:
        maker = async_sessionmaker(resources.engine, expire_on_commit=False)
        async with maker() as session:
            for path in bundles:
                bundle = load_bundle(path)
                if bundle.rejected_because:
                    # Reported as errored, not scored. The report in a rejected bundle is the
                    # draft that was refused; scoring it would put a number on an answer nobody
                    # received, and that number would look like a measurement.
                    run.outcomes.append(
                        ScenarioOutcome(
                            scenario=bundle.scenario,
                            card=None,
                            error=bundle.rejected_because,
                            attempt=bundle.attempt,
                        )
                    )
                    sys.stderr.write(f"{path.name}: rejected, not scored\n")
                    continue
                scenario = by_name(bundle.scenario)
                (
                    tenant,
                    investigation_id,
                    investigation,
                    gate_result,
                    verification,
                    applied,
                ) = await replay_bundle(session, bundle)
                card = await scorer.score(
                    session,
                    tenant,
                    investigation_id=investigation_id,
                    scenario=scenario,
                    investigation=investigation,
                    gate_result=gate_result,
                    verification=verification,
                    sufficiency=applied,
                )
                run.outcomes.append(
                    ScenarioOutcome(scenario=bundle.scenario, card=card, attempt=bundle.attempt)
                )
                sys.stderr.write(f"rescored {path.name}\n")
            # Rolled back for the same reason the run itself is: a replay must not leave
            # scratch tenants behind in whatever database it was pointed at.
            await session.rollback()
    return run


async def _main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    if args.variance:
        from cortex.eval.variance import measure_spread, render_spread

        async with open_resources() as resources:
            maker = async_sessionmaker(resources.engine, expire_on_commit=False)
            async with maker() as session:
                spread = await measure_spread(session, _resolve_run(args.variance))
                # Rolled back like every other path that replays bundles: measuring spread must
                # not leave scratch tenants behind.
                await session.rollback()
        sys.stdout.write(render_spread(spread))
        return 0

    if args.rescore:
        run = await _rescore(_resolve_run(args.rescore))
        sys.stdout.write(run.render())
        return 0 if run.passed else 1

    scenarios = tuple(by_name(name) for name in args.scenario) if args.scenario else SCENARIOS
    capture = _capture_directory(args)

    harness = EvalHarness(
        llm=_provider(args),
        verify=not args.no_verify,
        investigator_factory=_investigator_factory(args),
        capture_to=capture,
        sufficiency=not args.no_sufficiency,
    )

    async with open_resources() as resources:
        maker = async_sessionmaker(resources.engine, expire_on_commit=False)
        async with maker() as session:
            run = await harness.run(session, scenarios, repeat=args.repeat)
            # Rolled back: an eval run must not leave tenants and evidence behind in
            # whatever database it was pointed at.
            await session.rollback()

    sys.stdout.write(run.render())
    # On stderr, so stdout stays the scorecard alone and stays pipeable. Printed because a
    # capture nobody can find is a capture nobody uses -- and the path is what `--rescore` and
    # `--variance` take.
    if capture is not None:
        sys.stderr.write(
            f"captured {len(list(capture.glob('*.json')))} bundle(s) to {capture}\n"
            + _failure_note(capture)
            + "re-score with: python -m cortex.eval --rescore latest\n"
        )
    if args.dump_dir:
        for path in _dump_failures(run, Path(args.dump_dir)):
            sys.stderr.write(f"wrote {path}\n")
    return 0 if run.passed else 1


def _failure_note(capture: Path) -> str:
    """Point at captured failures, which are the artifacts most worth reading and least likely
    to be looked for.

    A run that errors prints the error inline, so the natural assumption is that the error was
    everything there was -- run 18's drafting failure left only its error string, and the
    question it raised was about the tool calls that came before it. Saying the record exists is
    what makes it get read.
    """
    failures = sorted((capture / FAILURES_DIRNAME).glob("*.json"))
    if not failures:
        return ""
    return (
        f"{len(failures)} failed attempt(s) recorded in {capture / FAILURES_DIRNAME} "
        "-- these carry the tool calls made before the failure, not a scoreable report\n"
    )


def _dump_failures(run: Any, directory: Path) -> list[Path]:
    """Write each failing attempt's delivered report, so it can be read afterwards."""
    written: list[Path] = []
    directory.mkdir(parents=True, exist_ok=True)
    for outcome in run.outcomes:
        if outcome.passed or not outcome.report_json:
            continue
        path = directory / f"{outcome.scenario}-attempt{outcome.attempt}.json"
        path.write_text(outcome.report_json)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
