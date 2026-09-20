"""Load extracted series off disk into `RealSeries`.

Separate from `generated.py` on purpose. The generator is arithmetic over series and is
tested against series the tests construct; this module is the only part that knows where an
extract lives or what shape a connector's dump has. Keeping them apart is what lets the
whole generator be tested without any real business data, which must never enter this
repository.

The extract directory is named by `WATCHER_DATA` or passed explicitly. There is no default
pointing at anyone's machine.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from cortex.eval.generated import RealSeries


def extract_root(explicit: str | Path | None = None) -> Path:
    root = Path(explicit) if explicit else Path(os.environ.get("WATCHER_DATA", ""))
    if not explicit and not os.environ.get("WATCHER_DATA"):
        raise RuntimeError("set WATCHER_DATA to the extract directory, or pass a path")
    if not root.is_dir():
        raise RuntimeError(f"{root} is not a directory")
    return root


def load_posthog(root: str | Path | None = None) -> tuple[RealSeries, ...]:
    """Every PostHog daily series in the extract that has at least one observation.

    A dump's `series` entries are `{bucket, value}`; buckets are ISO timestamps at day
    granularity. Days are sorted and de-duplicated here rather than trusted, because an
    extractor that windowed its requests can emit a boundary day twice.
    """
    base = extract_root(root) / "posthog"
    if not base.is_dir():
        return ()
    out: list[RealSeries] = []
    for project_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for path in sorted(project_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("interval") != "day":
                continue
            by_day: dict[date, int] = {}
            for bucket in payload.get("series") or ():
                stamp = bucket.get("bucket")
                value = bucket.get("value")
                if not stamp or value is None:
                    continue
                by_day[date.fromisoformat(stamp[:10])] = int(value)
            if not by_day:
                continue
            out.append(
                RealSeries(
                    source="posthog",
                    project=project_dir.name,
                    event=payload.get("event") or path.stem,
                    days=tuple(sorted(by_day.items())),
                )
            )
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    """Materialise the labelled set from an extract, for calibration to fit against.

    **The output carries real business data** -- event names and daily counts -- so it is
    written only where the caller names, never to a default inside the repository, and the
    repository ignores nothing on its behalf because nothing of it belongs here.
    """
    import argparse
    import json as _json

    from cortex.eval.generated import balanced, class_counts, generate

    parser = argparse.ArgumentParser(
        # `watcher questions`, not `cortex-questions`: this repository ships one command with
        # subcommands, and a founder who runs it and is shown a name they did not type has
        # been handed a word that means nothing to them. See `TestNothingUserFacingSaysCortex`.
        prog="watcher questions",
        description="Generate labelled questions whose answers are computed from real series.",
    )
    parser.add_argument("--data", help="extract directory; defaults to $WATCHER_DATA")
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="cap each premise verdict at this many cases (default: the scarcest class)",
    )
    parser.add_argument("--seed", type=int, default=0, help="makes the selection reproducible")
    parser.add_argument("--out", help="write the set as JSONL to this path (outside the repo)")
    args = parser.parse_args(argv)

    series = load_posthog(args.data)
    if not series:
        print("no daily series found in the extract", flush=True)
        return 1
    cases = generate(series)
    chosen = balanced(cases, per_class=args.per_class, seed=args.seed)

    print(f"{len(series)} series -> {len(cases)} cases, {class_counts(cases)}")
    print(f"selected {len(chosen)}, {class_counts(chosen)}")
    if args.out:
        with open(args.out, "w") as handle:
            for case in chosen:
                handle.write(
                    _json.dumps(
                        {
                            "name": case.name,
                            "question": case.question,
                            "premise_verdict": case.premise_verdict,
                            "basis": case.basis,
                            "event": case.series.event,
                            "project": case.series.project,
                            "required_signals": case.required_signals,
                            "refutation_signals": case.refutation_signals,
                        }
                    )
                    + "\n"
                )
        print(f"wrote {args.out}")
    return 0
