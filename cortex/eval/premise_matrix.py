"""What the report *said* the premise was, against what the data says it is.

**Why this is not the scorer.** `Scorer` grades an investigation: did it find the cause, is
every claim grounded, did it use the calls that separate cause from decoy. Generated cases
have no labelled cause -- they are built from real series nobody annotated -- so most of
that machinery has nothing to grade them with, and `_premise_accuracy` only runs on the
false-premise branch. Read through the scorer, a generated set would produce numbers that
look like measurements and are not.

What a generated case *does* carry is a computed premise verdict, for all four values of the
enum. So this compares one field against one label and reports the confusion matrix. That is
a smaller claim than a scorecard and a true one.

**Why the matrix rather than an accuracy figure.** The measured instability this set exists
to address is a three-way verdict re-rolling on a fact the analyst finds every time. An
accuracy number cannot distinguish "says `holds` when it should say `false`" -- delivering a
confident wrong answer -- from "says `unverifiable` when it should say `holds`", which is
over-abstention and merely unhelpful. Those call for opposite fixes, and a conformal
threshold fitted without separating them would trade one for the other blind.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

#: Every value a report may state, in the order the matrix prints them.
VERDICTS = ("holds", "false", "unverifiable", "none_asserted")


@dataclass(frozen=True, slots=True)
class PremiseOutcome:
    """One attempt: what the case was labelled, and what the report stated."""

    scenario: str
    expected: str
    #: None when no report was delivered -- an error, or a draft the gate refused. Kept
    #: distinct from a wrong verdict: a run that could not answer is a different problem
    #: from one that answered wrongly, and averaging them hides an outage.
    stated: str | None
    #: Verdicts that are also defensible on this case's evidence. See
    #: `generated.LabelledQuestion.also_acceptable`; empty for every hand-written scenario.
    also_acceptable: tuple[str, ...] = ()
    #: Which of the three answers a correct report would have given differently. Empty for a
    #: correct report, and for a capture written before the verdict was decomposed.
    diverged_on: tuple[str, ...] = ()


def premise_of(report_json: str | None) -> str | None:
    """The premise verdict a captured report states, or None if it has none."""
    if not report_json:
        return None
    try:
        report = json.loads(report_json)
    except (TypeError, ValueError):
        return None
    stated = report.get("premise")
    return stated if stated in VERDICTS else None


def expected_of(label: str | dict) -> tuple[str, tuple[str, ...]]:
    """The verdict a label requires and the ones it also accepts.

    Labels are written as a plain string or as `{expected, also_acceptable}`. Both are read,
    because captures written before the second form existed are still worth re-scoring.
    """
    if isinstance(label, str):
        return label, ()
    return label["expected"], tuple(label.get("also_acceptable") or ())


def outcomes_for(
    labels: dict[str, str | dict], outcomes: Iterable[object]
) -> tuple[PremiseOutcome, ...]:
    """Pair each attempt with the label of the case it ran.

    Attempts whose scenario is not in `labels` are dropped rather than guessed at: a run may
    mix generated cases with hand-written ones, and the hand-written suite has no computed
    premise label to compare against.
    """
    out: list[PremiseOutcome] = []
    for outcome in outcomes:
        name = getattr(outcome, "scenario", None)
        if name not in labels:
            continue
        expected, acceptable = expected_of(labels[name])
        raw = getattr(outcome, "report_json", None)
        try:
            report = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            report = {}
        out.append(
            PremiseOutcome(
                scenario=name,
                expected=expected,
                also_acceptable=acceptable,
                stated=premise_of(raw),
                diverged_on=diverged(expected, report) if isinstance(report, dict) else (),
            )
        )
    return tuple(out)


#: Confusions that cost a reader nothing, as `(expected, stated)`.
#:
#: Observed rather than assumed: the first live run answered "Which month had the highest
#: daily rate of X?" with `holds` where the label says `none_asserted`,
#: and named the right month. A wh-question asserts nothing, so the label is right by the
#: schema's own wording -- but reading "the premise holds" as "there is a well-defined
#: answer here" is a defensible reading of a question that makes no claim, and nobody is
#: misled either way.
#:
#: Kept explicit and small. Counting this the same as accepting a false premise would push a
#: threshold toward abstaining on ordinary questions to buy down an error that never hurt
#: anyone; folding it into "correct" would hide a real schema confusion. It is reported as
#: its own line.
BENIGN = frozenset({("none_asserted", "holds"), ("holds", "none_asserted")})


def is_benign(expected: str, stated: str | None) -> bool:
    return (expected, stated) in BENIGN


#: What each verdict implies about the three answers that now determine it. `None` means the
#: answer does not affect the verdict and cannot be wrong.
IMPLIED: dict[str, tuple[bool | None, bool | None, bool | None]] = {
    "none_asserted": (False, None, None),
    "unverifiable": (True, False, None),
    "false": (True, True, True),
    "holds": (True, True, False),
}

ANSWERS = ("premise_asserted", "premise_measured", "premise_contradicted")


def diverged(expected: str, report: dict) -> tuple[str, ...]:
    """Which of the three answers a correct report would have given differently.

    **The diagnostic the four-way label could not produce.** "Said `false`, wanted
    `unverifiable`" names a symptom; "answered `premise_measured` true where the evidence
    does not reach the window" names the step that went wrong, and that is the one a fix can
    be aimed at. Fifteen of sixteen reports in run 9 were wrong in exactly one answer, and
    the same one.

    Answers the verdict does not depend on are skipped rather than counted: once
    `premise_measured` is false, nothing turns on `premise_contradicted`, and reporting it as
    a disagreement would invent an error out of a field nobody read.
    """
    implied = IMPLIED.get(expected)
    if implied is None:
        return ()
    return tuple(
        name
        for name, want in zip(ANSWERS, implied, strict=True)
        if want is not None and name in report and bool(report[name]) is not want
    )


def confusion(outcomes: Iterable[PremiseOutcome]) -> Counter[tuple[str, str]]:
    """Counts keyed by `(expected, stated)`, with "none" for an undelivered report."""
    return Counter((o.expected, o.stated or "none") for o in outcomes)


def render(outcomes: tuple[PremiseOutcome, ...]) -> str:
    """The matrix, plus the two figures that mean opposite things.

    Printed rather than returned as numbers because the point is to be read: a 4x4 matrix
    shows which confusion is happening, and the two rates below it name the two that matter.
    """
    if not outcomes:
        return "\nNo generated cases in this run.\n"

    counts = confusion(outcomes)
    stated_values = [*VERDICTS, "none"]
    width = max(len(v) for v in stated_values) + 2

    lines = ["", "Premise verdict: expected (row) against stated (column)", ""]
    lines.append(" " * 15 + "".join(v.rjust(width) for v in stated_values))
    for expected in VERDICTS:
        row = [counts[(expected, stated)] for stated in stated_values]
        if not any(row):
            continue
        lines.append(expected.ljust(15) + "".join(str(n).rjust(width) for n in row))

    delivered = [o for o in outcomes if o.stated is not None]
    correct = sum(1 for o in delivered if o.stated == o.expected)
    # Counted apart from `correct`: the case was labelled one way and answered another, and
    # the evidence supports both. Folding it into either column would overstate something.
    defensible = sum(
        1 for o in delivered if o.stated != o.expected and o.stated in o.also_acceptable
    )
    benign = sum(1 for o in delivered if is_benign(o.expected, o.stated))
    # The two errors that are not interchangeable. A wrong confident verdict is a delivered
    # falsehood; an unnecessary abstention only costs an answer. A single accuracy figure
    # prices them the same, and a threshold fitted on it will trade one for the other.
    confident_wrong = sum(
        1
        for o in delivered
        if o.stated != o.expected
        and o.stated != "unverifiable"
        and o.stated not in o.also_acceptable
        and not is_benign(o.expected, o.stated)
    )
    # The one that puts a falsehood in front of a reader: the question asserted a movement
    # that did not happen and the report went along with it. Separated from the rest because
    # it is the only error this product exists to prevent -- every claim in such a report can
    # be individually grounded while the report answers a question nobody should have asked.
    accepted_false_premise = sum(
        1
        for o in delivered
        if o.expected == "false"
        and o.stated in ("holds", "none_asserted")
        and o.stated not in o.also_acceptable
    )
    # Answered a window the data does not reach. A grounded-looking answer about months that
    # were never collected is the other way to mislead.
    answered_the_unreachable = sum(
        1
        for o in delivered
        if o.expected == "unverifiable"
        and o.stated != "unverifiable"
        and o.stated not in o.also_acceptable
    )
    over_abstained = sum(
        1 for o in delivered if o.stated == "unverifiable" and o.expected != "unverifiable"
    )

    lines += [
        "",
        f"delivered      {len(delivered)}/{len(outcomes)}",
        f"correct        {correct}/{len(delivered)}"
        + (f" ({correct / len(delivered):.0%})" if delivered else ""),
        f"benign         {benign}  -- holds/none_asserted, a question that asserts nothing",
        f"defensible     {defensible}  -- the evidence supports the other reading too",
        "",
        f"accepted a false premise  {accepted_false_premise}  -- the error that misleads a reader",
        f"answered the unreachable  {answered_the_unreachable}  -- claimed a window with no data",
        f"confidently wrong (all)   {confident_wrong}",
        f"over-abstained            {over_abstained}  -- unverifiable where the data answers",
        "",
    ]

    steps: Counter[str] = Counter()
    for outcome in delivered:
        if outcome.stated != outcome.expected and outcome.stated not in outcome.also_acceptable:
            steps.update(outcome.diverged_on)
    if steps:
        # Which step failed, not only which verdict was wrong. Errors concentrated in one
        # answer are a defect with something to aim a fix at; errors spread evenly across the
        # three are noise, and the two call for opposite responses. A verdict-only matrix
        # cannot tell them apart -- fifteen of sixteen wrong reports in run 9 turned out to be
        # wrong in the same single answer, which is what made the fix obvious once it was
        # visible.
        lines.append("wrong answers by step")
        for name, count in steps.most_common():
            lines.append(f"  {name:22} {count}")
        lines.append("")
    return "\n".join(lines)
