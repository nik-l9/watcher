"""Retrieved-but-unread state: a bulk observation goes behind a handle, and the loop can
enumerate what it fetched and never read.

Our bug in one sentence (ADR 0005, decision 8): `posthog.list_events` was fetched, contained
the answer -- thirteen server-side event types whose `last_seen_at` all fell inside a
72-minute window on 2026-08-04, while browser autocapture kept flowing -- and was never read.
The analyst then explained the signup cessation with a pageview collapse that began seven days
later. The measured base rate for this class of failure is 13-25% for frontier models when the
answer *is* in context (ICLR 2025, Table 4a), so it is not a
prompt-quality problem to be argued away.

**The mechanism is RCAgent's OBSK** (CIKM 2024, §3 of arXiv:2310.16340): show the controller
only the head of an observation, keep the full thing
in a key-value store behind a hash id, and make analysing it a separate deliberate act. Their
ablation puts OBSK at G-Correctness 4.53 -> 5.22 and Invalid Rate 18.34 -> 7.93. Roy et al.
(FSE 2024, §2.3) supply the complementary negative: their retrieval tool "is stateless, and
does not take into consideration documents that have already been retrieved in prior steps",
so their agent re-pulled and re-ignored. An agent that cannot enumerate what it holds cannot
notice what it has ignored.

Two properties matter more than the mechanism, and both are testable:

  1. **The unread set is enumerable.** `ReadingLedger.unread()` is the point of the whole
     module. It is what makes an omission visible -- to the next turn, to the drafting call,
     and to the trace.
  2. **Reading is cheap and obvious.** One tool, one argument, no upstream request, no wall
     clock. If reading is expensive or obscure the model reasons from the head instead, and
     then this change has made things worse rather than better.

## What the head contains, and why it is not a head

Showing the first N rows of a series whose interesting fact is at the end is our original bug
in a new costume: it is *positional*, so ordering decides what the model sees. So what is
shown is not a prefix at all. Every bulk row collection is replaced, in place, by a digest
computed over **every** row:

  - **Non-row content is never elided.** Only lists longer than `BULK_ROWS` are replaced.
    Every scalar and every small object survives verbatim -- which is where the connectors'
    own resolutions live (`movement`, `blast_radius`, `series_ends_early`, `partial_buckets`,
    `total_available`). Those were the last four fixes; hiding one of them behind a handle
    would undo a shipped fix to ship this one.
  - **Per-field facts, order-independent.** Distinct values with counts, numeric min/max/sum
    and a count of zeros, and for a temporal field the earliest and latest value plus the
    day-buckets at both ends and the busiest one.
  - **Both ends of a time field, plus the mode -- not the top-k days.** "A group of things
    stopped together" shows up as a cluster at one *end* of a `last_seen_at` column. Ranking
    day-buckets by count would hide thirteen events among four busy days; naming the earliest
    day, the latest day and the largest always shows a cluster at either end. This is the one
    place the digest is designed against a specific incident, and it is still a mechanical
    rule rather than a judgement.
  - **One row verbatim, labelled as a shape.** The model needs the row's keys to formulate a
    follow-up query. It is explicitly *not* offered as a sample of the content.

## When it applies

Mechanically, on the payload's own size: a rendering over `BULK_CHARS` characters that
contains a list over `BULK_ROWS` long. No model judgement, nothing per-connector, and absent
by construction from every call that does not need it -- a small payload is rendered exactly
as it was before this module existed, and an investigation that never fetches a bulk result
never sees the read tool, the notice, or the reminder.

The threshold is deliberately high enough that a daily series is *not* summarised. A 60-day
GA4 series renders at ~5.4k characters and the analyst has to read it row by row to measure
anything; a PostHog event catalogue at 100 rows renders at ~12k and its useful content is
distributional. 8,000 characters separates those two in practice. The consequence worth
stating plainly: our eval's largest fixture payload is 5,390 characters, so **the eval suite
cannot measure this change at all** -- every scenario stays byte-identical. It fires on real
data only, which is also where the bug was found.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cortex.tools.executor import ExecutedTool

#: Rendered characters above which a payload is a candidate for a digest.
#:
#: ~2,000 tokens. Below this the whole payload is cheaper than the machinery for hiding it,
#: and the reasoning above explains why the line sits above a daily series rather than below.
BULK_CHARS = 8000

#: List length above which a list is a row collection rather than a field with a few values.
#:
#: 24 rather than 10, so a month of daily buckets, a week of hourly ones, or a funnel's steps
#: are never digested -- those are read row by row or they are not read at all.
BULK_ROWS = 24

#: Full observations returned per turn, however many the model asks for.
#:
#: A turn that asks for eight full payloads is not a deliberate read; it is the pile of
#: context that licensed the confident wrong answer in the first place -- the pile itself is
#: what licensed it. Three matches the middle of the tool-call width schedule.
#: Extra reads come back as a correctable error naming what was skipped, so nothing is lost --
#: the model asks again next turn.
MAX_READS_PER_STEP = 3

#: The read tool's flattened name. Not a registered capability: reading makes no upstream
#: request, needs no credential, mints no `Evidence` row and must not write one -- the row
#: already exists and re-recording it would put a second hash on the same observation. So the
#: loop intercepts this name before dispatch and the executor never sees it.
READ_TOOL = "cortex__read_observation"

#: Fields digested per row collection, and distinct values named per field.
_MAX_FIELDS = 16
_MAX_DISTINCT = 6
#: Characters of the one verbatim row shown for its shape.
_ROW_SHAPE_CHARS = 400
#: Longest single value rendered inside a digest line.
_VALUE_CHARS = 60
#: How deep the walk looks for row collections. Payloads are three levels deep at most.
_MAX_DEPTH = 6


def render_payload(payload: Any) -> str:
    """Render an observation body exactly as the loop has always rendered it.

    Shared with the investigator so a digest's claimed saving is measured against the string
    the model would otherwise have received, rather than against a differently-formatted
    approximation of it.

    `ensure_ascii=False` because this string is read by a model, not parsed by a machine.
    The default renders a customer called "Sch\u00f6nherr" as `Sch\\u00f6nherr` and an
    em-dash as `\\u2014`, and the model does not merely misread them -- it *copies the
    convention*, and writes escapes into its own prose. A real HubSpot run answered "the
    closed-won rate was 12.3% \\u2014 9 deals won", which is what a customer would have
    read in their report. No fixture caught it in 2,800 tests because every fixture is
    pure ASCII; only real data has umlauts and typographic dashes in it.
    """
    return json.dumps(payload, indent=2, sort_keys=True, default=str, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Digest:
    """A bulk payload as the model first sees it: every field described, no rows shown."""

    text: str
    #: Rows the digest covers and the model has not seen.
    rows: int
    #: Characters the full rendering would have cost.
    full_chars: int

    @property
    def chars(self) -> int:
        return len(self.text)


def digest_payload(payload: dict[str, Any]) -> Digest | None:
    """A digest of `payload`, or None when the payload should be shown in full.

    None on three separate grounds, and each of them is a case where hiding the rows would
    cost more than it saves:

      - the rendering is small enough to read whole;
      - it is large but contains no row collection, so there is nothing a digest can
        honestly replace -- a wall of prose stays a wall of prose;
      - the digest did not come out smaller, which happens for a list of long distinct
        strings. A summary that saves nothing has only added a hop.
    """
    full = render_payload(payload)
    if len(full) <= BULK_CHARS:
        return None
    reduced, rows = _reduce(payload, 0)
    if rows == 0:
        return None
    text = render_payload(reduced)
    if len(text) >= len(full):
        return None
    return Digest(text=text, rows=rows, full_chars=len(full))


def _reduce(value: Any, depth: int) -> tuple[Any, int]:
    """Replace every bulk list inside `value` with a digest, keeping the payload's shape.

    The shape is kept rather than flattened so the model can still see *where* the rows live:
    `{"events": {...digest...}}` reads as "the events key holds rows I have not seen", which
    is also the argument it needs for a follow-up query.
    """
    if depth > _MAX_DEPTH:
        return value, 0
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        rows = 0
        for key, item in value.items():
            reduced, n = _reduce(item, depth + 1)
            out[key] = reduced
            rows += n
        return out, rows
    if isinstance(value, list):
        if len(value) > BULK_ROWS:
            return _digest_rows(value), len(value)
        out_list: list[Any] = []
        rows = 0
        for item in value:
            reduced, n = _reduce(item, depth + 1)
            out_list.append(reduced)
            rows += n
        return out_list, rows
    return value, 0


def _digest_rows(rows: list[Any]) -> dict[str, Any]:
    """One row collection, described over all of its rows."""
    digest: dict[str, Any] = {
        "ROWS_NOT_SHOWN": len(rows),
        "digest_covers": (
            "every row, so nothing here depends on the order they came in; the rows "
            f"themselves are unread -- see {READ_TOOL}"
        ),
    }
    columns = _columns(rows)
    if columns is None:
        digest["values"] = _describe(rows, len(rows))
        return digest

    names = sorted(columns)
    digest["fields"] = {name: _describe(columns[name], len(rows)) for name in names[:_MAX_FIELDS]}
    if len(names) > _MAX_FIELDS:
        digest["fields_not_described"] = ", ".join(names[_MAX_FIELDS:])
    if isinstance(rows[0], dict):
        # The shape as a key-to-type map computed over *every* row, not the first row verbatim.
        #
        # Taking `rows[0]` broke this module's own order-independence promise -- shuffling the
        # rows changed the digest, which its own test caught. A verbatim row was also more than
        # was wanted: the stated purpose is that the model can see the keys and value shapes to
        # formulate a follow-up query, explicitly *not* a sample of the content. A type map is
        # order-independent by construction and answers the question better, because a field
        # that is a string in some rows and null in others says so instead of hiding behind
        # whichever row happened to arrive first.
        digest["row_shape"] = _shape(rows)
        digest["row_shape_note"] = (
            "field names and value types across all rows, for formulating a follow-up query "
            "-- not a sample of the data"
        )
    return digest


def _shape(rows: list[Any]) -> dict[str, str]:
    """Field name to the value types seen for it, across every row."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            seen.setdefault(str(key), set()).add("null" if value is None else type(value).__name__)
    return {name: "|".join(sorted(kinds)) for name, kinds in sorted(seen.items())}


def _columns(rows: list[Any]) -> dict[str, list[Any]] | None:
    """Rows as columns, or None when the rows are scalars.

    Positional lists become columns named `[0]`, `[1]` -- a warehouse result that returns
    arrays rather than objects still has columns, and describing them beats reporting only
    that there were N lists.
    """
    if all(isinstance(row, dict) for row in rows):
        columns: dict[str, list[Any]] = {}
        for row in rows:
            for key, value in row.items():
                columns.setdefault(str(key), []).append(value)
        return columns or None
    if all(isinstance(row, list) for row in rows):
        widths = {len(row) for row in rows}
        if len(widths) == 1 and widths.pop() <= _MAX_FIELDS:
            positional: dict[str, list[Any]] = {}
            for row in rows:
                for index, value in enumerate(row):
                    positional.setdefault(f"[{index}]", []).append(value)
            return positional or None
    return None


def _describe(values: list[Any], total: int) -> str:
    """One field, described over every value it has.

    Typed by what is actually in the column rather than by a schema, because a connector's
    payload is JSON and a column of numbers arrives as numbers, strings, or both.
    """
    present = [v for v in values if v is not None]
    missing = total - len(present)
    tail = f"; {missing} row(s) have no value" if missing else ""
    if not present:
        return f"no values in any of {total} row(s)"

    if all(isinstance(v, bool) for v in present):
        true = sum(1 for v in present if v)
        return f"true x{true}, false x{len(present) - true}" + tail
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in present):
        return _numeric(present) + tail
    if all(isinstance(v, str) for v in present):
        temporal = _temporal(present)
        if temporal is not None:
            return temporal + tail
    counts = Counter(_clip(str(v), _VALUE_CHARS) for v in present)
    return _counted(counts, len(present)) + tail


def _numeric(values: list[Any]) -> str:
    """Numbers: the range, the total, and how many are zero.

    Zeros are counted separately because "how many of these are zero" is the question behind
    most of the wrong answers this codebase has shipped -- a series that fell to zero and a
    series that was never collected look identical in a min/max.
    """
    zeros = sum(1 for v in values if v == 0)
    parts = [
        f"{len(values)} number(s)",
        f"min {_number(min(values))}",
        f"max {_number(max(values))}",
        f"sum {_number(sum(values))}",
    ]
    if zeros:
        parts.append(f"{zeros} are zero")
    return ", ".join(parts)


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _counted(counts: Counter[str], present: int) -> str:
    """Distinct values with their counts, or a count of distinct values plus an example.

    Ranked by count rather than by position, because which values are frequent does not depend
    on the order the rows arrived in and "the first three" would. An all-distinct column --
    an id, a name, a url -- says so and shows one value alphabetically, since ranking by count
    means nothing when every count is one.
    """
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(counts) <= _MAX_DISTINCT:
        return ", ".join(f"{value!r} x{count}" for value, count in ranked)
    if len(counts) == present:
        return f"{present} value(s), all distinct; e.g. {min(counts)!r}"
    commonest = ", ".join(f"{value!r} x{count}" for value, count in ranked[:3])
    return f"{present} value(s), {len(counts)} distinct; commonest {commonest}"


def _temporal(values: list[str]) -> str | None:
    """A date or timestamp column: both ends, their day-buckets, and the busiest day.

    None unless nearly every value parses, so a column of ordinary strings is never described
    as if it were a clock.

    **Why both ends and the mode rather than the top three days.** The fact that would have
    resolved our incident is thirteen event types whose last activity fell inside one
    72-minute window, in a payload of a hundred events whose other days were busier. Ranking
    buckets by count hides exactly that. The earliest day, the latest day and the largest
    always name a cluster at either end of the column, which is the shape "a group of things
    started or stopped together" takes.
    """
    days: list[str] = []
    for value in values:
        day = _day(value)
        if day is None:
            continue
        days.append(day)
    if len(days) < len(values) * 0.8:
        return None

    buckets = Counter(days)
    earliest_day, latest_day = min(buckets), max(buckets)
    busiest = max(buckets.items(), key=lambda kv: (kv[1], kv[0]))[0]
    named = []
    for day in (earliest_day, busiest, latest_day):
        if day not in named:
            named.append(day)
    listed = ", ".join(f"{day} x{buckets[day]}" for day in named)
    remainder = len(days) - sum(buckets[day] for day in named)
    more = (
        f" (+{remainder} on {len(buckets) - len(named)} other day(s))"
        if remainder and len(buckets) > len(named)
        else ""
    )
    return (
        f"{len(values)} timestamp(s) from {min(values)} to {max(values)}, "
        f"across {len(buckets)} day(s); earliest/busiest/latest day: {listed}{more}"
    )


def _day(value: str) -> str | None:
    """The calendar day of an ISO-8601-ish value, or None.

    Deliberately a prefix check rather than a parse: connectors return
    `2026-08-04T11:03:00Z`, `2026-08-04 11:03:00+00:00` and `2026-08-04`, and every one of
    those carries its day in the first ten characters. A full parse would add a dependency on
    every dialect of offset formatting for a substring we already have.
    """
    if len(value) < 10:
        return None
    day = value[:10]
    if day[4] != "-" or day[7] != "-":
        return None
    if not (day[:4].isdigit() and day[5:7].isdigit() and day[8:10].isdigit()):
        return None
    if len(value) > 10 and value[10] not in ("T", " "):
        return None
    return day


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True, slots=True)
class Unread:
    """One observation that was fetched and has not been read."""

    evidence_id: uuid.UUID
    source: str
    rows: int

    @property
    def line(self) -> str:
        return f"  - {self.evidence_id}  {self.source}  ({self.rows} unread row(s))"


class UnknownHandle(Exception):
    """The model asked to read something that is not behind a handle.

    Carries the sentence to feed back, because the useful answer differs by case: an
    observation shown in full needs no read, and an unrecognised id needs the list of ids
    that do exist.
    """


@dataclass(slots=True)
class _Held:
    executed: ExecutedTool
    digest: Digest
    read_at_step: int | None = None


@dataclass(slots=True)
class ReadingLedger:
    """What this investigation fetched, what it summarised, and what it has read.

    One per investigation, held by the loop rather than by the executor: this is a
    context-management decision belonging to whoever composes the prompt, and the executor's
    job -- validate, audit, hash, persist -- must stay identical whether or not a payload is
    later shown as a digest. Nothing here touches `Evidence` or `payload_hash`.
    """

    _held: dict[str, _Held] = field(default_factory=dict)
    #: Ids shown in full, so "read this" can answer "you already have every row of it"
    #: instead of "no such handle", which reads as a bug the model tries to work around.
    _whole: set[str] = field(default_factory=set)

    def offer(self, executed: ExecutedTool) -> Digest | None:
        """Register one observation; return its digest, or None to render it in full."""
        key = str(executed.evidence_id)
        digest = digest_payload(executed.payload)
        if digest is None:
            self._whole.add(key)
            return None
        self._held[key] = _Held(executed=executed, digest=digest)
        return digest

    def read(self, raw_id: Any, *, step: int) -> tuple[_Held, bool]:
        """Mark a handle read; return it and whether this was the first read of it.

        Raises `UnknownHandle` carrying the sentence to feed back. The first-read flag is
        returned rather than inferred from `read_at_step`, because two reads of the same
        handle in one turn would be indistinguishable by step number.
        """
        key = _normalise(raw_id)
        held = self._held.get(key)
        if held is None:
            if key in self._whole:
                raise UnknownHandle(
                    f"observation {key} was shown to you in full -- you already have every "
                    "row of it. Nothing further to read."
                )
            available = self.unread()
            if not available:
                raise UnknownHandle(
                    f"no observation is behind a handle under {raw_id!r}. Every observation "
                    "you have received was shown in full."
                )
            listed = "\n".join(item.line for item in available)
            raise UnknownHandle(
                f"{raw_id!r} is not one of the summarised observations. These are:\n{listed}"
            )
        first_read = held.read_at_step is None
        if first_read:
            held.read_at_step = step
        return held, first_read

    def unread(self) -> tuple[Unread, ...]:
        """Fetched, summarised, never read. The whole point of the module.

        In fetch order, because that is the order the analyst saw them and the order in which
        a forgotten observation is easiest to place.
        """
        return tuple(
            Unread(
                evidence_id=held.executed.evidence_id,
                source=f"{held.executed.tool_name}.{held.executed.capability}",
                rows=held.digest.rows,
            )
            for held in self._held.values()
            if held.read_at_step is None
        )

    @property
    def has_handles(self) -> bool:
        """Whether anything has ever gone behind a handle.

        Monotone on purpose. The read tool's spec is offered from the first digest onwards and
        never withdrawn, even after everything has been read, because the tool list sits in the
        cached prefix of every request: a spec that came and went would invalidate the prompt
        cache on each transition, and the loop's caching is worth more than the handful of
        tokens the spec costs. An investigation with no bulk result never offers it at all.
        """
        return bool(self._held)

    @property
    def read_count(self) -> int:
        return sum(1 for held in self._held.values() if held.read_at_step is not None)

    def reminder(self) -> str:
        """The standing disclosure of what is unread, or an empty string.

        Empty is the common case and it matters that it is empty: a turn with nothing unread
        must read exactly as it did before this module existed.
        """
        unread = self.unread()
        if not unread:
            return ""
        listed = "\n".join(item.line for item in unread)
        return (
            f"FETCHED BUT NOT READ ({len(unread)}). You retrieved these observations and have "
            "seen only a computed digest of each:\n"
            f"{listed}\n"
            f"Call {READ_TOOL} with one of those evidence_ids to get every row. It makes no "
            "upstream request, costs no wall clock, and does not use up a tool call. A digest "
            "reports counts and ranges; it cannot tell you which rows are in there. If your "
            "answer depends on that, read it before you conclude."
        )


def _normalise(raw_id: Any) -> str:
    """A handle as the ledger keys it.

    Tolerant of what a model actually emits -- surrounding whitespace, a stray `evidence_id:`
    prefix, a uuid without dashes -- because a read that fails on formatting is a read the
    model will not retry, and then it reasons from the digest.
    """
    text = str(raw_id).strip().removeprefix("evidence_id:").strip().strip("\"'")
    try:
        return str(uuid.UUID(text))
    except (ValueError, AttributeError, TypeError):
        return text


def read_tool_spec() -> dict[str, Any]:
    """The read tool, as the model sees it.

    The description carries the two things that decide whether it gets used: that reading is
    free, and that a digest is not a substitute for the rows. RCAgent passes its snapshot key
    to an expert agent for analysis; ours returns the rows to the same controller, because the
    controller is a frontier model and the thing it lacked was not analysis but the data.
    """
    return {
        "name": READ_TOOL,
        "description": (
            "Read one observation in full. An observation too large to paste into the "
            "conversation is shown to you as a digest -- counts, ranges and clusters over "
            "every row -- with its evidence_id. This returns all of its rows. It makes no "
            "upstream request, takes no measurable time, and does not count against your "
            "tool-call budget, so read anything whose rows could change your answer rather "
            "than reasoning from the digest. Reading is also the only way to quote or "
            "reconcile individual rows."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["evidence_id"],
            "properties": {
                "evidence_id": {
                    "type": "string",
                    "description": (
                        "The evidence_id of a summarised observation, exactly as it was "
                        "given to you."
                    ),
                }
            },
        },
    }
