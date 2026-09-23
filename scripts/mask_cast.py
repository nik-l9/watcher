#!/usr/bin/env python3
"""Mask the figures in a recording of a real run, so it can be published.

A `--real` investigation answers with its tenant's actual numbers, and publishing a
terminal recording of one is irreversible.

Two kinds of thing have to go. The figures -- a win rate, a deal count, two dollar totals
-- are usable competitor intelligence on their own. And sometimes the names: one run of
the same question answered purely in aggregates, and the next named four of the tenant's
customers in a single claim. Whether names appear is a property of the run, not of the
question, so both mechanisms always apply.

What the demo is actually for survives masking completely: two independent searches
cross-checking one claim, the verifier removing a claim and flagging another as
overreaching, and a data-quality block volunteering that the prior period was read only as
a digest and that six syncs are fifteen days stale. None of that needs a real dollar
figure.

No digit rule touches a company name, so names need their own mechanism: a term list
produced by `sensitive_terms.py`, which collects them from the investigation's own stored
evidence -- provenance rather than pattern, because there is no regular expression for "is
a customer".

**Safe by construction.** Rather than hunting for the sensitive values -- where one missed
occurrence leaks -- this masks *every digit* in the report's prose and then carves out the
few patterns that must survive:

  - dates (`2026-06-22`), without which no claim can be read;
  - evidence ids (`5b42dd2d`), which are the whole point of a cited report;
  - anything inside a citation bracket.

`verify()` then asserts the invariant directly: outside those carve-outs, no digit remains
in a masked section. A leak is a failed assertion, not something a reviewer has to notice.

Left alone deliberately: the timings and step numbers in the progress lines, the run header
(`Took 85s over 3 steps, 79,657 tokens`), the source table and the phase breakdown. They
carry no business figure and they are what makes a recording look like a run rather than a
slideshow. The one exception is a row count -- `(73 rows)` is a deal count whatever line it
appears on -- which is masked wherever it occurs.

Usage:
    sensitive_terms.py > terms.txt
    mask_cast.py in.cast out.cast terms.txt

The term list is optional but is not optional in practice for a run against a CRM.
"""

from __future__ import annotations

import json
import re
import sys

BLOCK = "█"

#: Sections whose prose states findings about the tenant's business.
#:
#: The source table and the phase breakdown are deliberately absent: one holds dates and
#: evidence ids, the other holds timings and token counts, and both are worth showing.
MASKED_SECTIONS = {
    "ANSWER",
    "FINDINGS",
    "CHARTS",
    "HYPOTHESES TESTED",
    "RECOMMENDATIONS",
    "RISKS AND CAVEATS",
    "DATA QUALITY",
}

#: Ends the masked region. Everything from here on is references and instrumentation.
END_SECTIONS = {"SOURCES"}

#: Counts that appear outside the report's prose and are still business figures.
#:
#: The progress lines sit before the report, so section-based masking never reached them --
#: and one of them reads `got read hubspot.pipeline in full (73 rows)`, which is the tenant's
#: open deal count in plain text, directly above a report whose every figure was blanked.
#: Timestamps and step numbers in those same lines are left alone: they are what makes a
#: recording look like a run rather than a slideshow.
_ROW_COUNT = re.compile(r"\((\d[\d,]*) rows?\)")

#: Record ids inside a source reference.
#:
#: The source table was exempt from digit masking on the grounds that it holds dates and
#: evidence ids, both worth showing. It also holds the URL each observation came from, and
#: those carry the tenant's own object ids:
#:
#:     hubspot://crm/v3/objects/companies/REDACTED/engagements
#:
#: That is a customer's CRM record, addressable by anyone with access to the portal, printed
#: beside a report whose every figure had been blanked. Five digits or more, and only inside
#: a URL: a date's longest run is its four-digit year, an API version is `v3`, and the
#: evidence id sits in its own column outside the URL, so all three survive.
_URL_IDS = re.compile(r"(?P<url>\w+://\S+)")
_LONG_RUN = re.compile(r"\d{5,}")

#: The value of a search query in a source reference.
#:
#: Blanked whole rather than term-matched, because the analyst does not always search for a
#: name it was given. It writes prefixes: a portal holding "Dennison Freight" produced
#: `?q=Denn`, which no term list contains and which still names the customer well enough to
#: guess. A query against a CRM is a customer name or it is nothing, so there is nothing to
#: lose by blanking all of it and a fragment to lose by being clever.
_QUERY_VALUE = re.compile(r"(?<=[?&]q=)[^&\s]+")

#: Patterns that must survive masking, in the order they are protected.
KEEP = (
    re.compile(r"\[[^\]]*\]"),  # a citation bracket, whole
    re.compile(r"\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?"),  # a date, optionally a timestamp
    re.compile(r"\b[0-9a-f]{8}\b"),  # an evidence id
)

_DIGIT = re.compile(r"\d")


#: Lines whose shape the rest of this module reads. Never term-masked.
STRUCTURAL = MASKED_SECTIONS | END_SECTIONS


def _mask_terms(text: str, terms: list[str]) -> str:
    """Blank each identifying term wherever it appears, whatever the section.

    Applied to the whole recording rather than only the masked sections: a customer name
    is no less identifying for appearing in a progress line or a source reference.

    Section headings are exempt, and that exemption is load-bearing. Matching is
    case-insensitive, so a portal with a deal named "Source ..." turned the heading
    `SOURCES` into `██████S`. `_sections` then never saw the end of the masked region,
    kept masking to the bottom of the report, and `verify` failed on the `v3` in a HubSpot
    URL. Two recordings were refused before the cause was obvious -- correctly refused,
    which is the only reason this was not published instead.
    """
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip() in STRUCTURAL:
            out.append(line)
            continue
        for term in terms:
            line = _blank(line, term)
        out.append(line)
    return "".join(out)


#: At or below this length a term is matched on a word boundary and case-sensitively.
#:
#: Short names are real -- AMD, SAP, HP -- and a substring rule cannot carry them: masking
#: "it" case-insensitively as a substring blanks a letter pair out of half the English in the
#: report. A boundary-anchored, case-sensitive match blanks the company and leaves the
#: pronoun alone.
_SHORT_TERM = 4


def _blank(line: str, term: str) -> str:
    """Replace one identifying term wherever it appears in this line."""
    block = BLOCK * min(len(term), 12)
    if len(term) <= _SHORT_TERM:
        return re.sub(rf"\b{re.escape(term)}\b", block, line)
    return re.sub(re.escape(term), block, line, flags=re.IGNORECASE)


def _mask_line(line: str) -> str:
    """Replace every digit outside a protected span with a block."""
    protected: list[tuple[int, int]] = []
    for pattern in KEEP:
        protected.extend(match.span() for match in pattern.finditer(line))

    def guarded(index: int) -> bool:
        return any(start <= index < end for start, end in protected)

    return "".join(
        BLOCK if character.isdigit() and not guarded(index) else character
        for index, character in enumerate(line)
    )


def _sections(lines: list[str]):
    """Yield (line, is_masked) for each line, tracking which section it sits in."""
    masking = False
    for line in lines:
        stripped = line.strip()
        if stripped in MASKED_SECTIONS:
            masking = True
        elif stripped in END_SECTIONS:
            masking = False
        yield line, masking


def mask(source: str, terms: list[str] | None = None) -> str:
    lines = source.splitlines(keepends=False)
    # The header is not output, and masking only the output left the tenant slug sitting in
    # plain text in `command` -- `--real --tenant <slug>` -- at the top of a file that was
    # about to be committed. A leak scan caught it; this is why the scan exists.
    header = _mask_terms(lines[0], terms or [])
    events = [json.loads(line) for line in lines[1:] if line.strip()]

    text = "".join(event[2] for event in events if event[1] == "o")
    out = "".join(
        _mask_line(line) if masked else line
        for line, masked in _sections(text.splitlines(keepends=True))
    )
    out = _mask_terms(out, terms or [])
    out = _ROW_COUNT.sub(lambda m: f"({BLOCK * len(m.group(1))} rows)", out)
    out = _URL_IDS.sub(
        lambda m: _QUERY_VALUE.sub(
            lambda q: BLOCK * min(len(q.group()), 12),
            _LONG_RUN.sub(lambda d: BLOCK * len(d.group()), m.group("url")),
        ),
        out,
    )

    # Re-emitted as one output event, plus whatever was not output (the exit status). A
    # recording being prepared for publication carries no information in its original
    # chunking, and pace_cast.py re-times it from the text regardless.
    trailing = [event for event in events if event[1] != "o"]
    return (
        "\n".join([header, json.dumps([0.0, "o", out]), *(json.dumps(e) for e in trailing)]) + "\n"
    )


def verify(masked: str, terms: list[str] | None = None) -> None:
    """Assert no digit survives in a masked section, and no identifying term anywhere."""
    lines = masked.splitlines()
    # The header is checked alongside the output, for the reason given in `mask`.
    text = lines[0] + "".join(
        event[2]
        for event in (json.loads(line) for line in lines[1:] if line.strip())
        if event[1] == "o"
    )
    for line, is_masked in _sections(text.splitlines()):
        if not is_masked:
            continue
        stripped = line
        for pattern in KEEP:
            stripped = pattern.sub("", stripped)
        if _DIGIT.search(stripped):
            raise AssertionError(f"unmasked figure survived: {line!r}")

    # Structural lines are exempt from masking, so they must be exempt from the check too.
    # They were not, and the result was a portal with a deal named "Source ..." failing
    # verification against the word SOURCES in its own report heading -- the exemption
    # arguing with the assertion that enforces it.
    text_checked = "\n".join(line for line in text.splitlines() if line.strip() not in STRUCTURAL)
    lowered = text_checked.lower()
    for term in terms or []:
        # Checked exactly as it is masked, including the word-boundary rule for short terms.
        if _blank(lowered if len(term) > _SHORT_TERM else text_checked, term) != (
            lowered if len(term) > _SHORT_TERM else text_checked
        ):
            raise AssertionError(f"identifying term survived: {term!r}")


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        sys.stderr.write(f"usage: {argv[0]} in.cast out.cast [terms.txt]\n")
        return 2
    terms: list[str] = []
    if len(argv) == 4:
        raw = open(argv[3], encoding="utf-8").read().splitlines()
        terms = sorted((t.strip() for t in raw if t.strip()), key=lambda t: -len(t))
    masked = mask(open(argv[1], encoding="utf-8").read(), terms)
    verify(masked, terms)
    open(argv[2], "w", encoding="utf-8").write(masked)
    sys.stderr.write(f"{argv[2]}: masked and verified ({len(terms)} term(s))\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
