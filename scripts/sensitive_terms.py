#!/usr/bin/env python3
"""List the identifying strings a real investigation touched, so they can be masked.

Masking digits is not enough. A real run's report named four of the tenant's customers in
one claim -- "several of the closed-won deals (e.g. ...) closed on the same date" -- which
is a good observation and an unpublishable sentence. Company names, deal owners and private
repository names carry no digits at all.

These cannot be found by pattern: there is no regular expression for "is a customer". They
can be found by *provenance*. A name in the report came from an observation, every
observation is a stored `Evidence` row, and this walks those rows and collects the values
of the fields that carry identity -- scoped per connector, because `name` means a customer
on a HubSpot deal and `$autocapture` on a PostHog event.

Output is one term per line, longest first so that masking a longer term cannot be
prevented by a shorter one nested inside it.

Usage:
    sensitive_terms.py [investigation-id] > terms.txt

With no argument it uses the most recent investigation.
"""

from __future__ import annotations

import asyncio
import re
import sys
import uuid
from typing import Any

from sqlalchemy import desc, select

from cortex.db.models import Evidence, Investigation
from cortex.runtime.resources import open_resources

#: Fields that carry identity, per connector. Scoped rather than global because the same
#: key means different things: `name` on a HubSpot deal is a customer, and on a PostHog
#: event definition it is `$pageview`.
IDENTIFYING: dict[str, frozenset[str]] = {
    "hubspot": frozenset(
        {
            "name",
            "dealname",
            "owner",
            "company",
            "contact",
            "email",
            "domain",
            "firstname",
            "lastname",
            "first_name",
            "last_name",
        }
    ),
    "github": frozenset({"repo", "repository", "description", "author", "login"}),
    "slack": frozenset({"user", "username", "real_name", "channel", "text"}),
}

#: Floor for a *derived token* -- a word pulled out of a longer name. Short fragments of a
#: deal title are noise, and masking them wrecks the prose around them.
MIN_LENGTH = 4

#: Floor for a *whole field value*, which is a different thing: a company literally named
#: "AMD" is identifying at three characters, and the shared floor of four let it through. It
#: reached a recording as `hubspot__companies query='AMD'` -- the model had read it from a
#: deal, so it was in the evidence, and the term list simply declined to carry it.
#:
#: Two is safe only because `mask_cast.py` matches a short term on a word boundary and
#: case-sensitively. Without both, a tenant with a customer called "IT" would blank the word
#: "it" across the whole report.
MIN_VALUE_LENGTH = 2

#: Connectors whose terms are also emitted broken into words.
#:
#: A model does not quote a deal name whole. Asked about closed-won deals a real run wrote
#: "several of the closed-won deals (e.g. Northwind, Initech, Umbrella)" while the stored
#: names were of the form "Northwind - Training Data" and "Initech - Self-hosted".
#: Masking whole terms caught none of them. The customer is the word, not the row label,
#: so the words are emitted as terms in their own right.
#:
#: The example names here are invented. Writing the real ones into a comment would leak
#: them from the very file that exists to stop that, in a public repository.
#:
#: Only for the CRM. Splitting a repository description into words and masking each would
#: shred ordinary prose to no purpose.
TOKENISED = frozenset({"hubspot"})

#: Words that appear inside deal names and identify nobody. Masking these would blank
#: half the report's ordinary sentences, and each one is here because it showed up in a
#: real portal's naming convention.
STOPWORDS = frozenset(
    """
    self hosted selfhosted self-hosted cloud saas paas iaas onprem on-prem
    training data intro demo trial pilot poc pov test testing sandbox
    starter standard premium enterprise business team teams
    package plan tier bundle licence license subscription contract
    dev devs developer developers seat seats user users
    renewal renewals expansion upsell upgrade downgrade migration
    new newbusiness existing annual monthly quarterly yearly
    deal deals opportunity account customer prospect
    inc ltd llc corp gmbh plc limited company
    the and for with from into via per and or not no yes
    q1 q2 q3 q4 fy phase stage round
    """.split()
)


def _tokens(term: str) -> set[str]:
    """The identifying words inside a term."""
    words = re.split(r"[^0-9A-Za-z]+", term)
    return {
        word
        for word in words
        if _identifying(word) and word.lower() not in STOPWORDS and not word.isdigit()
    }


def _identifying(word: str) -> bool:
    """Whether a word pulled out of a longer name is worth masking on its own.

    Four characters, or two if the word is all capitals.

    The second clause is the fix for a leak that survived one round of this. Lowering the
    floor for whole field values caught a company literally named "AMD", but the company
    arrived here as a *token* of a longer deal name -- three characters, below the token
    floor -- and reached a recording as "the three largest open deals ($Nk each: ..., AMD)"
    with every other name blanked beside it.

    All-caps is the discriminator that makes a lower floor safe. An acronym in a CRM is
    nearly always an organisation -- AMD, IBM, SAP, JPMC -- while the short lowercase tokens
    a deal title produces are prepositions. Matching is case-sensitive and word-anchored in
    `mask_cast.py`, so "AMD" blanks the company and leaves "amd" inside another word alone.
    """
    return len(word) >= MIN_LENGTH or (word.isupper() and len(word) >= 2)


def _collect(payload: Any, fields: frozenset[str], found: set[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and key.lower() in fields:
                text = value.strip()
                if len(text) >= MIN_VALUE_LENGTH and not text.startswith("$"):
                    found.add(text)
            else:
                _collect(value, fields, found)
    elif isinstance(payload, list):
        for item in payload:
            _collect(item, fields, found)


async def terms_for(investigation_id: uuid.UUID | None) -> list[str]:
    async with open_resources() as resources:
        async with resources.sessionmaker() as session:
            if investigation_id is None:
                investigation = await session.scalar(
                    select(Investigation).order_by(desc(Investigation.created_at)).limit(1)
                )
                if investigation is None:
                    raise SystemExit("no investigations stored")
                investigation_id = investigation.id
            rows = (
                await session.scalars(
                    select(Evidence).where(Evidence.investigation_id == investigation_id)
                )
            ).all()

    found: set[str] = set()
    for row in rows:
        fields = IDENTIFYING.get(row.tool_name)
        if not fields:
            continue
        here: set[str] = set()
        _collect(row.payload, fields, here)
        found |= here
        if row.tool_name in TOKENISED:
            for term in here:
                found |= _tokens(term)
    # Longest first, so masking a term cannot be pre-empted by a shorter one nested in it.
    return sorted(found, key=lambda term: (-len(term), term))


def main(argv: list[str]) -> int:
    given = uuid.UUID(argv[1]) if len(argv) > 1 else None
    terms = asyncio.run(terms_for(given))
    sys.stdout.write("\n".join(terms) + "\n")
    sys.stderr.write(f"{len(terms)} identifying term(s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
