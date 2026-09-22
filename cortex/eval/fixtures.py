"""Labeled evaluation fixtures.

A scenario is a synthetic GTM history with a **known ground truth**: a planted
causal chain the analyst is supposed to find, and decoy correlations it is supposed
to reject. That labelling is the whole point — accuracy cannot be scored against
real data, because nobody knows the real answer either.

Two design rules make the scores mean something:

  - **Every scenario carries decoys.** A scenario with one plausible explanation
    scores a coin-flip guesser as perfect. Each decoy here correlates with the
    outcome as strongly as the real cause and is ruled out only by evidence the
    analyst has to go and fetch.
  - **The data is generated, not written.** Hand-written rows tend to make the
    intended answer the only *coherent* story, which tests reading comprehension
    rather than investigation. These are generated from a spec so the decoys are
    genuinely as visible as the cause.

Deterministic: the generator is seeded, so a score change means the analyst changed,
not the fixture.
"""

from __future__ import annotations

import dataclasses
import enum
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

# The connector's own change arithmetic and coverage check, imported rather than
# reimplemented. A fixture that computes percentage change its own way is a second
# implementation of the thing the contract test compares, and the two would drift on the
# first edge case (a zero previous value).
from cortex.tools.ga4 import _absolute, _percent, _window_coverage


class Difficulty(enum.StrEnum):
    #: One clear cause, decoys present but weakly correlated.
    STRAIGHTFORWARD = "straightforward"
    #: Decoys correlate as strongly as the cause; only a specific tool call separates them.
    CONFOUNDED = "confounded"
    #: The evidence genuinely cannot establish a cause. The correct answer is to say so.
    UNANSWERABLE = "unanswerable"
    #: The question asserts something that did not happen. The correct answer contradicts
    #: the question, in its first sentence.
    FALSE_PREMISE = "false_premise"


#: One requirement, which a tuple satisfies by any of its alternatives.
#:
#: A plain string is a single requirement; a tuple is any-of. Both dimensions that use
#: this were previously all-of over plain strings, and that turned out to score the
#: *route* rather than the *result*. Two concrete cases, both real:
#:
#:   - An investigation identified the change as "PR #913, merged 2026-07-14, whose
#:     reviewer flagged the button below the fold on a 375px viewport", recommended the
#:     exact fix, and was scored 0.5 for accuracy because the label demanded the deploy
#:     sha. The sha and the PR number identify the same change equally well.
#:   - The same investigation reached the cause through `commits` and
#:     `pull_request_activity` instead of `deployment_history`, and lost a third of
#:     tool_selection for taking the better-evidenced path.
#:
#: A label that punishes a correct answer for arriving differently measures conformity.
Requirement = str | tuple[str, ...]


def alternatives_of(requirement: Requirement) -> tuple[str, ...]:
    return (requirement,) if isinstance(requirement, str) else requirement


def describe_requirement(requirement: Requirement) -> str:
    """How an unmet requirement is reported, so any-of reads as any-of."""
    alternatives = alternatives_of(requirement)
    return alternatives[0] if len(alternatives) == 1 else "any of [" + ", ".join(alternatives) + "]"


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """What a correct answer looks like.

    `required_signals` are substrings the report must contain to count as having
    found the cause — a specific deploy sha, a named segment. Deliberately
    string-matched rather than judged by a model: the scorer's grounding and accuracy
    dimensions must not themselves depend on an LLM's opinion.

    An entry in either requirement tuple may itself be a tuple, meaning **any of these
    satisfies it**. See `Requirement`.
    """

    cause: str
    required_signals: tuple[Requirement, ...]
    #: Explanations that correlate but are not causal. Naming one as *the* cause is wrong.
    decoys: tuple[str, ...]
    #: Tool capabilities a competent investigation must use to separate cause from decoy.
    required_capabilities: tuple[Requirement, ...]
    #: True when the honest answer is that the data cannot say.
    is_unanswerable: bool = False

    #: True when the question asserts a movement that did not happen.
    #:
    #: Distinct from `is_unanswerable`, which is "the data cannot say why". Here the data
    #: says clearly, and what it says is *that the thing being asked about did not occur*. No
    #: grounding mechanism catches a wrong answer to this: every claim can be supported by
    #: real evidence while the report answers a question whose premise nobody checked.
    is_false_premise: bool = False

    #: What refuting the premise looks like, checked against the **executive summary**.
    #:
    #: The position is the point, which is why this is a separate field rather than more
    #: `required_signals`. A report that reaches the right answer in its fourth bullet, after
    #: leading with a figure that appears to confirm the premise, has misinformed every reader
    #: who stopped early — and that is a real delivered answer, not a hypothetical.
    refutation_signals: tuple[Requirement, ...] = ()

    @property
    def declines_a_cause(self) -> bool:
        """True when naming a cause at all is the wrong answer.

        Both classes share this and it is scored the same way: an unanswerable scenario has no
        establishable cause, and a false-premise scenario has no effect to explain. In both, a
        recommendation to act is the failure.
        """
        return self.is_unanswerable or self.is_false_premise

    @property
    def capability_names(self) -> tuple[str, ...]:
        """Every capability that appears in a requirement, alternatives included.

        For callers that need the flat list — the fixture drift test, which checks each
        one still exists on a real tool.
        """
        return tuple(
            name
            for requirement in self.required_capabilities
            for name in alternatives_of(requirement)
        )


#: What the environment survey sees when a scenario does not plant it.
#:
#: Every scenario runs against a tenant that has repositories and projects, because every real
#: one does. Without these, the survey would open each investigation by reporting an empty
#: estate — and "you have no repositories" is a false premise rather than an empty observation,
#: which would push a competent analyst towards concluding the data does not exist.
#:
#: Deliberately generic: names a scenario needs to reason about belong in that scenario's own
#: `responses`, where they can be checked. `tests/db/test_eval.py` asserts every declared
#: discovery capability has either a plant or an entry here, so the next one cannot silently
#: survey an empty world.
DISCOVERY_DEFAULTS: dict[str, Any] = {
    "github__list_repositories": {
        "count": 3,
        "repositories": [
            {
                "repo": "acme/product",
                "description": "The SaaS application.",
                "pushed_at": "2026-07-16T09:12:00Z",
                "private": True,
                "archived": False,
                "default_branch": "main",
            },
            {
                "repo": "acme/company-website",
                "description": "Marketing site and blog.",
                "pushed_at": "2026-07-15T18:40:00Z",
                "private": False,
                "archived": False,
                "default_branch": "main",
            },
            {
                "repo": "acme/mobile",
                "description": "iOS and Android clients.",
                "pushed_at": "2026-07-14T21:05:00Z",
                "private": True,
                "archived": False,
                "default_branch": "main",
            },
        ],
    },
    # The fixture tenant is a PostHog shop, which is the ordinary case: a company runs *one*
    # product-analytics product. So these two report zero events, and that is the truth rather
    # than a silence -- it steers the analyst to the source that has the data instead of leaving
    # it to guess which of three to try.
    #
    # This is the one place an empty discovery result is correct. Elsewhere an empty survey is a
    # false premise, because the estate really does exist; here the tenant genuinely has no
    # Mixpanel or Amplitude project, and saying so is more useful than saying nothing. The `note`
    # is what makes it a statement rather than an absence.
    "mixpanel__list_events": {
        "event_count": 0,
        "events": [],
        "note": "This tenant has no Mixpanel project connected. Product analytics are in PostHog.",
    },
    "amplitude__list_events": {
        "event_count": 0,
        "events": [],
        "note": (
            "This tenant has no Amplitude project connected. Product analytics are in PostHog."
        ),
    },
    "posthog__list_projects": {
        "project_count": 1,
        "default_project": "100001",
        "projects": [{"id": "100001", "label": "product", "is_default": True}],
    },
    # The generic catalogue, for a scenario that does not plant its own. Three decoys, and none
    # of them is a rival name for the event the scenario plants.
    #
    # **`signup_completed` used to be the first entry and it cost real quality.** Every scenario
    # here asks about signups and every one plants `user signed up`, so the catalogue was offering
    # a *better-looking* name for the metric being asked about, attached to nothing. The analyst
    # queried it first every time -- reasonably -- got zero rows, and reported that the primary
    # signup event had returned no data for the whole period: a data incident that does not exist.
    # On `campaign_traffic_drop` that produced four claims retained at reduced confidence, a
    # sufficiency warning saying the evidence "doesn't clearly show a signup drop at all", and a
    # data-quality note about a pipeline that is fine, on a run that otherwise named the cause
    # exactly. On `partial_month_false_premise` it went further and cost the accuracy gate: the
    # analyst hedged the premise to `unverifiable` because "the signup event used in the
    # platform's canonical funnel returned zero data for the whole period".
    #
    # A trial start and a checkout are near misses an analyst still has to think about. They are
    # not impostors for the same fact, which is the difference between a decoy and a landmine.
    "posthog__list_events": {
        "count": 3,
        "events": [
            {"name": "trial_started", "last_seen_at": "2026-07-16T10:00:00Z"},
            {"name": "checkout_started", "last_seen_at": "2026-07-16T10:00:00Z"},
            {"name": "$pageview", "last_seen_at": "2026-07-16T10:00:00Z"},
        ],
    },
    "bigquery__list_queries": {
        "count": 1,
        "queries": [
            {"name": "signups_by_day", "description": "Daily signup counts.", "params": ["days"]}
        ],
    },
}


@dataclass(frozen=True, slots=True)
class Scenario:
    """One labeled evaluation case."""

    name: str
    question: str
    difficulty: Difficulty
    ground_truth: GroundTruth
    #: Canned tool responses, keyed by `tool__capability`. The eval provider serves
    #: these instead of calling a real API.
    responses: dict[str, Any] = field(default_factory=dict)

    #: Dated records a capability filters by subject and window, keyed by `tool__capability`.
    #: See `DatedRecords`. ADR 0006's first landing; the canned payloads it replaces could not
    #: honour `repo`, `since` or `until` except by remembering to.
    dated_records: dict[str, DatedRecords] = field(default_factory=dict)

    #: Daily counts a trend capability derives every interval from, keyed by `tool__capability`.
    #:
    #: A canned payload answers `interval="day"` with whatever buckets it was typed with, so a
    #: scenario whose trap is a monthly series answered the daily call -- the discriminating one
    #: -- with monthly data. See `DailyTruth`.
    #:
    #: A tuple rather than one series per capability, because a scenario often needs several:
    #: `measurement_stopped` asserts the site is fine and only its measurement stopped, which
    #: takes a healthy second event to say. One series per capability would have forced that
    #: scenario to stay canned and kept the defect alive in the one place it matters most.
    daily_truth: dict[str, tuple[DailyTruth, ...]] = field(default_factory=dict)

    #: Responses that differ by *subject* — which event, which repository — keyed by
    #: `tool__capability` then by the subject value.
    #:
    #: Needed because `responses` holds one payload per capability, so a scenario could plant
    #: exactly one series. `_for_subject` then returns an empty series for every other subject,
    #: which is honest but can be actively misleading: `measurement_stopped` asserts that the
    #: site is fine and only its *measurement* stopped, and a world where one product event has
    #: data and every other returns nothing says the opposite — that everything stopped, which is
    #: the site-outage reading the scenario exists to rule out.
    #:
    #: A real deployment has many healthy events. This lets a fixture say so.
    subject_responses: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: Whether the connector's own disclosures are computed over this scenario's payloads.
    #:
    #: True by default, so a scenario sees what production would send it. It was effectively
    #: False everywhere until now -- the eval replaces each capability's handler, so no
    #: connector method ran and no scenario ever saw `series_ends_early`, `partial_buckets`,
    #: `data_trust` or a movement description.
    #:
    #: **Set False deliberately, to keep a scenario harder than production.**
    #: `partial_month_false_premise` measures whether the analyst notices a month is incomplete
    #: with no help at all. Once the connector discloses it, that scenario stops measuring the
    #: skill and starts measuring compliance -- both worth knowing, and the disclosure is
    #: already unit-tested, so the scarcer one wins here.
    #:
    #: **A frozenset lets exactly those disclosures through**, which is what makes one of them
    #: measurable. A bool prices the whole set at once: turning it on for the truncation
    #: scenario would hand over `series_ends_early`, `partial_buckets`, `data_trust` and the
    #: movement note along with the coverage warning, so a paired comparison against it would
    #: measure four changes and attribute them to one. Naming a set is one perturbation, which
    #: is the same discipline the decline twins are built on.
    compute_disclosures: bool | frozenset[str] = True

    def discloses(self, name: str) -> bool:
        """Whether this scenario lets the connector's `name` disclosure reach the analyst."""
        if isinstance(self.compute_disclosures, bool):
            return self.compute_disclosures
        return name in self.compute_disclosures

    @property
    def discloses_anything(self) -> bool:
        """Whether any disclosure is allowed through, so a caller can skip computing them."""
        return bool(self.compute_disclosures)

    #: The scenario's GA4 world as one daily session series plus how it divides. See
    #: `MetricSeries`. Every `ga4__*` capability it declares is projected from this, so the
    #: funnel, the period comparison and the session series cannot disagree about the same
    #: window -- which three of them did, by up to 15%.
    metric_series: MetricSeries | None = None

    @property
    def as_of(self) -> date | None:
        """The last day this scenario's world has any data for.

        **The problem it solves.** A fixture answers every requested range with one fixed series.
        Once the eval started computing the connectors' real disclosures, that made every
        scenario report a collection failure to any analyst who asked a wider range than was
        planted: `onboarding_regression` reported its series stopping 46 days ago if the analyst
        asked through August, and `insufficient_evidence` 101 days. A scenario about a mobile
        onboarding regression would have been handed a data-incident verdict, which outranks and
        replaces the real answer.

        The fixture's own data is the authority on when its world ends, so this is derived from
        it rather than declared beside it -- a declared date is a second thing to keep in step,
        and the repository has already been bitten twice by exactly that (the repository listing,
        and the connected-tools inference this sits next to).

        Returns None when the scenario plants no dated series, in which case nothing clamps
        because nothing is being disclosed either.

        Note this is the maximum across *all* series, not per series, and that is the point:
        `measurement_stopped` has GA4 stopping on 3 August while PostHog runs to the 15th, so
        its world demonstrably has data through the 15th and GA4's stop is a real twelve-day gap.
        A per-series horizon would clamp the gap away and delete the scenario's whole subject.

        **`daily_truth` counts too, and forgetting it deleted that scenario once.** Moving the
        PostHog series out of `responses` and into derived form dropped this horizon from the
        15th to the 3rd, which made GA4's truncated series look complete and removed the only
        thing the scenario is about. That is the third derived invariant a new planting field has
        broken -- `events_described` twice before it -- so both are asserted directly rather than
        trusted to a list somebody remembers to extend.
        """
        latest: date | None = None
        if self.metric_series is not None:
            latest = max(day for day, _ in self.metric_series.daily)
        for planted in self.daily_truth.values():
            for truth in planted:
                for day, _ in truth.days:
                    if latest is None or day > latest:
                        latest = day
        for response in self.responses.values():
            if not isinstance(response, dict):
                continue
            for row in (response.get("series") or []) + (response.get("rows") or []):
                if not isinstance(row, dict):
                    continue
                raw = row.get("bucket") or (row.get("dimensions") or {}).get("date")
                if not isinstance(raw, str):
                    continue
                try:
                    parsed = date.fromisoformat(raw[:10])
                except ValueError:
                    continue
                if latest is None or parsed > latest:
                    latest = parsed
        return latest

    @property
    def connected_tools(self) -> frozenset[str]:
        """Which connectors this scenario's tenant has, inferred from what it plants.

        **Why the eval needs this at all.** `registry_for_tenant` exists in production because
        offering an analyst tools it cannot reach is expensive: every real investigation was
        offered `ga4__*` and `bigquery__*`, spent steps calling them, got `CredentialMissing`, and
        then apologised at length in the report for an absence that was never relevant. The eval
        was mirroring the *whole* registry instead, so scenarios were paying that cost the product
        no longer pays -- and the bill arrived the moment two connectors were added, with
        `onboarding_regression` going from 70.7s to 95.0s and `campaign_traffic_drop` from 78.7s
        to 98.7s on a 90s budget.

        Inferred rather than declared, so a scenario cannot drift out of step with its own data:
        a tenant has a connector exactly when the scenario has something for it to return, or when
        the ground truth requires calling it. Declaring it separately would create a second place
        to update and a new way for the two to disagree.
        """
        required = {
            alternative
            for requirement in self.ground_truth.required_capabilities
            for alternative in alternatives_of(requirement)
        }
        return frozenset(name.split("__")[0] for name in self.planted_capabilities | required)

    @property
    def planted_capabilities(self) -> frozenset[str]:
        """Every capability this scenario can answer for, whichever field it was planted in.

        **One place, because four separate derivations each learned about a new planting field
        the hard way.** `daily_truth` broke `events_described` (twice, once through
        `subject_responses` before it), then `as_of`, and then this -- and this one was the
        worst: moving `campaign_traffic_drop`'s only PostHog planting into `daily_truth` left
        the connector *disconnected*, so the analyst was never offered `posthog__event_trend`
        and the daily series built for it could not be reached at all. The scenario still scored
        1.00, from GA4 alone, which is why nothing noticed.

        Every derivation that asks "what does this scenario plant" now asks here. A field added
        later is wired in one place, and `TestEveryPlantingFieldIsSeenByEveryDerivation` fails
        if it is not.
        """
        return frozenset(
            set(self.responses)
            | set(self.subject_responses)
            | set(self.daily_truth)
            | set(self.dated_records)
            | (self.metric_series.capabilities if self.metric_series else frozenset())
        )

    def response_for(self, qualified_name: str, params: dict[str, Any] | None = None) -> Any:
        """The canned response, or an empty-but-valid shape for an unplanted call.

        An unplanted call returns empty rather than erroring: a competent analyst
        exploring a hypothesis that turns out to be unsupported should see "nothing
        here", which is a real observation, not a tool failure.

        **Discovery capabilities are the exception**, and fall back to
        `DISCOVERY_DEFAULTS` instead. The investigator surveys every discovery capability
        before the first step, so an unplanted one would open every investigation by telling
        the analyst that this tenant has no repositories and no projects — which is not a
        neutral empty result, it is a false premise that invites the analyst to conclude the
        data does not exist. A scenario that cares plants its own; the rest get a plausible
        environment.
        """
        # Ahead of every other layer: where a scenario declares a GA4 world, that world is the
        # authority on it. A payload planted for the same capability would otherwise reintroduce
        # exactly the second, contradicting figure `MetricSeries` exists to make unrepresentable.
        if self.metric_series is not None and qualified_name in GA4_SERIES_CAPABILITIES:
            projection = _GA4_PROJECTIONS.get(qualified_name)
            if projection is not None and qualified_name in self.metric_series.capabilities:
                answered = projection(self.metric_series, params)
                # The coverage disclosure is a disclosure, so a scenario that withholds them
                # withholds this one too. `partial_month_false_premise` is the case: it exists
                # to measure whether the analyst notices a truncated month unaided, and a note
                # saying "these windows hold 12 days and 31" hands over its whole answer.
                if qualified_name == "ga4__compare_periods" and self.discloses("window_coverage"):
                    answered.update(_coverage_for(self.metric_series, params))
                return answered
        if self._names_another_project(qualified_name, params):
            # A PostHog project this tenant has and this scenario has no data in. The projects
            # listing advertises two -- `web-app` and `oss-client` -- and every capability
            # answered both with the same series, so an analyst that queried the wrong project
            # was rewarded exactly as well as one that read the listing and chose. The last
            # place in the suite where a payload answered a question it was not asked.
            return self._nothing_for(qualified_name, params)
        by_subject = self.subject_responses.get(qualified_name)
        if by_subject and params:
            for key in self.SUBJECT_KEYS:
                asked = params.get(key)
                # Compared as a string because a pull-request number arrives as an int and an
                # event name as a str, and the layer is keyed the same way for both.
                if isinstance(asked, str | int) and str(asked) in by_subject:
                    return by_subject[str(asked)]
            planted = next(iter(by_subject.values()))
            entity = next((key for key in _ENTITY_KEYS if key in planted), None)
            if entity is not None:
                # A record this scenario does not describe, on a capability that looks records
                # up one at a time. Answered in the planted shape with every field about *some
                # other* record nulled -- because the payload this replaces returned pull
                # request 913 whatever number was asked for, so an analyst opening the pricing
                # decoy was handed the mobile onboarding modal's review thread under the
                # decoy's number. On a required capability, which made the scenario reward the
                # wrong lookup with the right answer.
                return _empty_like(planted, params, self.SUBJECT_KEYS)
        # The two derived listings come *before* the plain lookup, and the ordering is the whole
        # point: a scenario that plants its own catalogue would otherwise bypass the derivation
        # and go back to advertising a world the rest of the fixture cannot answer for. Both
        # builders take a planted listing as their base, so planting still decides the contents
        # -- it just no longer decides whether the invariant holds.
        if qualified_name == "github__list_repositories":
            return self._repository_listing()
        if qualified_name == "posthog__list_events":
            return self._event_listing()
        # Derived from daily counts, so the interval the caller asked for is the interval it
        # gets. Ahead of the plain lookup for the same reason the listings are: a scenario that
        # also planted a payload here would otherwise go back to serving one fixed granularity.
        records = self.dated_records.get(qualified_name)
        if records is None:
            # Inferred from a canned payload's own shape, so a capability not yet migrated by
            # hand still honours its parameters. Explicit `dated_records` wins where both exist.
            planted_payload = self.responses.get(qualified_name)
            if isinstance(planted_payload, dict):
                records = _auto_records(qualified_name, planted_payload)
        if records is not None:
            return _project_records(records, params)
        planted = self.daily_truth.get(qualified_name)
        if planted:
            asked = (params or {}).get("event")
            for truth in planted:
                if not isinstance(asked, str) or asked == truth.event:
                    return _resample(truth, params, matched=True)
            # Named an event this scenario does not plant. Answered empty in the first series'
            # shape, which keeps the payload readable -- an empty result that also lost its
            # interval and window says "nothing here" about an unknown question.
            return _resample(planted[0], params, matched=False)
        if qualified_name in self.responses:
            return self._for_subject(self.responses[qualified_name], params)
        return DISCOVERY_DEFAULTS.get(qualified_name, {"rows": [], "count": 0})

    def _repository_listing(self) -> dict[str, Any]:
        """Every repository this scenario describes, plus the standing decoys.

        **The bug this fixes.** All five scenarios described a repository that discovery said
        did not exist. `DISCOVERY_DEFAULTS` advertised `acme/product`, `acme/company-website`
        and `acme/mobile`; four scenarios answered every github call with payloads labelled
        `acme/web` and the fifth with `acme/marketing-site`. So the analyst surveyed the tenant,
        was told which repositories exist, asked about one of them, and got back a payload
        describing a different one.

        It cost real score. `tempting_coincidence` read `draft_reliability` 0.77, and the
        verifier's rejections were things like *"the payload's `repo` field returns
        'acme/marketing-site' in all three items, not acme/product"* -- correct every time, and
        the claim it was rejecting was the analyst faithfully reporting what it had asked for.
        A gating dimension was measuring a contradiction in the fixture. Nothing in the
        scorecard could show this; it came out of reading the captured payloads.

        **Derived rather than declared**, because a second hand-maintained list is how the
        first one drifted. A repository the scenario describes is advertised by construction.

        The decoys stay. A listing containing only the repository that matters would hand over
        the answer -- `tempting_coincidence` exists to see whether the analyst resists a
        plausible coincidence, and narrowing the field to one candidate removes the choice.
        """
        described = self.repositories_described()
        planted = self.responses.get("github__list_repositories") or {}
        decoys = (
            planted.get("repositories")
            or (DISCOVERY_DEFAULTS["github__list_repositories"]["repositories"])
        )
        known = {entry["repo"]: entry for entry in decoys}
        listing = [
            known.get(
                repo,
                {
                    "repo": repo,
                    "description": "",
                    "pushed_at": "",
                    "private": False,
                    "archived": False,
                    "default_branch": "main",
                },
            )
            for repo in sorted(described)
        ]
        listing += [entry for entry in decoys if entry["repo"] not in described]
        return {"count": len(listing), "repositories": listing}

    #: Payload fields naming what a response is *about*, checked against the request.
    #:
    #: A fixture is keyed by capability, so without this it answers every request for a series
    #: with the one series it holds -- whatever was asked for.
    SUBJECT_KEYS = ("event", "repo", "number")

    def _names_another_project(self, qualified_name: str, params: dict[str, Any] | None) -> bool:
        """True when a PostHog call named a project other than this scenario's own.

        Read from the projects listing rather than declared, so the catalogue the analyst is
        shown and the project the fixture answers for cannot drift apart -- which is the
        invariant the repository and event listings each had to learn the hard way.
        """
        if not qualified_name.startswith("posthog__") or not params:
            return False
        asked = params.get("project")
        if not isinstance(asked, str | int):
            return False
        listing = self.responses.get("posthog__list_projects") or DISCOVERY_DEFAULTS.get(
            "posthog__list_projects", {}
        )
        default = listing.get("default_project")
        return default is not None and str(asked) != str(default)

    def _nothing_for(self, qualified_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
        """An empty answer in whatever shape this capability would have returned.

        Shape rather than `{}`, for the reason this fixture keeps restating: "I looked and there
        is nothing" and "I could not look" are different observations, and only the first is
        something an analyst can reason from.
        """
        planted = self.daily_truth.get(qualified_name)
        if planted:
            return _resample(planted[0], params, matched=False)
        if qualified_name == "posthog__list_events":
            # The derived catalogue, emptied. A bare `{rows: [], count: 0}` here would be a
            # shape no PostHog capability returns, and the analyst reads the shape.
            return _empty_like(self._event_listing(), params or {}, self.SUBJECT_KEYS)
        by_subject = self.subject_responses.get(qualified_name)
        if by_subject:
            return _empty_like(next(iter(by_subject.values())), params or {}, self.SUBJECT_KEYS)
        canned = self.responses.get(qualified_name)
        if isinstance(canned, dict):
            return _empty_like(canned, params or {}, self.SUBJECT_KEYS)
        return {"rows": [], "count": 0}

    def _asks_about(self, event: str, params: dict[str, Any] | None) -> bool:
        """True when the call named this event, or named none at all.

        A call with no event still gets the series: the capability requires one in production,
        and refusing here would turn a fixture gap into a silent empty result.
        """
        asked = (params or {}).get("event")
        return not isinstance(asked, str) or asked == event

    #: Payload fields holding the rows a series or listing returned.
    _ROW_KEYS = ("series", "rows", "pull_requests", "deployments", "events", "messages")

    def _for_subject(
        self, response: dict[str, Any], params: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The planted response, or an empty one when it is about something else entirely.

        **The third and last layer of one bug.** A fixture is keyed by `tool__capability` and was
        ignoring the parameters that decide what the answer should be. Fixed at the discovery
        layer, so the analyst can now learn the right event name -- and then it asked for two
        names, got byte-identical payloads, and the sufficiency gate correctly refused the report:
        *"two different event names ... returning identical payloads suggests a tracking/query
        issue, not independent signals"*. It was right. They were not independent signals; they
        were the same fixture answering twice.

        A real connector asked about an event that has no data returns an empty series, not
        somebody else's. So that is what this returns: the planted response's own scalar metadata,
        its rows emptied, and the subject set to what was actually requested. Emptying rather than
        erroring is deliberate -- "nothing here" is a real observation an analyst should be able
        to make, and it is what makes a decoy event *disprovable* rather than unanswerable.
        """
        if not params:
            return response
        for key in self.SUBJECT_KEYS:
            planted, asked = response.get(key), params.get(key)
            if not isinstance(planted, str) or not isinstance(asked, str) or planted == asked:
                continue
            empty = {k: v for k, v in response.items() if k not in self._ROW_KEYS}
            empty[key] = asked
            for row_key in self._ROW_KEYS:
                if row_key in response:
                    empty[row_key] = []
            for count_key in ("row_count", "count", "total", "total_matching"):
                if count_key in response:
                    empty[count_key] = 0
            return empty
        return response

    def _event_listing(self) -> dict[str, Any]:
        """Every event this scenario can answer a trend for, plus whatever else it advertises.

        **The bug this fixes, and it is the same bug twice.** In all six scenarios the event the
        fixture plants was *not in the catalogue the analyst is shown*. It could not ask for the
        right name, so it picked a plausible one -- `signup_completed`, `$pageview_0` -- and
        `response_for` served the planted series anyway, because a fixture is keyed by capability
        and ignores the parameters that decide what the answer should be.

        The analyst then described what it received in the terms it had asked for, which is the
        honest thing to do and was wrong. The verifier caught it: *"the evidence's actual `event`
        field is 'user signed up', not a pageview event as claimed"*. So a correct report lost a
        claim to the fixture's inconsistency -- and in `measurement_stopped` that claim was the
        second line of evidence, leaving the report resting on the single source that had stopped.

        Exactly the shape of the repository-listing bug above: discovery advertising a world the
        rest of the fixture cannot answer for. Derived rather than declared, for the same reason.

        Decoys are kept. A catalogue holding only the answer removes the choice, and
        `tempting_coincidence` deliberately ships ninety-six events whose one visible cluster is
        *not* the answer to its question.
        """
        planted = self.responses.get("posthog__list_events")
        base = planted or DISCOVERY_DEFAULTS["posthog__list_events"]
        events = list(base.get("events") or [])
        if planted is None and self.as_of is not None:
            # The generic catalogue carries one hardcoded date, so on every scenario whose world
            # ends earlier it advertised events that last fired *after* the data runs out --
            # 16 July on a scenario ending 30 June, and on one ending 28 May. `last_seen_at` is
            # the field an analyst reads to decide which event is live, and this made every decoy
            # look more recent than the event that has the data.
            #
            # A scenario that plants its own catalogue is left alone: `tempting_coincidence`
            # dates ninety-six events deliberately and the cluster among them is its whole
            # subject.
            events = [
                {**event, "last_seen_at": _last_seen(self.as_of, str(event.get("name") or ""))}
                for event in events
            ]
        advertised = {event.get("name") for event in events}
        for name in sorted(self.events_described() - advertised):
            # Dated from the scenario's own horizon rather than left blank: a blank reads as
            # "never seen", which would make the planted event look like the dead option.
            events.append({"name": name, "last_seen_at": _last_seen(self.as_of, name, live=True)})
        return {**base, "count": len(events), "events": events}

    def events_described(self) -> frozenset[str]:
        """Event names this scenario's own trend payloads claim to be about.

        Covers every field an event name can be planted in, not just `responses`. Each new one
        has reintroduced the bug this method exists to prevent: moving
        `partial_month_false_premise`'s series into `daily_truth` emptied this set, so the
        catalogue stopped advertising `user signed up` and the analyst was told the event it
        needed did not exist. A test asserts the invariant directly rather than trusting this
        list to stay complete.
        """
        found = set()
        sources = list(self.responses.values()) + [
            payload
            for by_subject in self.subject_responses.values()
            for payload in by_subject.values()
        ]
        for response in sources:
            if isinstance(response, dict) and isinstance(response.get("event"), str):
                found.add(response["event"])
        found.update(truth.event for planted in self.daily_truth.values() for truth in planted)
        return frozenset(found)

    def repositories_described(self) -> frozenset[str]:
        """Repositories this scenario's own payloads claim to be about.

        Public so a test can assert the invariant directly: everything described is advertised.
        """
        found = set()
        # Both layers that can carry a `repo`. `events_described` already read the subject
        # layer and this did not, which is the same asymmetry that has now broken a derived
        # invariant four times -- and the pull-request records moved into that layer today.
        sources = list(self.responses.values()) + [
            payload
            for by_subject in self.subject_responses.values()
            for payload in by_subject.values()
        ]
        for response in sources:
            if isinstance(response, dict) and isinstance(response.get("repo"), str):
                found.add(response["repo"])
        return frozenset(found)


@dataclass(frozen=True, slots=True)
class DailyTruth:
    """One event's daily counts, from which any requested interval is derived.

    **Why a fixture needs this at all.** `posthog.event_trend` takes an `interval` and defaults
    it to `"day"`. A canned payload cannot: it answers `interval="day"` with whatever buckets it
    was written with, and `partial_month_false_premise` was written with three monthly ones. So
    the analyst asked for daily granularity three times in run 26, was handed a monthly series
    each time, never saw a run-rate, and hedged the premise to "unverifiable" -- while the
    scenario's own comment called the daily call "the one that separates a real fall from a
    calendar artefact". The fixture was punishing the correct query.

    Deriving also removes the second copy. The monthly totals used to be typed out beside the
    daily rates they were supposed to summarise, which is two statements of one fact.
    """

    event: str
    #: Ordered `(day, count)`. The planted window; a request is clamped to it, so `end_date`
    #: reports where the data really stops rather than where the caller hoped it would.
    days: tuple[tuple[date, int], ...]

    #: The property this series can be broken down by, and how each day divides across it.
    #:
    #: The last thing a canned payload was still needed for. `onboarding_regression` planted a
    #: nine-day, per-device series by hand -- eighteen rows typed out -- and it answered
    #: `interval="week"` with daily buckets, because a canned payload answers with the
    #: granularity it was typed at. It was the only remaining entry on the open-violations list.
    #:
    #: Shares rather than counts, for the reason `Segment` gives: the day's total is stated
    #: once, in `days`, and the breakdown divides it.
    breakdown_property: str | None = None
    segments: tuple[Segment, ...] = ()


_BUCKET_STARTS: dict[str, Callable[[date], date]] = {
    "day": lambda d: d,
    "week": lambda d: d - timedelta(days=d.weekday()),
    "month": lambda d: d.replace(day=1),
}


@dataclass(frozen=True, slots=True)
class DatedRecords:
    """Records a capability returns, filtered by subject and window at request time.

    **ADR 0006, first landing.** A canned payload returns whatever it was typed with, so
    honouring `repo`, `since` and `until` is something each fixture must remember. Six defects
    came from that, and every guard written for one arrived a field too late for the next. Here
    the filtering is the only way to produce a response at all.

    Deliberately narrow. It covers one shape -- dated records belonging to a subject, bounded by
    a window -- which is what five GitHub capabilities and three message capabilities all are.
    It is not a general query engine, because eighteen capabilities cluster into five shapes and
    a parameterisation covering all of them would be harder to read than five small projections.

    `envelope` carries the fields the connector echoes back unchanged: the repository it was
    asked about, the environments that exist, whether pull requests were excluded. Those are what
    make "looked and found nothing" distinguishable from "nobody looked", which the decline twins
    depend on and which an empty list alone cannot say.
    """

    #: The payload key holding the list -- "commits", "deployments", "messages".
    key: str
    #: Each record's date field. Records outside the requested window are not returned.
    date_field: str
    records: tuple[dict[str, Any], ...] = ()
    #: Echoed back unchanged, so the shape matches the connector's even when nothing matches.
    envelope: dict[str, Any] = field(default_factory=dict)
    #: The subject this scenario planted, and the parameter naming it. A request for a different
    #: subject returns empty rather than these records -- the `disjoint` property, which four of
    #: the six defects violated.
    subject: str | None = None
    subject_param: str = "repo"
    #: Which parameters bound the window. `until` is absent on capabilities that take only a
    #: lower bound, which is most of them.
    since_param: str | None = "since"
    until_param: str | None = "until"

    #: When set, the subject parameter is a *search term* matched as a substring against these
    #: record fields rather than compared for equality. Slack's `query` and PostHog's `search`
    #: are searches: asking for "campaign" must match a message containing it, not one equal to
    #: it, and treating a search as an exact match would return nothing for every real query.
    search_fields: tuple[str, ...] = ()

    #: What the connector calls its counts. PostHog says `annotation_count` and `total_available`
    #: where GitHub and Slack say `count` and `total_matching`. Declared rather than assumed,
    #: because a fixture whose envelope differs from the connector's is drift the analyst reads.
    count_key: str = "count"
    #: Matches *before* the limit is applied, where the connector reports one. None where it
    #: does not -- inventing the field would be its own kind of infidelity.
    total_key: str | None = None


#: Date fields a record can carry, most specific first. A projection filters on whichever one
#: the records actually use; connectors are not consistent about it and there is no reason they
#: should be.
_DATE_FIELDS = ("date", "created_at", "merged_at", "submitted_at", "timestamp", "ts", "date_marker")

#: Window parameters, by the names each connector gives them. GitHub says since/until, Slack says
#: after/before, and a capability taking neither filters on subject alone.
_WINDOW_PARAMS = (("since", "until"), ("after", "before"))

#: The parameter naming a capability's subject, and whether it is an exact match or a search.
_SUBJECT_PARAMS = {"repo": False, "query": True, "topic": True, "search": True}


#: Every GA4 capability a `MetricSeries` can answer for. The set is closed deliberately: a
#: capability listed here is one whose numbers are *derived* from the daily series, so a
#: scenario cannot plant a second, contradicting figure for it.
GA4_SERIES_CAPABILITIES = frozenset(
    {
        "ga4__get_sessions",
        "ga4__compare_periods",
        "ga4__get_funnel",
        "ga4__top_pages",
    }
)


@dataclass(frozen=True, slots=True)
class Segment:
    """One value of a breakdown dimension, and how it behaves across the series.

    A segment states two things about itself -- what fraction of the day's sessions it takes,
    and what fraction of those convert -- and optionally that both change on one date. It never
    states a session count, because the daily series already does.
    """

    value: str
    share: float
    #: None where the series being divided has no conversions to speak of -- a `DailyTruth`
    #: breakdown divides an event count, and an event count does not convert.
    conversion_rate: float | None = None
    #: The day the segment changes. A shift is what makes a scenario's story: paid search
    #: losing its share, mobile losing its conversion rate.
    shifts_on: date | None = None
    share_after: float | None = None
    conversion_rate_after: float | None = None

    def at(self, day: date) -> tuple[float, float | None]:
        """This segment's share and conversion rate on `day`."""
        shifted = self.shifts_on is not None and day >= self.shifts_on
        return (
            self.share_after if shifted and self.share_after is not None else self.share,
            self.conversion_rate_after
            if shifted and self.conversion_rate_after is not None
            else self.conversion_rate,
        )


@dataclass(frozen=True, slots=True)
class PageShare:
    """One page's traffic, stated as a rate against sessions rather than a view count.

    A declared view count is a second statement of the same fact the session series makes, and
    it answers every window with one number. A rate answers the window it is asked about.
    """

    path: str
    views_per_session: float
    engagement_rate: float


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """A scenario's GA4 world: one daily session series, and how it divides.

    **The defect this type removes.** Every scenario declared its session count three times --
    once as a daily series under `get_sessions`, once as a device breakdown under `get_funnel`,
    once as a period comparison under `compare_periods` -- and the three disagreed. On
    `campaign_traffic_drop` the series said 13,239 sessions in the current period while the
    comparison said 11,300 and the funnel agreed with the comparison: a 15% contradiction
    between two payloads describing the same sixteen days, both of which an analyst would cite.
    Each payload also declared a `totals.sessions` its own rows contradicted, by 13% on
    `onboarding_regression` and 30% on `measurement_stopped`.

    None of that is findable by inspection and none of it is a bug in the analyst. It is what
    happens when one fact is written down four times.

    Here it is written once. `daily` is the only place a session count exists; every capability
    is a projection of it, so `get_funnel` summing to something other than `get_sessions` over
    the same window is not a defect to catch but a state that cannot be represented.

    The breakdowns are shares rather than counts for the same reason, and the last segment of
    each takes the rounding remainder so a day's split adds up to the day exactly.
    """

    #: (day, sessions). The scenario's one statement of how much traffic there was.
    daily: tuple[tuple[date, int], ...]
    #: Dimension name -> its segments. Shares within one dimension must sum to 1.
    breakdowns: dict[str, tuple[Segment, ...]] = field(default_factory=dict)
    #: Pages, for `top_pages`. Empty means this scenario does not answer that capability.
    pages: tuple[PageShare, ...] = ()
    property_id: str = "123456789"

    @property
    def capabilities(self) -> frozenset[str]:
        """Which GA4 capabilities this series can answer, given what it declares.

        Read by `Scenario.planted_capabilities`, so a series that declares no pages does not
        advertise `top_pages` -- the connected-tools inference and the reachability guard both
        read that set, and a capability advertised with nothing behind it is the defect
        `TestEveryConnectorAScenarioOffersIsLoadBearing` exists to catch.
        """
        offered = {"ga4__get_sessions", "ga4__compare_periods"}
        if self.breakdowns:
            offered.add("ga4__get_funnel")
        if self.pages:
            offered.add("ga4__top_pages")
        return frozenset(offered)

    @property
    def default_dimension(self) -> str | None:
        return next(iter(self.breakdowns), None)

    def window(
        self, params: dict[str, Any] | None, start_key: str, end_key: str
    ) -> tuple[date, date]:
        """The requested window. An unasked bound means the whole series.

        A blanked series has no days to fall back on, so an unasked bound resolves to a window
        that selects nothing rather than raising -- an emptied GA4 world still has to answer.
        """
        asked = params or {}
        first = self.daily[0][0] if self.daily else date.max
        last = self.daily[-1][0] if self.daily else date.min
        return _as_date(asked.get(start_key)) or first, _as_date(asked.get(end_key)) or last


#: Subject keys that name a *thing* rather than a filter. A missing one has to answer in the
#: planted shape with the other thing's details removed; a missing event or repository is
#: already handled by emptying the rows, because the scalars there describe the query rather
#: than the subject.
_ENTITY_KEYS = frozenset({"number"})

#: Keys whose value is a count of the rows beside them, so an emptied payload has to zero them
#: rather than null them. Shared by `_for_subject` and `_empty_like`.
_COUNT_KEYS = ("row_count", "count", "total", "total_matching")


def _without_rows(payload: dict[str, Any]) -> dict[str, Any]:
    """The same subject, with nothing recorded against it.

    Distinct from `_empty_like`, and the distinction is the twin's whole perturbation. A pull
    request that exists and has no review thread is one observation; a pull request number that
    matches nothing is another. The twin needs the first: the change is still visible, and what
    is gone is the human record that explained it.
    """
    emptied = dict(payload)
    for name, value in payload.items():
        if isinstance(value, list):
            emptied[name] = []
        elif name in _COUNT_KEYS:
            emptied[name] = 0
    return emptied


def _empty_like(
    payload: dict[str, Any], asked: dict[str, Any], keys: tuple[str, ...]
) -> dict[str, Any]:
    """The planted shape, describing nothing, for a record this scenario does not have.

    Every field that described the planted record is nulled rather than carried over: a payload
    that kept `title` and `author` while changing `number` would be a *worse* answer than the
    wrong one, because it reads as a real record of the thing that was asked about.

    What is echoed instead is the request's own subject -- the repository and the number it
    named -- so the answer says which lookup came back empty. Two lookups that found nothing
    must still be distinguishable from each other, or "nothing here" degenerates into one
    payload that answers everything.

    The shape survives because the shape is the honest part. A connector asked for a pull
    request with no review activity returns the envelope with empty lists, and an analyst can
    tell that from "I could not look" -- the distinction this whole fixture layer keeps having
    to preserve.
    """
    emptied: dict[str, Any] = {}
    for name, value in payload.items():
        if name in keys:
            emptied[name] = asked.get(name)
        elif isinstance(value, list):
            emptied[name] = []
        elif isinstance(value, dict):
            emptied[name] = {}
        elif name in _COUNT_KEYS:
            emptied[name] = 0
        elif isinstance(value, bool):
            emptied[name] = False
        else:
            emptied[name] = None
    return emptied


def _last_seen(horizon: date | None, name: str, *, live: bool = False) -> str:
    """When a catalogue says an event last fired, on the last day the scenario's world has data.

    The time of day is spread across that day by the event's own name rather than fixed, and
    neither of the two things this does is cosmetic.

    A catalogue in which every event last fired at exactly midnight reads as a *synchronised
    pipeline stop* -- the other misreading this field keeps inviting, and `measurement_stopped`
    exists to make one real instance of it findable. A live project reports each event a few
    hours apart.

    And an event the scenario can actually answer for is dated in the evening, after the decoys.
    Spreading everything through one window put `campaign_traffic_drop`'s planted `user signed
    up` at 09:54 with all three decoys later, so the answerable event looked like the stalest
    thing in the catalogue -- the same defect as the impostor name, in miniature. An analyst
    picks the live event by recency because that is what the field is for, so the live one has
    to be the most recent.

    Deterministic in the name, so a catalogue does not change between runs.
    """
    if horizon is None:
        return ""
    first, span = (18, 5 * 60) if live else (5, 12 * 60)
    minutes = sum(ord(character) for character in name) % span
    stamp = datetime(horizon.year, horizon.month, horizon.day, first, 0, tzinfo=UTC) + timedelta(
        minutes=minutes
    )
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _split(segments: tuple[Segment, ...], day: date, sessions: int) -> list[tuple[Segment, int]]:
    """A day's sessions divided across one dimension's segments.

    The last segment takes the remainder rather than its own rounded share, so the parts sum to
    `sessions` exactly. A breakdown that does not add up to its own total is the contradiction
    this module exists to remove, and rounding is enough to create one.
    """
    allocated = 0
    split: list[tuple[Segment, int]] = []
    for index, segment in enumerate(segments):
        share, _ = segment.at(day)
        part = sessions - allocated if index == len(segments) - 1 else round(sessions * share)
        allocated += part
        split.append((segment, part))
    return split


def _daily_conversions(series: MetricSeries, day: date, sessions: int) -> float | None:
    """A day's conversions, summed over the default breakdown.

    Undimensioned conversions are computed from the same split a dimensioned request gets, so
    the two agree by construction rather than by two authors agreeing.
    """
    segments = series.breakdowns.get(series.default_dimension or "")
    if not segments:
        return None
    total = 0.0
    for segment, part in _split(segments, day, sessions):
        _, rate = segment.at(day)
        if rate is None:
            return None
        total += part * rate
    return total


def _series_rows(
    series: MetricSeries, dimension: str | None, start: date, end: date
) -> list[dict[str, Any]]:
    """The fact table at its finest grain: one row per (date, segment) inside the window."""
    segments = series.breakdowns.get(dimension) if dimension else None
    rows: list[dict[str, Any]] = []
    for day, sessions in series.daily:
        if day < start or day > end:
            continue
        if not segments:
            metrics: dict[str, Any] = {"sessions": sessions}
            conversions = _daily_conversions(series, day, sessions)
            if conversions is not None:
                metrics["conversions"] = conversions
            rows.append({"dimensions": {"date": day.isoformat()}, "metrics": metrics})
            continue
        for segment, part in _split(segments, day, sessions):
            _, rate = segment.at(day)
            metrics = {"sessions": part}
            if rate is not None:
                metrics["conversions"] = part * rate
            rows.append(
                {
                    "dimensions": {"date": day.isoformat(), dimension: segment.value},
                    # Unrounded: these rows are an internal grain, and rounding each of sixty
                    # of them before summing put the window's conversion rate 2.4% off the rate
                    # that produced it -- enough to make a scenario whose whole claim is "the
                    # rate held flat" report a rate that moved.
                    "metrics": metrics,
                }
            )
    return rows


def _rounded(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round the aggregate, having summed the exact parts. GA4 reports whole conversions."""
    for row in rows:
        conversions = row["metrics"].get("conversions")
        if isinstance(conversions, float):
            row["metrics"]["conversions"] = round(conversions)
    return rows


def _group_rows(rows: list[dict[str, Any]], by: list[str]) -> list[dict[str, Any]]:
    """Aggregate fact rows to the dimensions asked for, summing every metric."""
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple((row.get("dimensions") or {}).get(name) for name in by)
        bucket = grouped.setdefault(
            key, {"dimensions": dict(zip(by, key, strict=True)), "metrics": {}}
        )
        for metric, value in (row.get("metrics") or {}).items():
            if isinstance(value, int | float):
                bucket["metrics"][metric] = bucket["metrics"].get(metric, 0) + value
    return list(grouped.values())


def _with_rate(row: dict[str, Any]) -> dict[str, Any]:
    """The conversion rate, computed after aggregation.

    A rate cannot be summed, so it is derived from the two extensive metrics once they have
    been. Mirrors what the connector does for the same reason: a rate carried through an
    aggregation is a rate for some other window.
    """
    sessions = row["metrics"].get("sessions")
    conversions = row["metrics"].get("conversions")
    rate = round(conversions / sessions, 6) if sessions and conversions is not None else None
    row["metrics"]["sessionConversionRate"] = rate
    return row


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The window's totals, summed from the rows that make it up.

    Rates are excluded from the sum and recomputed: adding two conversion rates together
    produces a number that is not a rate at all, and a `totals` block declaring one is the same
    class of defect as a declared total that disagrees with its own rows.
    """
    totals: dict[str, float] = {}
    for row in rows:
        for metric, value in (row.get("metrics") or {}).items():
            if isinstance(value, int | float) and not metric.endswith("Rate"):
                totals[metric] = totals.get(metric, 0) + value
    summed = {
        name: round(value, 6) if isinstance(value, float) else value
        for name, value in totals.items()
    }
    sessions, conversions = summed.get("sessions"), summed.get("conversions")
    if sessions and conversions is not None:
        summed["sessionConversionRate"] = round(conversions / sessions, 6)
    return summed


def _asked_dimensions(params: dict[str, Any] | None, key: str = "dimensions") -> list[str]:
    asked = (params or {}).get(key)
    return [name for name in asked if isinstance(name, str)] if isinstance(asked, list) else []


def _breakdown_for(series: MetricSeries, asked: list[str]) -> tuple[str | None, bool]:
    """Which declared breakdown answers this request, and whether one was asked for in vain.

    A dimension this world does not record answers empty rather than silently by date: returning
    a date series to a request for a country breakdown is answering a question that was not
    asked, which is the whole family of defect ADR 0006 is closing.
    """
    wanted = [name for name in asked if name != "date"]
    known = [name for name in wanted if name in series.breakdowns]
    return (known[0] if known else None, bool(wanted) and not known)


def _ga4_get_sessions(series: MetricSeries, params: dict[str, Any] | None) -> dict[str, Any]:
    start, end = series.window(params, "start_date", "end_date")
    asked = _asked_dimensions(params)
    dimension, unanswerable = _breakdown_for(series, asked)
    by = asked or ["date"]
    rows = (
        []
        if unanswerable
        else _rounded(_group_rows(_series_rows(series, dimension, start, end), by))
    )
    for row in rows:
        # `get_sessions` is a traffic capability; conversions belong to the funnel. Dropped
        # after aggregation rather than never computed, so the two capabilities still share
        # one split.
        row["metrics"].pop("conversions", None)
    return {
        "property_id": series.property_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "dimensions": asked,
        "metrics": ["sessions"],
        "totals": _totals(rows),
        "row_count": len(rows),
        "rows": rows,
    }


def _ga4_get_funnel(series: MetricSeries, params: dict[str, Any] | None) -> dict[str, Any]:
    start, end = series.window(params, "start_date", "end_date")
    asked = (params or {}).get("dimension")
    # The connector's own default when the caller omits it. Falling back to whichever
    # breakdown the scenario happens to declare first would answer a channel question with a
    # device breakdown on any scenario that declares both.
    fallback = "deviceCategory" if "deviceCategory" in series.breakdowns else None
    dimension = asked if isinstance(asked, str) else (fallback or series.default_dimension)
    rows = (
        _rounded(_group_rows(_series_rows(series, dimension, start, end), [dimension]))
        if dimension in series.breakdowns
        else []
    )
    for row in rows:
        _with_rate(row)
        # The connector recomputes this rather than trusting the reported rate, and the
        # analyst reads it. A fixture that omits it hides the field the scenario turns on.
        row["derived_conversion_rate"] = row["metrics"]["sessionConversionRate"]
    return {
        "property_id": series.property_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "dimension": dimension,
        "metrics": ["sessions", "conversions", "sessionConversionRate"],
        "totals": _totals(rows),
        "row_count": len(rows),
        "rows": rows,
    }


def _ga4_compare_periods(series: MetricSeries, params: dict[str, Any] | None) -> dict[str, Any]:
    current_start, current_end = series.window(params, "current_start", "current_end")
    previous_start, previous_end = series.window(params, "previous_start", "previous_end")
    asked = _asked_dimensions(params)
    dimension, unanswerable = _breakdown_for(series, asked)
    metrics = _asked_dimensions(params, "metrics") or ["sessions", "conversions"]
    by = asked

    def _side(start: date, end: date) -> dict[tuple[Any, ...], dict[str, Any]]:
        if unanswerable:
            return {}
        rows = [
            _with_rate(row)
            for row in _rounded(_group_rows(_series_rows(series, dimension, start, end), by))
        ]
        return {tuple(row["dimensions"].get(name) for name in by): row for row in rows}

    current = _side(current_start, current_end)
    previous = _side(previous_start, previous_end)
    comparison = []
    for key in sorted(set(current) | set(previous), key=lambda k: tuple(str(part) for part in k)):
        current_row = current.get(key, {}).get("metrics", {})
        previous_row = previous.get(key, {}).get("metrics", {})
        entry: dict[str, Any] = {"dimensions": dict(zip(by, key, strict=True))}
        for metric in metrics:
            now, before = current_row.get(metric), previous_row.get(metric)
            entry[metric] = {
                "current": now,
                "previous": before,
                # The connector's own arithmetic, imported rather than reimplemented: a
                # fixture that computes percentage change its own way is a second
                # implementation, and the contract test would be comparing two of them.
                "absolute_change": _absolute(now, before),
                "percent_change": _percent(now, before),
            }
        comparison.append(entry)
    payload = {
        "property_id": series.property_id,
        "current_period": {"start": current_start.isoformat(), "end": current_end.isoformat()},
        "previous_period": {"start": previous_start.isoformat(), "end": previous_end.isoformat()},
        "metrics": metrics,
        "dimensions": asked,
        "row_count": len(comparison),
        "comparison": comparison,
    }
    return payload


def _coverage_for(series: MetricSeries, params: dict[str, Any] | None) -> dict[str, Any]:
    """The connector's own coverage check, over the days this series actually holds.

    Computed through the same function production calls, not hand-written: a fixture that states
    its own disclosure passes whether or not the connector computes one, which is the mistake
    `measurement_stopped` exists to avoid making twice.

    Separate from the projection because it is a *disclosure*, and a scenario is allowed to
    withhold those. `partial_month_false_premise` sets `compute_disclosures=False` precisely so
    that noticing a truncated month stays a skill it measures rather than a note it hands over.
    """
    current_start, current_end = series.window(params, "current_start", "current_end")
    previous_start, previous_end = series.window(params, "previous_start", "previous_end")
    return (
        _window_coverage(
            [{"dimensions": {"date": day.isoformat()}} for day, _ in series.daily],
            {
                "current": (current_start.isoformat(), current_end.isoformat()),
                "previous": (previous_start.isoformat(), previous_end.isoformat()),
            },
        )
        or {}
    )


def _ga4_top_pages(series: MetricSeries, params: dict[str, Any] | None) -> dict[str, Any]:
    start, end = series.window(params, "start_date", "end_date")
    sessions = sum(count for day, count in series.daily if start <= day <= end)
    rows = [
        {
            "dimensions": {"pagePath": page.path},
            "metrics": {
                "screenPageViews": round(sessions * page.views_per_session),
                "engagementRate": page.engagement_rate,
            },
        }
        for page in series.pages
    ]
    # Sorted, because the capability is named for it. The canned payloads were not, so
    # `top_pages` returned its second-largest page first.
    rows.sort(key=lambda row: row["metrics"]["screenPageViews"], reverse=True)
    return {
        "property_id": series.property_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "metrics": ["screenPageViews", "engagementRate"],
        "row_count": len(rows),
        "rows": rows,
    }


#: Which projection answers which capability. One dispatch, so a capability added to
#: `GA4_SERIES_CAPABILITIES` without a projection fails loudly rather than falling through to
#: the canned lookup it was supposed to replace.
_GA4_PROJECTIONS: dict[str, Callable[[MetricSeries, dict[str, Any] | None], dict[str, Any]]] = {
    "ga4__get_sessions": _ga4_get_sessions,
    "ga4__compare_periods": _ga4_compare_periods,
    "ga4__get_funnel": _ga4_get_funnel,
    "ga4__top_pages": _ga4_top_pages,
}


def _auto_records(capability: str, payload: dict[str, Any]) -> DatedRecords | None:
    """A projection inferred from a canned payload's own shape, or None if it is not this shape.

    **Why inferred rather than hand-written twenty-seven times.** Migrating each payload by hand
    is twenty-seven chances to make the mistake the migration exists to remove -- and the last
    one made it: a twin blanked a layer that had stopped being consulted. Inference reads what
    the payload already says, and the metamorphic properties assert the result behaves, so a
    wrong inference fails a test rather than passing quietly.

    Returns None where the shape is genuinely ambiguous rather than guessing. Two cases, both
    real: a payload with several dict-bearing lists (`pull_request_activity` carries reviews and
    comments, and which one is *the* result is not recoverable from the shape), and a payload
    whose records carry no recognisable date. Those stay canned and are listed as open debt.
    """
    from cortex.tools.registry import gtm_analyst_registry

    # A list of strings is metadata, not the result: `environments_available` says which
    # environments exist. Excluded by content where there is any, and by name where the list is
    # empty and content cannot say -- an empty payload is the common case, because a scenario
    # planting "nothing happened here" is how half of them establish it.
    listed = {
        key: value
        for key, value in payload.items()
        if isinstance(value, list)
        and not any(isinstance(item, str) for item in value)
        and not key.endswith("_available")
    }
    if len(listed) != 1:
        return None
    key, records = next(iter(listed.items()))

    # An empty result has no date to infer from and needs none: nothing survives any window.
    # The field is still declared so the projection is well-formed if records are added later.
    dated = next((f for f in _DATE_FIELDS if any(f in record for record in records)), None)
    if dated is None and records:
        return None
    dated = dated or "date"

    tool_name, capability_name = capability.split("__", 1)
    try:
        schema = gtm_analyst_registry().get(tool_name).capability(capability_name).params_schema
    except (KeyError, AttributeError):  # pragma: no cover - registry drift
        return None
    accepted = set((schema or {}).get("properties", {}))

    since = until = None
    for lower, upper in _WINDOW_PARAMS:
        if lower in accepted:
            since, until = lower, (upper if upper in accepted else None)
            break

    subject_param = next((p for p in _SUBJECT_PARAMS if p in accepted), "repo")
    searches = _SUBJECT_PARAMS.get(subject_param, False)
    text_fields = tuple(
        f
        for f in ("text", "content", "title", "subject", "channel", "channel_name")
        if any(f in record for record in records)
    )

    envelope = {
        name: value
        for name, value in payload.items()
        if name != key
        and name not in {"count", "total_matching", "total_available", "annotation_count"}
        and name not in {subject_param, since, until}
    }
    return DatedRecords(
        key=key,
        date_field=dated,
        records=tuple(records),
        envelope=envelope,
        subject=payload.get(subject_param) if not searches else None,
        subject_param=subject_param,
        since_param=since,
        until_param=until,
        search_fields=text_fields if searches else (),
        count_key="annotation_count" if "annotation_count" in payload else "count",
        total_key=next((t for t in ("total_matching", "total_available") if t in payload), None),
    )


def _matches_search(query: str, record: dict[str, Any], fields: tuple[str, ...]) -> bool:
    """Whether a record answers a search, matched by terms rather than as one string.

    **Substring matching was impersonating search, and it cost a scenario its cause.**
    `campaign_traffic_drop` plants one Slack message -- *"spring campaign budget is exhausted,
    pausing ads today"* -- and the whole query string had to appear in it contiguously. So
    `budget exhausted` found nothing, because the message says "budget **is** exhausted". An
    investigation made four Slack searches, every one came back empty, and the sufficiency gate
    correctly withheld the cause on the grounds that no dated record explained why paid search
    stopped. The record was there. The fixture was answering "is this exact phrase present?" to
    a question that asked "which messages are about these terms?".

    Every term must appear, which is what a real search connector does and what keeps the search
    a real step: a query of unrelated words still returns nothing, so the planted message is
    findable rather than handed over.
    """
    haystack = " ".join(str(record.get(field, "")) for field in fields).lower()
    words = {_stem(word) for word in re.split(r"[^\w]+", haystack) if word}
    terms = [term for term in re.split(r"[^\w]+", query.lower()) if term]
    return all(_matches_term(term, haystack, words) for term in terms) if terms else True


#: How much of two words has to agree before they count as the same word. Four characters, which
#: is enough to separate "pause"/"pausing" from "pause"/"paid".
_ROOT = 4


def _matches_term(term: str, haystack: str, words: set[str]) -> bool:
    """Whether one search term is answered by a record's text.

    Slack's own search stems, and this fixture stands in for Slack's search -- so "ads paused"
    against a message reading "pausing ads today" has to match, or the fixture is stricter than
    the thing it simulates and the difference shows up as evidence that does not exist.

    Stemming alone was not enough: "pause" strips to "pause" and "pausing" to "paus", so the two
    still missed each other. Either root prefixing the other is what closes that, with a
    four-character floor so short words match exactly rather than promiscuously.
    """
    if term in haystack:
        return True
    root = _stem(term)
    return any(
        min(len(word), len(root)) >= _ROOT and (word.startswith(root) or root.startswith(word))
        for word in words
    )


def _stem(word: str) -> str:
    """A crude suffix strip. Length-guarded, so short words are left alone."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _project_records(planted: DatedRecords, params: dict[str, Any] | None) -> dict[str, Any]:
    """The records matching this request, in the connector's envelope.

    Empty is a real observation and must stay readable: the envelope survives, `count` is zero,
    and the caller can tell "this repository has no commits in that window" from "no repository
    was named".
    """
    asked = params or {}
    matching = list(planted.records)

    subject_asked = asked.get(planted.subject_param)
    if planted.search_fields:
        # A search: empty or absent returns everything, which is what these connectors do.
        if isinstance(subject_asked, str) and subject_asked.strip():
            matching = [
                record
                for record in matching
                if _matches_search(subject_asked, record, planted.search_fields)
            ]
    elif planted.subject is not None and isinstance(subject_asked, str):
        if subject_asked != planted.subject:
            matching = []

    since = _as_date(asked.get(planted.since_param)) if planted.since_param else None
    until = _as_date(asked.get(planted.until_param)) if planted.until_param else None
    if since or until:
        matching = [
            record
            for record in matching
            if (dated := _as_date(record.get(planted.date_field))) is None
            or ((since is None or dated >= since) and (until is None or dated <= until))
        ]

    before_limit = len(matching)
    limit = asked.get("limit")
    if isinstance(limit, int) and limit > 0:
        matching = matching[:limit]

    echoed = {
        planted.subject_param: subject_asked if subject_asked is not None else planted.subject,
        **planted.envelope,
    }
    if planted.since_param:
        echoed[planted.since_param] = asked.get(planted.since_param)
    if planted.until_param:
        echoed[planted.until_param] = asked.get(planted.until_param)
    if planted.total_key:
        echoed[planted.total_key] = before_limit
    return {**echoed, planted.count_key: len(matching), planted.key: matching}


def _resample(
    truth: DailyTruth, params: dict[str, Any] | None, *, matched: bool = True
) -> dict[str, Any]:
    """A trend payload at the requested interval, over the requested window.

    Clamped to the planted days at both ends, which is what makes the artefact *findable*: an
    analyst asking for 1 July to 26 August is told the series ends on 12 August, and counting
    the buckets it got back is then enough. Nothing here announces that the last month is
    partial -- `partial_month_false_premise` deliberately withholds that disclosure, and this
    keeps the withholding while making the honest calculation reachable.

    `matched=False` is a call about some other event: an empty series, but keeping the interval
    and the window, because an empty payload that also lost its scalars says "nothing here"
    about an unknown question rather than about the one that was asked. The default interval is
    `posthog.event_trend`'s own -- a fixture that echoes a different one is answering as a
    connector this project does not have.
    """
    asked = params or {}
    interval = asked.get("interval") or "day"
    bucket_of = _BUCKET_STARTS.get(interval, _BUCKET_STARTS["day"])
    start = _as_date(asked.get("start_date"))
    end = _as_date(asked.get("end_date"))
    days = (
        [
            (day, value)
            for day, value in truth.days
            if (start is None or day >= start) and (end is None or day <= end)
        ]
        if matched
        else []
    )
    # Segmented only when the caller asks for it, and only for the property this series can
    # actually divide by. PostHog returns a flat series otherwise, and a fixture that always
    # segmented would be answering a question that was not asked.
    wanted = asked.get("breakdown_property")
    segmented = bool(truth.segments) and wanted == truth.breakdown_property and wanted is not None

    buckets: dict[tuple[date, str | None], int] = {}
    for day, value in days:
        key = bucket_of(day)
        if not segmented:
            buckets[(key, None)] = buckets.get((key, None), 0) + value
            continue
        # Split the *day* and then bucket, so a weekly request aggregates each segment over its
        # own days rather than dividing a week's total by a single day's shares.
        for segment, part in _split(truth.segments, day, value):
            buckets[(key, segment.value)] = buckets.get((key, segment.value), 0) + part
    planted = [day for day, _ in truth.days]
    _order = {segment.value: index for index, segment in enumerate(truth.segments)}
    return {
        # Named for what was asked, so the analyst is not left inferring which event it holds.
        "event": asked.get("event") if not matched else truth.event,
        "measure": "count",
        "interval": interval,
        # Falls back to the requested window, then to the planted one, so an empty result still
        # reports the range it found nothing in.
        "start_date": (days[0][0] if days else start or planted[0]).isoformat(),
        "end_date": (days[-1][0] if days else end or planted[-1]).isoformat(),
        "breakdown_property": wanted if segmented else None,
        "row_count": len(buckets),
        "series": [
            {"bucket": f"{key.isoformat()}T00:00:00", "value": value}
            if segment is None
            else {"bucket": f"{key.isoformat()}T00:00:00", "segment": segment, "value": value}
            # Declaration order within a bucket, not alphabetical: the scenario lists the
            # segment its story is about first, and an analyst reads the first row.
            for (key, segment), value in sorted(
                buckets.items(), key=lambda item: (item[0][0], _order.get(item[0][1], 0))
            )
        ],
        "total": sum(buckets.values()),
    }


def _as_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _daily(
    start: date,
    days: int,
    baseline: float,
    *,
    change_on: int | None = None,
    change_to: float | None = None,
    noise: float = 0.03,
    rng: random.Random,
) -> tuple[tuple[date, int], ...]:
    """A daily session series with optional step change and multiplicative noise.

    Noise matters: a perfectly flat series with one clean step is trivially readable,
    and would let a weak analyst score as well as a good one.

    Returns `(day, sessions)` pairs rather than GA4 rows, because the rows are a *view* of this
    -- one of four, and the other three used to be written out separately and disagree. See
    `MetricSeries`.
    """
    series = []
    for offset in range(days):
        level = baseline
        if change_on is not None and offset >= change_on and change_to is not None:
            level = change_to
        value = level * (1 + rng.uniform(-noise, noise))
        series.append((start + timedelta(days=offset), round(value)))
    return tuple(series)


def onboarding_regression(seed: int = 1) -> Scenario:
    """The doc's motivating case: signups fall, mobile onboarding is the cause.

    Two decoys, both as visible as the cause in the aggregate:
      - a pricing-page change shipped the same week
      - a paid campaign ended two days earlier, cutting traffic

    Only a device-level conversion breakdown plus the deploy timeline separates the
    real cause from either. An analyst that stops at the aggregate cannot tell them
    apart, which is the point.
    """
    rng = random.Random(seed)
    start = date(2026, 7, 1)
    #: The day the regression lands. Read by both the daily step and the mobile conversion
    #: shift, so the traffic series and the breakdown cannot pivot on different days.
    onset = start + timedelta(days=14)

    return Scenario(
        name="onboarding_regression",
        # Dated absolutely, not "last week". The planted data sits in July 2026, so a
        # relative question drifts away from it as the calendar moves: a run on 30 July
        # asked for 23-29 July, got the 15-21 July payload the fixture always serves, and
        # spent two steps reconciling the mismatch before reporting the discrepancy
        # honestly. The analyst behaved correctly; the question was measuring the fixture.
        question="Why did signups fall in the week of 15 July 2026?",
        difficulty=Difficulty.CONFOUNDED,
        ground_truth=GroundTruth(
            cause=(
                "The onboarding modal shipped in deploy 91c3e4a regressed mobile "
                "conversion from 4.2% to 2.9%."
            ),
            required_signals=(
                # The change, identified. The sha and the PR number name the same commit,
                # and an analyst that says "PR #913, merged 14 July, reworked the mobile
                # onboarding modal" has identified it exactly as precisely as one quoting
                # the sha -- so both count. Requiring the sha alone failed a report that
                # had the whole story right.
                ("91c3e4a", "913"),
                "mobile",
            ),
            decoys=("pricing page", "campaign", "seasonality"),
            required_capabilities=(
                # The breakdown that isolates the segment.
                "ga4__get_funnel",
                # How the change was dated. Either the deploy timeline or the commit list
                # answers "what shipped, and when"; the commit list arguably answers it
                # better, since a deploy record does not say what was in it.
                ("github__deployment_history", "github__commits"),
                # What a senior analyst does that a dashboard cannot: read the change
                # itself. The review thread on PR 913 names the mobile viewport, which is
                # the difference between "a deploy correlates with the drop" and "this
                # code did this". Requires the two-step a human makes -- list the PRs,
                # then open the one that matches -- so it is not free, and there is no
                # alternative route to it.
                "github__pull_request_activity",
            ),
        ),
        subject_responses={
            # The human record around the change: a reviewer raised the exact failure mode
            # before it shipped and was overruled on timing -- the kind of evidence no metric
            # contains, and the reason a senior analyst reads the thread.
            #
            # Keyed by pull-request number, because the capability is a lookup and the payload
            # this replaces returned pull request 913 whatever number was asked for. An analyst
            # opening the pricing decoy was handed the mobile onboarding modal's review thread
            # under the decoy's number -- on a *required* capability, so the scenario was
            # rewarding the wrong lookup with the right answer, and nothing in the scorecard
            # could show it.
            #
            # Both pull requests are described, which is the stronger fixture as well as the
            # correct one: the decoy becomes disprovable on its own evidence rather than
            # indistinguishable from the cause.
            "github__pull_request_activity": {
                "913": {
                    "repo": "acme/web",
                    "number": 913,
                    "title": "Rework mobile onboarding modal",
                    "state": "closed",
                    "merged_at": "2026-07-14T10:00:00Z",
                    "author": "dwhitfield",
                    "body": (
                        "Replaces the three-step onboarding modal with a single scrolling "
                        "sheet. Desktop unchanged."
                    ),
                    "reviews": [
                        {
                            "author": "sbeck",
                            "verdict": "CHANGES_REQUESTED",
                            "submitted_at": "2026-07-13T16:22:00Z",
                            "body": (
                                "The continue button sits below the fold on a 375px viewport "
                                "with the keyboard open -- on an iPhone SE you cannot reach "
                                "it. Needs a sticky footer before this goes out."
                            ),
                        },
                        {
                            "author": "dwhitfield",
                            "verdict": "APPROVED",
                            "submitted_at": "2026-07-14T09:50:00Z",
                            "body": "Shipping to hit the launch date; sticky footer to follow.",
                        },
                    ],
                    "comments": [
                        {
                            "author": "sbeck",
                            "created_at": "2026-07-14T09:58:00Z",
                            "body": (
                                "Merging without the footer fix, noted. Watch mobile signups."
                            ),
                        }
                    ],
                },
                # The pricing decoy, described honestly: a copy change nobody raised anything
                # about. What rules it out is that there is nothing here to rule in.
                "908": {
                    "repo": "acme/web",
                    "number": 908,
                    "title": "Update pricing page copy",
                    "state": "closed",
                    "merged_at": "2026-07-12T08:00:00Z",
                    "author": "praman",
                    "body": "Rewrites the three plan descriptions. No layout or form changes.",
                    "reviews": [
                        {
                            "author": "dwhitfield",
                            "verdict": "APPROVED",
                            "submitted_at": "2026-07-11T15:40:00Z",
                            "body": "Copy reads well.",
                        }
                    ],
                    "comments": [],
                },
            },
        },
        daily_truth={
            # The decisive product-side series, and the last canned payload in the suite that
            # could still answer a question it was not asked. It was eighteen rows typed out by
            # hand covering nine days, and it answered `interval="week"` with daily buckets.
            #
            # Extended to cover the same 21 days as the session series. The hand-written version
            # started on the 13th, and once the eval began computing the connectors' real
            # disclosures that narrowness became the scenario's answer: an analyst asking any
            # wider range was told "onboarding completed has no data after ..., the event may
            # have stopped firing", and a data-incident verdict outranks the real one.
            "posthog__event_trend": (
                DailyTruth(
                    event="onboarding completed",
                    days=tuple(
                        (
                            start + timedelta(days=offset),
                            round((85.0 if offset < 14 else 65.0) * (1 + rng.uniform(-0.04, 0.04))),
                        )
                        for offset in range(21)
                    ),
                    breakdown_property="$device_type",
                    # Mobile loses share on the day of the deploy and desktop absorbs it: in
                    # counts, mobile falls by a third and desktop does not move. The same fact
                    # the GA4 funnel states as a conversion rate, said by the product's own
                    # instrumentation -- two systems agreeing is what makes it evidence.
                    segments=(
                        Segment("Mobile", share=0.71, shifts_on=onset, share_after=0.615),
                        Segment("Desktop", share=0.29, shifts_on=onset, share_after=0.385),
                    ),
                ),
            )
        },
        metric_series=MetricSeries(
            daily=_daily(start, 21, 2400, change_on=14, change_to=1970, rng=rng),
            # The decisive breakdown: mobile keeps its share of the traffic and loses its
            # conversion rate, desktop keeps both. Stated as rates, so the funnel, the period
            # comparison and the session series are three views of one fact rather than three
            # figures that have to be kept in step -- they were not, and disagreed by 13%.
            breakdowns={
                "deviceCategory": (
                    Segment(
                        "mobile",
                        share=0.705,
                        conversion_rate=0.035,
                        shifts_on=onset,
                        conversion_rate_after=0.029,
                    ),
                    # Flat rate, falling count: desktop conversions drop with the traffic while
                    # the rate holds, so an analyst reading counts sees both segments fall and
                    # only one reading *rates* can name mobile. That discrimination is what the
                    # scenario measures.
                    Segment("desktop", share=0.295, conversion_rate=0.0353),
                ),
            },
            pages=(
                PageShare("/signup", views_per_session=0.712, engagement_rate=0.34),
                # The pricing decoy: heavily engaged, and nothing to do with the drop.
                PageShare("/pricing", views_per_session=0.225, engagement_rate=0.71),
            ),
        ),
        responses={
            # The timeline. Both the real cause and the pricing decoy appear, so the
            # deploy list alone does not give the answer away.
            #
            # The environment is deliberately **not** called "production". Every fixture
            # here used to call it that, and a live investigation against the real
            # repository filtered on environment="production", got nothing, and had to
            # reason around the silence -- the repository's environments are "dev-deploy"
            # and "staging - docs". The suite was teaching the analyst a name that does
            # not generalise, so it now teaches the opposite: read
            # `environments_available` rather than assuming what an environment is called.
            "github__deployment_history": {
                "repo": "acme/web",
                "environment": None,
                "count": 2,
                "environments_available": ["prod-web", "staging"],
                "note": None,
                "deployments": [
                    {
                        "id": 40122,
                        "sha": "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a",
                        "ref": "main",
                        "environment": "prod-web",
                        "created_at": "2026-07-14T11:04:00Z",
                        "state": "success",
                        "creator": "ci-bot",
                    },
                    {
                        "id": 40098,
                        "sha": "77aa21bcd3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8",
                        "ref": "main",
                        "environment": "prod-web",
                        "created_at": "2026-07-12T09:20:00Z",
                        "state": "success",
                        "creator": "ci-bot",
                    },
                ],
            },
            # What actually landed, which is not the same as what was deployed. Deploy
            # 91c3e4a carried two commits; only one of them touches onboarding, and the
            # other is a dependency bump -- so "the deploy did it" is still one step
            # short of an answer.
            "github__commits": {
                "repo": "acme/web",
                "since": "2026-07-12",
                "until": "2026-07-15",
                "path": None,
                "count": 3,
                "commits": [
                    {
                        "sha": "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a",
                        "short_sha": "91c3e4a",
                        "date": "2026-07-14T10:02:00Z",
                        "author": "Dana Whitfield",
                        "subject": "Rework mobile onboarding modal (#913)",
                        "url": "https://github.com/acme/web/commit/91c3e4a",
                    },
                    {
                        "sha": "a41b8c92de3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b",
                        "short_sha": "a41b8c9",
                        "date": "2026-07-14T09:41:00Z",
                        "author": "dependabot[bot]",
                        "subject": "Bump tailwindcss from 4.1.2 to 4.1.3",
                        "url": "https://github.com/acme/web/commit/a41b8c9",
                    },
                    {
                        "sha": "77aa21bcd3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8",
                        "short_sha": "77aa21b",
                        "date": "2026-07-12T08:00:00Z",
                        "author": "Priya Raman",
                        "subject": "Update pricing page copy (#908)",
                        "url": "https://github.com/acme/web/commit/77aa21b",
                    },
                ],
            },
            # Reported breakage, which predates the metric noticing. The pricing issue is
            # a decoy: it is real, it is the same week, and it has nothing to do with the
            # drop.
            "github__issues": {
                "repo": "acme/web",
                "since": "2026-07-14",
                "labels": None,
                "count": 2,
                "pull_requests_excluded": True,
                "issues": [
                    {
                        "number": 921,
                        "title": "Cannot finish signup on iPhone -- button off screen",
                        "state": "open",
                        "created_at": "2026-07-15T07:12:00Z",
                        "author": "external-user-44",
                        "labels": ["bug", "mobile", "onboarding"],
                        "comments": 4,
                        "body": (
                            "Since yesterday the signup sheet scrolls but the Continue "
                            "button never appears. Safari on iOS 18."
                        ),
                    },
                    {
                        "number": 919,
                        "title": "Typo in enterprise pricing tier",
                        "state": "closed",
                        "created_at": "2026-07-14T13:40:00Z",
                        "author": "priya",
                        "labels": ["docs"],
                        "comments": 1,
                        "body": "Says $49/mo in one place and $490/yr in another.",
                    },
                ],
            },
            # Product-side corroboration from a second, independent source. GA4 measures
            # the website; PostHog measures the product, and a story that holds in both
            # is much harder to explain away as instrumentation.
            "posthog__list_projects": {
                "project_count": 2,
                "default_project": "100001",
                "projects": [
                    {"id": "100001", "label": "web-app", "is_default": True},
                    {"id": "100002", "label": "oss-client", "is_default": False},
                ],
            },
            "github__recent_prs": {
                "repo": "acme/web",
                "pull_requests": [
                    {
                        "number": 913,
                        "title": "Rework mobile onboarding modal",
                        "merged_at": "2026-07-14T10:00:00Z",
                        "labels": ["onboarding", "mobile"],
                    },
                    {
                        "number": 908,
                        "title": "Update pricing page copy",
                        "merged_at": "2026-07-12T08:00:00Z",
                        "labels": ["marketing"],
                    },
                ],
            },
            # Rules the pricing decoy out: its own conversion did not move.
            "slack__find_decision": {
                "topic": "onboarding",
                "messages": [
                    {
                        "ts": "1784889840.000100",
                        "timestamp": "2026-07-15T09:14:00+00:00",
                        "user": "U123",
                        "text": "mobile signups look off since yesterday's modal change",
                        "channel_name": "growth",
                        "decision_signals": [],
                    }
                ],
            },
            # The campaign decoy: traffic did fall, but conversion is the story.
            "hubspot__contacts": {
                "start_date": "2026-07-15",
                "end_date": "2026-07-21",
                "by_lifecycle_stage": {"lead": 284},
                "contacts": [],
            },
        },
    )


def campaign_traffic_drop(seed: int = 2) -> Scenario:
    """A volume story, not a conversion story.

    Included specifically to punish an analyst that has learned "the answer is always
    a deploy". Conversion is flat; a campaign ended. The deploy timeline contains a
    deploy on the same day as the drop, so the wrong answer is readily available.
    """
    rng = random.Random(seed)
    start = date(2026, 6, 1)
    #: The day the campaign budget runs out. One date, read by both the traffic step and the
    #: channel shift, so the volume and the channel mix cannot pivot a day apart.
    onset = date(2026, 6, 15)

    return Scenario(
        name="campaign_traffic_drop",
        question="Why did signups drop in the second half of June?",
        difficulty=Difficulty.STRAIGHTFORWARD,
        ground_truth=GroundTruth(
            cause=(
                "Paid traffic fell after the spring campaign ended on 14 June. "
                "Conversion was unchanged; the drop is volume, not quality."
            ),
            required_signals=("campaign",),
            decoys=("deploy", "onboarding", "conversion rate"),
            required_capabilities=(
                "ga4__compare_periods",
                # The deploy on the day of the drop is the trap. Reading what was in it
                # -- a CI config change and a dependency bump, no user-facing code --
                # is how the trap is disarmed with evidence instead of with an argument
                # from flat conversion alone.
                "github__commits",
            ),
        ),
        # Signups, daily, so the interval the analyst asks for is the interval it gets.
        #
        # This was four weekly buckets, and a request for `interval="day"` received them --
        # the same defect `partial_month_false_premise` carried, fixed there and left here.
        # It cost real score twice in run 31: both the sufficiency gate and the verifier
        # objected that no signups series established the decline day by day, and on the
        # attempt that did retrieve this series they were describing four weekly points
        # spanning the very boundary the question turns on. The weekly sums are unchanged
        # within noise -- 402, 391, 236, 228 -- so a weekly call still sees what it saw.
        daily_truth={
            "posthog__event_trend": (
                DailyTruth(
                    event="user signed up",
                    days=tuple(
                        (day, round(rate * (1 + rng.uniform(-0.06, 0.06))))
                        for first, count, rate in (
                            (date(2026, 6, 1), 14, 56.6),
                            # The campaign ended on the 14th; volume steps down from the 15th.
                            (date(2026, 6, 15), 16, 33.2),
                        )
                        for day in (first + timedelta(days=offset) for offset in range(count))
                    ),
                ),
            )
        },
        # GitHub, projected rather than canned. The repository, the window and the limit are
        # honoured because filtering is the only way this produces a response -- which is what
        # ADR 0006 is for. The two commits are the decoy: a CI runner pin and a lockfile bump on
        # the day the campaign ended, real enough to tempt and disarming once read.
        dated_records={
            # Slack, searched rather than returned. The announcement is findable by any query
            # that names the campaign or the budget, and absent for one that does not -- which is
            # what makes "no record explains it" an observation the decline twin can rest on.
            "slack__search_messages": DatedRecords(
                key="messages",
                date_field="timestamp",
                subject_param="query",
                search_fields=("text", "channel_name"),
                since_param="after",
                until_param="before",
                total_key="total_matching",
                records=(
                    {
                        "ts": "1781000000.000100",
                        "timestamp": "2026-06-14T12:00:00+00:00",
                        "text": "spring campaign budget is exhausted, pausing ads today",
                        "channel_name": "marketing",
                    },
                ),
            ),
            "github__commits": DatedRecords(
                key="commits",
                date_field="date",
                subject="acme/web",
                envelope={"path": None},
                records=(
                    {
                        "sha": "beef123cafe4567890abcdef1234567890abcdef",
                        "short_sha": "beef123",
                        "date": "2026-06-14T15:50:00Z",
                        "author": "ops-bot",
                        "subject": "ci: pin runner image to ubuntu-24.04",
                        "url": "https://github.com/acme/web/commit/beef123",
                    },
                    {
                        "sha": "0f1e2d3c4b5a69788796a5b4c3d2e1f009182736",
                        "short_sha": "0f1e2d3",
                        "date": "2026-06-14T15:12:00Z",
                        "author": "dependabot[bot]",
                        "subject": "Bump @types/node from 22.9.0 to 22.9.1",
                        "url": "https://github.com/acme/web/commit/0f1e2d3",
                    },
                ),
            ),
            "github__deployment_history": DatedRecords(
                key="deployments",
                date_field="created_at",
                subject="acme/web",
                # Named as the repository actually names them. `environments_available` survives
                # an empty result on purpose: it is what tells an analyst the environment exists
                # and had no deployments, rather than that it was never asked about.
                envelope={
                    "environment": None,
                    "environments_available": ["prod-web", "staging"],
                    "note": None,
                },
                since_param=None,
                until_param=None,
                records=(
                    {
                        "id": 38771,
                        "sha": "beef123cafe4567890abcdef1234567890abcdef",
                        "ref": "main",
                        "environment": "prod-web",
                        "created_at": "2026-06-14T16:00:00Z",
                        "state": "success",
                        "creator": "ci-bot",
                    },
                ),
            ),
            "github__issues": DatedRecords(
                key="issues",
                date_field="created_at",
                subject="acme/web",
                envelope={"labels": None, "pull_requests_excluded": True},
                until_param=None,
            ),
        },
        metric_series=MetricSeries(
            daily=_daily(start, 30, 1400, change_on=14, change_to=820, rng=rng),
            # Two breakdowns of the same traffic. The channel split carries the story -- paid
            # search loses three quarters of its share on the day the budget runs out, organic
            # holds its per-day volume -- and the device split carries the decoy: nothing about
            # the collapse is device-specific, which is what rules out a funnel regression.
            #
            # Every segment converts at the same flat rate, and that is load-bearing twice
            # over. It is the scenario's discriminating observation (volume fell, rate did
            # not), and it is what makes total conversions the same number whichever dimension
            # is asked for -- two breakdowns of one series that disagreed on the total would be
            # the contradiction this type removes, reintroduced one level down.
            breakdowns={
                "sessionDefaultChannelGroup": (
                    Segment(
                        "Paid Search",
                        share=0.55,
                        conversion_rate=0.041,
                        shifts_on=onset,
                        share_after=0.232,
                    ),
                    Segment(
                        "Organic Search",
                        share=0.45,
                        conversion_rate=0.041,
                        shifts_on=onset,
                        share_after=0.768,
                    ),
                ),
                "deviceCategory": (
                    Segment("mobile", share=0.626, conversion_rate=0.041),
                    Segment("desktop", share=0.374, conversion_rate=0.041),
                ),
                # The campaign by name, because `campaign` is this scenario's required signal
                # and `sessionCampaignName` is the most direct question an analyst can ask about
                # it. Without this the breakdown came back empty -- on a scenario about a
                # campaign -- and the name had to be inferred from the channel group instead.
                # An investigation asked for it, got nothing, and wrote that "GA4
                # campaign-level breakdowns returned no data", which is a statement about the
                # fixture.
                #
                # The same numbers as the channel split, deliberately: this tenant's paid search
                # traffic *is* the spring campaign, so two breakdowns describing it differently
                # would be the contradiction `MetricSeries` exists to prevent.
                "sessionCampaignName": (
                    Segment(
                        "spring-2026-brand",
                        share=0.55,
                        conversion_rate=0.041,
                        shifts_on=onset,
                        share_after=0.232,
                    ),
                    Segment(
                        "(not set)",
                        share=0.45,
                        conversion_rate=0.041,
                        shifts_on=onset,
                        share_after=0.768,
                    ),
                ),
            },
        ),
        responses={
            # The decoy: a real deploy on the day the drop began. Environment named as
            # the repository actually names it, not "production" -- see the note on the
            # onboarding scenario's deployment payload.
            # What disarms the deploy decoy on evidence rather than on inference: the
            # deploy contained no user-facing change at all.
            # No breakage reported around the drop. Stated explicitly rather than left
            # empty, so "nobody complained" is an observation the analyst can cite
            # instead of a silence it has to interpret.
            "hubspot__contacts": {
                "by_lifecycle_stage": {"lead": 463},
                "contacts": [],
            },
        },
    )


def insufficient_evidence(seed: int = 3) -> Scenario:
    """The case where the honest answer is "the data cannot tell us".

    Signups moved within noise, no deploys, no campaign changes, nothing in Slack.
    A confident causal story here is a hallucination even if every number cited is
    real — which is exactly the failure a grounding-only score would miss, and the
    reason this scenario exists.
    """
    rng = random.Random(seed)
    start = date(2026, 5, 1)

    return Scenario(
        name="insufficient_evidence",
        question="Why did enterprise signups fall 3% in the week of 22 May 2026?",
        difficulty=Difficulty.UNANSWERABLE,
        ground_truth=GroundTruth(
            cause=(
                "A 3% move is within normal weekly variation and no candidate cause "
                "is visible. The data cannot establish a reason."
            ),
            required_signals=(),
            decoys=("deploy", "campaign", "onboarding", "pricing"),
            required_capabilities=("ga4__compare_periods",),
            is_unanswerable=True,
        ),
        # The product source agrees with the website source: within noise. Two independent
        # sources both saying "nothing to see" is what makes "insufficient evidence" a finding
        # rather than a shrug.
        #
        # Derived daily rather than typed as four weekly buckets, which is what it was. A
        # request for `interval="day"` received weeks -- the defect that cost
        # `campaign_traffic_drop` real score twice, and the last plain series still carrying it.
        # 27/day flat across May: the weekly sums it replaces were 188, 194, 191, 185.
        daily_truth={
            "posthog__event_trend": (
                DailyTruth(
                    event="user signed up",
                    days=tuple(
                        (
                            date(2026, 5, 1) + timedelta(days=offset),
                            round(27.1 * (1 + rng.uniform(-0.07, 0.07))),
                        )
                        for offset in range(28)
                    ),
                ),
            )
        },
        metric_series=MetricSeries(
            # No step change: noise only. The whole scenario is that nothing here is
            # distinguishable from noise, so nothing about the breakdown shifts either -- an
            # analyst that finds a segment story in this data has invented one.
            daily=_daily(start, 28, 675, noise=0.06, rng=rng),
            breakdowns={
                "deviceCategory": (
                    Segment("mobile", share=0.62, conversion_rate=0.04),
                    Segment("desktop", share=0.38, conversion_rate=0.04),
                ),
            },
        ),
        responses={
            # Empty, but in the shape the real connectors return, including the fields
            # that say *why* it is empty. This scenario is the one where the analyst has
            # to distinguish "nothing happened" from "I could not look", and a payload
            # missing `environments_available` or `count` leaves it guessing -- which is
            # the same confusion that produced a wrong conclusion on real data three
            # times today.
            "github__deployment_history": {
                "repo": "acme/web",
                "environment": None,
                "count": 0,
                "environments_available": ["prod-web", "staging"],
                "note": None,
                "deployments": [],
            },
            "github__recent_prs": {"repo": "acme/web", "pull_requests": []},
            "github__commits": {
                "repo": "acme/web",
                "since": "2026-05-18",
                "until": "2026-05-28",
                "path": None,
                "count": 0,
                "commits": [],
            },
            "github__issues": {
                "repo": "acme/web",
                "since": "2026-05-18",
                "labels": None,
                "count": 0,
                "pull_requests_excluded": True,
                "issues": [],
            },
            "posthog__annotations": {
                "annotation_count": 0,
                "total_available": 0,
                "annotations": [],
            },
            "slack__search_messages": {"messages": [], "total_matching": 0},
            "hubspot__closed_won": {"deals": [], "total_amount": None},
        },
    )


def partial_month_false_premise(seed: int = 4) -> Scenario:
    """The question that asserts a fall which never happened.

    Taken from a live failure, near-verbatim. Asked *"did our signups fell from last month?"* in
    Slack on the 12th, the analyst compared 619 events in the partial month against 4,849 in the
    complete one, reported that figure first, and spent four paragraphs looking for a cause —
    deploy history, Slack incident channels, a mid-June trend break. Every claim it made was
    cited and every citation resolved. The answer was still wrong, because signups had not
    fallen: the periods were 12 days and 31 days long.

    This scenario is deliberately not `UNANSWERABLE`. There the data cannot say why; here it
    says clearly, and what it says is that the premise is false. The two need separating because
    the failure modes differ: an unanswerable scenario tempts a report into inventing a cause,
    while this one tempts it into *confirming the question* — and confirmation is the more
    dangerous mistake, because the reader asked for it.

    The trap is one monthly series, which is what a first tool call naturally returns. The
    refutation requires a second call at daily granularity, where the run-rate is flat: 156.4/day
    across July against 157.0/day across the first twelve days of August. The decoys are real —
    there *was* a deploy on the 4th and there *is* a Slack thread worrying about signups — so a
    report that goes looking for a cause will find a coherent story to tell.
    """
    rng = random.Random(seed)
    july = date(2026, 7, 1)
    august = date(2026, 8, 1)

    return Scenario(
        name="partial_month_false_premise",
        question="Did our signups fall from last month?",
        difficulty=Difficulty.FALSE_PREMISE,
        ground_truth=GroundTruth(
            cause=(
                "Signups did not fall. August's total is lower only because the month is 12 "
                "days old: 157.0 signups/day across 1-12 August against 156.4/day across all "
                "of July. Comparing a partial month's total to a complete one is the entire "
                "apparent decline."
            ),
            required_signals=(),
            # Both are real events in the fixture, and both are the kind of thing a report
            # hunting a cause will reach for. Naming either as the reason signups fell is wrong
            # twice over: it is not the cause, and there was nothing to cause.
            decoys=("deploy", "pricing page", "campaign", "onboarding"),
            # The daily-granularity call is the one that separates a real fall from a calendar
            # artefact. Either source establishes it, so either satisfies the requirement.
            required_capabilities=(("posthog__event_trend", "ga4__get_sessions"),),
            is_false_premise=True,
            refutation_signals=(
                # Somewhere in the summary it has to actually say no.
                (
                    "did not fall",
                    "have not fallen",
                    "has not fallen",
                    "not fallen",
                    "no fall",
                    "did not decline",
                    "no decline",
                    "did not drop",
                    "have not dropped",
                    "no, signups",
                    "signups have held",
                    "flat",
                ),
                # And it has to say why the figure looked otherwise, or it is a bare assertion
                # against a number the reader can see.
                (
                    "partial",
                    "incomplete",
                    "12 days",
                    "twelve days",
                    "run rate",
                    "run-rate",
                    "per day",
                    "daily average",
                    "not comparable",
                    "different lengths",
                    "same length",
                ),
            ),
        ),
        # Daily counts, one per day, from which every requested interval is derived. The rates
        # are the ground truth's own: 156.4/day through June and July, 157.0/day across the
        # twelve days of August that exist. A monthly call still sums them into the trap -- one
        # full month against a third of one -- and a daily call now answers at the granularity
        # where the run-rate is visible, instead of returning three monthly buckets to a caller
        # that asked for days.
        daily_truth={
            "posthog__event_trend": (
                DailyTruth(
                    event="user signed up",
                    days=tuple(
                        (day, round(rate * (1 + rng.uniform(-0.05, 0.05))))
                        for first, count, rate in (
                            (date(2026, 6, 1), 30, 156.4),
                            (july, 31, 156.4),
                            (august, 12, 157.0),
                        )
                        for day in (first + timedelta(days=offset) for offset in range(count))
                    ),
                ),
            )
        },
        dated_records={
            "slack__search_messages": DatedRecords(
                key="messages",
                date_field="ts",
                subject_param="query",
                search_fields=("text", "channel"),
                since_param="after",
                until_param="before",
                total_key="total_matching",
                records=(
                    {
                        "channel": "growth",
                        "user": "priya",
                        "ts": "2026-08-11T16:02:00Z",
                        "text": (
                            "signups look way down this month vs July \u2014 is the pricing "
                            "redesign hurting us?"
                        ),
                    },
                    {
                        "channel": "growth",
                        "user": "sam",
                        "ts": "2026-08-11T16:09:00Z",
                        "text": "checking now, might just be the month being young",
                    },
                ),
            ),
        },
        metric_series=MetricSeries(
            # The refutation, at the granularity where a run-rate is visible. Flat across both
            # months, noise only -- and the comparison is now a projection of it, so an analyst
            # that asks for the truncated month against the complete one is shown the truncation
            # rather than a hand-written pair of totals that concealed it.
            daily=_daily(july, 31, 156.4, noise=0.05, rng=rng)
            + _daily(august, 12, 157.0, noise=0.05, rng=rng),
            breakdowns={
                "deviceCategory": (
                    Segment("mobile", share=0.60, conversion_rate=0.041),
                    Segment("desktop", share=0.40, conversion_rate=0.041),
                ),
            },
        ),
        responses={
            # Decoy one. A real deploy, on a date that fits the story, changing something a
            # person would believe could affect signups.
            "github__deployment_history": {
                "repo": "acme/web",
                "environment": "prod-web",
                "count": 2,
                "environments_available": ["prod-web", "staging"],
                "note": None,
                "deployments": [
                    {
                        "sha": "6b1f0ac",
                        "environment": "prod-web",
                        "created_at": "2026-08-04T09:12:00Z",
                        "description": "Pricing page redesign",
                        "state": "success",
                    },
                    {
                        "sha": "a04c39e",
                        "environment": "prod-web",
                        "created_at": "2026-07-19T14:40:00Z",
                        "description": "Dependency bumps",
                        "state": "success",
                    },
                ],
            },
            "github__recent_prs": {
                "repo": "acme/web",
                "pull_requests": [
                    {
                        "number": 1204,
                        "title": "Pricing page redesign",
                        "merged_at": "2026-08-04T08:55:00Z",
                        "author": "dana",
                        "files_changed": 14,
                    }
                ],
            },
            # Decoy two, and the most tempting one: a colleague has already asserted the
            # premise in writing. A report that treats this as corroboration has mistaken
            # someone else's worry for evidence.
            "posthog__annotations": {
                "annotation_count": 1,
                "total_available": 1,
                "annotations": [
                    {
                        "date_marker": "2026-08-04T00:00:00Z",
                        "content": "Pricing page redesign to 100%",
                        "creation_type": "USR",
                    }
                ],
            },
            "hubspot__closed_won": {"deals": [], "total_amount": None},
        },
        # The one deliberate exception. See the `partial_buckets` note above: this scenario
        # measures whether the analyst notices an incomplete month unaided, which is the
        # scarcer signal, and the disclosure is unit-tested in tests/tools/test_posthog.py.
        compute_disclosures=False,
    )


#: The suite. Ordered so a partial run still covers the difficulty classes.


def _event_catalogue() -> dict[str, Any]:
    """An event catalogue large enough to be digested, with a cluster at one end.

    Ninety-six events. Eighty-four browser-autocapture ones spread across the recent fortnight,
    and twelve product events whose `last_seen_at` all fall inside a ninety-minute window six
    weeks earlier. The cluster is real and is not the answer to this scenario's question: those
    twelve stopped long before the signup movement, so an analyst that seizes on them has found
    a second, older incident and misattributed it.

    That is deliberate. A fixture where the buried fact *is* the answer would reward reading it;
    this one rewards reading it and then saying what it does and does not explain.
    """
    events: list[dict[str, Any]] = [
        {
            "name": f"$autocapture_{index}" if index % 4 else f"$pageview_{index}",
            "description": None,
            "verified": False,
            "last_seen_at": (
                datetime(2026, 6, 16, 9, 0, tzinfo=UTC) + timedelta(minutes=index * 137)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for index in range(84)
    ]
    stopped = (
        "credit purchased",
        "settings saved",
        "onboarding completed",
        "conversation created",
        "conversation finished",
        "conversation deleted",
        "trajectory downloaded",
        "create pr button clicked",
        "team members invited",
        "api key created",
        "billing portal opened",
        "workspace renamed",
    )
    events += [
        {
            "name": name,
            "description": f"Server-side {name}.",
            "verified": True,
            "last_seen_at": (
                datetime(2026, 5, 4, 20, 5, tzinfo=UTC) + timedelta(minutes=position * 8)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for position, name in enumerate(stopped)
    ]
    return {
        "event_count": len(events),
        "total_available": len(events),
        "excluded_stale": False,
        "events": events,
    }


def partial_month_disclosed(seed: int = 4) -> Scenario:
    """The truncation case with the coverage warning turned on, and nothing else.

    **Built because the measurement said the suite could not measure the thing it had just
    shipped.** `ga4.compare_periods` gained a `window_coverage` disclosure after the same wrong
    headline appeared three times -- twelve days of August compared against thirty-one of July,
    reported as a 61.8% decline. Auditing two forty-attempt captures then showed that **34% and
    28% of every `compare_periods` call in the suite compared windows of unequal coverage**, and
    that the mistake is systematic in exactly two scenarios: `measurement_stopped`, which was
    already scoring 100% and had no headroom to show an improvement, and
    `partial_month_false_premise`, which withholds every disclosure on purpose.

    So no scenario both made the mistake and was allowed to see the warning, and the
    disclosure's effect was unmeasurable by construction. That is a gap in the eval rather than
    a fact about the disclosure.

    This is the same world as its parent with one thing changed: `window_coverage` is allowed
    through, and every other disclosure stays withheld. The pair prices the warning the way the
    decline twins price abstention -- the parent measures whether the analyst notices a
    truncated month unaided, the twin measures whether it acts on being told.

    The seed is the parent's, for the reason every twin here shares one: a different seed
    changes the planted series, and then the pair differs in the perturbation *and* in the
    data.
    """
    return dataclasses.replace(
        partial_month_false_premise(seed=seed),
        name="partial_month_disclosed",
        compute_disclosures=frozenset({"window_coverage"}),
    )


def tempting_coincidence(seed: int = 5) -> Scenario:
    """A real movement, a change on the same day, and no way to connect them.

    **Built to provoke the failure, because the existing unanswerable scenario does not do it
    reliably.** `insufficient_evidence` gives the analyst nothing to blame, so a well-behaved
    report and a lazy one both decline a cause and the sufficiency gate is never needed --
    measured: two consecutive runs of that scenario produced a causal summary claim once and a
    declining one the next time, which is variance rather than a test.

    Here the temptation is maximal and the identifiability is still absent. Signups step down
    115/day on 2026-06-17, which is real and 16x the noise. One deploy landed on 2026-06-16,
    the day before, so the onset rule cannot eliminate it. Nothing else shipped in the window.
    And there is still no way to establish the link: no per-segment series to compare against,
    no experiment, no error report, no rollback, and the deploy touched copy rather than the
    signup path.

    So the correct answer names the deploy as the only candidate, states that the movement is
    established, and refuses to call it the cause. Every one of those three parts is checked:
    `required_signals` demands the candidate and the movement, `decoys` catches asserting it,
    and `is_unanswerable` makes naming a cause score down however well-cited it is.

    This is the scenario the sufficiency gate exists for. A report that says "signups fell
    because of the 16 June deploy" has every citation resolving and is still wrong.
    """
    rng = random.Random(seed)
    # The window starts well before the movement on purpose. A first draft ran 2026-05-20 to
    # 07-18 with the break in the middle, and the conformal test returned "unresolvable": it has
    # no power unless the post period is shorter than the pre period, so a break at day 28 of 60
    # cannot be established however large it is. The fixture had walked into the power cliff
    # documented in `cortex.analysis.conformal`, which is a useful demonstration that the cliff
    # is a practical constraint on window choice and not a theoretical footnote.
    start = date(2026, 5, 1)

    return Scenario(
        name="tempting_coincidence",
        question="Why did signups drop in mid-June 2026?",
        difficulty=Difficulty.UNANSWERABLE,
        ground_truth=GroundTruth(
            cause=(
                "Signups did step down in mid-June and the only candidate is a copy deploy "
                "merged on 2026-06-16. Nothing connects them: no comparable unaffected series, no "
                "experiment, no error report, and the deploy did not touch the signup path. "
                "The movement is established; its cause is not."
            ),
            # The movement and the candidate must both be reported. A report that mentions
            # neither has not investigated; one that names only the deploy has jumped.
            # The onset as *detected*, which is a day earlier than the day the level actually
            # changes: the segmenter reports 2026-06-16 for a step at index 47, and demanding
            # the true date would fail a report that faithfully quoted the evidence it was
            # given. Both are accepted rather than picking one, because the scenario tests
            # whether a cause is asserted, not date precision.
            required_signals=(
                ("2026-06-16", "2026-06-17", "16 June", "17 June", "June 16", "June 17"),
                ("copy", "deploy", "812"),
            ),
            # Asserting the deploy *as the cause* is the failure. The decoy is the assertion,
            # not the mention -- `_asserted_text` is what the scorer searches, so naming it in
            # a contradicted or inconclusive hypothesis is not penalised.
            decoys=("caused by the deploy", "the deploy caused", "driven by the deploy"),
            required_capabilities=("posthog__event_trend", "github__recent_prs"),
            is_unanswerable=True,
        ),
        # A real, unmistakable level shift: ~215/day to ~100/day on 2026-06-17, which is 16x the
        # noise. Derived so a weekly or monthly call aggregates it instead of receiving sixty
        # daily rows in answer to a question about months.
        daily_truth={
            "posthog__event_trend": (
                DailyTruth(
                    event="user signed up",
                    days=tuple(
                        (
                            start + timedelta(days=index),
                            # Day 47 is 2026-06-17: 47 days before, 13 after.
                            round((215 if index < 47 else 100) * (1 + rng.uniform(-0.07, 0.07))),
                        )
                        for index in range(60)
                    ),
                ),
            )
        },
        responses={
            # Exactly one change in the window, the day before the movement. Copy only.
            "github__recent_prs": {
                "repo": "acme/marketing-site",
                "count": 1,
                "pull_requests": [
                    {
                        "number": 812,
                        "title": "Homepage and pricing copy refresh",
                        "merged_at": "2026-06-16T14:20:00Z",
                        "files_changed": 6,
                        "additions": 118,
                        "deletions": 94,
                        "paths": [
                            "content/home.mdx",
                            "content/pricing.mdx",
                            "content/_meta.json",
                        ],
                    }
                ],
            },
            # Nothing corroborating anywhere. Shaped as the real connectors return an empty
            # result, so "nobody looked" stays distinguishable from "nothing happened".
            "github__deployment_history": {
                "repo": "acme/marketing-site",
                "count": 0,
                "deployments": [],
                # Present even though the list is empty, and that is the point: it separates
                # "this repo deploys to these two places and neither had one in the window"
                # from "nobody looked". An empty list on its own reads as the second.
                "environments_available": ["web-preview", "web-live"],
                "note": (
                    "no deployments to any environment inside this window; the repository does "
                    "record deployments outside it"
                ),
            },
            "slack__search_messages": {
                "query": "signup drop",
                "total_matching": 0,
                "count": 0,
                "messages": [],
                "authored_by": {"person": 0, "app": 0, "self": 0},
            },
            "posthog__experiments": {"count": 0, "experiments": [], "note": "none running"},
            # A bulk payload, and the only one in the suite.
            #
            # Phase 5's digest fires above 8,000 rendered characters and every other fixture
            # payload is under 5,400, so the retrieved-but-unread machinery was unmeasurable by
            # this suite -- every scenario byte-identical with it and without it. This catalogue
            # renders past the threshold, so the digest engages here and nowhere else.
            #
            # Shaped like the real `list_events` result that carried the answer nobody read: the
            # decisive fact is a cluster at one *end* of a `last_seen_at` column, which is the
            # arrangement a positional head would hide. Nothing about it explains the signup
            # movement -- this scenario stays unanswerable -- but it is the payload on which
            # "was it read" becomes a question with an answer.
            "posthog__list_events": _event_catalogue(),
        },
    )


def measurement_stopped(seed: int = 6) -> Scenario:
    """The metric did not fall. The instrument that measures it stopped.

    **Why this scenario exists.** Every other scenario here hands the analyst a series that means
    what it appears to mean. This is the one that does not, and it is the class of failure this
    project has actually shipped: a `user signed up` series ran 1-3 August against a range ending
    on the 15th, and the analyst computed 206/day from those three days, described it as "the
    August run rate", and reported no real decline. Every figure in that answer was real and
    correctly cited. It was wrong because nobody asked whether the data covered the question.

    The connectors disclose it now -- `series_ends_early`, and a `data_trust` verdict over it --
    and until today the evaluation suite could not see any of that, because it replaces each
    capability's handler and no connector method ran. This scenario is the first that fails if
    the disclosure stops working.

    **Two independent lines, because one is not enough.** DOE-NE-STD-1004-92 is blunt about this:
    two independent lines of evidence, or the tree does not narrow. Noticing that GA4 stops on the
    3rd establishes only that GA4 stops on the 3rd -- it cannot distinguish a measurement failure
    from a site that genuinely went dark. PostHog is the second line: both signups and pageviews
    keep arriving at their usual rate through the 15th, which is impossible if traffic actually
    stopped. The pair is what makes "the tag broke" an evidenced conclusion rather than the more
    comfortable guess.

    **Either PostHog series is a correct second line, and the scoring says so.** Pageviews measure
    the same quantity GA4 lost, so they refute the collapse directly; signups are a proxy, arriving
    only if visitors did. A report that reaches for either has corroborated, and requiring one
    particular event name would score which series it picked rather than whether it corroborated.
    Run 25 failed exactly that way: the answer was right, cited, and named pageviews.

    **A cause, not an absence.** Unlike `insufficient_evidence` this is not unanswerable, and
    unlike `partial_month_false_premise` the premise is not false -- measured sessions really did
    fall off a cliff. There is a specific, findable cause and a specific right action, which is
    why neither `is_unanswerable` nor `is_false_premise` is set: recommending that someone restore
    the measurement tag is the correct answer here, and a scenario flagged as declining a cause
    would have scored that correct recommendation as a failure.

    **The decoys are the comfortable answers.** A deploy landed on 2 August, one day before the
    cliff -- the tightest possible coincidence, and the shape every analyst is trained to seize
    on. Reading what was in it (a CI cache change, no user-facing code) is how it gets ruled out
    with evidence rather than by argument.
    """
    rng = random.Random(seed)
    july = date(2026, 7, 16)

    return Scenario(
        name="measurement_stopped",
        question="Why did site sessions collapse in August?",
        difficulty=Difficulty.CONFOUNDED,
        ground_truth=GroundTruth(
            cause=(
                "GA4 session collection stopped after 3 August; the range runs to the 15th. "
                "PostHog kept recording throughout -- pageviews and signups both -- so traffic "
                "did not stop, the measurement did. The answer is the data incident, not a "
                "business cause."
            ),
            required_signals=(
                # The stop itself, however the report phrases it -- including the tense.
                # "stopped" alone missed a summary reading "GA4's session data simply *stops*
                # after 2026-08-03", which is the answer, stated in the present tense because
                # the series still is not reporting. Scoring which tense a correct report chose
                # is the same over-specification the corroboration signal below already cost
                # once.
                (
                    "stopped",
                    "stops",
                    "stop reporting",
                    "no data after",
                    "no rows",
                    "collection",
                    "truncated",
                    "ends",
                ),
                # And the second line, which is what turns noticing into concluding. A report
                # naming only the gap has found a symptom and stopped. Any of the healthy
                # PostHog series satisfies it: they are alternatives, not a checklist, and the
                # dimension is scoring whether the report corroborated -- not which event it
                # chose to corroborate with.
                ("signup", "signups", "pageview", "pageviews"),
            ),
            decoys=("deploy", "campaign", "seasonal"),
            required_capabilities=(
                "ga4__get_sessions",
                # The independent line. Without it the report cannot rule out a site outage,
                # and the honest ceiling on a GA4-only answer is "something stopped".
                "posthog__event_trend",
            ),
        ),
        # The second line, and there are two of them on purpose.
        #
        # Signups arrive normally right through the 15th, flat with noise: a perfectly flat
        # series would let a weak analyst score as well as a good one, and the claim being
        # supported is "unchanged", which needs visible ordinary variation. `$pageview` is
        # planted alongside because this scenario asserts the site is fine and only its
        # measurement stopped -- a world where one product event has data while every other
        # returns nothing says the opposite, which is the site-outage reading it exists to
        # rule out.
        #
        # Both derived rather than canned, which is what the tuple form of this field is for.
        # A single series per capability would have kept this scenario on a payload that
        # answers `interval="week"` with daily rows, in the one place a second series is the
        # whole argument.
        daily_truth={
            "posthog__event_trend": (
                DailyTruth(
                    event="user signed up",
                    days=tuple(
                        (july + timedelta(days=offset), round(21 * (1 + rng.uniform(-0.12, 0.12))))
                        for offset in range(31)
                    ),
                ),
                DailyTruth(
                    event="$pageview",
                    days=tuple(
                        (
                            july + timedelta(days=offset),
                            round(1_450 * (1 + rng.uniform(-0.09, 0.09))),
                        )
                        for offset in range(31)
                    ),
                ),
            )
        },
        metric_series=MetricSeries(
            # Three weeks of healthy sessions, then nothing. The series stops on 3 August while
            # the request runs to the 15th, which is what `_gap` turns into `series_ends_early`
            # -- and the fixture states no disclosure of its own, so the scenario fails if the
            # connector's own computation stops working. That is the whole point of it.
            daily=_daily(july, 19, 158.0, noise=0.05, rng=rng)
            + _daily(date(2026, 8, 1), 3, 161.0, noise=0.05, rng=rng),
            breakdowns={
                "deviceCategory": (
                    Segment("mobile", share=0.61, conversion_rate=0.04),
                    Segment("desktop", share=0.39, conversion_rate=0.04),
                ),
            },
        ),
        responses={
            # The decoy, one day before the cliff. Contents are the disarming evidence: a CI
            # cache key and a lockfile bump reach no user and cannot move sessions.
            "github__recent_prs": {
                "repo": "acme/web",
                "count": 1,
                "pull_requests": [
                    {
                        "number": 947,
                        "title": "Bump CI cache key and refresh lockfile",
                        "merged_at": "2026-08-02T11:05:00Z",
                        "files_changed": 2,
                        "additions": 214,
                        "deletions": 209,
                        "paths": [".github/workflows/ci.yml", "pnpm-lock.yaml"],
                    }
                ],
            },
            # Nothing to find, shaped the way a real connector returns an empty result so that
            # "nobody looked" stays distinguishable from "nothing happened".
            "github__deployment_history": {
                "repo": "acme/web",
                "count": 0,
                "deployments": [],
                "environments_available": ["staging", "production"],
                "note": (
                    "no deployments to any environment inside this window; the repository does "
                    "record deployments outside it"
                ),
            },
            "slack__search_messages": {
                "query": "sessions drop",
                "total_matching": 0,
                "count": 0,
                "messages": [],
                "authored_by": {"person": 0, "app": 0, "self": 0},
            },
            # No large catalogue here, deliberately, and the first version had one.
            #
            # It reused `_event_catalogue()` -- ninety-six events, built for
            # `tempting_coincidence` where digesting a big catalogue *is* the test. That buried
            # `user signed up` as one entry in ninety-seven, and both attempts of run 24 failed:
            # one never queried a series at all, the other asked for `$pageview` and correctly
            # got nothing. The corroboration this scenario is built on had only ever worked
            # because the fixture used to answer any event name with the planted series.
            #
            # So the standing four-event default applies instead, with the planted event added to
            # it, which leaves the analyst able to find the second line of evidence. This scenario
            # tests whether it *corroborates*; making it also a needle-hunt through autocapture
            # noise conflates two skills and measures neither.
        },
        # Every product event is healthy, because that is what "the site is fine" means. With one
        # series planted and the rest empty, an analyst asking about pageviews is told they
        # stopped too -- which points at a site outage, the exact reading this scenario exists to
        # rule out. The fixture must not argue against its own ground truth.
    )


def _blank(parent: Scenario, capability: str) -> tuple[str, Any]:
    """An emptied version of whatever `parent` planted for `capability`, in its own layer.

    **Which layer matters, and getting it wrong disarms the twin silently.** A twin used to blank
    `responses` by name. Once `campaign_traffic_drop`'s Slack search moved to `dated_records`,
    `response_for` consulted the projection first and the blanked dict was never reached -- so the
    twin returned the campaign announcement, its cause was identifiable again, and the scenario
    stopped being undecidable while still passing. Found two commits into ADR 0006, which is the
    exact failure that ADR exists to stop.

    So a twin names a *capability*, not a payload, and this empties it wherever it lives. For a
    projection that means keeping the envelope and dropping the records, which is also the more
    honest empty: "this repository has no commits in that window" rather than a bare `{}`.
    """
    if capability in parent.dated_records:
        return "dated_records", dataclasses.replace(parent.dated_records[capability], records=())
    if capability in parent.daily_truth:
        return "daily_truth", ()
    if capability in parent.subject_responses:
        # Every subject emptied rather than the layer removed: a pull request that was opened
        # and had no review activity is a different observation from one nobody looked at, and
        # the twin's whole claim is that the repository was looked at and had nothing in it.
        return "subject_responses", {
            subject: _without_rows(payload)
            for subject, payload in parent.subject_responses[capability].items()
        }
    if parent.metric_series is not None and capability in parent.metric_series.capabilities:
        # A GA4 world with no traffic in it: the projections still answer, and answer empty.
        # Blanking the capability alone is not available -- every one of them reads the same
        # series, which is the property that made them consistent in the first place.
        return "metric_series", dataclasses.replace(parent.metric_series, daily=())
    planted = parent.responses.get(capability, {})
    emptied = {
        key: (
            [] if isinstance(value, list) else 0 if key.endswith(("count", "matching")) else value
        )
        for key, value in planted.items()
    }
    return "responses", emptied


def _undecidable_twin(
    parent: Scenario,
    *,
    cause: str,
    blanked: tuple[str, ...],
    decoys: tuple[str, ...] | None = None,
) -> Scenario:
    """A scenario identical to its parent except that the cause is no longer identifiable.

    **Derived rather than written out, so "one controlled perturbation" is structural.** A
    hand-copied twin drifts from its parent the first time either is edited, and then the pair
    stops being a pair -- a comparison between two scenarios that differ in ways nobody tracked.
    Here the difference is the `blanked` argument and nothing else, readable in one place.

    **The twin must share its parent's seed**, which is why each twin defaults to the parent's
    rather than to a fresh one. Every planted series is generated with multiplicative noise from
    `random.Random(seed)`, so a different seed changes the movement itself -- and then the pair
    differs in the perturbation *and* in the data, which is no longer a controlled comparison.
    Caught by the test asserting the difference touches exactly one connector.

    Why the suite needs these at all: its hard bar is `delivered_hallucinations == 0`, which
    **rewards silence**. A system that declined every question would score perfectly on
    grounding, and nothing here could tell that from one that answers well. Every fix that makes
    a gate stricter -- and several landed today -- buys accuracy at some unmeasured cost in
    over-abstention. Scoring a should-answer question beside its should-decline twin is what
    prices that cost. arXiv 2607.10059 measures the same gap directly: the best agent it tested
    reached 59.5% paired accuracy, and abstention was "largely independent of general
    task-solving capability".

    The parent keeps its signals and its cause; the twin declines. `is_unanswerable` is what
    tells `accuracy` that declining is the correct answer here, and `veto_precision` and
    `verifier_precision` both exempt a scenario with no findable cause, so a gate withholding a
    causal claim on the twin is the gate working rather than a defect.
    """
    return dataclasses.replace(
        parent,
        name=f"{parent.name}_undecidable",
        difficulty=Difficulty.UNANSWERABLE,
        ground_truth=dataclasses.replace(
            parent.ground_truth,
            cause=cause,
            # Nothing to name. `accuracy` switches to scoring whether the report declined, and
            # `required_signals` is what `veto_precision` reads to decide there is a cause worth
            # protecting -- so it has to be empty, or the twin would penalise a gate for
            # withholding a cause the scenario says cannot be established.
            required_signals=(),
            decoys=decoys if decoys is not None else parent.ground_truth.decoys,
            is_unanswerable=True,
        ),
        **_blanked_layers(parent, blanked),
    )


def _blanked_layers(parent: Scenario, capabilities: tuple[str, ...]) -> dict[str, Any]:
    """The scenario fields a twin overrides, grouped by the layer each capability lives in."""
    layers: dict[str, Any] = {
        "responses": dict(parent.responses),
        "dated_records": dict(parent.dated_records),
        "daily_truth": dict(parent.daily_truth),
        "subject_responses": dict(parent.subject_responses),
    }
    for capability in capabilities:
        layer, emptied = _blank(parent, capability)
        if layer == "metric_series":
            # Not keyed by capability: the series is the layer, and emptying it empties every
            # capability projected from it.
            layers[layer] = emptied
            continue
        layers[layer][capability] = emptied
    return layers


def onboarding_regression_undecidable(seed: int = 1) -> Scenario:
    """The mobile regression with no change to pin it on.

    One perturbation: the repository shows no commits, no deployments and no pull-request
    activity in the onset window. The movement is untouched -- signups fall, and the fall is
    still concentrated in mobile, which the funnel and the device breakdown both establish. What
    is gone is any dated change that could have produced it.

    So the honest answer is available and specific: *the drop is real and it is mobile-only, and
    nothing in the window explains it.* That is not the same as the parent's answer with lower
    confidence, and it is not `insufficient_evidence` either -- there the movement itself is
    within noise, so there is nothing to explain. Here there is plainly something to explain and
    no candidate to explain it with, which is the case an analyst most wants to dress up.
    """
    parent = onboarding_regression(seed=seed)
    return _undecidable_twin(
        parent,
        cause=(
            "Mobile signups really did fall, and no dated change in the window can account "
            "for it: the repository shows no commits, no deployments and no pull-request "
            "activity. The honest answer names the segment and declines the cause."
        ),
        # A repository that was looked at and had nothing in it, which is a different
        # observation from one nobody asked about -- the shape a real connector returns.
        # Named as capabilities, not payloads: `_blank` empties each wherever the parent
        # planted it, so a capability later moved to a projection stays blanked.
        blanked=(
            "github__commits",
            "github__deployment_history",
            "github__recent_prs",
            "github__pull_request_activity",
            # The issue tracker too. The parent files a signup-blocking iOS bug on the day the
            # mobile decline begins, which is a cause plainly stated whatever the repository
            # shows -- a twin whose cause is only mostly removed measures nothing.
            "github__issues",
        ),
        # The parent's own cause becomes a decoy here: a report naming the onboarding modal is
        # naming something the evidence no longer contains.
        decoys=("pricing page", "campaign", "seasonality", "modal", "onboarding"),
    )


def campaign_traffic_drop_undecidable(seed: int = 2) -> Scenario:
    """The paid-search collapse with no record of why it stopped.

    One perturbation: the Slack search returns nothing. The channel collapse is untouched --
    paid search sessions fall away, signups follow, and the daily series still shows the step --
    but the announcement that the campaign budget was exhausted is gone.

    That is a genuine identifiability failure rather than a missing lookup. Paid search stopping
    is consistent with an exhausted budget, an ad-platform outage, a tracking break on that
    channel, or a deliberate pause, and nothing in the evidence separates them. The analyst can
    establish *which channel* carries the whole movement and cannot establish why -- so the
    honest answer names the channel and stops.
    """
    parent = campaign_traffic_drop(seed=seed)
    return _undecidable_twin(
        parent,
        cause=(
            "The fall is entirely in paid search, and nothing observed says why that channel "
            "stopped: an exhausted budget, an ad-platform outage, a tracking break and a "
            "deliberate pause all fit equally. The honest answer names the channel and "
            "declines the cause."
        ),
        blanked=("slack__search_messages",),
        decoys=("deploy", "onboarding", "conversion rate", "budget", "campaign"),
    )


def won_accounts_not_activated(seed: int = 7) -> Scenario:
    """A state question whose answer exists in no single tool.

    **Why this scenario exists.** Every other scenario in this suite asks why something
    *changed*, and the analyst's system prompt teaches a method for exactly that shape. Six
    live runs against a real CRM were asked *state* questions instead -- how much pipeline,
    what is our win rate, which deals are stale -- and five of the six answered from HubSpot
    alone, never touching PostHog, GitHub or Slack. That is not the model being lazy: no
    method it was given applies, so it falls back to retrieval from the one obvious source.

    Nothing in the suite could detect that, because a suite made entirely of change
    questions shares the blind spot it would need to measure.

    So this question is answerable **only** by joining two connectors. HubSpot knows which
    accounts were won. PostHog knows which organisations generate events. Neither knows
    whether the accounts that were won are the organisations that show up, and the honest
    answer -- six of the nine won accounts have produced no product event at all since
    closing -- is invisible from either side.

    The decoys are built so that each single-source view looks *fine*:

    - HubSpot alone: nine deals closed won for $1.23M, a good quarter.
    - PostHog alone: total events flat and healthy across the window, no incident.

    An analyst that stops at one tool does not get a wrong number. It gets a right number
    and misses the finding, which is the failure mode this whole project is about.
    """
    rng = random.Random(seed)
    start = date(2026, 7, 1)

    # The nine accounts closed in the prior quarter. Three of them show up in the product;
    # six never do. Invented names -- a fixture that used real customers would put them in
    # the repository forever.
    won = [
        ("Northwind Traders", 210_000, "2026-06-28"),
        ("Globex", 185_000, "2026-06-30"),
        ("Initech", 160_000, "2026-05-19"),
        ("Umbrella Industries", 145_000, "2026-06-11"),
        ("Hooli", 140_000, "2026-04-27"),
        ("Soylent Corp", 125_000, "2026-06-30"),
        ("Vandelay Industries", 110_000, "2026-05-06"),
        ("Wonka Industries", 95_000, "2026-06-02"),
        ("Cyberdyne Systems", 60_000, "2026-06-30"),
    ]
    activated = {"Globex", "Hooli", "Wonka Industries"}

    return Scenario(
        name="won_accounts_not_activated",
        question="Are the accounts we closed last quarter actually using the product?",
        difficulty=Difficulty.CONFOUNDED,
        ground_truth=GroundTruth(
            cause=(
                "Nine accounts closed won last quarter. Only three of them -- Globex, Hooli "
                "and Wonka Industries -- have produced a single product event since. The "
                "other six, carrying $810,000 of the $1,230,000 closed, have never appeared "
                "in the product at all. Total event volume is flat and healthy, which is why "
                "neither source shows this on its own."
            ),
            required_signals=(
                # The gap, however it is counted. An analyst reporting "only three of nine
                # are active" has found exactly what one reporting "six have never used it"
                # found, and demanding one phrasing would fail the other.
                ("6 of", "six of", "6 of the 9", "six of the nine", "3 of", "three of"),
                # Named, so a report cannot pass by asserting a gap it never located.
                ("Northwind", "Initech", "Umbrella", "Soylent", "Vandelay", "Cyberdyne"),
            ),
            # Each is a true statement about one source and a wrong answer to the question.
            decoys=(
                "usage is healthy",
                "all accounts are active",
                "every account is using",
                "no activation problem",
            ),
            # Both, and that is the entire point of the scenario. `connected_tools` is
            # inferred from this, so the tenant gets exactly the two connectors needed.
            required_capabilities=("hubspot__closed_won", "posthog__event_trend"),
        ),
        responses={
            "hubspot__closed_won": {
                "count": len(won),
                "deals": [
                    {
                        "id": str(70_000_000 + index),
                        "name": name,
                        "stage": "closedwon",
                        "amount": amount,
                        "close_date": f"{closed}T12:00:00Z",
                        "pipeline": "default",
                        "deal_type": "newbusiness",
                    }
                    for index, (name, amount, closed) in enumerate(won)
                ],
                "total_amount": sum(amount for _, amount, _ in won),
            },
        },
        subject_responses={
            # Keyed by event, so asking for the breakdown returns the per-organisation view
            # and asking for anything else does not. `segment`/`value` is the shape the real
            # connector returns for a breakdown query -- see `posthog.event_trend`.
            "posthog__event_trend": {
                "workspace opened": {
                    "event": "workspace opened",
                    "measure": "count",
                    "interval": "day",
                    "breakdown_property": "organization",
                    "row_count": len(activated),
                    "series": [
                        {"segment": name, "value": 400 + rng.randrange(0, 600)}
                        for name, _, _ in won
                        if name in activated
                    ],
                    "total": 2_100,
                },
            },
        },
        # Distractors. The tenant has four connectors, as the live one does, and two of them
        # hold nothing that bears on the question.
        #
        # **Why they are here.** Built with only the two connectors it needs, this scenario was
        # passed by calling everything available -- which is not the behaviour it exists to
        # measure, and not the situation that produced the failure. The live investigations had
        # four connectors and three went untouched. A fixture that makes the right pair the only
        # pair is testing retrieval, not judgement.
        #
        # Deliberately plausible and deliberately silent on activation: GitHub shows a team
        # shipping normally, Slack shows people talking about renewals and pricing. Neither
        # contains any hint of which accounts are or are not using the product, so an analyst
        # that stops in either has nothing, and one that reasons from them is reasoning from
        # noise.
        dated_records={
            "github__commits": DatedRecords(
                key="commits",
                date_field="date",
                subject="acme/web",
                subject_param="repo",
                # No `repo` here. Putting it in the envelope made every response claim to be
                # about acme/web whatever repository was asked for -- the disjoint violation
                # `test_a_different_subject_gives_a_different_answer` exists to catch, and it
                # caught it. The projection echoes the requested subject on its own.
                envelope={"path": None},
                records=tuple(
                    {
                        "sha": f"c{index:06x}",
                        "date": (start + timedelta(days=index * 9)).isoformat(),
                        "message": message,
                        "author": "a.developer",
                    }
                    for index, message in enumerate(
                        [
                            "Bump dependency versions",
                            "Fix flaky test in the billing suite",
                            "Add an index to the sessions table",
                            "Tidy up the settings page layout",
                            "Upgrade the logging library",
                        ]
                    )
                ),
            ),
            "slack__search_messages": DatedRecords(
                key="messages",
                date_field="timestamp",
                records=tuple(
                    {
                        "timestamp": (start + timedelta(days=day)).isoformat(),
                        "text": text,
                        "user": "U0GTM0001",
                        "channel_name": "gtm",
                        "author_kind": "person",
                    }
                    for day, text in [
                        (3, "renewal paperwork for the Q3 cohort is with legal"),
                        (12, "anyone got the latest pricing one-pager?"),
                        (28, "moving the pipeline review to Thursdays"),
                        (41, "reminder: log your calls before month end"),
                    ]
                ),
            ),
        },
        daily_truth={
            # The decoy: overall product usage is flat and unremarkable across the window, so
            # a PostHog-only reading finds nothing wrong and stops.
            "posthog__event_trend": (
                DailyTruth(
                    event="$pageview",
                    days=tuple(
                        (
                            start + timedelta(days=index),
                            round(1_450 * (1 + rng.uniform(-0.06, 0.06))),
                        )
                        for index in range(80)
                    ),
                ),
            )
        },
    )


#: Deal-stage win probabilities, as a CRM actually carries them.
_STAGE_PROBABILITY = {
    "qualifiedtobuy": 0.4,
    "presentationscheduled": 0.6,
    "contractsent": 0.8,
}


def forecast_ignores_the_decision(seed: int = 8) -> Scenario:
    """A CRM question whose answer is in Slack, and nothing in the question says so.

    **What this measures that `won_accounts_not_activated` does not.** That scenario also
    needs two connectors, but the question hands over the second one: *"are the accounts we
    closed actually using the product"* points at product analytics in its own wording, and
    both arms of a paired measurement found it. It tests whether the analyst *can* join two
    sources. It does not test whether it *thinks to look*.

    This one asks a pipeline question. Every word of it belongs to the CRM. The CRM answers
    it completely and confidently -- twelve open deals, $2.4M, close dates inside the
    quarter -- and that answer is wrong by $900,000, because three of those accounts froze
    procurement and the only record of it is a Slack thread. No HubSpot field carries it.

    So an analyst that opens the CRM and stops gets a clean, well-cited, plausible number,
    and `accuracy` catches it. That is the shape of the failure observed live: six real
    investigations, five of them answered out of a single connector while three others sat
    connected and unopened.

    The Slack route is deliberately reachable two ways -- `find_decision` and
    `search_messages` both return the thread -- because requiring one endpoint would score
    the route rather than the result.
    """
    rng = random.Random(seed)
    start = date(2026, 7, 1)

    frozen = {"Stark Industries", "Tyrell Corp", "Massive Dynamic"}
    open_deals = [
        ("Stark Industries", 400_000, "presentationscheduled"),
        ("Tyrell Corp", 300_000, "contractsent"),
        ("Massive Dynamic", 200_000, "presentationscheduled"),
        ("Gringotts Bank", 260_000, "contractsent"),
        ("Duff Brewing", 240_000, "presentationscheduled"),
        ("Pied Piper", 210_000, "qualifiedtobuy"),
        ("Bluth Company", 180_000, "presentationscheduled"),
        ("Prestige Worldwide", 165_000, "qualifiedtobuy"),
        ("Dunder Mifflin", 150_000, "contractsent"),
        ("Sterling Cooper", 130_000, "qualifiedtobuy"),
        ("Los Pollos Hermanos", 90_000, "presentationscheduled"),
        ("Paper Street Soap", 75_000, "qualifiedtobuy"),
    ]
    assert sum(a for _, a, _ in open_deals) == 2_400_000
    assert sum(a for n, a, _ in open_deals if n in frozen) == 900_000

    # Earlier quarters, at a win rate that makes the open pipeline unremarkable rather than
    # remarkable in either direction.
    closed_won = [
        ("Initrode", 120_000, "2026-05-22"),
        ("Vehement Capital", 95_000, "2026-04-14"),
        ("Bluth Original", 80_000, "2026-06-05"),
        ("Cogswell Cogs", 75_000, "2026-03-27"),
        ("Spacely Sprockets", 60_000, "2026-05-08"),
    ]

    decision = {
        "ts": "1788350400.000100",
        "timestamp": "2026-08-15T14:00:00+00:00",
        "user": "U0VP0SALES",
        # Worded to be findable by the terms an analyst actually reaches for -- pipeline,
        # forecast, quarter, close, commit, deal -- because `_matches_search` requires *every*
        # query term to appear, as a real search connector does. Planted with a narrower
        # vocabulary, this scenario measured whether the analyst guessed the fixture's wording:
        # a run searching "Q3 pipeline forecast" got zero rows and missed the answer, while one
        # searching "forecast commit" found it. That is search luck, and this scenario is about
        # whether the analyst thinks to open Slack at all.
        "text": (
            "Q3 pipeline forecast, commit review: Stark Industries, Tyrell Corp and Massive "
            "Dynamic have all frozen procurement until their new fiscal year. None of those "
            "three deals will close this quarter -- taking them out of the forecast now."
        ),
        "channel_name": "revenue",
        "decision_signals": ["confirmed", "please take them out"],
    }

    return Scenario(
        name="forecast_ignores_the_decision",
        question="How much of our open pipeline is realistically going to close this quarter?",
        difficulty=Difficulty.CONFOUNDED,
        ground_truth=GroundTruth(
            cause=(
                "Twelve deals are open for the quarter totalling $2,400,000, but three of them "
                "-- Stark Industries, Tyrell Corp and Massive Dynamic, $900,000 between them "
                "-- were pulled from the commit on 2026-08-15 because those accounts froze "
                "procurement until their next fiscal year. The realistic figure is $1,500,000. "
                "No CRM field records the freeze; the only record is a Slack thread."
            ),
            required_signals=(
                # The adjustment, by either figure. An analyst reporting "$1.5M" has found the
                # same thing as one reporting "exclude $900K", and demanding one phrasing would
                # fail the other.
                ("1,500,000", "1.5M", "1.5 million", "900,000", "900K", "0.9M"),
                # Why, which is the part only Slack can supply.
                ("procurement", "frozen", "freeze", "commit", "fiscal year"),
            ),
            # Each is the CRM answer stated as the answer. The raw total is not a decoy on its
            # own -- a correct report quotes it before adjusting it -- so these are phrasings
            # that only appear when the adjustment was never made.
            decoys=(
                "all twelve deals",
                "all 12 deals",
                "entire pipeline will close",
                "full $2,400,000 is expected",
            ),
            # Any-of on the Slack side: both endpoints return the thread, and requiring one
            # would score conformity to a route rather than reaching the fact.
            required_capabilities=(
                "hubspot__pipeline",
                ("slack__find_decision", "slack__search_messages"),
            ),
        ),
        responses={
            "hubspot__pipeline": {
                "pipeline_id": "default",
                "total_matching": len(open_deals),
                "count": len(open_deals),
                "total_amount": sum(amount for _, amount, _ in open_deals),
                "deals": [
                    {
                        "id": str(80_000_000 + index),
                        "name": name,
                        "stage": stage,
                        "amount": amount,
                        "close_date": (start + timedelta(days=45 + index * 3)).isoformat()
                        + "T12:00:00Z",
                        "pipeline": "default",
                        # Varied by stage. A flat 0.6 on every deal is not what a CRM looks
                        # like, and two runs spotted it and made it the story -- correctly,
                        # since "every probability is identical regardless of stage" is a real
                        # data-quality finding. It was a fixture artifact, and a planted false
                        # lead more salient than the planted answer.
                        "probability": _STAGE_PROBABILITY[stage],
                    }
                    for index, (name, amount, stage) in enumerate(open_deals)
                ],
            },
            "slack__find_decision": {"topic": "forecast", "messages": [decision]},
            # Closed history, so the portal is a coherent world rather than one with twelve
            # open deals and no record of ever having closed anything.
            #
            # Planted after three runs failed here for a reason that was entirely the
            # fixture's: the analyst searched closed-won and closed-lost, got zero rows from
            # both, and reported -- reasonably -- that "no closes recorded at all in 2025-2026"
            # looked like a broken sync rather than a true absence. The sufficiency gate then
            # refused the forecast, correctly, because a definitive answer resting on a world
            # that appears instrumented-wrong is not supportable. Nothing here bears on the
            # question; it exists so that looking around does not turn up an apparent incident.
            "hubspot__closed_won": {
                "count": len(closed_won),
                "deals": [
                    {
                        "id": str(79_000_000 + index),
                        "name": name,
                        "stage": "closedwon",
                        "amount": amount,
                        "close_date": f"{closed}T12:00:00Z",
                        "pipeline": "default",
                        "deal_type": "newbusiness",
                    }
                    for index, (name, amount, closed) in enumerate(closed_won)
                ],
                "total_amount": sum(amount for _, amount, _ in closed_won),
            },
        },
        dated_records={
            # The same thread, reachable by search as well as by the decision finder.
            "slack__search_messages": DatedRecords(
                key="messages",
                date_field="timestamp",
                records=(
                    decision,
                    {
                        "ts": "1787832000.000200",
                        "timestamp": "2026-07-20T09:30:00+00:00",
                        "user": "U0GTM0002",
                        "text": "reminder to keep close dates current before the forecast call",
                        "channel_name": "revenue",
                        "decision_signals": [],
                    },
                ),
            ),
            # Distractors, as in `won_accounts_not_activated`: a team shipping normally, with
            # nothing to say about procurement or forecasting.
            "github__commits": DatedRecords(
                key="commits",
                date_field="date",
                subject="acme/web",
                subject_param="repo",
                envelope={"path": None},
                records=tuple(
                    {
                        "sha": f"d{index:06x}",
                        "date": (start + timedelta(days=index * 11)).isoformat(),
                        "message": message,
                        "author": "a.developer",
                    }
                    for index, message in enumerate(
                        [
                            "Raise the default page size",
                            "Cache the account settings lookup",
                            "Drop an unused column",
                            "Refresh the marketing footer",
                        ]
                    )
                ),
            ),
        },
        daily_truth={
            # Product usage is flat and healthy and has nothing to do with the question.
            #
            # Deliberately *not* `$pageview`. That name is already in `DISCOVERY_DEFAULTS`, and
            # `_event_listing` only awards the evening `live=True` slot to events it has to add
            # to the catalogue -- so a scenario planting a default name gets a decoy-range
            # timestamp and can look staler than its own decoys. A latent trap for any scenario
            # planting a standard event, avoided here rather than fixed, because fixing it moves
            # the catalogue every other scenario is measured against.
            "posthog__event_trend": (
                DailyTruth(
                    event="workspace opened",
                    days=tuple(
                        (start + timedelta(days=index), round(980 * (1 + rng.uniform(-0.05, 0.05))))
                        for index in range(90)
                    ),
                ),
            )
        },
    )


SCENARIOS: tuple[Scenario, ...] = (
    onboarding_regression(),
    campaign_traffic_drop(),
    insufficient_evidence(),
    partial_month_false_premise(),
    # Its disclosure twin, differing in one allowed disclosure. See `partial_month_disclosed`.
    partial_month_disclosed(),
    tempting_coincidence(),
    measurement_stopped(),
    # The decline twins, each derived from the scenario above it. They roughly double the cost
    # of a full run, which is the price of measuring over-abstention at all: without them a
    # stricter gate always looks like an improvement.
    onboarding_regression_undecidable(),
    campaign_traffic_drop_undecidable(),
    # The first state question in the suite. Every scenario above it asks why something
    # changed; this one asks what is true now, and can only be answered across two
    # connectors. See its docstring.
    won_accounts_not_activated(),
    # The second state question, and the harder one: its answer sits in a connector the
    # question gives no reason to open. See its docstring.
    forecast_ignores_the_decision(),
)


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"no scenario named {name!r}; have: {', '.join(s.name for s in SCENARIOS)}")
