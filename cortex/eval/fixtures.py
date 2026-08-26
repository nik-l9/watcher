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

import enum
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any


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
    "posthog__list_events": {
        "count": 4,
        "events": [
            {"name": "signup_completed", "last_seen_at": "2026-07-16T10:00:00Z"},
            {"name": "checkout_started", "last_seen_at": "2026-07-16T10:00:00Z"},
            {"name": "$pageview", "last_seen_at": "2026-07-16T10:00:00Z"},
            {"name": "trial_started", "last_seen_at": "2026-07-16T10:00:00Z"},
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

    #: Responses that must differ before and after the change, keyed by
    #: `tool__capability` then by `"before"` / `"after"`.
    #:
    #: Needed because a single canned payload per capability is served to *every* call,
    #: including two calls that ask about different periods. A live investigation asked
    #: for the funnel on each side of the drop, received byte-identical figures, and
    #: correctly reported that the device totals could not be reconciled with the channel
    #: totals — a contradiction planted by the fixture, not found in the data. A scenario
    #: about a change has to be able to answer differently on each side of it.
    period_responses: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: The day the planted change takes effect. A request whose start date is on or after
    #: this resolves to the `"after"` variant.
    change_date: date | None = None

    #: Daily counts a trend capability derives every interval from, keyed by `tool__capability`.
    #:
    #: A canned payload answers `interval="day"` with whatever buckets it was typed with, so a
    #: scenario whose trap is a monthly series answered the daily call -- the discriminating one
    #: -- with monthly data. See `DailyTruth`.
    daily_truth: dict[str, DailyTruth] = field(default_factory=dict)

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
    compute_disclosures: bool = True

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
        """
        latest: date | None = None
        for response in list(self.responses.values()) + [
            variant for variants in self.period_responses.values() for variant in variants.values()
        ]:
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
        planted = set(self.responses) | set(self.period_responses)
        required = {
            alternative
            for requirement in self.ground_truth.required_capabilities
            for alternative in alternatives_of(requirement)
        }
        return frozenset(name.split("__")[0] for name in planted | required)

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
        variants = self.period_responses.get(qualified_name)
        if variants:
            return variants["after" if self._is_after(params) else "before"]
        # Subject before period: a scenario planting several series wants the one that was asked
        # for, and only a scenario planting *one* has a before/after to choose between.
        by_subject = self.subject_responses.get(qualified_name)
        if by_subject and params:
            for key in self.SUBJECT_KEYS:
                asked = params.get(key)
                if isinstance(asked, str) and asked in by_subject:
                    return by_subject[asked]
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
        truth = self.daily_truth.get(qualified_name)
        if truth is not None:
            return _resample(truth, params, matched=self._asks_about(truth.event, params))
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
    SUBJECT_KEYS = ("event", "repo")

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
        base = (
            self.responses.get("posthog__list_events") or DISCOVERY_DEFAULTS["posthog__list_events"]
        )
        events = list(base.get("events") or [])
        advertised = {event.get("name") for event in events}
        last_seen = f"{self.as_of.isoformat()}T00:00:00Z" if self.as_of else ""
        for name in sorted(self.events_described() - advertised):
            # Dated from the scenario's own horizon rather than left blank: `last_seen_at` is a
            # field the analyst reads to decide whether an event is live, and a blank one reads
            # as "never seen" -- which would make the planted event look like the dead option.
            events.append({"name": name, "last_seen_at": last_seen})
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
        sources = (
            list(self.responses.values())
            + [
                variant
                for variants in self.period_responses.values()
                for variant in variants.values()
            ]
            + [
                payload
                for by_subject in self.subject_responses.values()
                for payload in by_subject.values()
            ]
        )
        for response in sources:
            if isinstance(response, dict) and isinstance(response.get("event"), str):
                found.add(response["event"])
        found.update(truth.event for truth in self.daily_truth.values())
        return frozenset(found)

    def repositories_described(self) -> frozenset[str]:
        """Repositories this scenario's own payloads claim to be about.

        Public so a test can assert the invariant directly: everything described is advertised.
        """
        found = set()
        for response in self.responses.values():
            if isinstance(response, dict) and isinstance(response.get("repo"), str):
                found.add(response["repo"])
        for variants in self.period_responses.values():
            for response in variants.values():
                if isinstance(response, dict) and isinstance(response.get("repo"), str):
                    found.add(response["repo"])
        return frozenset(found)

    def _is_after(self, params: dict[str, Any] | None) -> bool:
        """Which side of the change a request is asking about.

        Defaults to the *after* period when no date is given, matching the question:
        an analyst asking without a range is asking about the change that prompted it.
        """
        if not params or self.change_date is None:
            return True
        for key in ("start_date", "current_start", "previous_start"):
            raw = params.get(key)
            if isinstance(raw, str):
                try:
                    return date.fromisoformat(raw) >= self.change_date
                except ValueError:
                    continue
        return True


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


_BUCKET_STARTS: dict[str, Callable[[date], date]] = {
    "day": lambda d: d,
    "week": lambda d: d - timedelta(days=d.weekday()),
    "month": lambda d: d.replace(day=1),
}


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
    buckets: dict[date, int] = {}
    for day, value in days:
        key = bucket_of(day)
        buckets[key] = buckets.get(key, 0) + value
    planted = [day for day, _ in truth.days]
    return {
        # Named for what was asked, so the analyst is not left inferring which event it holds.
        "event": asked.get("event") if not matched else truth.event,
        "measure": "count",
        "interval": interval,
        # Falls back to the requested window, then to the planted one, so an empty result still
        # reports the range it found nothing in.
        "start_date": (days[0][0] if days else start or planted[0]).isoformat(),
        "end_date": (days[-1][0] if days else end or planted[-1]).isoformat(),
        "breakdown_property": None,
        "row_count": len(buckets),
        "series": [
            {"bucket": f"{key.isoformat()}T00:00:00", "value": value}
            for key, value in sorted(buckets.items())
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
) -> list[dict[str, Any]]:
    """A daily series with optional step change and multiplicative noise.

    Noise matters: a perfectly flat series with one clean step is trivially readable,
    and would let a weak analyst score as well as a good one.
    """
    rows = []
    for offset in range(days):
        level = baseline
        if change_on is not None and offset >= change_on and change_to is not None:
            level = change_to
        value = level * (1 + rng.uniform(-noise, noise))
        rows.append(
            {
                "dimensions": {"date": (start + timedelta(days=offset)).isoformat()},
                "metrics": {"sessions": round(value)},
            }
        )
    return rows


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
        responses={
            "ga4__get_sessions": {
                "property_id": "123456789",
                "start_date": "2026-07-01",
                "end_date": "2026-07-21",
                "totals": {"sessions": 41200},
                "rows": _daily(start, 21, 2400, change_on=14, change_to=1970, rng=rng),
            },
            # The decisive call: conversion split by device.
            "ga4__get_funnel": {
                "property_id": "123456789",
                "start_date": "2026-07-15",
                "end_date": "2026-07-21",
                "rows": [
                    {
                        "dimensions": {"deviceCategory": "mobile"},
                        "metrics": {"sessions": 9800, "conversions": 284},
                        "derived_conversion_rate": 0.029,
                    },
                    {
                        "dimensions": {"deviceCategory": "desktop"},
                        "metrics": {"sessions": 4100, "conversions": 178},
                        "derived_conversion_rate": 0.0434,
                    },
                ],
            },
            # The same breakdown a week earlier, establishing what changed.
            "ga4__compare_periods": {
                "current_period": {"start": "2026-07-15", "end": "2026-07-21"},
                "previous_period": {"start": "2026-07-08", "end": "2026-07-14"},
                "comparison": [
                    {
                        "dimensions": {"deviceCategory": "mobile"},
                        "conversions": {
                            "current": 284,
                            "previous": 412,
                            "percent_change": -31.07,
                        },
                    },
                    {
                        "dimensions": {"deviceCategory": "desktop"},
                        "conversions": {
                            "current": 178,
                            "previous": 174,
                            "percent_change": 2.3,
                        },
                    },
                ],
            },
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
            # The human record around the change. A reviewer raised the exact failure
            # mode before it shipped and was overruled on timing -- the kind of evidence
            # no metric contains, and the reason a senior analyst reads the thread.
            "github__pull_request_activity": {
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
                        "body": "Merging without the footer fix, noted. Watch mobile signups.",
                    }
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
            "posthog__event_trend": {
                "event": "onboarding completed",
                "measure": "count",
                "interval": "day",
                "start_date": "2026-07-13",
                # Extended to the 21st, which is where this scenario's other series end.
                #
                # It stopped on the 16th, and once the eval began computing the connectors' real
                # disclosures that became a five-day gap against a world with GA4 sessions
                # through the 21st -- so an analyst asking a range wider than this narrow window
                # was told "onboarding completed has no data after 2026-07-16... the event may
                # have stopped firing". In a scenario about a mobile onboarding regression, a
                # data-incident verdict outranks and replaces the real answer.
                #
                # The five added days sit at the post-deploy level, so the planted signal is
                # unchanged and there is more of it: four days after the change instead of two,
                # which is what a real analyst would have to read a level shift from.
                "end_date": "2026-07-21",
                "breakdown_property": "$device_type",
                "row_count": 18,
                "series": [
                    {"bucket": "2026-07-13T00:00:00", "segment": "Mobile", "value": 61},
                    {"bucket": "2026-07-13T00:00:00", "segment": "Desktop", "value": 24},
                    {"bucket": "2026-07-14T00:00:00", "segment": "Mobile", "value": 58},
                    {"bucket": "2026-07-14T00:00:00", "segment": "Desktop", "value": 25},
                    {"bucket": "2026-07-15T00:00:00", "segment": "Mobile", "value": 39},
                    {"bucket": "2026-07-15T00:00:00", "segment": "Desktop", "value": 26},
                    {"bucket": "2026-07-16T00:00:00", "segment": "Mobile", "value": 41},
                    {"bucket": "2026-07-16T00:00:00", "segment": "Desktop", "value": 23},
                    {"bucket": "2026-07-17T00:00:00", "segment": "Mobile", "value": 40},
                    {"bucket": "2026-07-17T00:00:00", "segment": "Desktop", "value": 25},
                    {"bucket": "2026-07-18T00:00:00", "segment": "Mobile", "value": 38},
                    {"bucket": "2026-07-18T00:00:00", "segment": "Desktop", "value": 24},
                    {"bucket": "2026-07-19T00:00:00", "segment": "Mobile", "value": 42},
                    {"bucket": "2026-07-19T00:00:00", "segment": "Desktop", "value": 26},
                    {"bucket": "2026-07-20T00:00:00", "segment": "Mobile", "value": 37},
                    {"bucket": "2026-07-20T00:00:00", "segment": "Desktop", "value": 23},
                    {"bucket": "2026-07-21T00:00:00", "segment": "Mobile", "value": 40},
                    {"bucket": "2026-07-21T00:00:00", "segment": "Desktop", "value": 25},
                ],
                "total": 617,
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
            "ga4__top_pages": {
                "rows": [
                    {
                        "dimensions": {"pagePath": "/pricing"},
                        "metrics": {"screenPageViews": 3100, "engagementRate": 0.71},
                    },
                    {
                        "dimensions": {"pagePath": "/signup"},
                        "metrics": {"screenPageViews": 9800, "engagementRate": 0.34},
                    },
                ]
            },
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
        responses={
            "ga4__get_sessions": {
                "totals": {"sessions": 30400},
                "rows": _daily(start, 30, 1400, change_on=14, change_to=820, rng=rng),
            },
            # Conversion flat, volume down: the discriminating observation.
            "ga4__compare_periods": {
                "current_period": {"start": "2026-06-15", "end": "2026-06-30"},
                "previous_period": {"start": "2026-06-01", "end": "2026-06-14"},
                "comparison": [
                    {
                        "dimensions": {"sessionDefaultChannelGroup": "Paid Search"},
                        "sessions": {"current": 3100, "previous": 9800, "percent_change": -68.4},
                        "sessionConversionRate": {
                            "current": 0.041,
                            "previous": 0.0405,
                            "percent_change": 1.2,
                        },
                    },
                    {
                        "dimensions": {"sessionDefaultChannelGroup": "Organic Search"},
                        "sessions": {"current": 8200, "previous": 8100, "percent_change": 1.2},
                        "sessionConversionRate": {
                            "current": 0.039,
                            "previous": 0.0392,
                            "percent_change": -0.5,
                        },
                    },
                ],
            },
            # The decoy: a real deploy on the day the drop began. Environment named as
            # the repository actually names it, not "production" -- see the note on the
            # onboarding scenario's deployment payload.
            "github__deployment_history": {
                "repo": "acme/web",
                "environment": None,
                "count": 1,
                "environments_available": ["prod-web", "staging"],
                "note": None,
                "deployments": [
                    {
                        "id": 38771,
                        "sha": "beef123cafe4567890abcdef1234567890abcdef",
                        "ref": "main",
                        "environment": "prod-web",
                        "created_at": "2026-06-14T16:00:00Z",
                        "state": "success",
                        "creator": "ci-bot",
                    }
                ],
            },
            # What disarms the deploy decoy on evidence rather than on inference: the
            # deploy contained no user-facing change at all.
            "github__commits": {
                "repo": "acme/web",
                "since": "2026-06-13",
                "until": "2026-06-16",
                "path": None,
                "count": 2,
                "commits": [
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
                ],
            },
            # No breakage reported around the drop. Stated explicitly rather than left
            # empty, so "nobody complained" is an observation the analyst can cite
            # instead of a silence it has to interpret.
            "github__issues": {
                "repo": "acme/web",
                "since": "2026-06-14",
                "labels": None,
                "count": 0,
                "pull_requests_excluded": True,
                "issues": [],
            },
            # Product-side confirmation that the funnel itself held: signup completions
            # per session are flat while sessions collapse.
            "posthog__event_trend": {
                "event": "user signed up",
                "measure": "count",
                "interval": "week",
                "start_date": "2026-06-01",
                "end_date": "2026-06-30",
                "breakdown_property": None,
                "row_count": 4,
                "series": [
                    {"bucket": "2026-06-01T00:00:00", "value": 402},
                    {"bucket": "2026-06-08T00:00:00", "value": 391},
                    {"bucket": "2026-06-15T00:00:00", "value": 236},
                    {"bucket": "2026-06-22T00:00:00", "value": 228},
                ],
                "total": 1257,
            },
            "hubspot__contacts": {
                "by_lifecycle_stage": {"lead": 463},
                "contacts": [],
            },
            "slack__search_messages": {
                "messages": [
                    {
                        "ts": "1781000000.000100",
                        "timestamp": "2026-06-14T12:00:00+00:00",
                        "text": "spring campaign budget is exhausted, pausing ads today",
                        "channel_name": "marketing",
                    }
                ]
            },
        },
        change_date=date(2026, 6, 15),
        period_responses={
            # Volume falls, rate does not. This is the discriminating observation in the
            # whole scenario: the analyst can only rule out a funnel regression by seeing
            # that conversion rate held while sessions collapsed.
            #
            # The device totals reconcile with the channel totals on each side --
            # 11,200 + 6,700 = 17,900 before, 7,100 + 4,200 = 11,300 after, matching
            # Paid 9,800 -> 3,100 plus Organic 8,100 -> 8,200. A previous single payload
            # served 11,300 for both periods, which made the fixture contradict itself
            # and was caught by an investigation rather than by a test.
            "ga4__get_funnel": {
                "before": {
                    "rows": [
                        {
                            "dimensions": {"deviceCategory": "mobile"},
                            "metrics": {"sessions": 11200, "conversions": 459},
                            "derived_conversion_rate": 0.041,
                        },
                        {
                            "dimensions": {"deviceCategory": "desktop"},
                            "metrics": {"sessions": 6700, "conversions": 275},
                            "derived_conversion_rate": 0.041,
                        },
                    ]
                },
                "after": {
                    "rows": [
                        {
                            "dimensions": {"deviceCategory": "mobile"},
                            "metrics": {"sessions": 7100, "conversions": 291},
                            "derived_conversion_rate": 0.041,
                        },
                        {
                            "dimensions": {"deviceCategory": "desktop"},
                            "metrics": {"sessions": 4200, "conversions": 172},
                            "derived_conversion_rate": 0.041,
                        },
                    ]
                },
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
        responses={
            "ga4__get_sessions": {
                "totals": {"sessions": 18900},
                # No step change: noise only.
                "rows": _daily(start, 28, 675, noise=0.06, rng=rng),
            },
            "ga4__compare_periods": {
                "comparison": [
                    {
                        "dimensions": {},
                        "sessions": {"current": 4620, "previous": 4763, "percent_change": -3.0},
                        "sessionConversionRate": {
                            "current": 0.0402,
                            "previous": 0.0399,
                            "percent_change": 0.8,
                        },
                    }
                ]
            },
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
            # The product source agrees with the website source: within noise. Two
            # independent sources both saying "nothing to see" is what makes
            # "insufficient evidence" a finding rather than a shrug.
            "posthog__event_trend": {
                "event": "user signed up",
                "measure": "count",
                "interval": "week",
                "start_date": "2026-05-01",
                "end_date": "2026-05-28",
                "breakdown_property": None,
                "row_count": 4,
                "series": [
                    {"bucket": "2026-05-01T00:00:00", "value": 188},
                    {"bucket": "2026-05-08T00:00:00", "value": 194},
                    {"bucket": "2026-05-15T00:00:00", "value": 191},
                    {"bucket": "2026-05-22T00:00:00", "value": 185},
                ],
                "total": 758,
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
            "posthog__event_trend": DailyTruth(
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
            )
        },
        responses={
            # The refutation, at the granularity where a run-rate is visible. Flat across both
            # months, noise only.
            "ga4__get_sessions": {
                "totals": {"sessions": 6733},
                "rows": _daily(july, 31, 156.4, noise=0.05, rng=rng)
                + _daily(august, 12, 157.0, noise=0.05, rng=rng),
            },
            # Equal-length windows, which is the comparison the question should have been
            # answered with: the last 12 days against the 12 before them.
            "ga4__compare_periods": {
                "comparison": [
                    {
                        "dimensions": {},
                        "sessions": {"current": 1884, "previous": 1871, "percent_change": 0.7},
                        "sessionConversionRate": {
                            "current": 0.0413,
                            "previous": 0.0409,
                            "percent_change": 1.0,
                        },
                    }
                ]
            },
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
            "slack__search_messages": {
                "total_matching": 2,
                "messages": [
                    {
                        "channel": "growth",
                        "user": "priya",
                        "ts": "2026-08-11T16:02:00Z",
                        "text": (
                            "signups look way down this month vs July — is the pricing "
                            "redesign hurting us?"
                        ),
                    },
                    {
                        "channel": "growth",
                        "user": "sam",
                        "ts": "2026-08-11T16:09:00Z",
                        "text": "checking now, might just be the month being young",
                    },
                ],
            },
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
        responses={
            # A real, unmistakable level shift: ~215/day to ~100/day on day 28 (2026-06-17).
            "posthog__event_trend": {
                "event": "user signed up",
                "interval": "day",
                "start_date": start.isoformat(),
                "end_date": (start + timedelta(days=59)).isoformat(),
                "measure": "total_events",
                "row_count": 60,
                "total": 9_300,
                # Generated here rather than through `_daily`, which returns GA4's
                # dimensions/metrics shape. A PostHog trend is bucket/value, and the analysis
                # layer parses `bucket` -- handing it a GA4 row would make the movement
                # invisible rather than wrong, which is the harder failure to notice.
                "series": [
                    {
                        "bucket": f"{(start + timedelta(days=index)).isoformat()}T00:00:00Z",
                        "value": round(
                            # Day 47 is 2026-06-17: 47 days before, 13 after.
                            (215 if index < 47 else 100) * (1 + rng.uniform(-0.07, 0.07))
                        ),
                    }
                    for index in range(60)
                ],
            },
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
                # The stop itself, however the report phrases it.
                ("stopped", "no data after", "collection", "truncated", "ends"),
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
        responses={
            # Thirty days of healthy sessions, then nothing. The rows stop on 3 August while the
            # request runs to the 15th, which is what `_gap` turns into `series_ends_early` --
            # and the fixture states no disclosure of its own, so the scenario fails if the
            # connector's own computation stops working. That is the whole point of it.
            "ga4__get_sessions": {
                "totals": {"sessions": 4991},
                "rows": _daily(july, 19, 158.0, noise=0.05, rng=rng)
                + _daily(date(2026, 8, 1), 3, 161.0, noise=0.05, rng=rng),
            },
            # The second line: signups arrive normally right through the 15th. Flat, with noise,
            # because a perfectly flat series would let a weak analyst score as well as a good
            # one -- and because the claim being supported is "unchanged", which needs a series
            # whose ordinary variation is visible.
            #
            # Also planted under `$pageview` in `subject_responses` below, because the world this
            # scenario asserts is "the site is fine, its measurement stopped" -- and a world where
            # one product event has data while every other returns nothing says the opposite.
            "posthog__event_trend": {
                "event": "user signed up",
                "measure": "count",
                "interval": "day",
                "start_date": "2026-07-16",
                "end_date": "2026-08-15",
                "breakdown_property": None,
                "row_count": 31,
                "series": [
                    {
                        "bucket": f"{(july + timedelta(days=offset)).isoformat()}T00:00:00",
                        "value": round(21 * (1 + rng.uniform(-0.12, 0.12))),
                    }
                    for offset in range(31)
                ],
                "total": 651,
            },
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
        subject_responses={
            "posthog__event_trend": {
                "$pageview": {
                    "event": "$pageview",
                    "measure": "count",
                    "interval": "day",
                    "start_date": "2026-07-16",
                    "end_date": "2026-08-15",
                    "breakdown_property": None,
                    "row_count": 31,
                    "series": [
                        {
                            "bucket": f"{(july + timedelta(days=offset)).isoformat()}T00:00:00",
                            "value": round(1_450 * (1 + rng.uniform(-0.09, 0.09))),
                        }
                        for offset in range(31)
                    ],
                    "total": 44_950,
                },
            },
        },
    )


SCENARIOS: tuple[Scenario, ...] = (
    onboarding_regression(),
    campaign_traffic_drop(),
    insufficient_evidence(),
    partial_month_false_premise(),
    tempting_coincidence(),
    measurement_stopped(),
)


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"no scenario named {name!r}; have: {', '.join(s.name for s in SCENARIOS)}")
