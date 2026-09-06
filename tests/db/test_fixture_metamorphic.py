"""Metamorphic properties every simulated capability must satisfy.

Six defects of one family turned up in these fixtures in a single day, each patched separately:
a date range ignored, daily buckets answered with weekly ones, one event's series returned for
another's, one repository's payload for another's. They are not six bugs. They are five
instances of a named pattern.

Segura et al., *Metamorphic Testing of RESTful Web APIs* (IEEE TSE 44, 2018,
DOI 10.1109/TSE.2017.2764464), catalogues six **Metamorphic Relation Output Patterns** —
equivalence, equality, subset, disjoint, complete, difference — generalised to query-based
systems in *Metamorphic Relation Patterns for Query-Based Systems* (MET 2019). Our connectors are
query-based systems. The defects map one-for-one: an ignored range is **subset**, daily-answered-
as-weekly is **complete**, a wrong subject is **disjoint**.

The degenerate case underneath all of them is the property this file leads with: **a response must
be a function of its request.** A canned payload cannot satisfy it, which is why every one of
those defects was possible and why a guard written per-instance always arrives one field too late.

Deliberately *not* using metamorphic testing's methodology, only its catalogue. MT exists to solve
the oracle problem where correct output is unknown; here the real connector is the oracle and we
know exactly what `interval="day"` means. This is a checklist, applied cheaply.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cortex.eval.fixtures import SCENARIOS, Scenario

#: Capabilities whose response legitimately does not vary with the parameter under test, with the
#: reason. An exemption without a reason is a place to hide a defect.
_SUBJECT_INVARIANT = {
    # Listings describe the tenant, not a queried subject: there is no subject to vary.
    "posthog__list_events",
    "posthog__list_projects",
    "github__list_repositories",
}


def _capabilities_taking(param: str) -> list[tuple[Scenario, str]]:
    """Every (scenario, capability) pair whose real schema accepts `param`."""
    from cortex.tools.registry import gtm_analyst_registry

    registry = gtm_analyst_registry()
    found: list[tuple[Scenario, str]] = []
    for scenario in SCENARIOS:
        for qualified in sorted(scenario.planted_capabilities):
            if qualified in _SUBJECT_INVARIANT:
                continue
            tool_name, capability_name = qualified.split("__", 1)
            try:
                capability = registry.get(tool_name).capability(capability_name)
            except (KeyError, AttributeError):  # pragma: no cover - registry drift
                continue
            if param in (capability.params_schema or {}).get("properties", {}):
                found.append((scenario, qualified))
    return found


class TestAResponseIsAFunctionOfItsRequest:
    """The degenerate metamorphic property, and the one that would have caught five of six.

    A fixture that returns the same payload whatever it is asked is not simulating a connector;
    it is impersonating one connector call. Asserted per subject-bearing capability, because the
    subject is what every one of those defects got wrong.
    """

    @pytest.mark.parametrize("param", ["event", "repo", "number"])
    def test_a_different_subject_gives_a_different_answer(self, param: str) -> None:
        checked = 0
        for scenario, qualified in _capabilities_taking(param):
            asked = _planted_subject(scenario, param)
            planted = scenario.response_for(
                qualified, {param: int(asked) if asked.isdigit() else asked}
            )
            other = scenario.response_for(
                qualified, {param: 0 if asked.isdigit() else "definitely-not-planted"}
            )
            assert planted != other, (
                f"{scenario.name}/{qualified} answers the same for any {param!r}. That is the "
                "disjoint MROP violated, and it is how one event's series came to be returned "
                "for another's."
            )
            checked += 1
        assert checked, f"no capability takes {param!r}: this property is asserting nothing"

    def test_the_unplanted_subject_answers_empty_rather_than_wrongly(self) -> None:
        """ "Nothing here" is a real observation and the analyst must be able to read it. Answering
        with *some other subject's* data is the failure; answering with an empty result of the
        right shape is correct."""
        for scenario, qualified in _capabilities_taking("event"):
            payload = scenario.response_for(qualified, {"event": "definitely-not-planted"})
            series = payload.get("series")
            if series is None:
                continue
            assert series == [], (scenario.name, qualified)
            assert payload.get("total") in (0, None), (scenario.name, qualified)


class TestNarrowingARangeCannotAddData:
    """The **subset** MROP. A window inside another window cannot return more than it.

    This is the property an ignored date range violates, and it is structurally guaranteed for a
    series planted as `DailyTruth` — `_resample` filters one planted list — which is exactly the
    argument for ADR 0006 converting the rest.
    """

    WIDE = {"start_date": "2026-01-01", "end_date": "2026-12-31"}
    NARROW = {"start_date": "2026-06-01", "end_date": "2026-06-07"}

    def test_a_narrower_window_is_a_subset(self) -> None:
        checked = 0
        for scenario, qualified in _capabilities_taking("start_date"):
            subject = _planted_subject(scenario, "event")
            wide = scenario.response_for(qualified, {**self.WIDE, "event": subject})
            narrow = scenario.response_for(qualified, {**self.NARROW, "event": subject})
            wide_buckets = {row["bucket"] for row in wide.get("series") or []}
            narrow_buckets = {row["bucket"] for row in narrow.get("series") or []}
            if not wide_buckets:
                continue
            assert narrow_buckets <= wide_buckets, (
                f"{scenario.name}/{qualified}: a narrower window returned buckets the wider one "
                "did not. The response is not a function of the requested range."
            )
            checked += 1
        assert checked, "no ranged series checked: this property is asserting nothing"


class TestDailyBucketsAggregateToTheirCoarserOnes:
    """The **complete** MROP: a finer partition must sum to the coarser one over the same window.

    This is what "asked for days, got weeks" violates. It also catches the inverse defect a canned
    payload made possible — a declared `total` disagreeing with the series beside it, which one
    fixture did by 18%.
    """

    WINDOW = {"start_date": "2026-06-01", "end_date": "2026-06-28"}

    def test_a_total_equals_its_own_series(self) -> None:
        for scenario, qualified in _capabilities_taking("interval"):
            subject = _planted_subject(scenario, "event")
            for interval in ("day", "week", "month"):
                payload = scenario.response_for(
                    qualified, {**self.WINDOW, "interval": interval, "event": subject}
                )
                series = payload.get("series") or []
                if not series or "total" not in payload:
                    continue
                assert payload["total"] == sum(row["value"] for row in series), (
                    scenario.name,
                    qualified,
                    interval,
                )

    #: Empty, and kept rather than deleted. It held two entries -- `onboarding_regression`'s
    #: segmented event trend and the twin that inherits it -- found by this property on its
    #: first run, two hours after the same payload was exempted from the canned-payload guard on
    #: the reasoning that it "is already daily, so the defect is not reachable there". It was
    #: reachable in the other direction: asking for weeks got days.
    #:
    #: `DailyTruth` grew `segments` so the last canned trend could be derived, and the list
    #: emptied itself. The test below still asserts the set exactly, so a new violation has to
    #: be written down here to pass rather than quietly joining a list nobody reads.
    OPEN_VIOLATIONS: set[tuple[str, str]] = set()

    def test_the_known_violations_are_still_exactly_these(self) -> None:
        """An exemption list nobody checks becomes a place to hide things. If one is fixed, this
        fails and the entry comes out."""
        still_broken = set()
        for scenario, qualified in _capabilities_taking("interval"):
            subject = _planted_subject(scenario, "event")
            payload = scenario.response_for(
                qualified, {**self.WINDOW, "interval": "week", "event": subject}
            )
            if payload.get("interval") not in (None, "week"):
                still_broken.add((scenario.name, qualified))
        assert still_broken == self.OPEN_VIOLATIONS

    def test_the_requested_interval_is_the_one_returned(self) -> None:
        for scenario, qualified in _capabilities_taking("interval"):
            if (scenario.name, qualified) in self.OPEN_VIOLATIONS:
                continue
            subject = _planted_subject(scenario, "event")
            for interval in ("day", "week", "month"):
                payload = scenario.response_for(
                    qualified, {**self.WINDOW, "interval": interval, "event": subject}
                )
                if "interval" not in payload:
                    continue
                assert payload["interval"] == interval, (
                    f"{scenario.name}/{qualified} answered {payload['interval']!r} for a request "
                    f"of {interval!r}. This exact defect cost score in two scenarios."
                )


def _planted_subject(scenario: Scenario, param: str) -> str:
    """A subject this scenario actually plants, so the comparison is planted-vs-absent."""
    if param == "event":
        described = sorted(scenario.events_described())
        return described[0] if described else "user signed up"
    if param == "number":
        # The records a subject-keyed capability is keyed by. Read from the scenario for the
        # same reason every other derivation is: a hardcoded number goes stale silently.
        numbers = sorted(
            subject
            for subjects in scenario.subject_responses.values()
            for subject in subjects
            if subject.isdigit()
        )
        return numbers[0] if numbers else "913"
    described = sorted(scenario.repositories_described())
    return described[0] if described else "acme/product"


class TestEveryConnectorAScenarioOffersIsLoadBearing:
    """Capability ablation, the cheap deterministic half.

    The defect this exists for: `campaign_traffic_drop` had PostHog **disconnected** — a
    capability planted in a newer field was invisible to `connected_tools` — so the analyst was
    never offered `posthog__event_trend`, the daily series built for that scenario could not be
    reached at all, and it still scored 1.00 from GA4 alone. The score was not wrong about the
    answer; it was measuring something else entirely, and nothing here could tell.

    The general form, from ABC (arXiv:2507.02825, T.9 and its trivial-agent baseline): if a
    scenario scores the same with a connector removed, it does not depend on that connector.
    The full version needs an eval run per connector per scenario and costs real tokens; these
    are the two halves that are free and deterministic, and either one alone would have caught it.
    """

    def test_every_planted_capability_is_reachable(self) -> None:
        """Data the analyst cannot reach is indistinguishable from data it chose not to use."""
        for scenario in SCENARIOS:
            for qualified in scenario.planted_capabilities:
                tool = qualified.split("__")[0]
                assert tool in scenario.connected_tools, (
                    f"{scenario.name} plants {qualified} and does not offer {tool}. The analyst "
                    "cannot call it, so the data is unreachable and the scenario scores on "
                    "whatever else it has."
                )

    def test_every_offered_connector_has_something_behind_it(self) -> None:
        """The converse, and it costs steps rather than correctness: a connector offered with
        nothing planted is one the analyst spends turns calling for an empty answer. Production
        does not do this — `registry_for_tenant` offers only connectors with credentials — so a
        scenario that does is measuring a loop the product does not have.
        """
        for scenario in SCENARIOS:
            planted_tools = {q.split("__")[0] for q in scenario.planted_capabilities}
            required_tools = {
                alternative.split("__")[0]
                for requirement in scenario.ground_truth.required_capabilities
                for alternative in _alternatives(requirement)
            }
            offered = set(scenario.connected_tools)
            assert offered <= planted_tools | required_tools, (
                f"{scenario.name} offers {sorted(offered - (planted_tools | required_tools))} "
                "with nothing planted and nothing required."
            )

    def test_every_required_capability_is_actually_planted(self) -> None:
        """`required_capabilities` is what `tool_selection` scores against. A requirement naming
        a capability the scenario plants nothing for asks the analyst to make a call that can
        only return empty — which it is then scored for making."""
        for scenario in SCENARIOS:
            for requirement in scenario.ground_truth.required_capabilities:
                alternatives = _alternatives(requirement)
                assert any(a in scenario.planted_capabilities for a in alternatives), (
                    f"{scenario.name} requires {alternatives} and plants none of them."
                )


def _alternatives(requirement: object) -> tuple[str, ...]:
    from cortex.eval.fixtures import alternatives_of

    return alternatives_of(requirement)  # type: ignore[arg-type]


class TestProjectedRecordsHonourTheirRequest:
    """ADR 0006's first landing, asserted as the properties it exists to make structural.

    These are the same MROPs the trend series already satisfies — disjoint on subject, subset
    under narrowing — now for dated records. The difference from every earlier guard is that a
    projection cannot fail them by omission: filtering is how the response is produced, so there
    is no path where a fixture author forgets.
    """

    def test_a_different_subject_returns_nothing(self) -> None:
        """**Disjoint.** Four of the six defects were this: repo A's payload for a request about
        repo B, one event's series for another's."""
        for scenario in SCENARIOS:
            for qualified, planted in scenario.dated_records.items():
                if planted.subject is None:
                    continue
                mine = scenario.response_for(qualified, {planted.subject_param: planted.subject})
                theirs = scenario.response_for(
                    qualified, {planted.subject_param: "acme/definitely-not-planted"}
                )
                assert theirs["count"] == 0, (scenario.name, qualified)
                assert theirs[planted.key] == [], (scenario.name, qualified)
                if planted.records:
                    assert mine["count"] > 0, (scenario.name, qualified)

    def test_a_narrower_window_cannot_add_records(self) -> None:
        """**Subset.** An ignored range is what made scenarios report collection failures to an
        analyst who asked wider than was planted."""
        for scenario in SCENARIOS:
            for qualified, planted in scenario.dated_records.items():
                if not planted.since_param or not planted.records:
                    continue
                subject = {planted.subject_param: planted.subject} if planted.subject else {}
                wide = scenario.response_for(
                    qualified, {**subject, planted.since_param: "2020-01-01"}
                )
                narrow = scenario.response_for(
                    qualified, {**subject, planted.since_param: "2099-01-01"}
                )
                assert narrow["count"] <= wide["count"], (scenario.name, qualified)
                assert narrow["count"] == 0, (scenario.name, qualified)

    def test_an_empty_result_stays_readable(self) -> None:
        """ "Looked and found nothing" must remain distinguishable from "nobody looked". The
        envelope survives an empty match — which is what the decline twins depend on, and what
        an empty list alone cannot say."""
        for scenario in SCENARIOS:
            for qualified, planted in scenario.dated_records.items():
                empty = scenario.response_for(
                    qualified, {planted.subject_param: "acme/definitely-not-planted"}
                )
                for field_name in planted.envelope:
                    assert field_name in empty, (scenario.name, qualified, field_name)
                assert planted.key in empty, (scenario.name, qualified)

    def test_a_limit_is_honoured(self) -> None:
        for scenario in SCENARIOS:
            for qualified, planted in scenario.dated_records.items():
                if len(planted.records) < 2:
                    continue
                subject = {planted.subject_param: planted.subject} if planted.subject else {}
                capped = scenario.response_for(qualified, {**subject, "limit": 1})
                assert capped["count"] == 1, (scenario.name, qualified)


class TestEveryGA4CapabilityAgreesAboutTheSameWindow:
    """The **equality** MROP, across capabilities rather than across requests.

    Four GA4 capabilities describe the same traffic. Asked about the same window they must
    report the same number of sessions, because there is only one number of sessions.

    They did not. `campaign_traffic_drop` served 13,239 sessions from the daily series, 11,300
    from the period comparison and 11,300 from the device funnel for 15-30 June: an analyst
    citing the series and the comparison in one paragraph cited a 15% contradiction, and did,
    and was marked down for it. Every scenario also declared a `totals.sessions` its own rows
    contradicted -- by 13% on `onboarding_regression`, 30% on `measurement_stopped`.

    None of it was findable by reading one payload. It is what happens when one fact is written
    down four times, and it is unrepresentable now: `MetricSeries` holds the fact once and each
    capability is a projection. This asserts the projections stay projections.
    """

    WINDOW = {"start_date": "2026-06-15", "end_date": "2026-06-30"}

    def _series(self) -> list[tuple[Scenario, object]]:
        return [(s, s.metric_series) for s in SCENARIOS if s.metric_series is not None]

    def test_every_scenario_declares_its_ga4_world_once(self) -> None:
        """A scenario must not plant a payload for a capability its series already answers.

        The second figure is the whole defect. A planted payload would be shadowed by the
        projection rather than served, which is worse than a contradiction: it is a number
        somebody maintains that nobody reads.
        """
        from cortex.eval.fixtures import GA4_SERIES_CAPABILITIES

        # Every scenario, not only the ones declaring a series: GA4 is answered by
        # `MetricSeries` or not at all, so a canned payload for one of these capabilities is a
        # second implementation whether or not a series sits beside it.
        for scenario in SCENARIOS:
            planted = set(scenario.responses) | set(scenario.subject_responses)
            overlap = planted & GA4_SERIES_CAPABILITIES
            assert not overlap, (
                f"{scenario.name} plants {sorted(overlap)} as a canned payload beside a "
                "MetricSeries. Every GA4 capability is a projection of the series, so a second "
                "payload is either shadowed and unread, or served and contradicting."
            )

    def test_the_funnel_and_the_session_series_report_the_same_traffic(self) -> None:
        checked = 0
        for scenario, series in self._series():
            if "ga4__get_funnel" not in series.capabilities:  # type: ignore[attr-defined]
                continue
            window = {
                "start_date": min(day for day, _ in series.daily).isoformat(),  # type: ignore[attr-defined]
                "end_date": max(day for day, _ in series.daily).isoformat(),  # type: ignore[attr-defined]
            }
            for asked in (window, self.WINDOW):
                sessions = scenario.response_for("ga4__get_sessions", asked)["totals"]
                funnel = scenario.response_for("ga4__get_funnel", asked)["totals"]
                assert sessions.get("sessions", 0) == funnel.get("sessions", 0), (
                    f"{scenario.name}: the funnel and the session series disagree about "
                    f"{asked}. They describe the same traffic."
                )
            checked += 1
        assert checked, "no scenario checked: this property is asserting nothing"

    def test_the_comparison_and_the_session_series_report_the_same_traffic(self) -> None:
        checked = 0
        for scenario, series in self._series():
            first = min(day for day, _ in series.daily)  # type: ignore[attr-defined]
            last = max(day for day, _ in series.daily)  # type: ignore[attr-defined]
            midpoint = first + (last - first) / 2
            comparison = scenario.response_for(
                "ga4__compare_periods",
                {
                    "current_start": midpoint.isoformat(),
                    "current_end": last.isoformat(),
                    "previous_start": first.isoformat(),
                    "previous_end": (midpoint - timedelta(days=1)).isoformat(),
                },
            )["comparison"]
            current = sum(row["sessions"]["current"] or 0 for row in comparison)
            expected = scenario.response_for(
                "ga4__get_sessions",
                {"start_date": midpoint.isoformat(), "end_date": last.isoformat()},
            )["totals"]["sessions"]
            assert current == expected, (
                f"{scenario.name}: the period comparison and the session series disagree about "
                f"{midpoint}..{last}."
            )
            checked += 1
        assert checked, "no scenario checked: this property is asserting nothing"

    def test_total_conversions_do_not_depend_on_the_dimension_asked_for(self) -> None:
        """A breakdown is a way of dividing a total, not a second total.

        Two breakdowns of one series can only disagree through rounding, so this allows one
        conversion per segment and no more. A wider gap means the shares or the rates do not
        describe the same population.
        """
        for scenario, series in self._series():
            breakdowns = series.breakdowns  # type: ignore[attr-defined]
            if len(breakdowns) < 2:
                continue
            window = {
                "start_date": min(day for day, _ in series.daily).isoformat(),  # type: ignore[attr-defined]
                "end_date": max(day for day, _ in series.daily).isoformat(),  # type: ignore[attr-defined]
            }
            totals = {
                dimension: scenario.response_for(
                    "ga4__get_funnel", {**window, "dimension": dimension}
                )["totals"]["conversions"]
                for dimension in breakdowns
            }
            slack = max(len(segments) for segments in breakdowns.values())
            assert max(totals.values()) - min(totals.values()) <= slack, (
                f"{scenario.name}: total conversions depend on which dimension was asked "
                f"for -- {totals}. A breakdown divides a total; it does not restate one."
            )


class TestAProjectIsPartOfTheQuestion:
    """The **disjoint** MROP on the dimension every PostHog capability carries and none honoured.

    `onboarding_regression` advertises two projects — `web-app` and `oss-client` — and every
    capability answered both with the same series. An analyst that queried the wrong project was
    rewarded exactly as well as one that read the listing and chose, which makes the listing a
    decoration rather than a step.

    Two directions, because either alone is satisfiable by a fixture that has stopped working:
    the tenant's own project must answer as if no project had been named, and another project
    must answer empty.
    """

    def test_the_tenants_own_project_answers_as_if_unnamed(self) -> None:
        checked = 0
        for scenario, qualified in _posthog_capabilities():
            default = _default_project(scenario)
            asked = {"event": _planted_subject(scenario, "event")}
            assert scenario.response_for(qualified, asked) == scenario.response_for(
                qualified, {**asked, "project": default}
            ), f"{scenario.name}/{qualified} answers its own project differently from no project"
            checked += 1
        assert checked, "no PostHog capability checked: this property is asserting nothing"

    def test_another_project_answers_empty(self) -> None:
        checked = 0
        for scenario, qualified in _posthog_capabilities():
            asked = {"event": _planted_subject(scenario, "event"), "project": "999999"}
            payload = scenario.response_for(qualified, asked)
            rows = (payload.get("series") or []) + (payload.get("rows") or [])
            rows += payload.get("events") or []
            assert not rows, (
                f"{scenario.name}/{qualified} returned this project's data for another "
                "project's question. The project is part of the question."
            )
            checked += 1
        assert checked, "no PostHog capability checked: this property is asserting nothing"


def _posthog_capabilities() -> list[tuple[Scenario, str]]:
    """Every (scenario, capability) pair a scenario plants behind a PostHog project."""
    return [
        (scenario, qualified)
        for scenario in SCENARIOS
        for qualified in sorted(scenario.planted_capabilities)
        if qualified.startswith("posthog__") and qualified != "posthog__list_projects"
    ]


def _default_project(scenario: Scenario) -> str:
    from cortex.eval.fixtures import DISCOVERY_DEFAULTS

    listing = (
        scenario.responses.get("posthog__list_projects")
        or DISCOVERY_DEFAULTS["posthog__list_projects"]
    )
    return str(listing["default_project"])


class TestNarrowingARangeCannotAddRows:
    """The **subset** MROP again, for row-shaped payloads rather than bucketed series.

    The bucket version above could not see GA4 at all: those payloads carry `rows`, not
    `series`, and every one of them answered a narrow window with the same thirty days it
    answered a wide one. Two properties, because the shapes are two and a defect in either is
    the same defect.
    """

    WIDE = {"start_date": "2026-01-01", "end_date": "2026-12-31"}
    NARROW = {"start_date": "2026-06-15", "end_date": "2026-06-21"}

    def test_a_narrower_window_is_a_subset(self) -> None:
        checked = 0
        for scenario, qualified in _capabilities_taking("start_date"):
            wide = scenario.response_for(qualified, self.WIDE)
            narrow = scenario.response_for(qualified, self.NARROW)
            if not isinstance(wide, dict) or not isinstance(narrow, dict):
                continue
            wide_dates = _dates_in(wide)
            narrow_dates = _dates_in(narrow)
            if not wide_dates:
                continue
            assert narrow_dates <= wide_dates, (
                f"{scenario.name}/{qualified}: a narrower window returned dates the wider one "
                "did not. The response is not a function of the requested range."
            )
            assert not (narrow_dates - set(_between(self.NARROW))), (
                f"{scenario.name}/{qualified}: returned rows outside the window it was asked "
                "for. This is the defect that made an analyst report a figure for the wrong "
                "seven days."
            )
            checked += 1
        assert checked, "no row-shaped capability checked: this property is asserting nothing"


def _dates_in(payload: dict) -> set[str]:
    """Every date a row-shaped payload places itself on."""
    return {
        (row.get("dimensions") or {}).get("date")
        for row in payload.get("rows") or []
        if isinstance(row, dict) and (row.get("dimensions") or {}).get("date")
    }


def _between(window: dict[str, str]) -> list[str]:
    start = date.fromisoformat(window["start_date"])
    end = date.fromisoformat(window["end_date"])
    return [(start + timedelta(days=n)).isoformat() for n in range((end - start).days + 1)]
