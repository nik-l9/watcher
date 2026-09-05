"""The evaluation suite.

The suite is what gates the first real investigation, so it needs to be trustworthy
before it is trusted. The tests that matter most are the ones that prove it **fails**
on a bad analyst — a scorer that only ever passes is worse than no scorer, because it
converts an unknown into false confidence.

Every run uses a scripted provider, so the harness is exercised deterministically and
for free. What is measured here is the harness, not a model.
"""

from __future__ import annotations

import json
import uuid
from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agents.investigator import Investigator
from cortex.agents.llm import (
    LLMOutputTruncated,
    LLMResponse,
    RecordedLLM,
    ToolRequest,
    Usage,
)
from cortex.db.models import Evidence, ToolCall
from cortex.eval.fixtures import SCENARIOS, Difficulty, alternatives_of, by_name
from cortex.eval.replay import FAILURES_DIRNAME
from cortex.eval.runner import EvalHarness, EvalRun, ScenarioOutcome, scenario_registry
from cortex.eval.scorer import Scorer
from cortex.tools.disclosures import grade_series, series_disclosures

#: Valid arguments per capability. The scenario tools carry the *real* schemas, so a
#: call with missing required parameters is rejected before it reaches the handler —
#: which is the schema/handler contract working, not something to route around.
_VALID_ARGS: dict[str, dict[str, object]] = {
    "ga4__get_sessions": {"start_date": "2026-07-01", "end_date": "2026-07-21"},
    "ga4__get_funnel": {"start_date": "2026-07-15", "end_date": "2026-07-21"},
    "ga4__top_pages": {"start_date": "2026-07-15", "end_date": "2026-07-21"},
    "ga4__compare_periods": {
        "current_start": "2026-07-15",
        "current_end": "2026-07-21",
        "previous_start": "2026-07-08",
        "previous_end": "2026-07-14",
    },
    "github__deployment_history": {"repo": "acme/web"},
    "github__recent_prs": {"repo": "acme/web"},
    "github__commits": {"repo": "acme/web", "since": "2026-07-12"},
    "github__pull_request_activity": {"repo": "acme/web", "number": 913},
    "github__issues": {"repo": "acme/web", "since": "2026-07-14"},
    "hubspot__contacts": {"start_date": "2026-07-15", "end_date": "2026-07-21"},
    "posthog__event_trend": {
        "event": "user signed up",
        "start_date": "2026-06-01",
        "end_date": "2026-08-12",
    },
    "slack__find_decision": {"topic": "onboarding"},
    "slack__search_messages": {"query": "onboarding"},
}


def _call(qualified: str, **overrides: object) -> LLMResponse:
    """One scripted tool-calling turn, with arguments the real schema accepts."""
    arguments = {**_VALID_ARGS.get(qualified, {}), **overrides}
    return LLMResponse(
        text=f"Checking {qualified}.",
        tool_requests=[
            ToolRequest(id=f"t_{uuid.uuid4().hex[:6]}", name=qualified, arguments=arguments)
        ],
        usage=Usage(input_tokens=400, output_tokens=90),
    )


def _done() -> LLMResponse:
    return LLMResponse(text="Enough to conclude.", usage=Usage(input_tokens=300, output_tokens=60))


def _competent_calls(scenario_name: str) -> list[LLMResponse]:
    """The turns a scenario's own ground truth says a competent analyst makes.

    Read from the scenario rather than written out here. Hardcoded call lists went stale
    the moment a scenario gained a required capability: two tests that meant "this
    analyst did everything right" started asserting a perfect tool_selection score for an
    analyst that had skipped a call.
    """
    scenario = by_name(scenario_name)
    # The first alternative of an any-of requirement. Which one is arbitrary: the point of
    # any-of is that either satisfies the requirement, so a test that means "did
    # everything right" only needs to pick one.
    return [_call(alternatives_of(r)[0]) for r in scenario.ground_truth.required_capabilities]


#: A hypothesis every scripted report carries, so a test double is not the stub shape the
#: drafting call now retries on.
#:
#: `_is_degenerate` rejects a draft with no findings *and* no hypotheses, because two live
#: attempts returned exactly that -- one "placeholder" claim and nothing else -- and nothing
#: noticed until the citation gate discarded the whole investigation. The test builders were
#: producing that same shape, so they now say what they tested.
def _tested(
    evidence_id: uuid.UUID,
    #: Deliberately not any scenario's decoy. The first version said "a deploy caused it", and
    #: `deploy` is a decoy in four of the five scenarios -- so contradicting it read as "tested
    #: and ruled out" and cancelled a summary that was blaming the deploy, turning a report the
    #: test wants penalised into one scoring 1.00 on decoy rejection.
    statement: str = "A bank holiday reduced traffic that week",
) -> dict:
    return {
        "statement": statement,
        "verdict": "contradicted",
        "contradicting_evidence_ids": [str(evidence_id)],
        "reasoning": "scripted",
    }


def _undraftable(_evidence_ids):  # type: ignore[no-untyped-def]
    """A report that never arrives, the way the real one didn't.

    Raises rather than returning a bad payload, because truncation is a provider-level
    failure: there is no draft to hand back and no validation error to explain. Both live
    deaths in run 18 came through this path.
    """
    raise LLMOutputTruncated("scripted: output truncated at max_tokens")


class _ScriptedAnalyst(RecordedLLM):
    """A provider whose report cites whatever evidence the loop actually gathered.

    The alternative — hardcoding ids — is impossible, since ids are minted at write
    time. `report_builder` receives the observed ids and returns a report payload, so a
    test can simulate a well-behaved analyst, a fabricating one, or a lazy one.
    """

    def __init__(
        self,
        completions,  # type: ignore[no-untyped-def]
        report_builder,  # type: ignore[no-untyped-def]
        verdict: str = "supported",
        sufficient: bool = True,
    ) -> None:
        super().__init__(completions=completions)
        self._build = report_builder
        self._verdict = verdict
        self._sufficient = sufficient
        self._seen_ids: list[uuid.UUID] = []

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        # Harvest the evidence ids the loop fed back, exactly as the real analyst
        # would read them.
        for message in kwargs.get("messages", []):
            for rendered in message.tool_results.values():
                for line in rendered.splitlines():
                    if line.startswith("evidence_id: "):
                        self._seen_ids.append(uuid.UUID(line.split(": ", 1)[1].strip()))
        return await super().complete(**kwargs)

    async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
        """Route by schema, and refuse to guess.

        Dispatch used to be "has a `verdict` property, or else it is the report", and the
        else-branch also cleared `_seen_ids`. Adding the sufficiency gate — a third kind of
        structured call — sent it down the report branch, which drafted a report early and wiped
        the harvested ids, so the real drafting call then cited nothing and three tests failed
        with `IndexError` several frames from the cause.

        So each schema is now matched explicitly and an unrecognised one raises. A test double
        that silently treats an unknown call as the one it knows about does not fail where the
        change was made.
        """
        properties = kwargs.get("schema", {}).get("properties", {})
        if "verdict" in properties:
            return {"verdict": self._verdict, "reason": "scripted"}, Usage(
                input_tokens=80, output_tokens=20
            )
        if "sufficient" in properties:
            return {
                "sufficient": self._sufficient,
                "missing": [] if self._sufficient else ["a control series"],
                "reason": "scripted",
            }, Usage(input_tokens=60, output_tokens=20)
        if "establishes_cause" in properties:
            # The fresh-context re-derivation: the verifier asks the evidence an open question
            # and never shows it the claim, so this answer is about the *evidence*, not about
            # whether the drafted sentence was right. Dates are ordered cause-then-movement so
            # the onset elimination rule does not fire on a scripted supported verdict.
            establishes = self._verdict == "supported"
            return {
                "establishes_cause": establishes,
                "cause": "a scripted change" if establishes else None,
                "cause_date": "2026-06-15" if establishes else None,
                "movement_date": "2026-06-17" if establishes else None,
                "what_the_evidence_shows": "scripted reading of the cited observations",
            }, Usage(input_tokens=70, output_tokens=20)
        if "executive_summary" not in properties:
            raise AssertionError(
                f"unrecognised structured schema with properties {sorted(properties)}; "
                "add a branch here rather than letting it fall through to the report builder"
            )
        payload = self._build(list(dict.fromkeys(self._seen_ids)))
        # Cleared once the report is drafted, so a second investigation through the same
        # provider cites its own evidence. Without this, a repeated run had attempt 2
        # citing attempt 1's ids -- and the gate correctly counted that as a
        # hallucination, since those rows belong to a different investigation.
        self._seen_ids.clear()
        return payload, Usage(input_tokens=500, output_tokens=200)


def _good_report(scenario_name: str):  # type: ignore[no-untyped-def]
    """A report that names the planted cause and cites real evidence."""
    scenario = by_name(scenario_name)
    truth = scenario.ground_truth

    def _build(ids: list[uuid.UUID]) -> dict:
        # Flattened, taking one alternative per requirement: a report satisfying any of
        # them satisfies the requirement.
        signals = " ".join(alternatives_of(r)[0] for r in truth.required_signals)
        if truth.is_false_premise:
            # Read from `refutation_signals` for the same reason `_competent_calls` reads from
            # the scenario: a hardcoded sentence would keep asserting "a good report passes"
            # after the label it is supposed to satisfy had moved.
            summary = (
                "Signups "
                + " and ".join(alternatives_of(r)[0] for r in truth.refutation_signals)
                + " — the run-rate is unchanged."
            )
        elif truth.required_signals:
            summary = f"Signups fell. The cause was deploy {signals}."
        else:
            summary = "A 3% move is within normal variation; the data cannot establish a cause."
        return {
            "question": scenario.question,
            "executive_summary": [{"text": summary, "evidence_ids": [str(i) for i in ids[:2]]}],
            "findings": [
                {
                    "title": "Segment breakdown",
                    "claims": [
                        {
                            "text": f"The change concentrates in {signals or 'no segment'}.",
                            "evidence_ids": [str(ids[0])],
                        }
                    ],
                }
            ],
            "hypotheses": [
                {
                    "statement": "The change has a single identifiable cause",
                    "verdict": "inconclusive" if truth.declines_a_cause else "supported",
                    **(
                        {} if truth.declines_a_cause else {"supporting_evidence_ids": [str(ids[0])]}
                    ),
                }
            ],
            # Empty for both classes that decline a cause. On a false premise there is no
            # effect to act on, and a recommendation naming the deploy would assert a decoy as
            # the reason for a fall that did not happen.
            "recommendations": (
                []
                if truth.declines_a_cause
                else [
                    {
                        "action": f"Roll back the change in {signals}",
                        "rationale": "It is the only change coincident with the drop.",
                        "evidence_ids": [str(ids[0])],
                    }
                ]
            ),
            "risks": [{"description": "GA4 figures may be sampled."}],
            "confidence": "insufficient_evidence" if truth.is_unanswerable else "high",
        }

    return _build


class TestReadingARecommendation:
    """Whether a recommendation proposes *acting* or proposes *looking*.

    This one function has now been wrong three times, each time penalising a correct
    report, so every sentence that has actually caused a misread is pinned here. All three
    failures shared a shape: a sentence about checking something, containing a noun that
    happened to look like an action verb.
    """

    def test_a_request_to_look_is_not_a_proposal_to_act(self) -> None:
        from cortex.eval.scorer import _proposes_action

        # Scored as "deploy" by substring matching, because "deployment" contains it.
        assert not _proposes_action(
            "Confirm HubSpot, GitHub deployment, and BigQuery connector health given "
            "that all returned empty result sets"
        )
        # Scored as "deploy" by word-boundary matching, because a hyphen is a boundary.
        assert not _proposes_action(
            "Confirm the correct repository and date range are being queried in GitHub "
            "before ruling out a deploy-related cause"
        )
        assert not _proposes_action(
            "Instrument the onboarding modal with PostHog events to capture the failure "
            "mode on small viewports"
        )
        assert not _proposes_action("Re-run the channel-mix comparison with segmentation")

    def test_a_proposal_to_act_is_still_recognised(self) -> None:
        from cortex.eval.scorer import _proposes_action

        assert _proposes_action("Roll back PR #913 on mobile")
        # An adverbial opener must not hide the verb.
        assert _proposes_action("Immediately revert the onboarding modal")
        assert _proposes_action("Disable the new sheet for small viewports")

    def test_advice_against_acting_is_not_advice_to_act(self) -> None:
        from cortex.eval.scorer import _is_advice_against, _proposes_action

        advice = "Do not roll back or alter the signup flow on the strength of this report"
        assert _is_advice_against(advice)
        assert not (_proposes_action(advice) and not _is_advice_against(advice))


class TestFixtures:
    def test_every_scenario_is_labeled(self) -> None:
        for scenario in SCENARIOS:
            assert scenario.question
            assert scenario.ground_truth.cause
            assert scenario.responses, scenario.name

    def test_every_scenario_carries_decoys(self) -> None:
        """A scenario with one plausible explanation scores a coin-flip guesser as
        perfect, so a decoy-free scenario measures nothing."""
        for scenario in SCENARIOS:
            assert scenario.ground_truth.decoys, scenario.name

    def test_the_difficulty_classes_are_all_represented(self) -> None:
        """Without an unanswerable case the suite rewards always naming a cause."""
        assert {s.difficulty for s in SCENARIOS} == set(Difficulty)

    def test_generation_is_deterministic(self) -> None:
        """A score change must mean the analyst changed, not the fixture."""
        from cortex.eval.fixtures import onboarding_regression

        assert onboarding_regression(seed=7) == onboarding_regression(seed=7)
        assert onboarding_regression(seed=7) != onboarding_regression(seed=8)

    def test_an_unplanted_call_returns_an_empty_observation(self) -> None:
        """Exploring an unsupported hypothesis should look like "nothing here", which
        is a real observation, not a tool failure.

        Uses a *measuring* capability. This test used `bigquery__list_queries` until the
        environment survey existed; a discovery capability now falls back to
        `DISCOVERY_DEFAULTS` instead, because an empty estate is a false premise rather than an
        empty observation — see the test below."""
        scenario = by_name("onboarding_regression")
        assert scenario.response_for("bigquery__run_named_query") == {"rows": [], "count": 0}

    def test_every_discovery_capability_surveys_something(self) -> None:
        """The drift guard on the survey.

        The investigator runs every discovery capability before the first step, so one with no
        planted response would open each investigation by reporting that the tenant has no
        repositories and no projects. That is not a neutral empty result — it is a false premise
        that pushes a competent analyst towards concluding the data does not exist.

        This fails on the commit that adds a discovery capability without an environment for it,
        which is the same class of drift that once let six new capabilities go unmeasured."""
        from cortex.eval.fixtures import DISCOVERY_DEFAULTS, SCENARIOS
        from cortex.tools.registry import gtm_analyst_registry

        declared = gtm_analyst_registry().discovery_capabilities()
        assert declared, "no discovery capabilities are declared at all"
        for scenario in SCENARIOS:
            for qualified in declared:
                planted = qualified in scenario.responses or qualified in DISCOVERY_DEFAULTS
                assert planted, f"{scenario.name} surveys {qualified} and gets nothing"
                payload = scenario.response_for(qualified)
                assert payload != {"rows": [], "count": 0}, (
                    f"{scenario.name}: {qualified} surveys an empty world"
                )

    def test_the_fixture_tools_mirror_the_discovery_flag(self) -> None:
        """A fixture that dropped the flag would exercise a loop the product does not have:
        no survey, a smaller opening prompt, and fewer tool calls than a real investigation."""
        from cortex.eval.fixtures import SCENARIOS
        from cortex.eval.runner import scenario_registry
        from cortex.tools.registry import gtm_analyst_registry

        scenario = SCENARIOS[0]
        surveyed = set(scenario_registry(scenario).discovery_capabilities())
        production = set(gtm_analyst_registry().discovery_capabilities())

        # Every discovery capability the scenario's tenant has must still be surveyed. The
        # comparison is a subset rather than equality because the surface is filtered to the
        # tenant's own connectors, so a discovery capability belonging to a connector this tenant
        # does not have is correctly absent rather than missing.
        assert surveyed <= production
        assert surveyed == {
            qualified
            for qualified in production
            if qualified.split("__")[0] in scenario.connected_tools
        }
        assert surveyed, "a scenario that surveys nothing would open blind"

    def test_the_campaign_fixture_answers_differently_on_each_side_of_the_change(
        self,
    ) -> None:
        """A scenario about a change must be able to answer per period.

        One canned payload per capability was served to every call, so an analyst asking
        for the funnel before and after the drop received byte-identical figures.
        """
        scenario = by_name("campaign_traffic_drop")
        before = scenario.response_for("ga4__get_funnel", {"start_date": "2026-06-01"})
        after = scenario.response_for("ga4__get_funnel", {"start_date": "2026-06-16"})
        assert before != after

    def test_the_campaign_fixture_reconciles_with_its_own_story(self) -> None:
        """The fixture must not contradict itself.

        A live investigation reported that the device funnel totals could not be
        reconciled with the channel totals, and it was right — the contradiction was
        planted by the fixture. An analyst that has to reason about impossible data is
        being scored on the wrong thing, so the arithmetic is asserted here.
        """
        scenario = by_name("campaign_traffic_drop")
        channels = scenario.responses["ga4__compare_periods"]["comparison"]
        expected_before = sum(row["sessions"]["previous"] for row in channels)
        expected_after = sum(row["sessions"]["current"] for row in channels)

        def _total(params: dict[str, str]) -> int:
            rows = scenario.response_for("ga4__get_funnel", params)["rows"]
            return sum(row["metrics"]["sessions"] for row in rows)

        assert _total({"start_date": "2026-06-01"}) == expected_before
        assert _total({"start_date": "2026-06-16"}) == expected_after

    def test_the_campaign_fixture_holds_conversion_rate_flat(self) -> None:
        """Volume moves, rate does not — the observation that rules out a regression."""
        scenario = by_name("campaign_traffic_drop")

        def _rates(params: dict[str, str]) -> list[float]:
            rows = scenario.response_for("ga4__get_funnel", params)["rows"]
            return [round(r["metrics"]["conversions"] / r["metrics"]["sessions"], 3) for r in rows]

        assert _rates({"start_date": "2026-06-01"}) == _rates({"start_date": "2026-06-16"})

    def test_the_planted_cause_is_not_the_only_deploy(self) -> None:
        """If the deploy list had one entry, naming it would require no reasoning."""
        scenario = by_name("onboarding_regression")
        deploys = scenario.responses["github__deployment_history"]["deployments"]
        assert len(deploys) > 1

    def test_every_planted_response_names_a_real_capability(self) -> None:
        """The fixtures must describe the tool surface that actually exists.

        Written after the connectors gained `commits`, `pull_request_activity`, `issues`
        and the PostHog project capabilities while the fixtures kept describing the old
        surface. Nothing failed: the scenarios just quietly stopped exercising the new
        calls, so the suite reported on a product that no longer existed. A typo in a
        capability name fails the same way — the fixture serves the generic empty
        payload and the scenario silently loses its planted evidence.
        """
        from cortex.tools.registry import gtm_analyst_registry

        registry = gtm_analyst_registry()
        for scenario in SCENARIOS:
            planted = set(scenario.responses) | set(scenario.period_responses)
            planted |= set(scenario.ground_truth.capability_names)
            for qualified in sorted(planted):
                tool_name, _, capability = qualified.partition("__")
                assert tool_name in registry.tool_names, f"{scenario.name}: {qualified}"
                assert capability in registry.get(tool_name).capability_names, (
                    f"{scenario.name}: {qualified}"
                )

    def test_an_any_of_requirement_is_satisfied_by_either_alternative(self) -> None:
        """A correct answer must not be marked wrong for how it names the change.

        A live run identified the cause as "PR #913, merged 2026-07-14, whose reviewer
        flagged the continue button below the fold on a 375px viewport", recommended the
        exact fix, and scored 0.5 for accuracy because the label demanded the deploy sha.
        The sha and the PR number identify the same commit.
        """
        from cortex.eval.fixtures import alternatives_of, describe_requirement

        assert alternatives_of("mobile") == ("mobile",)
        assert alternatives_of(("91c3e4a", "913")) == ("91c3e4a", "913")
        assert describe_requirement("mobile") == "mobile"
        assert describe_requirement(("a", "b")) == "any of [a, b]"

        truth = by_name("onboarding_regression").ground_truth
        assert any(isinstance(r, tuple) for r in truth.required_signals), (
            "the scenario should express the sha/PR alternatives as any-of"
        )

    def test_no_fixture_deployment_is_called_production(self) -> None:
        """The suite must not teach an environment name that does not generalise.

        Every fixture called it "production". A live investigation copied that habit,
        filtered the real repository on `environment=production`, got nothing back, and
        had to reason around the silence — the environments there are `dev-deploy` and
        `staging - docs`. The fixtures now use a different name and always carry
        `environments_available`, so the analyst learns to read what exists.
        """
        for scenario in SCENARIOS:
            payload = scenario.responses.get("github__deployment_history")
            if payload is None:
                continue
            assert "environments_available" in payload, scenario.name
            for deployment in payload["deployments"]:
                assert deployment["environment"] != "production", scenario.name

    def test_the_onboarding_scenario_exercises_the_human_record(self) -> None:
        """A senior analyst reads the change, not only the metric.

        The review thread on the planted PR names the exact failure mode, which is
        evidence no metric contains — so the scenario requires the call that finds it.
        """
        scenario = by_name("onboarding_regression")
        assert "github__pull_request_activity" in scenario.ground_truth.required_capabilities
        activity = scenario.responses["github__pull_request_activity"]
        assert any("viewport" in review["body"] for review in activity["reviews"])

    def test_the_campaign_deploy_decoy_is_refutable_from_evidence(self) -> None:
        """The trap deploy must be disprovable by reading it, not only by inference.

        Flat conversion is an argument; "the deploy contained a CI pin and a dependency
        bump" is evidence. The scenario requires the call that produces the second.
        """
        scenario = by_name("campaign_traffic_drop")
        assert "github__commits" in scenario.ground_truth.required_capabilities
        subjects = [c["subject"] for c in scenario.responses["github__commits"]["commits"]]
        assert subjects and not any("onboarding" in s.lower() for s in subjects)

    def test_the_unanswerable_scenario_has_no_candidate_cause(self) -> None:
        scenario = by_name("insufficient_evidence")
        assert scenario.responses["github__deployment_history"]["deployments"] == []
        assert scenario.responses["slack__search_messages"]["messages"] == []


class TestTheScriptedCallsAreActuallyValid:
    def test_every_required_capability_has_arguments_the_schema_accepts(self) -> None:
        """A scripted call with missing required parameters is rejected before the handler, so
        it writes no evidence — and a test meaning "a competent analyst" then scores a report
        that cited nothing.

        The failure is silent in the direction that matters. Adding the false-premise scenario
        made `posthog__event_trend` a required capability with no entry here, and three tests
        died on `ids[0]` in the report builder rather than on anything describing the cause.
        """
        missing = sorted(
            {
                alternative
                for scenario in SCENARIOS
                for requirement in scenario.ground_truth.required_capabilities
                for alternative in alternatives_of(requirement)
                if alternative not in _VALID_ARGS
            }
        )
        assert not missing, f"no scripted arguments for: {', '.join(missing)}"


class TestTheSurfaceMatchesTheTenant:
    """The eval must offer what the tenant has, exactly as production does.

    `registry_for_tenant` filters the surface per tenant because offering unreachable tools costs
    steps and produces honest-looking caveats about irrelevant absences. The eval mirrored the
    whole registry instead, so it was measuring a surface the product does not present -- and the
    cost was invisible until two connectors were added for a different tenant and every scenario
    slowed down.
    """

    def test_a_scenario_offers_only_the_tools_it_has_data_for(self) -> None:
        for scenario in SCENARIOS:
            offered = set(scenario_registry(scenario).tool_names)
            assert offered == set(scenario.connected_tools), scenario.name

    def test_a_connector_added_for_another_tenant_does_not_widen_every_scenario(self) -> None:
        """The regression this exists to prevent, stated as the property rather than the
        instance: shipping a connector no fixture plants must not change any scenario's surface.
        Mixpanel and Amplitude were the first to prove it could."""
        from cortex.tools.registry import gtm_analyst_registry

        unplanted = set(gtm_analyst_registry().tool_names) - {
            tool for scenario in SCENARIOS for tool in scenario.connected_tools
        }
        assert unplanted, "expected at least one connector no fixture uses"
        for scenario in SCENARIOS:
            assert not (set(scenario_registry(scenario).tool_names) & unplanted), scenario.name

    def test_every_required_capability_is_still_reachable(self) -> None:
        """The filter must never remove a tool the ground truth demands, or `tool_selection`
        would score an analyst for failing to make a call it was never offered."""
        for scenario in SCENARIOS:
            offered = set(scenario_registry(scenario).tool_names)
            for requirement in scenario.ground_truth.required_capabilities:
                tools = {alt.split("__")[0] for alt in alternatives_of(requirement)}
                assert tools & offered, f"{scenario.name}: {requirement} is unreachable"

    def test_a_planted_response_implies_a_connected_tool(self) -> None:
        """Inferred rather than declared, so the two cannot drift. A scenario that plants a
        payload for a tool is a scenario whose tenant has that tool."""
        for scenario in SCENARIOS:
            for qualified in set(scenario.responses) | set(scenario.period_responses):
                assert qualified.split("__")[0] in scenario.connected_tools, qualified


class TestScenarioRegistry:
    def test_a_connected_tool_mirrors_its_real_capabilities_exactly(self) -> None:
        """Tool selection is being measured, so for every tool the tenant *has*, the model must
        choose from the same capabilities it sees in production.

        The set of tools is filtered per scenario -- see `TestTheSurfaceMatchesTheTenant` -- which
        is itself mirroring production, where `registry_for_tenant` offers only what the tenant
        holds credentials for. What must not differ is any connected tool's own surface."""
        from cortex.tools.registry import gtm_analyst_registry

        real = gtm_analyst_registry()
        mirrored = scenario_registry(by_name("onboarding_regression"))
        assert set(mirrored.tool_names) <= set(real.tool_names)
        assert mirrored.tool_names, "a scenario with no tools would measure nothing"
        for name in mirrored.tool_names:
            assert mirrored.get(name).capability_names == real.get(name).capability_names

    def test_mirrored_capabilities_keep_the_real_schemas(self) -> None:
        from cortex.tools.registry import gtm_analyst_registry

        real = gtm_analyst_registry().get("ga4").capability("get_funnel")
        mirrored = (
            scenario_registry(by_name("onboarding_regression")).get("ga4").capability("get_funnel")
        )
        assert mirrored.params_schema == real.params_schema
        assert mirrored.description == real.description

    def test_mirrored_capabilities_are_read_only(self) -> None:
        mirrored = scenario_registry(by_name("onboarding_regression"))
        for tool_name in mirrored.tool_names:
            tool = mirrored.get(tool_name)
            for capability_name in tool.capability_names:
                assert tool.capability(capability_name).read_only


class TestHarnessEndToEnd:
    async def test_a_competent_analyst_passes(self, session: AsyncSession) -> None:
        scenario = by_name("onboarding_regression")
        llm = _ScriptedAnalyst(
            completions=[
                _call("ga4__get_sessions", start_date="2026-07-01", end_date="2026-07-21"),
                *_competent_calls("onboarding_regression"),
                _done(),
            ],
            report_builder=_good_report("onboarding_regression"),
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.hallucinations == 0
        assert outcome.card.dimension("grounding").score == 1.0
        assert outcome.card.dimension("accuracy").score == 1.0
        assert outcome.card.dimension("tool_selection").score == 1.0
        assert outcome.passed, outcome.card.failures

    async def test_evidence_and_audit_rows_are_really_written(self, session: AsyncSession) -> None:
        """The harness must exercise the real executor, or the grounding score is
        meaningless — the gate resolves against the rows the executor writes."""
        scenario = by_name("campaign_traffic_drop")
        llm = _ScriptedAnalyst(
            completions=[
                _call(
                    "ga4__compare_periods",
                    current_start="2026-06-15",
                    current_end="2026-06-30",
                    previous_start="2026-06-01",
                    previous_end="2026-06-14",
                ),
                _done(),
            ],
            report_builder=_good_report("campaign_traffic_drop"),
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)
        assert outcome.card is not None, outcome.error

        assert await session.scalar(select(func.count()).select_from(Evidence)) > 0
        assert await session.scalar(select(func.count()).select_from(ToolCall)) > 0

    async def test_an_alternative_loop_can_be_scored_on_the_same_scenario(
        self, session: AsyncSession
    ) -> None:
        """The seam that makes any alternative-loop comparison meaningful.

        Everything except the loop must stay constant — same scenario, tools, evidence
        store, drafting call, gate and verifier — or a score difference says nothing
        about the loop. Asserted here because an injection point nothing exercises is
        one that silently stops being used.
        """
        scenario = by_name("campaign_traffic_drop")
        llm = _ScriptedAnalyst(
            completions=[
                _call(
                    "ga4__compare_periods",
                    current_start="2026-06-15",
                    current_end="2026-06-30",
                    previous_start="2026-06-01",
                    previous_end="2026-06-14",
                ),
                _done(),
            ],
            report_builder=_good_report("campaign_traffic_drop"),
        )

        built: list[str] = []

        def _factory(**kwargs: object) -> Investigator:
            # Records that the harness asked for a loop, and that it was handed the
            # same collaborators the default would have received.
            built.append("called")
            assert set(kwargs) == {"llm", "registry", "executor", "employee"}
            return Investigator(**kwargs)  # type: ignore[arg-type]

        outcome = await EvalHarness(llm=llm, investigator_factory=_factory).run_one(
            session, scenario
        )

        assert built == ["called"]
        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("grounding").score == 1.0

    async def test_repeats_score_a_scenario_more_than_once(self, session: AsyncSession) -> None:
        """One attempt is a sample, not a measurement.

        `campaign_traffic_drop` passed at 0.85 in one live run and failed in the next on
        unchanged analyst code. Any comparison drawn from single attempts is therefore
        reading variance, which is what repeats exist to expose.
        """
        scenario = by_name("campaign_traffic_drop")
        call = _call(
            "ga4__compare_periods",
            current_start="2026-06-15",
            current_end="2026-06-30",
            previous_start="2026-06-01",
            previous_end="2026-06-14",
        )
        llm = _ScriptedAnalyst(
            # Two attempts, so the script has to cover both.
            completions=[call, _done(), call, _done()],
            report_builder=_good_report("campaign_traffic_drop"),
        )

        run = await EvalHarness(llm=llm).run(session, (scenario,), repeat=2)

        assert [o.attempt for o in run.outcomes] == [1, 2]
        assert all(o.card is not None for o in run.outcomes), [o.error for o in run.outcomes]
        rendered = run.render()
        # The spread is the whole point of repeating; a longer list is not a measurement.
        assert "Across attempts" in rendered
        assert "2/2 passed" in rendered
        assert "min=" in rendered and "mean=" in rendered and "max=" in rendered

    async def test_repeats_are_interleaved_across_scenarios(self, session: AsyncSession) -> None:
        """A provider having a bad few minutes should degrade every scenario a little,
        not destroy one scenario's entire sample."""
        llm = RecordedLLM()  # every attempt errors; order is what is under test
        run = await EvalHarness(llm=llm).run(
            session,
            (by_name("campaign_traffic_drop"), by_name("insufficient_evidence")),
            repeat=2,
        )
        assert [(o.scenario, o.attempt) for o in run.outcomes] == [
            ("campaign_traffic_drop", 1),
            ("insufficient_evidence", 1),
            ("campaign_traffic_drop", 2),
            ("insufficient_evidence", 2),
        ]

    async def test_a_single_attempt_prints_no_spread_section(self, session: AsyncSession) -> None:
        """Otherwise every ordinary run grows a section that says nothing."""
        run = await EvalHarness(llm=RecordedLLM()).run(
            session, (by_name("campaign_traffic_drop"),), repeat=1
        )
        assert "Across attempts" not in run.render()

    async def test_each_scenario_gets_its_own_tenant(self, session: AsyncSession) -> None:
        """Sharing one would let an earlier scenario's rows inflate a later
        scenario's grounding and tool-selection scores."""
        llm_a = _ScriptedAnalyst(
            completions=[
                *_competent_calls("onboarding_regression"),
                _done(),
            ],
            report_builder=_good_report("onboarding_regression"),
        )
        llm_b = _ScriptedAnalyst(
            completions=[*_competent_calls("campaign_traffic_drop"), _done()],
            report_builder=_good_report("campaign_traffic_drop"),
        )
        harness_a = EvalHarness(llm=llm_a)
        harness_b = EvalHarness(llm=llm_b)

        first = await harness_a.run_one(session, by_name("onboarding_regression"))
        second = await harness_b.run_one(session, by_name("campaign_traffic_drop"))

        assert first.card and second.card
        # The second scenario only made one call, so a leaked tenant would have shown
        # the first scenario's calls too and inflated tool_selection.
        assert second.card.dimension("tool_selection").score == 1.0


class TestTheSuiteCatchesBadAnalysts:
    """The tests that make the suite worth trusting. A scorer that only ever passes
    converts an unknown into false confidence."""

    async def test_a_fabricated_citation_is_removed_and_counted_as_caught(
        self, session: AsyncSession
    ) -> None:
        """A fabricated citation is caught before delivery, so it is not a *delivered*
        hallucination — but it must still be visible.

        The distinction is the point of the split: the gate filters the claim out, so
        nothing unsupported reaches the reader, and failing the run on it would mean
        failing the product for defending itself. What must not happen is the fabrication
        becoming invisible, so `caught_and_removed` records it and `draft_reliability`
        scores it.
        """
        scenario = by_name("onboarding_regression")

        def _fabricating(ids: list[uuid.UUID]) -> dict:
            good = _good_report("onboarding_regression")(ids)
            # A real claim plus one citing an id that was never observed.
            good["executive_summary"].append(
                {
                    "text": "Enterprise revenue also fell 40%.",
                    "evidence_ids": [str(uuid.uuid4())],
                }
            )
            return good

        llm = _ScriptedAnalyst(
            completions=[
                _call("ga4__get_funnel"),
                _call("github__deployment_history", repo="acme/web"),
                _done(),
            ],
            report_builder=_fabricating,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.caught_and_removed >= 1, "the fabrication must be recorded"
        assert outcome.card.hallucinations == 0, "it was removed, so nothing shipped"
        # Visible in the score even though it did not fail the run.
        assert outcome.card.dimension("draft_reliability").score < 1.0

    async def test_a_claim_that_ships_unverified_is_a_delivered_hallucination(
        self, session: AsyncSession
    ) -> None:
        """The exposure the delivered count exists to measure.

        When the verifier cannot reach a verdict the claim is deliberately kept — an
        outage must not silently delete grounded work — and disclosed as a risk. But it
        reaches the reader unchecked, and that is the one thing the suite must fail on.
        """
        from cortex.agents.llm import LLMError

        class _VerifierDown(_ScriptedAnalyst):
            async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
                if "verdict" in kwargs.get("schema", {}).get("properties", {}):
                    raise LLMError("verifier unavailable")
                return await super().structured(**kwargs)

        scenario = by_name("onboarding_regression")
        llm = _VerifierDown(
            completions=[
                *_competent_calls("onboarding_regression"),
                _done(),
            ],
            report_builder=_good_report("onboarding_regression"),
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.hallucinations >= 1
        assert outcome.card.caught_and_removed == 0, "nothing was judged, so nothing was caught"
        assert not outcome.passed
        assert any("unverified" in f for f in outcome.card.failures)

    async def test_a_draft_that_is_mostly_fabricated_fails(self, session: AsyncSession) -> None:
        """The hole that gating on delivery alone would leave.

        An analyst whose every claim is fabricated would otherwise pass, because both
        mechanisms caught everything — and "the defenses held this time" is not the same
        as "the analyst is sound". The floor exists so a collapsed drafter fails even
        while nothing unsupported reaches the reader.
        """
        scenario = by_name("onboarding_regression")

        def _mostly_fabricated(ids: list[uuid.UUID]) -> dict:
            good = _good_report("onboarding_regression")(ids)
            good["executive_summary"] += [
                {"text": f"Invented claim {n}.", "evidence_ids": [str(uuid.uuid4())]}
                for n in range(6)
            ]
            return good

        llm = _ScriptedAnalyst(
            completions=[
                _call("ga4__get_funnel"),
                _call("github__deployment_history", repo="acme/web"),
                _done(),
            ],
            report_builder=_mostly_fabricated,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("draft_reliability").score < 0.5
        assert not outcome.passed
        assert any("draft_reliability" in f for f in outcome.card.failures)

    async def test_naming_a_decoy_as_the_cause_scores_down(self, session: AsyncSession) -> None:
        """The dimension that separates investigation from pattern-matching."""
        scenario = by_name("campaign_traffic_drop")

        def _decoy_blaming(ids: list[uuid.UUID]) -> dict:
            return {
                "question": scenario.question,
                "hypotheses": [_tested(ids[0])],
                "executive_summary": [
                    {
                        "text": "Signups fell because of a deploy that shipped on 14 June.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "recommendations": [
                    {
                        "action": "Roll back the 14 June deploy",
                        "rationale": "It coincides with the drop.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "confidence": "high",
            }

        llm = _ScriptedAnalyst(
            completions=[_call("ga4__compare_periods"), _done()],
            report_builder=_decoy_blaming,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("decoy_rejection").score < 1.0
        assert outcome.card.dimension("accuracy").score < 1.0

    async def test_ruling_a_decoy_out_is_not_penalised(self, session: AsyncSession) -> None:
        """Naming a decoy in order to *dismiss* it is the behaviour we want.

        A live report was scored 0.33 on this dimension for exactly the right conduct: it
        identified the 14 June deploy as the plausible suspect, killed it with flat
        conversion rates, and recommended "do not roll back the deploy". A substring
        match counted both the word "deploy" and the phrase "conversion rate" as decoys
        asserted as the cause, so the suite punished the single best judgement in the
        report — and, because three independently built loops all scored 0.33, it looked
        like an analyst weakness rather than a measurement bug.
        """
        scenario = by_name("campaign_traffic_drop")

        def _rules_decoys_out(ids: list[uuid.UUID]) -> dict:
            return {
                "question": scenario.question,
                "executive_summary": [
                    {
                        "text": "Signups fell because the spring campaign ended and paid "
                        "volume collapsed; conversion rate was flat.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "hypotheses": [
                    {
                        "statement": "The 14 June deploy broke the onboarding flow.",
                        "verdict": "contradicted",
                        "reasoning": "Conversion rate is identical before and after on "
                        "both devices, so no regression is present.",
                        "contradicting_evidence_ids": [str(ids[0])],
                    }
                ],
                "recommendations": [
                    {
                        "action": "Do not roll back the 14 June deploy",
                        "rationale": "Conversion rate did not move, so the deploy is not "
                        "the cause and a rollback would not help.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "confidence": "medium",
            }

        llm = _ScriptedAnalyst(
            completions=[_call("ga4__compare_periods"), _done()],
            report_builder=_rules_decoys_out,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        decoy = outcome.card.dimension("decoy_rejection")
        assert decoy.score == 1.0, decoy.detail
        assert "ruled out" in decoy.detail

    async def test_guessing_the_right_answer_without_the_evidence_scores_down(
        self, session: AsyncSession
    ) -> None:
        """Names the planted cause but never makes the discriminating call. A guess
        that happens to be right is not a repeatable capability."""
        scenario = by_name("onboarding_regression")
        llm = _ScriptedAnalyst(
            # Only one unrelated call: neither required capability is used.
            completions=[
                _call("hubspot__contacts", start_date="2026-07-15", end_date="2026-07-21"),
                _done(),
            ],
            report_builder=_good_report("onboarding_regression"),
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("accuracy").score == 1.0
        assert outcome.card.dimension("tool_selection").score == 0.0
        assert outcome.card.overall < 1.0

    async def test_a_confident_story_on_unanswerable_data_scores_zero_accuracy(
        self, session: AsyncSession
    ) -> None:
        """The failure a grounding-only score would miss entirely: every number real,
        every citation valid, and the conclusion invented."""
        scenario = by_name("insufficient_evidence")

        def _overconfident(ids: list[uuid.UUID]) -> dict:
            return {
                "question": scenario.question,
                "hypotheses": [_tested(ids[0])],
                "executive_summary": [
                    {
                        "text": "Enterprise signups fell because of a pricing page change.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "recommendations": [
                    {
                        "action": "Revert the pricing page",
                        "rationale": "It caused the decline.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "confidence": "high",
            }

        llm = _ScriptedAnalyst(
            completions=[_call("ga4__compare_periods"), _done()],
            report_builder=_overconfident,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        # Grounding is perfect — the citation is real — and the answer is still wrong.
        assert outcome.card.dimension("grounding").score == 1.0
        assert outcome.card.dimension("accuracy").score == 0.0
        assert outcome.card.dimension("actionability").score == 0.0
        assert not outcome.passed

    async def test_an_analyst_that_gathers_nothing_is_errored_not_scored(
        self, session: AsyncSession
    ) -> None:
        """A loop that could not run is a different problem from one that answered
        badly; collapsing them would hide an outage behind a quality regression."""
        outcome = await EvalHarness(llm=RecordedLLM()).run_one(
            session, by_name("onboarding_regression")
        )
        assert outcome.card is None
        assert outcome.error
        assert not outcome.passed


class TestTheSuiteCatchesAConfirmedFalsePremise:
    """The failure class the suite had no coverage of until a live answer exposed it.

    Asked *"did our signups fell from last month?"*, the analyst compared a 12-day total to a
    31-day one, reported that figure first, and hunted a cause. Every claim was cited and every
    citation resolved, so grounding scored 1.00 and the hallucination count was zero. Nothing in
    the suite could see it, because nothing in the suite asked a question whose premise was false.
    """

    async def test_a_report_that_refutes_the_premise_up_front_passes(
        self, session: AsyncSession
    ) -> None:
        scenario = by_name("partial_month_false_premise")
        llm = _ScriptedAnalyst(
            completions=[
                *_competent_calls("partial_month_false_premise"),
                _done(),
            ],
            report_builder=_good_report("partial_month_false_premise"),
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("accuracy").score == 1.0
        assert outcome.card.dimension("summary_placement").score == 1.0
        # The point of making completeness shape-aware: this report has no hypotheses and no
        # recommendations, which is what a factual question is told to produce, and it must not
        # be docked for that.
        assert outcome.card.dimension("completeness").score == 1.0
        assert outcome.card.dimension("actionability").score == 1.0
        assert outcome.passed, outcome.card.failures

    async def test_a_report_that_confirms_the_premise_fails(self, session: AsyncSession) -> None:
        """The live answer, reproduced. It names a real deploy as the cause of a fall that did
        not happen, cites real evidence for every claim, and must fail anyway — accuracy gates,
        so a confidently wrong conclusion fails the run even with perfect grounding."""
        scenario = by_name("partial_month_false_premise")

        def _confirming(ids: list[uuid.UUID]) -> dict:
            return {
                "question": scenario.question,
                "executive_summary": [
                    {
                        "text": (
                            "Signups fell sharply: 1,884 in August against 4,849 in July. "
                            "The pricing page redesign deployed on 4 August is the most "
                            "likely cause."
                        ),
                        "evidence_ids": [str(i) for i in ids[:2]],
                    }
                ],
                "findings": [
                    {
                        "title": "Deploy coincides with the decline",
                        "claims": [
                            {
                                "text": "A pricing page redesign shipped on 4 August.",
                                "evidence_ids": [str(ids[0])],
                            }
                        ],
                    }
                ],
                "hypotheses": [
                    {
                        "statement": "The pricing page redesign reduced signups",
                        "verdict": "supported",
                        "supporting_evidence_ids": [str(ids[0])],
                    }
                ],
                "recommendations": [
                    {
                        "action": "Roll back the pricing page redesign",
                        "rationale": "It is the only change coincident with the decline.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "risks": [{"description": "GA4 figures may be sampled."}],
                "confidence": "high",
            }

        llm = _ScriptedAnalyst(
            completions=[*_competent_calls("partial_month_false_premise"), _done()],
            report_builder=_confirming,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        # Grounding cannot see this, which is the whole reason accuracy exists as a separate
        # gating dimension. Asserted rather than implied: if grounding ever started catching
        # this, the two dimensions would be measuring the same thing.
        assert outcome.card.dimension("grounding").score == 1.0
        assert outcome.card.hallucinations == 0
        assert outcome.card.dimension("accuracy").score == 0.0
        assert not outcome.passed
        # And the recommendation to act on a non-existent decline is scored too.
        assert outcome.card.dimension("actionability").score == 0.0

    async def test_the_right_answer_buried_passes_accuracy_and_fails_placement(
        self, session: AsyncSession
    ) -> None:
        """What actually shipped, and why the two are scored apart.

        The investigation was correct and the delivery was not, and those deserve different
        consequences. Folding them into one gating dimension failed a run in which the analyst
        established the premise was false, made every discriminating call and recommended no
        action -- because the day-count mechanism sat in a finding rather than the summary.
        Gating on where a correct answer was printed measures house style, not capability.

        So accuracy sees a correct investigation and passes it; `summary_placement` sees a buried
        answer and scores zero, weighted rather than gating."""
        scenario = by_name("partial_month_false_premise")

        def _buried(ids: list[uuid.UUID]) -> dict:
            payload = _good_report("partial_month_false_premise")(ids)
            correct = payload["executive_summary"][0]["text"]
            payload["executive_summary"] = [
                {
                    "text": "August shows 1,884 signups against July's total of 4,849.",
                    "evidence_ids": [str(ids[0])],
                }
            ]
            payload["findings"].append(
                {
                    "title": "Period lengths",
                    "claims": [{"text": correct, "evidence_ids": [str(ids[0])]}],
                }
            )
            return payload

        llm = _ScriptedAnalyst(
            completions=[*_competent_calls("partial_month_false_premise"), _done()],
            report_builder=_buried,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("accuracy").score == 1.0
        placement = outcome.card.dimension("summary_placement")
        assert placement.score == 0.0
        assert "not in the summary" in placement.detail
        # The distinction that makes the split worth having: a buried answer is a defect worth
        # scoring and not a reason to fail an otherwise sound investigation.
        assert placement.gates is False
        assert outcome.passed


class TestScorerDimensions:
    def test_grounding_and_hallucination_never_consult_a_model(self) -> None:
        """The product's central claim must not be scored by the same kind of
        component it exists to constrain."""
        import inspect

        source = inspect.getsource(Scorer._grounding)
        for model_ish in ("llm", "structured", "complete(", "verdict"):
            assert model_ish not in source.lower(), model_ish

    def test_grounding_gates_and_soft_dimensions_do_not(self) -> None:
        """A soft score that can fail a build teaches people to distrust the build."""
        from cortex.eval.scorer import Dimension

        assert Dimension(name="grounding", score=0.5, detail="", gates=True).passed is False
        assert Dimension(name="actionability", score=0.0, detail="").passed is True

    def test_overall_weights_grounding_and_accuracy_highest(self) -> None:
        from cortex.eval.scorer import Dimension, Scorecard

        strong = Scorecard(
            scenario="x",
            dimensions=[
                Dimension("grounding", 1.0, ""),
                Dimension("accuracy", 1.0, ""),
                Dimension("latency", 0.0, ""),
            ],
        )
        weak = Scorecard(
            scenario="x",
            dimensions=[
                Dimension("grounding", 0.0, ""),
                Dimension("accuracy", 0.0, ""),
                Dimension("latency", 1.0, ""),
            ],
        )
        assert strong.overall > weak.overall
        assert strong.overall > 0.8

    def test_a_hallucination_fails_the_card_regardless_of_scores(self) -> None:
        from cortex.eval.scorer import Dimension, Scorecard

        card = Scorecard(
            scenario="x",
            dimensions=[Dimension("grounding", 1.0, "", gates=True)],
            hallucinations=1,
        )
        assert card.passed is False
        assert "must be 0" in card.failures[0]

    def test_latency_never_gates(self) -> None:
        """A slow correct answer is worth more than a fast wrong one, and gating on
        wall clock would fail the suite on a loaded CI runner."""
        import inspect

        source = inspect.getsource(Scorer._latency)
        assert "gates=True" not in source


class TestRunRendering:
    def test_the_scorecard_names_what_failed(self) -> None:
        """An operator should not need the code open to act on a failure."""
        from cortex.eval.scorer import Dimension, Scorecard

        run = EvalRun(
            outcomes=[
                ScenarioOutcome(
                    scenario="onboarding_regression",
                    card=Scorecard(
                        scenario="onboarding_regression",
                        dimensions=[
                            Dimension("grounding", 0.5, "1 of 2 citations unresolvable", gates=True)
                        ],
                        hallucinations=1,
                    ),
                )
            ]
        )
        rendered = run.render()
        assert "FAIL" in rendered
        assert "hallucinations=1" in rendered
        assert "1 of 2 citations unresolvable" in rendered
        assert "0/1 scenarios passed" in rendered

    def test_an_errored_scenario_is_reported_distinctly(self) -> None:
        run = EvalRun(
            outcomes=[ScenarioOutcome(scenario="x", card=None, error="the analyst could not run")]
        )
        rendered = run.render()
        assert "ERRORED" in rendered
        assert "could not run" in rendered

    def test_an_empty_run_does_not_pass(self) -> None:
        """A suite that ran nothing must not report success."""
        assert EvalRun().passed is False


class TestCLI:
    def test_argument_parsing(self) -> None:
        from cortex.eval.__main__ import _parse

        args = _parse(["--scenario", "onboarding_regression", "--effort", "low", "--no-verify"])
        assert args.scenario == ["onboarding_regression"]
        assert args.effort == "low"
        assert args.no_verify is True

    def test_an_unknown_scenario_is_named(self) -> None:
        with pytest.raises(KeyError, match="no scenario named"):
            by_name("does_not_exist")


class TestActionabilityReadsNegation:
    """Advising *against* the tempting wrong move is the ideal output on an
    unanswerable scenario, not a failure.

    A live run scored 0.00 for "Do not roll back or alter the signup or onboarding flow on
    the strength of this report" — advice to hold, matched as advice to act because the
    substring "roll back" was present. Fourth measurement defect of the same shape.
    """

    async def test_advising_against_action_is_not_penalised(self, session: AsyncSession) -> None:
        scenario = by_name("insufficient_evidence")

        def _counsels_restraint(ids: list[uuid.UUID]) -> dict:
            return {
                "question": scenario.question,
                "hypotheses": [_tested(ids[0])],
                "executive_summary": [
                    {
                        "text": "A 3% move is within normal variation; the data cannot "
                        "establish a cause.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "recommendations": [
                    {
                        "action": "Do not roll back or alter the signup flow on the "
                        "strength of this report",
                        "rationale": "No cause is established, so any change would be "
                        "acting on noise.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "confidence": "insufficient_evidence",
            }

        llm = _ScriptedAnalyst(
            completions=[_call("ga4__get_sessions"), _done()],
            report_builder=_counsels_restraint,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("actionability").score == 1.0

    async def test_recommending_action_is_still_penalised(self, session: AsyncSession) -> None:
        """The guard must still fire: acting on an unestablished cause is the failure this
        dimension exists to catch."""
        scenario = by_name("insufficient_evidence")

        def _acts_anyway(ids: list[uuid.UUID]) -> dict:
            return {
                "question": scenario.question,
                "hypotheses": [_tested(ids[0])],
                "executive_summary": [
                    {"text": "The data cannot establish a cause.", "evidence_ids": [str(ids[0])]}
                ],
                "recommendations": [
                    {
                        "action": "Roll back the onboarding deploy",
                        "rationale": "It is the most likely candidate.",
                        "evidence_ids": [str(ids[0])],
                    }
                ],
                "confidence": "insufficient_evidence",
            }

        llm = _ScriptedAnalyst(
            completions=[_call("ga4__get_sessions"), _done()],
            report_builder=_acts_anyway,
        )
        outcome = await EvalHarness(llm=llm).run_one(session, scenario)

        assert outcome.card is not None, outcome.error
        assert outcome.card.dimension("actionability").score == 0.0


class TestAFailedAttemptLeavesSomethingToRead:
    """Run 18 lost two attempts to a drafting failure and captured nothing.

    A bundle needs a report and there was none, so the whole surviving trace was one line of
    error text -- and the first question a drafting failure raises is about what the loop did
    beforehand: did it work and the draft blow up, or did it wander and gather more than any
    draft could fit? That is answerable from the tool calls and from nothing else.
    """

    async def test_the_tool_calls_before_the_failure_are_kept(
        self, session: AsyncSession, tmp_path
    ) -> None:
        scenario = by_name("campaign_traffic_drop")
        llm = _ScriptedAnalyst(
            completions=[
                _call(
                    "ga4__compare_periods",
                    current_start="2026-06-15",
                    current_end="2026-06-30",
                    previous_start="2026-06-01",
                    previous_end="2026-06-14",
                ),
                _done(),
            ],
            # A draft that cannot be produced at all, which is what truncation looks like
            # from the harness: the loop finished, the report did not arrive.
            report_builder=_undraftable,
        )
        outcome = await EvalHarness(llm=llm, capture_to=tmp_path).run_one(session, scenario)

        assert outcome.card is None
        assert outcome.error
        records = sorted((tmp_path / FAILURES_DIRNAME).glob("*.json"))
        assert len(records) == 1
        record = json.loads(records[0].read_text())
        assert record["failed"] is True
        assert record["error"] == outcome.error
        # The point of the record: the loop is visible even though the report is not, down to
        # the survey calls the loop makes before the analyst asks for anything -- which is how
        # "the loop wandered" would show up here.
        made = [(c["tool_name"], c["capability"]) for c in record["tool_calls"]]
        assert ("ga4", "compare_periods") in made
        assert record["observations"] == len(made)

    async def test_it_is_not_left_where_the_scorer_looks_for_bundles(
        self, session: AsyncSession, tmp_path
    ) -> None:
        """Three separate places glob `*.json` here and hand what they find to `load_bundle`.

        A filename convention would make each of them -- and every future one -- responsible
        for remembering an exclusion. A subdirectory means they cannot see it at all.
        """
        scenario = by_name("campaign_traffic_drop")
        llm = _ScriptedAnalyst(
            completions=[_done()],
            report_builder=_undraftable,
        )
        await EvalHarness(llm=llm, capture_to=tmp_path).run_one(session, scenario)

        assert list(tmp_path.glob("*.json")) == []
        assert list((tmp_path / FAILURES_DIRNAME).glob("*.json"))


class TestTheFixtureUniverseIsSelfConsistent:
    """A scenario must not describe a repository its own discovery survey denies exists.

    All five did. `DISCOVERY_DEFAULTS` advertised acme/product, acme/company-website and
    acme/mobile; four scenarios answered every github call with payloads labelled acme/web and
    the fifth with acme/marketing-site. The analyst surveyed the tenant, asked about a
    repository it had been told existed, and got back a payload about a different one.

    It cost real score on a *gating* dimension. `tempting_coincidence` read `draft_reliability`
    0.77, and the verifier's rejections were of the form "the payload's `repo` field returns
    'acme/marketing-site', not acme/product" -- right every time, rejecting an analyst that was
    faithfully reporting what it had asked for. No scorecard could show this. It came out of
    reading the captured payloads.
    """

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_every_repository_described_is_advertised(self, scenario) -> None:  # type: ignore[no-untyped-def]
        listing = scenario.response_for("github__list_repositories")
        advertised = {entry["repo"] for entry in listing["repositories"]}
        assert scenario.repositories_described() <= advertised

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_the_listing_still_offers_a_choice(self, scenario) -> None:  # type: ignore[no-untyped-def]
        """A listing narrowed to the one repository that matters hands over the answer.

        `tempting_coincidence` exists to see whether the analyst resists a plausible
        coincidence; with a single candidate there is nothing to resist.
        """
        listing = scenario.response_for("github__list_repositories")
        assert len(listing["repositories"]) > len(scenario.repositories_described())
        assert listing["count"] == len(listing["repositories"])

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_a_described_repository_keeps_its_metadata_when_it_is_a_known_one(
        self, scenario
    ) -> None:  # type: ignore[no-untyped-def]
        """A synthesised entry must not overwrite a decoy's description with a blank one."""
        listing = scenario.response_for("github__list_repositories")
        for entry in listing["repositories"]:
            if entry["repo"] in {"acme/product", "acme/company-website", "acme/mobile"}:
                assert entry["description"]


class TestTheScenarioThatFailsIfADisclosureStops:
    """`measurement_stopped` is the first scenario whose answer depends on a connector disclosure.

    The class of failure it encodes has actually shipped: a `user signed up` series ran 1-3 August
    against a range ending on the 15th, the analyst computed 206/day from those three days, called
    it "the August run rate", and reported no real decline. Every figure was real and correctly
    cited. It was wrong because nobody asked whether the data covered the question.

    The fixture states no disclosure of its own on purpose. If `_gap` or the trust gate stops
    working, this scenario's payload silently becomes an ordinary-looking cliff and the suite
    should stop passing -- which is the property no other scenario had.
    """

    def _ga4(self) -> dict:
        scenario = by_name("measurement_stopped")
        payload = dict(scenario.responses["ga4__get_sessions"])
        params = {"start_date": "2026-07-16", "end_date": "2026-08-15", "dimensions": ["date"]}
        payload.update(series_disclosures("ga4__get_sessions", payload, params))
        payload.update(grade_series("ga4__get_sessions", payload))
        return payload

    def test_the_fixture_declares_no_disclosure_itself(self) -> None:
        """Otherwise it would pass whether or not the connector computes one, which is the whole
        thing this scenario was added to detect."""
        raw = by_name("measurement_stopped").responses["ga4__get_sessions"]
        assert "series_ends_early" not in raw
        assert "data_trust" not in raw

    def test_the_connector_supplies_the_gap(self) -> None:
        payload = self._ga4()
        assert payload["series_ends_early"] == {
            "last_bucket": "2026-08-03",
            "requested_end": "2026-08-15",
            "days_missing": 12,
        }

    def test_ga4_degrades_and_does_not_block(self) -> None:
        """A GA4 series alone cannot distinguish a broken tag from a site that went dark, so it
        must not claim the authority to end the investigation."""
        trust = self._ga4()["data_trust"]
        assert trust["state"] == "degraded"
        assert trust["may_answer_the_business_question"] is True

    def test_the_second_line_of_evidence_is_healthy(self) -> None:
        """DOE-NE-STD-1004-92: two independent lines, or the tree does not narrow.

        Noticing GA4 stops on the 3rd establishes only that GA4 stops on the 3rd. Signups
        arriving normally through the 15th is what makes "the measurement broke" evidenced
        rather than merely the more comfortable of two guesses.
        """
        scenario = by_name("measurement_stopped")
        # Through the resolver: this series is derived from daily counts now, so there is no
        # canned payload to read and asking for one would test the wrong thing.
        payload = dict(
            scenario.response_for(
                "posthog__event_trend", {"event": "user signed up", "interval": "day"}
            )
        )
        request = {
            k: payload.get(k) for k in ("start_date", "end_date", "interval", "breakdown_property")
        }
        payload.update(series_disclosures("posthog__event_trend", payload, request))
        payload.update(grade_series("posthog__event_trend", payload))

        assert "series_ends_early" not in payload
        assert "data_trust" not in payload
        # And it must not invent a level shift in a flat series, which would hand the analyst a
        # signups movement to explain and destroy the corroboration.
        assert payload["movement"]["level_shifts"] == []

    def test_it_is_neither_unanswerable_nor_a_false_premise(self) -> None:
        """Measured sessions really did fall, and there is a specific right action -- restore the
        tag. A scenario flagged as declining a cause would score that correct recommendation as a
        failure, which is why this is an ordinary causal scenario whose cause is a data incident.
        """
        truth = by_name("measurement_stopped").ground_truth
        assert not truth.is_unanswerable
        assert not truth.is_false_premise
        assert not truth.declines_a_cause

    def test_the_decoy_deploy_lands_before_the_cliff_and_is_disarmable(self) -> None:
        """One day before, the tightest possible coincidence. Its contents are what rules it
        out: a CI cache key and a lockfile reach no user and cannot move sessions."""
        prs = by_name("measurement_stopped").responses["github__recent_prs"]["pull_requests"]
        assert prs[0]["merged_at"].startswith("2026-08-02")
        assert all(
            path.startswith(".github/") or path.endswith("lock.yaml") for path in prs[0]["paths"]
        )


class TestNoScenarioReportsAGapItDidNotPlant:
    """A fixture answers every requested range with one fixed series.

    Once the eval began computing the connectors' real disclosures, that turned every scenario
    into a collection failure for any analyst who asked wider than was planted:
    `onboarding_regression` reported its series stopping 46 days earlier if asked through August,
    `insufficient_evidence` 101 days. A data-incident verdict outranks and replaces the real
    answer, so those scenarios would have failed on the harness rather than on the analyst.
    """

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    @pytest.mark.parametrize("probe", ["2026-08-31", "2026-09-15", "2026-12-31"])
    def test_asking_past_the_planted_window_discloses_nothing_new(
        self,
        scenario,  # type: ignore[no-untyped-def]
        probe: str,
    ) -> None:
        if not scenario.compute_disclosures:
            pytest.skip("this scenario withholds disclosures deliberately")
        for capability in ("posthog__event_trend", "ga4__get_sessions"):
            payload = scenario.responses.get(capability)
            if payload is None:
                variants = scenario.period_responses.get(capability) or {}
                payload = variants.get("after")
            if payload is None:
                continue
            request = {
                "start_date": "2026-05-01",
                "end_date": probe,
                "interval": payload.get("interval", "day"),
            }
            found = series_disclosures(capability, payload, request, as_of=scenario.as_of)
            if scenario.name == "measurement_stopped" and capability == "ga4__get_sessions":
                # The one scenario where the gap is the subject rather than an artefact.
                assert found["series_ends_early"]["days_missing"] == 12
                continue
            assert "series_ends_early" not in found, (scenario.name, capability, probe)

    def test_the_horizon_is_the_whole_world_not_one_series(self) -> None:
        """`measurement_stopped` depends on this. GA4 stops on 3 August and PostHog runs to the
        15th, so the world demonstrably has data through the 15th and GA4's stop is real. A
        per-series horizon would clamp the gap away and delete the scenario's subject."""
        assert by_name("measurement_stopped").as_of == date(2026, 8, 15)

    def test_the_horizon_is_derived_from_the_data(self) -> None:
        """Declared beside the fixtures it would be a second thing to keep in step, and this
        repository has already been bitten twice by exactly that -- the repository listing, and
        the connected-tools inference this sits next to."""
        for scenario in SCENARIOS:
            planted = [
                raw[:10]
                for response in scenario.responses.values()
                if isinstance(response, dict)
                for row in (response.get("series") or []) + (response.get("rows") or [])
                if isinstance(row, dict)
                for raw in [row.get("bucket") or (row.get("dimensions") or {}).get("date")]
                if isinstance(raw, str)
            ]
            # Derived series count toward the horizon, and leaving them out is what dropped
            # `measurement_stopped` from the 15th to the 3rd -- making GA4's truncated series
            # look complete and deleting the only thing that scenario is about. This test is
            # what caught it.
            planted += [
                day.isoformat()
                for series in scenario.daily_truth.values()
                for truth in series
                for day, _ in truth.days
            ]
            if planted:
                assert scenario.as_of == date.fromisoformat(max(planted)), scenario.name


class TestDiscoveryAdvertisesWhatTheFixtureCanAnswer:
    """Twice the same bug: discovery describing a world the rest of the fixture cannot answer for.

    First the repositories. Then the events — in **all six** scenarios the event the fixture
    planted was not in the catalogue the analyst is shown. So it could not ask for the right name,
    picked a plausible one (`signup_completed`, `$pageview_0`), and `response_for` served the
    planted series anyway, because a fixture is keyed by capability and ignores the parameters
    that decide what the answer should be.

    The analyst then described what it received in the terms it had asked for — honest, and wrong.
    The verifier caught it: *"the evidence's actual `event` field is 'user signed up', not a
    pageview event as claimed"*. In `measurement_stopped` the claim it destroyed was the second
    line of evidence, which left a correct report resting on the single source that had stopped,
    and the scenario failed on `accuracy` for a defect in its own fixture.
    """

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_every_event_described_is_advertised(self, scenario) -> None:  # type: ignore[no-untyped-def]
        listing = scenario.response_for("posthog__list_events")
        advertised = {entry["name"] for entry in listing["events"]}
        assert scenario.events_described() <= advertised

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_the_catalogue_still_offers_a_choice(self, scenario) -> None:  # type: ignore[no-untyped-def]
        """A catalogue holding only the answer removes the choice. `tempting_coincidence` ships
        ninety-six events whose one visible cluster is deliberately *not* its answer."""
        listing = scenario.response_for("posthog__list_events")
        assert len(listing["events"]) > len(scenario.events_described())
        assert listing["count"] == len(listing["events"])

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_a_synthesised_entry_is_dated(self, scenario) -> None:  # type: ignore[no-untyped-def]
        """`last_seen_at` decides whether an analyst treats an event as live. Blank reads as
        "never seen", which would make the planted event look like the dead option."""
        listing = scenario.response_for("posthog__list_events")
        for entry in listing["events"]:
            if entry["name"] in scenario.events_described():
                assert entry["last_seen_at"], (scenario.name, entry)

    def test_planting_a_catalogue_does_not_bypass_the_derivation(self) -> None:
        """The ordering inside `response_for`, which is where the first attempt at this failed.

        `tempting_coincidence` plants its own ninety-six-event catalogue, where digesting a large
        catalogue is the test. With the plain `self.responses` lookup first, that planted listing
        won and the derivation never ran — leaving the scenario most affected still broken.
        """
        scenario = by_name("tempting_coincidence")
        assert "posthog__list_events" in scenario.responses
        listing = scenario.response_for("posthog__list_events")
        planted = len(scenario.responses["posthog__list_events"]["events"])
        # The planted contents survive, with the described event added to them.
        assert len(listing["events"]) == planted + len(scenario.events_described())
        assert scenario.events_described() <= {e["name"] for e in listing["events"]}


class TestAFixtureAnswersOnlyWhatItWasAsked:
    """The third and last layer of one bug: a fixture keyed by capability, ignoring parameters.

    Fixed at the discovery layer first, so the analyst could learn the right event name. It then
    asked for two names, received byte-identical payloads, and the sufficiency gate refused the
    report — correctly: *"two different event names ... returning identical payloads suggests a
    tracking/query issue, not independent signals"*. They were not independent signals. They were
    the same fixture answering twice.
    """

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_the_planted_subject_still_resolves(self, scenario) -> None:  # type: ignore[no-untyped-def]
        """The risk in this change: strictness that also breaks the scenario's own question."""
        for capability, key in (
            ("posthog__event_trend", "event"),
            ("github__recent_prs", "repo"),
            ("github__deployment_history", "repo"),
        ):
            planted = scenario.responses.get(capability)
            if not planted or not isinstance(planted.get(key), str):
                continue
            assert scenario.response_for(capability, {key: planted[key]}) == planted

    def test_a_different_event_returns_an_empty_series_not_another_one(self) -> None:
        scenario = by_name("partial_month_false_premise")
        planted = scenario.daily_truth["posthog__event_trend"][0].event
        asked = {"event": "signup_completed", "interval": "day"}
        decoy = scenario.response_for("posthog__event_trend", asked)

        assert decoy != scenario.response_for("posthog__event_trend", {**asked, "event": planted})
        assert decoy["series"] == []
        assert decoy["total"] == 0
        assert decoy["row_count"] == 0
        # Named for what was asked, so the analyst is not left inferring which event it holds.
        assert decoy["event"] == "signup_completed"

    def test_the_metadata_survives_so_the_empty_result_is_readable(self) -> None:
        """An empty payload that also lost its interval and range says "nothing here" about an
        unknown question. The scalars are what make it a usable observation.

        The interval echoes the request rather than a granularity the fixture chose, and a
        request naming none gets `posthog.event_trend`'s own default. A fixture answering at
        some other default is standing in for a connector this project does not have -- which
        is how the daily call ended up being served monthly buckets.
        """
        scenario = by_name("partial_month_false_premise")
        decoy = scenario.response_for("posthog__event_trend", {"event": "checkout_started"})
        assert decoy["interval"] == "day"
        assert decoy["start_date"] and decoy["end_date"]

        weekly = scenario.response_for(
            "posthog__event_trend", {"event": "checkout_started", "interval": "week"}
        )
        assert weekly["interval"] == "week"

    def test_emptying_rather_than_erroring(self) -> None:
        """ "Nothing here" is a real observation and is what makes a decoy disprovable. An error
        would teach the analyst that tools fail for no reason, which cost real steps before."""
        decoy = by_name("measurement_stopped").response_for(
            "github__recent_prs", {"repo": "acme/nonexistent"}
        )
        assert decoy["pull_requests"] == []
        assert decoy["repo"] == "acme/nonexistent"

    def test_a_request_with_no_subject_is_unchanged(self) -> None:
        """A capability whose params name no subject must not be filtered by accident.

        With several series planted, "no subject named" resolves to the first — the one the
        scenario leads with. Answering empty instead would make a call that omits the event
        look like a call about an event nobody planted, which are different observations.
        """
        scenario = by_name("measurement_stopped")
        leading = scenario.daily_truth["posthog__event_trend"][0].event
        for params in ({}, None):
            payload = scenario.response_for("posthog__event_trend", params)
            assert payload["event"] == leading
            assert payload["series"], params


class TestAFixtureMustNotArgueAgainstItsOwnGroundTruth:
    """`measurement_stopped` asserts the site is fine and only its measurement stopped.

    With one series planted and every other event empty, an analyst asking about pageviews is
    told they stopped too — which points at a site outage, the exact reading the scenario exists
    to rule out. Both attempts of run 24 failed here: one never queried a series, the other asked
    for `$pageview` and got nothing.

    A real deployment has many healthy events, so `subject_responses` lets the fixture say so.
    """

    def test_more_than_one_product_event_is_healthy(self) -> None:
        scenario = by_name("measurement_stopped")
        for event in ("user signed up", "$pageview"):
            series = scenario.response_for("posthog__event_trend", {"event": event})["series"]
            assert len(series) == 31, event
            assert series[-1]["bucket"].startswith("2026-08-15"), event

    def test_they_run_past_the_date_ga4_stops(self) -> None:
        """The whole argument. GA4 stops on the 3rd; PostHog carrying on to the 15th is what makes
        "the measurement broke" evidenced rather than the more comfortable of two guesses."""
        scenario = by_name("measurement_stopped")
        rows = scenario.responses["ga4__get_sessions"]["rows"]
        ga4_last = max(r["dimensions"]["date"] for r in rows)
        assert ga4_last == "2026-08-03"
        posthog = scenario.response_for("posthog__event_trend", {"event": "$pageview"})
        assert max(r["bucket"][:10] for r in posthog["series"]) == "2026-08-15"

    def test_the_planted_event_is_findable_rather_than_buried(self) -> None:
        """The first version reused the ninety-six-event catalogue built for
        `tempting_coincidence`, which made finding the one useful event a needle-hunt and
        conflated two skills. This scenario tests corroboration, not catalogue digestion."""
        catalogue = by_name("measurement_stopped").response_for("posthog__list_events")
        assert len(catalogue["events"]) <= 8
        assert "user signed up" in {e["name"] for e in catalogue["events"]}

    def test_a_subject_response_is_still_advertised_by_discovery(self) -> None:
        """The invariant, reintroduced through the newer field if `events_described` missed it."""
        scenario = by_name("measurement_stopped")
        assert "$pageview" in scenario.events_described()
        advertised = {e["name"] for e in scenario.response_for("posthog__list_events")["events"]}
        assert scenario.events_described() <= advertised


class TestTheDiscriminatingCallMustBeAnswerable:
    """`partial_month_false_premise` names the daily call as the one that separates a real fall
    from a calendar artefact, and could not answer it.

    Its trend payload was three monthly buckets. `posthog.event_trend` defaults `interval` to
    `"day"`, so the fixture was both serving a granularity nobody asked for and standing in for
    a connector that behaves differently. Run 26 asked for daily granularity three times, was
    handed monthly buckets each time, never saw a run-rate, and hedged the premise to
    "unverifiable" — while `required_capabilities` recorded the discriminating call as made.
    """

    SCENARIO = "partial_month_false_premise"

    def test_a_daily_request_gets_daily_buckets(self) -> None:
        scenario = by_name(self.SCENARIO)
        payload = scenario.response_for(
            "posthog__event_trend",
            {
                "event": "user signed up",
                "interval": "day",
                "start_date": "2026-07-01",
                "end_date": "2026-08-26",
            },
        )
        assert payload["interval"] == "day"
        # 31 days of July and the 12 of August that exist. Not 57: the window is clamped to the
        # data, which is the fact the analyst has to notice.
        assert payload["row_count"] == 43
        assert payload["end_date"] == "2026-08-12"

    def test_the_run_rate_the_ground_truth_states_is_computable_from_it(self) -> None:
        """156.4/day against 157.0/day is the answer. Both runs of the scenario instead divided
        August's total by the 26 days the calendar had, having asked for a window ending on the
        26th — so the number they reported was 72/day and the comparison was meaningless."""
        scenario = by_name(self.SCENARIO)
        payload = scenario.response_for(
            "posthog__event_trend", {"event": "user signed up", "interval": "day"}
        )
        by_month: dict[str, list[int]] = {}
        for row in payload["series"]:
            by_month.setdefault(row["bucket"][:7], []).append(row["value"])
        july = by_month["2026-07"]
        august = by_month["2026-08"]
        assert len(july) == 31
        assert len(august) == 12
        # Flat within noise, which is what "signups did not fall" means here.
        assert abs(sum(august) / len(august) - sum(july) / len(july)) < 8

    def test_the_monthly_trap_still_works(self) -> None:
        """The scenario is only worth running if the naive call still looks like a collapse. One
        full month against a third of one, with nothing in the payload calling it partial."""
        scenario = by_name(self.SCENARIO)
        payload = scenario.response_for(
            "posthog__event_trend", {"event": "user signed up", "interval": "month"}
        )
        buckets = {row["bucket"][:7]: row["value"] for row in payload["series"]}
        assert buckets["2026-08"] < buckets["2026-07"] / 2
        assert "partial_buckets" not in payload

    def test_both_stated_alternatives_can_establish_the_run_rate(self) -> None:
        """`required_capabilities` offers `posthog__event_trend` or `ga4__get_sessions`, meaning
        either establishes it. That was false while only GA4 carried daily rows: an analyst
        picking the other alternative scored as having made the discriminating call while
        holding nothing that could answer it."""
        scenario = by_name(self.SCENARIO)
        # Same window on both sides: PostHog plants June as well, GA4 starts in July.
        trend = scenario.response_for(
            "posthog__event_trend",
            {"event": "user signed up", "interval": "day", "start_date": "2026-07-01"},
        )
        sessions = scenario.response_for("ga4__get_sessions", {})
        assert len(trend["series"]) == 43
        assert len({row["dimensions"]["date"] for row in sessions["rows"]}) == 43


class TestEveryPlantedEventStaysAdvertised:
    """The invariant every new planting field has broken in turn.

    Moving this scenario's series into `daily_truth` emptied `events_described`, so discovery
    stopped advertising `user signed up` and the analyst would have been told the event it
    needed did not exist. Asserted across all scenarios rather than for the field that broke it
    most recently, because the next field will break it the same way.
    """

    def test_described_events_are_advertised_everywhere(self) -> None:
        for scenario in SCENARIOS:
            advertised = {
                e["name"] for e in scenario.response_for("posthog__list_events")["events"]
            }
            missing = scenario.events_described() - advertised
            assert not missing, f"{scenario.name} describes unadvertised events: {sorted(missing)}"

    def test_a_daily_truth_event_is_among_them(self) -> None:
        scenario = by_name("partial_month_false_premise")
        assert "user signed up" in scenario.events_described()
        advertised = {e["name"] for e in scenario.response_for("posthog__list_events")["events"]}
        assert "user signed up" in advertised


class TestTheCampaignScenarioAnswersTheIntervalAsked:
    """The fifth-instance defect, fixed in one scenario and left in another.

    `campaign_traffic_drop` planted four weekly buckets and handed them to a request for
    `interval="day"`. It cost score twice in run 31: the sufficiency gate withheld the planted
    cause and the verifier cut a claim, both objecting that no signups series established the
    decline — and on the attempt that did retrieve the series, they were describing four weekly
    points spanning the very boundary the question turns on.
    """

    SCENARIO = "campaign_traffic_drop"

    def test_a_daily_request_gets_daily_buckets(self) -> None:
        scenario = by_name(self.SCENARIO)
        payload = scenario.response_for(
            "posthog__event_trend",
            {
                "event": "user signed up",
                "interval": "day",
                "start_date": "2026-06-01",
                "end_date": "2026-06-30",
            },
        )
        assert payload["interval"] == "day"
        assert payload["row_count"] == 30

    def test_the_step_lands_where_the_campaign_ended(self) -> None:
        """The campaign ended on 14 June, so the step belongs on the 15th. A series whose
        movement sits anywhere else would let a correct answer be graded against wrong data."""
        scenario = by_name(self.SCENARIO)
        payload = scenario.response_for(
            "posthog__event_trend", {"event": "user signed up", "interval": "day"}
        )
        before = [r["value"] for r in payload["series"] if r["bucket"][:10] <= "2026-06-14"]
        after = [r["value"] for r in payload["series"] if r["bucket"][:10] >= "2026-06-15"]
        rate_before = sum(before) / len(before)
        rate_after = sum(after) / len(after)
        assert rate_before > rate_after
        # A fall of roughly 40%: large enough to find, small enough to need the daily view.
        assert 0.35 < (rate_before - rate_after) / rate_before < 0.50

    def test_a_weekly_call_still_sees_what_it_saw(self) -> None:
        """The weekly totals are the fixture's own history — 402, 391, 236, 228 — and deriving
        them from daily counts must not move them, or a scenario tuned against those numbers
        would start measuring the change rather than the analyst."""
        scenario = by_name(self.SCENARIO)
        payload = scenario.response_for(
            "posthog__event_trend",
            {
                "event": "user signed up",
                "interval": "week",
                "start_date": "2026-06-01",
                "end_date": "2026-06-30",
            },
        )
        weekly = {r["bucket"][:10]: r["value"] for r in payload["series"]}
        for bucket, expected in (
            ("2026-06-01", 402),
            ("2026-06-08", 391),
            ("2026-06-15", 236),
            ("2026-06-22", 228),
        ):
            assert abs(weekly[bucket] - expected) < 20, (bucket, weekly[bucket])

    def test_the_event_stays_advertised(self) -> None:
        """The invariant every planting field has broken in turn."""
        scenario = by_name(self.SCENARIO)
        advertised = {e["name"] for e in scenario.response_for("posthog__list_events")["events"]}
        assert scenario.events_described() <= advertised
        assert "user signed up" in advertised


class TestAPlainTrendSeriesMustBeDerived:
    """The class, closed. Six times a fixture answered a question it was not asked.

    Repository, date range, event name, subject, interval, and interval again — each found
    separately, each fixed in the one scenario that exposed it. The interval one cost real score
    twice, in `partial_month_false_premise` and then in `campaign_traffic_drop`, because fixing
    it in the first did nothing for the second.

    A canned payload cannot answer the interval it was asked for: it returns whatever buckets
    were typed into it. So a plain trend series has to be planted as `daily_truth` and derived.
    The exemption is a *segmented* series, which `DailyTruth` has no shape for — and that is
    recorded here rather than left as an absence, so the next reader knows it was considered.
    """

    #: `onboarding_regression` plants a per-device breakdown, where each row carries a `segment`
    #: alongside bucket and value. `DailyTruth` holds one series of `(day, count)` and cannot
    #: express that. Its payload is already daily, so the defect this test guards is not
    #: reachable there — a request for days gets days.
    SEGMENTED = {"onboarding_regression"}

    def test_no_scenario_plants_a_canned_plain_trend(self) -> None:
        for scenario in SCENARIOS:
            planted = scenario.responses.get("posthog__event_trend")
            if planted is None or scenario.name in self.SEGMENTED:
                continue
            raise AssertionError(
                f"{scenario.name} plants a canned posthog__event_trend payload. A canned series "
                "answers whatever interval it was typed with, whatever the caller asked for. "
                "Plant it as `daily_truth` instead, or add it to SEGMENTED with the reason."
            )

    def test_the_segmented_exemption_is_really_segmented(self) -> None:
        """An exemption nobody checks becomes a place to hide things. If that payload ever stops
        carrying segments, it has no reason to stay canned and this says so."""
        for name in self.SEGMENTED:
            series = by_name(name).responses["posthog__event_trend"]["series"]
            assert all("segment" in row for row in series), name

    def test_every_derived_series_answers_the_interval_asked(self) -> None:
        """The property itself, across every scenario that has one, rather than per fixture."""
        for scenario in SCENARIOS:
            for truth in scenario.daily_truth.get("posthog__event_trend", ()):
                for interval in ("day", "week", "month"):
                    payload = scenario.response_for(
                        "posthog__event_trend",
                        {"event": truth.event, "interval": interval},
                    )
                    assert payload["interval"] == interval, (scenario.name, truth.event, interval)
                    assert payload["series"], (scenario.name, truth.event, interval)

    def test_a_derived_series_declares_the_total_it_actually_sums_to(self) -> None:
        """`tempting_coincidence` declared `total: 9300` beside a series summing to 11,405 — an
        18% contradiction handed to any analyst who read the field instead of adding the rows
        up. Derivation computes it, so the two cannot disagree."""
        for scenario in SCENARIOS:
            for truth in scenario.daily_truth.get("posthog__event_trend", ()):
                payload = scenario.response_for(
                    "posthog__event_trend", {"event": truth.event, "interval": "day"}
                )
                assert payload["total"] == sum(r["value"] for r in payload["series"])
