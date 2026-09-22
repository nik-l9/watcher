"""The suite's first state question, and the property that makes it worth running.

Every other scenario asks why something *changed*. Six live runs against a real CRM were
asked state questions instead -- how much pipeline, what is our win rate -- and five of the
six answered out of HubSpot alone, never touching the other three connectors they had.

A suite made entirely of change questions cannot detect that, which is why it went unnoticed.
`won_accounts_not_activated` is built so that a single-source answer is not merely worse, it
is *wrong*: each source on its own reports a healthy picture, and only the join shows six of
nine won accounts have never produced a product event.

These tests protect that construction. A future edit that made either source sufficient would
leave a scenario that still passes and measures nothing.
"""

from __future__ import annotations

import asyncio

import pytest

from cortex.eval.fixtures import alternatives_of, by_name
from cortex.eval.runner import scenario_registry

SCENARIO = "won_accounts_not_activated"


def _call(capability, **params):
    return asyncio.run(capability.handler(None, **params)).payload


@pytest.fixture
def scenario():
    return by_name(SCENARIO)


@pytest.fixture
def registry(scenario):
    return scenario_registry(scenario)


class TestNeitherSourceCanAnswerAlone:
    def test_the_crm_shows_a_good_quarter_and_nothing_else(self, registry) -> None:
        # A HubSpot-only reading is not wrong about anything. It is just not an answer to
        # the question that was asked, which is the trap.
        payload = _call(registry.get("hubspot").capability("closed_won"))
        assert payload["count"] == 9
        assert payload["total_amount"] == 1_230_000
        blob = repr(payload).lower()
        for absent in ("usage", "event", "active", "activation"):
            assert absent not in blob, f"the CRM payload must not hint at {absent!r}"

    def test_product_analytics_alone_shows_a_flat_healthy_series(self, registry) -> None:
        payload = _call(
            registry.get("posthog").capability("event_trend"),
            event="$pageview",
            start_date="2026-07-01",
            end_date="2026-09-18",
        )
        values = [row["value"] for row in payload["series"]]
        assert len(values) >= 60
        # Flat within noise: no change-point to find, so the change-question method that the
        # analyst is actually taught yields nothing here.
        assert max(values) / min(values) < 1.5

    def test_the_answer_exists_only_in_the_join(self, registry) -> None:
        won = {
            deal["name"]
            for deal in _call(registry.get("hubspot").capability("closed_won"))["deals"]
        }
        seen = {
            row["segment"]
            for row in _call(
                registry.get("posthog").capability("event_trend"),
                event="workspace opened",
                breakdown_property="organization",
            )["series"]
        }
        silent = won - seen
        assert len(silent) == 6
        # And the overlap is real, so the finding is "six of nine" rather than "no data".
        assert won & seen == {"Globex", "Hooli", "Wonka Industries"}


class TestTheLabelMatchesTheConstruction:
    def test_it_requires_a_capability_from_each_connector(self, scenario) -> None:
        required = {
            name.split("__")[0]
            for requirement in scenario.ground_truth.required_capabilities
            for name in alternatives_of(requirement)
        }
        assert required == {"hubspot", "posthog"}, (
            "if this scenario stops requiring both connectors it stops measuring the only "
            "thing it was built to measure"
        )

    def test_the_tenant_is_offered_distractors_too(self, scenario) -> None:
        # Built with only the two connectors it needs, this scenario could be passed by calling
        # everything available -- which is not the behaviour it exists to measure, and not the
        # situation that produced the failure. The live investigations had four connectors and
        # left three untouched, so the fixture has four.
        assert scenario.connected_tools == frozenset({"hubspot", "posthog", "github", "slack"})

    def test_the_distractors_say_nothing_about_activation(self, registry) -> None:
        # A distractor that hinted at the answer would make the scenario easier, not harder.
        won = {
            deal["name"]
            for deal in _call(registry.get("hubspot").capability("closed_won"))["deals"]
        }
        commits = _call(
            registry.get("github").capability("commits"), repo="acme/web", since="2026-07-01"
        )
        messages = _call(registry.get("slack").capability("search_messages"), query="renewal")
        for payload in (commits, messages):
            blob = repr(payload)
            assert not [name for name in won if name.split()[0] in blob]
            for word in ("activat", "onboard", "adopt"):
                assert word not in blob.lower()
        # ...and they are not empty, which would make them no distraction at all.
        assert len(commits["commits"]) >= 4
        assert len(messages["messages"]) >= 3

    def test_a_correct_answer_must_name_an_account_it_found(self, scenario) -> None:
        groups = [alternatives_of(r) for r in scenario.ground_truth.required_signals]
        named = [g for g in groups if any(a in {"Northwind", "Initech", "Cyberdyne"} for a in g)]
        assert named, "a report can otherwise pass by asserting a gap it never located"

    def test_the_decoys_are_the_single_source_readings(self, scenario) -> None:
        # Each decoy is a true statement about one tool and a wrong answer to the question.
        assert any("healthy" in decoy for decoy in scenario.ground_truth.decoys)
        assert any("all accounts" in decoy for decoy in scenario.ground_truth.decoys)

    def test_it_is_not_labelled_unanswerable(self, scenario) -> None:
        # The data answers this clearly. Only a single-source reading cannot.
        assert not scenario.ground_truth.is_unanswerable
        assert not scenario.ground_truth.is_false_premise


class TestTheQuestionDoesNotPointAtTheSecondSource:
    """`forecast_ignores_the_decision`, and why it is the harder of the two.

    `won_accounts_not_activated` needs two connectors, but its question hands over the second
    one -- "are the accounts we closed actually *using the product*" names product analytics
    in its own wording, and both arms of a paired measurement found it. It tests whether the
    analyst can join two sources, not whether it thinks to look.

    Here every word of the question belongs to the CRM, and the CRM answers it completely and
    wrongly. That is the shape of the live failure: six real investigations, five answered out
    of one connector while three others sat connected and unopened.
    """

    @pytest.fixture
    def forecast(self):
        return by_name("forecast_ignores_the_decision")

    @pytest.fixture
    def forecast_registry(self, forecast):
        return scenario_registry(forecast)

    def test_the_question_names_no_source(self, forecast) -> None:
        # If a future edit puts "slack", "decision" or "commit" in the question, the scenario
        # stops measuring whether the analyst thinks to look and starts measuring whether it
        # can follow an instruction.
        question = forecast.question.lower()
        for pointer in ("slack", "message", "thread", "decision", "discussed", "commit"):
            assert pointer not in question, "the question must not point at the second source"

    def test_the_crm_alone_gives_a_confident_wrong_answer(self, forecast_registry) -> None:
        pipeline = _call(forecast_registry.get("hubspot").capability("pipeline"))
        assert pipeline["count"] == 12
        assert pipeline["total_amount"] == 2_400_000
        # Nothing in the CRM payload hints that any of it is at risk.
        blob = repr(pipeline).lower()
        for absent in ("frozen", "procurement", "freeze", "commit"):
            assert absent not in blob

    @pytest.mark.parametrize(
        "capability,params",
        [("find_decision", {"topic": "forecast"}), ("search_messages", {"query": "commit"})],
    )
    def test_either_slack_route_reaches_the_decision(self, forecast_registry, capability, params):
        # Any-of on purpose: requiring one endpoint scores the route, not the result.
        payload = _call(forecast_registry.get("slack").capability(capability), **params)
        assert any("frozen procurement" in m["text"] for m in payload["messages"])

    def test_the_adjustment_is_exactly_the_planted_figure(self, forecast_registry) -> None:
        pipeline = _call(forecast_registry.get("hubspot").capability("pipeline"))
        frozen = {"Stark Industries", "Tyrell Corp", "Massive Dynamic"}
        excluded = sum(d["amount"] for d in pipeline["deals"] if d["name"] in frozen)
        assert excluded == 900_000
        assert pipeline["total_amount"] - excluded == 1_500_000

    def test_the_distractors_do_not_carry_it(self, forecast_registry) -> None:
        commits = _call(
            forecast_registry.get("github").capability("commits"),
            repo="acme/web",
            since="2026-07-01",
        )
        blob = repr(commits).lower()
        for word in ("stark", "tyrell", "massive", "procurement", "forecast"):
            assert word not in blob
