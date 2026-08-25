"""The conditions under which a queued borrow becomes worth building.

ADR 0002 lists three mechanisms from the OpenHands SDK that were deliberately *not* adopted,
each with a written trigger. A trigger written in a document is a trigger nobody checks, so the
one that can be checked mechanically is checked here.

`SecretRegistry` is that one. Theirs exists because their agent runs arbitrary bash in a sandbox
and a tool parameter can carry a credential, so secrets have to be registered and masked. Ours
has nothing to mask: every connector reads its credential from the vault by provider, and no
capability accepts one as a parameter. That is a property of the tool surface rather than an
opinion, so this test asserts it — and fails on the commit that first adds a capability taking a
token, which is precisely the commit at which the borrow becomes worth making.

The other two triggers are measurements rather than properties, and are recorded in ADR 0002
with the numbers behind them:

  - **Event-sourced resumable state** — trigger: an investigation running over five minutes, or
    a UI that has to replay one. Longest completed run to date: 123 seconds. The replay half is
    answered without it, because `/investigations/{id}/trace` is derived from the `tool_calls`
    audit rows, so history is durable without a second event log.
  - **`LLMSummarizingCondenser`** — trigger: the first context-limit hit. Largest context
    reached to date: 129,605 tokens against a 1,000,000-token window, 13%. `ModelCard`'s
    `room_for_output()` exists so that number can be computed rather than guessed when it
    matters.
"""

from __future__ import annotations

import re

from cortex.tools.registry import gtm_analyst_registry

#: Parameter names that would mean a tool call can carry a credential in its arguments.
#:
#: Matched on the name rather than the value, because the value only exists at call time and by
#: then it is already in a `tool_calls.params` JSONB column and in a model's context. The name
#: is visible at import.
_CREDENTIAL_SHAPED = re.compile(
    r"token|api[_-]?key|secret|password|passwd|credential|bearer|cookie|private[_-]?key|"
    r"access[_-]?key|session[_-]?id",
    re.I,
)


def _parameters() -> list[tuple[str, str]]:
    registry = gtm_analyst_registry()
    return [
        (spec["name"], parameter)
        for spec in registry.llm_tool_specs()
        for parameter in ((spec.get("input_schema") or {}).get("properties") or {})
    ]


def test_no_capability_accepts_a_credential_as_a_parameter() -> None:
    """The `SecretRegistry` trigger, as a test rather than a note.

    Two reasons this matters beyond the borrow. A credential in a tool parameter is written to
    the `tool_calls` audit row, which is displayed on the report page — and it is placed in the
    model's context, where prompt caching means it is also sent again on every subsequent step
    of the investigation.
    """
    offenders = [
        (capability, parameter)
        for capability, parameter in _parameters()
        if _CREDENTIAL_SHAPED.search(parameter)
    ]
    assert not offenders, (
        f"these capabilities accept a credential-shaped parameter: {offenders}. "
        "This is the trigger for adopting their SecretRegistry (ADR 0002): a secret in a tool "
        "argument is written to the tool_calls audit row, shown on the report page, and kept in "
        "the model's cached context for the rest of the investigation. Read the credential from "
        "the vault by provider instead, as every existing connector does."
    )


def test_the_tool_surface_is_large_enough_for_that_to_be_meaningful() -> None:
    """A guard on the guard. If the registry were empty the test above would pass while
    asserting nothing, which is the failure mode of every check written as an absence."""
    parameters = _parameters()
    assert len({capability for capability, _ in parameters}) >= 20
    assert len(parameters) >= 60
