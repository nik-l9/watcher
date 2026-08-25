"""Tool framework invariants.

The properties asserted here are the ones the whole grounding story rests on:
capabilities are read-only, they return structured JSON rather than prose, and
hallucinated arguments are rejected instead of ignored.
"""

from __future__ import annotations

from typing import Any

import pytest

from cortex.db.models import CredentialProvider
from cortex.tools.base import (
    Capability,
    CapabilityNotFound,
    Freshness,
    Tool,
    ToolNotFound,
    ToolRegistry,
    ToolResult,
)
from cortex.tools.registry import gtm_analyst_registry

OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"x": {"type": "integer"}},
}


async def _handler(ctx: object, **kwargs: object) -> ToolResult:
    del ctx, kwargs
    return ToolResult(payload={"ok": True})


def _capability(name: str = "probe", **kwargs: Any) -> Capability:
    defaults: dict[str, Any] = {
        "name": name,
        "description": "probe",
        "params_schema": dict(OBJECT_SCHEMA),
        "handler": _handler,
        "result_key": "rows",
    }
    return Capability(**{**defaults, **kwargs})


class _ProbeTool(Tool):
    name = "probe"
    provider = None

    def capabilities(self) -> list[Capability]:
        return [_capability("alpha"), _capability("beta")]


class TestReadOnlyEnforcement:
    def test_write_capability_cannot_be_declared(self) -> None:
        """V1 guarantees "humans approve" by the absence of write tools, not by a
        prompt. The type system is the enforcement point."""
        with pytest.raises(ValueError, match="read_only=False"):
            _capability(read_only=False)

    def test_every_shipped_capability_is_read_only(self) -> None:
        registry = gtm_analyst_registry()
        for tool_name in registry.tool_names:
            tool = registry.get(tool_name)
            for capability_name in tool.capability_names:
                assert tool.capability(capability_name).read_only, (
                    f"{tool_name}.{capability_name} is not read-only"
                )


class TestSchemaDiscipline:
    def test_non_object_schema_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="object schema"):
            _capability(params_schema={"type": "array"})

    def test_open_schema_is_rejected(self) -> None:
        """Without additionalProperties=false a hallucinated argument is silently
        ignored instead of surfacing as a correctable error."""
        with pytest.raises(ValueError, match="additionalProperties"):
            _capability(params_schema={"type": "object", "properties": {}})

    def test_every_shipped_schema_is_closed(self) -> None:
        registry = gtm_analyst_registry()
        for spec in registry.llm_tool_specs():
            schema = spec["input_schema"]
            assert schema["type"] == "object", spec["name"]
            assert schema.get("additionalProperties") is False, spec["name"]


class TestToolResult:
    def test_payload_must_be_structured(self) -> None:
        """Tools return JSON. A tool returning prose would leave nothing to cite."""
        with pytest.raises(TypeError, match="never prose"):
            ToolResult(payload="Signups fell 18% because of the deploy.")  # type: ignore[arg-type]

    def test_defaults_to_live(self) -> None:
        assert ToolResult(payload={}).freshness is Freshness.LIVE

    def test_meta_default_is_not_shared(self) -> None:
        a, b = ToolResult(payload={}), ToolResult(payload={})
        a.meta["x"] = 1
        assert b.meta == {}


class TestToolConstruction:
    def test_tool_without_capabilities_is_rejected(self) -> None:
        class _Empty(Tool):
            name = "empty"
            provider = None

            def capabilities(self) -> list[Capability]:
                return []

        with pytest.raises(ValueError, match="no capabilities"):
            _Empty()

    def test_tool_without_a_name_is_rejected(self) -> None:
        class _Nameless(Tool):
            name = ""
            provider = None

            def capabilities(self) -> list[Capability]:
                return [_capability()]

        with pytest.raises(ValueError, match="must declare a name"):
            _Nameless()

    def test_unknown_capability_names_the_alternatives(self) -> None:
        tool = _ProbeTool()
        with pytest.raises(CapabilityNotFound, match="alpha, beta"):
            tool.capability("gamma")


class TestRegistry:
    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_ProbeTool())

    def test_unknown_tool_names_the_alternatives(self) -> None:
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        with pytest.raises(ToolNotFound, match="probe"):
            registry.get("ga4")

    def test_resolves_qualified_names(self) -> None:
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        tool, capability = registry.resolve("probe__alpha")
        assert (tool.name, capability.name) == ("probe", "alpha")

    def test_unqualified_name_is_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        with pytest.raises(ToolNotFound, match="not a qualified tool call"):
            registry.resolve("probe")

    def test_required_providers_excludes_credential_free_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_ProbeTool())
        assert registry.required_providers() == set()


class TestLLMSpecs:
    def test_names_are_flattened_and_unique(self) -> None:
        """A flat list of concrete operations produces better tool selection than
        asking a model to pick a tool and then an operation."""
        specs = gtm_analyst_registry().llm_tool_specs()
        names = [s["name"] for s in specs]
        assert len(names) == len(set(names))
        assert all("__" in name for name in names)

    def test_every_spec_has_a_usable_description(self) -> None:
        """Descriptions are the only thing driving tool selection."""
        for spec in gtm_analyst_registry().llm_tool_specs():
            assert len(spec["description"]) >= 40, spec["name"]


class TestShippedRegistry:
    def test_covers_the_v1_connectors(self) -> None:
        assert gtm_analyst_registry().tool_names == [
            "amplitude",
            "bigquery",
            "ga4",
            "github",
            "hubspot",
            "mixpanel",
            "posthog",
            "slack",
        ]

    def test_requires_every_v1_credential(self) -> None:
        assert gtm_analyst_registry().required_providers() == {
            CredentialProvider.AMPLITUDE,
            CredentialProvider.GA4,
            CredentialProvider.BIGQUERY,
            CredentialProvider.GITHUB,
            CredentialProvider.HUBSPOT,
            CredentialProvider.MIXPANEL,
            CredentialProvider.POSTHOG,
            CredentialProvider.SLACK,
        }

    def test_capabilities_match_the_specification(self) -> None:
        registry = gtm_analyst_registry()
        expected = {
            "ga4": ["compare_periods", "get_funnel", "get_sessions", "run_report", "top_pages"],
            # commits, issues and pull_request_activity were added after a live
            # investigation could not answer "what changed that day": deployment_history
            # returned nothing because the repository has no environment called
            # "production", and there was no way to list commits by date or read the
            # discussion on a change.
            # list_repositories was added after a live investigation searched
            # acme/acme for the company website -- the only repository name it had
            # ever seen -- and honestly reported it could not confirm an answer. Every other
            # capability here takes a repo name, so nothing in the surface could tell the
            # analyst which repositories exist.
            "github": [
                "commit_diff",
                "commits",
                "deployment_history",
                "find_feature",
                "issues",
                "list_repositories",
                "pull_request_activity",
                "recent_prs",
                "release_summary",
            ],
            "hubspot": ["activities", "closed_won", "companies", "contacts", "pipeline"],
            "slack": ["find_decision", "recent_threads", "search_messages"],
            "bigquery": ["list_queries", "run_named_query"],
            # The three "what changed" capabilities are the reason this connector
            # matters: annotations, flags and experiments are a change record living
            # beside the metrics, which no other source provides.
            # list_projects exists because an organisation splits analytics across
            # projects that do not share events — a marketing site, a SaaS product, an
            # open-source client — so querying the wrong one returns an empty result that
            # looks exactly like a real absence.
            "posthog": [
                "annotations",
                "event_trend",
                "experiments",
                "feature_flags",
                "funnel",
                "list_events",
                "list_projects",
            ],
        }
        for tool_name, capabilities in expected.items():
            assert registry.get(tool_name).capability_names == capabilities, tool_name

    def test_registry_instances_are_independent(self) -> None:
        """A shared mutable registry is the kind of global that becomes a
        cross-tenant bug."""
        assert gtm_analyst_registry() is not gtm_analyst_registry()


class TestSchemaMatchesHandler:
    """F-06. Three capabilities declared parameters their handlers required but the
    schema did not list as required, so a model omitting one got a Python TypeError
    wrapped as a generic ToolError — an opaque internal failure instead of a
    correctable validation error.

    Derived by introspection rather than enumerated, so the whole class of drift is
    caught instead of those three instances.
    """

    @staticmethod
    def _mandatory(handler: object) -> set[str]:
        """Parameters the handler cannot default."""
        import inspect

        return {
            name
            for name, param in inspect.signature(handler).parameters.items()  # type: ignore[arg-type]
            if param.default is inspect.Parameter.empty
            and name not in ("self", "ctx")
            and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        }

    def test_every_mandatory_parameter_is_declared_required(self) -> None:
        registry = gtm_analyst_registry()
        drift = []
        for tool_name in registry.tool_names:
            tool = registry.get(tool_name)
            for capability_name in tool.capability_names:
                capability = tool.capability(capability_name)
                missing = self._mandatory(capability.handler) - set(
                    capability.params_schema.get("required", [])
                )
                if missing:
                    drift.append(f"{tool_name}.{capability_name} -> {sorted(missing)}")
        assert not drift, (
            "these capabilities require parameters their schema does not declare, so "
            "omitting one raises TypeError instead of InvalidParams:\n  " + "\n  ".join(drift)
        )

    def test_every_required_parameter_exists_on_the_handler(self) -> None:
        """The other direction: a schema requiring a parameter the handler does not
        accept would fail as an unexpected keyword argument."""
        import inspect

        registry = gtm_analyst_registry()
        drift = []
        for tool_name in registry.tool_names:
            tool = registry.get(tool_name)
            for capability_name in tool.capability_names:
                capability = tool.capability(capability_name)
                accepted = set(inspect.signature(capability.handler).parameters)
                for declared in capability.params_schema.get("required", []):
                    if declared not in accepted:
                        drift.append(f"{tool_name}.{capability_name} -> {declared}")
        assert not drift, "schema requires parameters the handler cannot accept:\n  " + "\n  ".join(
            drift
        )

    def test_every_schema_property_exists_on_the_handler(self) -> None:
        """An optional parameter the handler does not accept is the same bug, just
        only triggered when a model happens to pass it."""
        import inspect

        registry = gtm_analyst_registry()
        drift = []
        for tool_name in registry.tool_names:
            tool = registry.get(tool_name)
            for capability_name in tool.capability_names:
                capability = tool.capability(capability_name)
                signature = inspect.signature(capability.handler)
                accepts_kwargs = any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
                )
                if accepts_kwargs:
                    continue
                for prop in capability.params_schema.get("properties", {}):
                    if prop not in signature.parameters:
                        drift.append(f"{tool_name}.{capability_name} -> {prop}")
        assert not drift, "schema declares parameters the handler cannot accept:\n  " + "\n  ".join(
            drift
        )
