"""A capability's declared `result_key` must name a key it actually returns.

`hubspot__pipeline` declared `result_key="stages"` and returns `deals`. Nothing in the
module ever set a "stages" key, so `Capability.is_empty` looked for a key that could not
exist and answered True for every pipeline result -- including a live one carrying 69 open
deals worth $2.4M.

Two things followed, both invisible:

  - the analyst was handed `THIS RESULT IS EMPTY. An empty result is not evidence that
    nothing happened ... widen or remove one filter and look again` stapled to a full
    pipeline;
  - the grounding gate counted that observation among the ones that "returned no results",
    so a report resting on it was disclosed as resting on silence.

`is_empty`'s own docstring anticipates this exact failure -- *"a payload with rows under an
unexpected key looked empty when it was not"* -- which is why it reads a declared key rather
than guessing. Declaring the wrong one puts the hole back.
"""

from __future__ import annotations

import pytest

from cortex.tools.base import NEVER_EMPTY
from cortex.tools.registry import gtm_analyst_registry


def _declared() -> list[tuple[str, str, str]]:
    registry = gtm_analyst_registry()
    out = []
    for tool_name in registry.tool_names:
        for capability in registry.get(tool_name).capabilities():
            key = capability.result_key
            if key and key != "" and not key.startswith(NEVER_EMPTY):
                out.append((tool_name, capability.name, key))
    return out


class TestADeclaredResultKeyIsOneTheConnectorSets:
    @pytest.mark.parametrize("tool,capability,key", _declared())
    def test_the_key_appears_in_the_connector_module(
        self, tool: str, capability: str, key: str
    ) -> None:
        import importlib
        import pathlib

        module = importlib.import_module(f"cortex.tools.{tool}")
        source = pathlib.Path(module.__file__).read_text()
        assert f'"{key}"' in source, (
            f"{tool}__{capability} declares result_key={key!r}, which {tool}.py never sets. "
            "is_empty() will report every result as empty."
        )


class TestThePipelineRegression:
    """The specific case, asserted both ways so a fix in one direction cannot break the other."""

    @pytest.fixture
    def pipeline(self):
        return gtm_analyst_registry().get("hubspot").capability("pipeline")

    def test_a_full_pipeline_is_not_empty(self, pipeline) -> None:
        payload = {
            "pipeline_id": None,
            "total_matching": 69,
            "count": 69,
            "total_amount": 2_400_000,
            "deals": [{"id": str(n)} for n in range(69)],
        }
        assert pipeline.is_empty(payload) is False

    def test_an_empty_pipeline_still_reads_as_empty(self, pipeline) -> None:
        # The fix must not buy a false negative: a tenant with nothing open should still be
        # told so, because "no open deals" is a real observation.
        payload = {"pipeline_id": None, "total_matching": 0, "count": 0, "deals": []}
        assert pipeline.is_empty(payload) is True
