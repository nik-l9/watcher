"""Graph vocabulary.

The vocabulary is closed on purpose: if labels could be invented at runtime,
traversal queries would silently stop matching and recall would rot with no test
failing. These tests pin the vocabulary and the label-injection guard.
"""

from __future__ import annotations

import pytest

from cortex.memory.entities import Edge, Node, NodeLabel, RelType
from cortex.memory.falkordb_store import _label, _rel


class TestVocabulary:
    def test_label_values_are_pascal_case_and_unique(self) -> None:
        values = [label.value for label in NodeLabel]
        assert len(values) == len(set(values))
        for v in values:
            assert v[0].isupper(), v
            assert v.isalnum(), v

    def test_rel_values_are_screaming_snake_and_unique(self) -> None:
        values = [rel.value for rel in RelType]
        assert len(values) == len(set(values))
        for v in values:
            assert v.replace("_", "").isalpha() and v.isupper(), v

    def test_causal_chain_vocabulary_is_present(self) -> None:
        """The doc's Campaign→Feature→PR→Ticket→Customer→Revenue walk must be expressible."""
        for required in (
            NodeLabel.CAMPAIGN,
            NodeLabel.FEATURE,
            NodeLabel.PR,
            NodeLabel.DEPLOY,
            NodeLabel.METRIC,
            NodeLabel.SUPPORT_TICKET,
            NodeLabel.CUSTOMER,
        ):
            assert required in NodeLabel
        for required_rel in (
            RelType.PROMOTES,
            RelType.DELIVERED_BY,
            RelType.SHIPPED_IN,
            RelType.AFFECTS,
            RelType.RAISED,
        ):
            assert required_rel in RelType


class TestNodeAndEdge:
    def test_node_props_default_to_empty_and_are_not_shared(self) -> None:
        a = Node(NodeLabel.DEPLOY, "d1")
        b = Node(NodeLabel.DEPLOY, "d2")
        a.props["x"] = 1
        assert b.props == {}, "dataclass default must not be a shared mutable"

    def test_edge_props_default_to_empty_and_are_not_shared(self) -> None:
        a = Edge(RelType.AFFECTS, NodeLabel.DEPLOY, "d1", NodeLabel.METRIC, "m1")
        b = Edge(RelType.AFFECTS, NodeLabel.DEPLOY, "d2", NodeLabel.METRIC, "m2")
        a.props["x"] = 1
        assert b.props == {}


class TestInjectionGuards:
    """Labels and relationship types cannot be parameterized in Cypher, so the
    only defence is proving they came from the closed enum."""

    def test_label_accepts_enum(self) -> None:
        assert _label(NodeLabel.DEPLOY) == "Deploy"

    def test_rel_accepts_enum(self) -> None:
        assert _rel(RelType.AFFECTS) == "AFFECTS"

    @pytest.mark.parametrize(
        "hostile",
        [
            "Deploy) MATCH (x) DETACH DELETE x //",
            "Deploy`",
            "Deploy {key:'x'}",
            "Deploy) RETURN 1 //",
            "Deploy",  # even a benign-looking bare string must be refused
        ],
    )
    def test_label_rejects_raw_strings(self, hostile: str) -> None:
        with pytest.raises(TypeError, match="expected NodeLabel"):
            _label(hostile)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "hostile", ["AFFECTS", "AFFECTS*0..99", "AFFECTS] () // ", "AFFECTS|ANYTHING"]
    )
    def test_rel_rejects_raw_strings(self, hostile: str) -> None:
        with pytest.raises(TypeError, match="expected RelType"):
            _rel(hostile)  # type: ignore[arg-type]

    def test_strenum_membership_does_not_weaken_the_guard(self) -> None:
        """NodeLabel is a StrEnum, so "Deploy" == NodeLabel.DEPLOY is True.

        The guard must therefore check the type, not equality — otherwise a plain
        string carrying Cypher would pass an equality-based check.
        """
        assert NodeLabel.DEPLOY == "Deploy"
        with pytest.raises(TypeError):
            _label("Deploy")  # type: ignore[arg-type]
