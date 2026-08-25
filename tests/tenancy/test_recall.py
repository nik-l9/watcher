"""Hybrid recall — semantic entry point, then graph expansion.

Two things are worth proving here, and they pull in opposite directions:

  - Recall must reach the structure a search alone cannot: from a Slack message to the PR
    it referenced, and from there to the deploy and the metric it affected.
  - Recall must **not** invent that structure. A wrong entity link puts another change's
    author, PR and deploy into the context for this question, and the analyst has no way
    to tell a manufactured association from a real one.

And one framing property: what recall returns is context, never evidence. A report can
only cite a resolvable `evidence_id`, so the rendered block has to say so at the point the
text is produced rather than relying on the prompt author to remember.
"""

from __future__ import annotations

from cortex.memory.entities import Edge, Node, NodeLabel, RelType
from cortex.memory.falkordb_store import FalkorDBGraphStore
from cortex.memory.recall import HybridRecall
from cortex.memory.vector_store import Document, QdrantVectorStore
from cortex.tenancy.context import TenantContext

_SLACK_TEXT = "pausing the spring campaign today, budget exhausted"


async def _seed_graph(graph: FalkorDBGraphStore, ctx: TenantContext) -> None:
    await graph.upsert_nodes(
        ctx,
        [
            Node(NodeLabel.PR, "913", {"title": "Rework mobile onboarding modal"}),
            Node(NodeLabel.DEPLOY, "91c3e4a", {"environment": "prod-web"}),
            Node(NodeLabel.METRIC, "signups", {"name": "signups"}),
            # Unconnected, so a test can prove traversal does not simply return
            # everything in the graph.
            Node(NodeLabel.PR, "908", {"title": "Update pricing page copy"}),
        ],
    )
    await graph.upsert_edges(
        ctx,
        [
            Edge(RelType.DEPLOYED_AS, NodeLabel.PR, "913", NodeLabel.DEPLOY, "91c3e4a"),
            Edge(RelType.AFFECTS, NodeLabel.DEPLOY, "91c3e4a", NodeLabel.METRIC, "signups"),
        ],
    )


async def _seed_text(
    vectors: QdrantVectorStore, ctx: TenantContext, *, link: dict[str, str] | None = None
) -> None:
    await vectors.upsert(
        ctx,
        [
            Document(
                kind="slack",
                source_id="1781000000.000100",
                text=_SLACK_TEXT,
                source_ref="https://slack.example/marketing/1",
                metadata={"channel": "marketing", **(link or {})},
            )
        ],
    )


async def test_a_linked_hit_pulls_in_the_graph_around_it(
    vectors: QdrantVectorStore,
    graph: FalkorDBGraphStore,
    vector_tenant_a: TenantContext,
) -> None:
    """The point of hybrid recall: the search finds the sentence, the graph supplies
    the structure nobody wrote down in it."""
    await graph.provision(vector_tenant_a)
    try:
        await _seed_graph(graph, vector_tenant_a)
        await _seed_text(vectors, vector_tenant_a, link={"entity_label": "PR", "entity_key": "913"})

        recall = await HybridRecall(vectors, graph).recall(vector_tenant_a, _SLACK_TEXT)

        assert [hit.source_id for hit in recall.hits] == ["1781000000.000100"]
        reached = {(entity.label, entity.key) for entity in recall.entities}
        # The linked PR, plus the deploy one hop away along a causal relation.
        assert ("PR", "913") in reached
        assert ("Deploy", "91c3e4a") in reached
        # Not the unrelated PR: expansion follows edges, not label membership.
        assert ("PR", "908") not in reached
    finally:
        await graph.drop(vector_tenant_a)


async def test_an_unlinked_hit_returns_text_without_inventing_entities(
    vectors: QdrantVectorStore,
    graph: FalkorDBGraphStore,
    vector_tenant_a: TenantContext,
) -> None:
    """A hit the ingest never resolved must not be guessed into the graph.

    The text says "campaign", and the graph has a PR about onboarding and another about
    pricing. Matching on prose would attach one of them, and a fabricated association is
    worse than a missing one: it presents an unrelated change as implicated.
    """
    await graph.provision(vector_tenant_a)
    try:
        await _seed_graph(graph, vector_tenant_a)
        await _seed_text(vectors, vector_tenant_a)

        recall = await HybridRecall(vectors, graph).recall(vector_tenant_a, _SLACK_TEXT)

        assert recall.hits, "the text itself is still recalled"
        assert recall.entities == []
    finally:
        await graph.drop(vector_tenant_a)


async def test_a_link_to_a_node_that_does_not_exist_is_dropped(
    vectors: QdrantVectorStore,
    graph: FalkorDBGraphStore,
    vector_tenant_a: TenantContext,
) -> None:
    """An ingest can reference a PR before the graph sync that creates it. That is a
    gap in memory, not licence to invent a node the system knows nothing about."""
    await graph.provision(vector_tenant_a)
    try:
        await _seed_text(
            vectors, vector_tenant_a, link={"entity_label": "PR", "entity_key": "99999"}
        )

        recall = await HybridRecall(vectors, graph).recall(vector_tenant_a, _SLACK_TEXT)

        assert recall.hits
        assert recall.entities == []
    finally:
        await graph.drop(vector_tenant_a)


async def test_a_label_outside_the_vocabulary_is_refused(
    vectors: QdrantVectorStore,
    graph: FalkorDBGraphStore,
    vector_tenant_a: TenantContext,
) -> None:
    """The enums exist so traversal cannot silently stop matching. A link naming
    something outside them is dropped rather than coerced into a near-miss."""
    await graph.provision(vector_tenant_a)
    try:
        await _seed_graph(graph, vector_tenant_a)
        await _seed_text(
            vectors,
            vector_tenant_a,
            link={"entity_label": "PullRequest", "entity_key": "913"},
        )

        recall = await HybridRecall(vectors, graph).recall(vector_tenant_a, _SLACK_TEXT)

        assert recall.entities == []
    finally:
        await graph.drop(vector_tenant_a)


async def test_recall_does_not_cross_tenants(
    vectors: QdrantVectorStore,
    graph: FalkorDBGraphStore,
    vector_tenant_a: TenantContext,
    vector_tenant_b: TenantContext,
) -> None:
    """Both stores are isolated per tenant; recall must not become the seam that joins
    them. It reads only from `ctx`, so tenant B sees nothing of tenant A."""
    await graph.provision(vector_tenant_a)
    await graph.provision(vector_tenant_b)
    try:
        await _seed_graph(graph, vector_tenant_a)
        await _seed_text(vectors, vector_tenant_a, link={"entity_label": "PR", "entity_key": "913"})

        recall = await HybridRecall(vectors, graph).recall(vector_tenant_b, _SLACK_TEXT)

        assert recall.is_empty
    finally:
        await graph.drop(vector_tenant_a)
        await graph.drop(vector_tenant_b)


async def test_empty_recall_says_so_rather_than_looking_like_absent_history(
    vectors: QdrantVectorStore, graph: FalkorDBGraphStore, vector_tenant_a: TenantContext
) -> None:
    recall = await HybridRecall(vectors, graph).recall(vector_tenant_a, "anything")
    rendered = recall.render()
    assert "nothing recalled" in rendered
    assert "not as an absence of history" in rendered


async def test_rendered_memory_is_labelled_as_not_evidence(
    vectors: QdrantVectorStore,
    graph: FalkorDBGraphStore,
    vector_tenant_a: TenantContext,
) -> None:
    """A recalled Slack message has no evidence row behind it. If the block did not say
    so, an analyst citing it would produce a claim the gate then strips — and the reader
    would see a thinner report for no visible reason."""
    await graph.provision(vector_tenant_a)
    try:
        await _seed_text(vectors, vector_tenant_a)
        recall = await HybridRecall(vectors, graph).recall(vector_tenant_a, _SLACK_TEXT)

        rendered = recall.render()
        assert "NOT evidence" in rendered
        assert "Do not cite it" in rendered
        # The provenance is still there, so a lead can be followed to its source.
        assert "https://slack.example/marketing/1" in rendered
    finally:
        await graph.drop(vector_tenant_a)
