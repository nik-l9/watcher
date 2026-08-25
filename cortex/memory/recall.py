"""Hybrid recall — semantic hit, then graph expansion, then ranking.

Neither store answers an investigation's question alone.

A vector search over Slack finds *"we're pausing the spring campaign today"* but cannot
tell you the campaign promoted the onboarding feature whose PR shipped in the deploy that
preceded the drop. A graph traversal knows all of that but cannot find the sentence,
because nobody labelled the message. So recall runs the search to get an entry point, and
the traversal to get the structure around it.

**Entity linking is deliberately conservative.** A hit is connected to a graph node when
the hit's own metadata names one — an ingest step that already resolved a Slack message to
a PR — or when a node's key appears verbatim in the text. It does **not** guess from
prose similarity. A wrong link is worse than a missing one here: it would put another
change's PR, author and deploy into the context for this question, and the analyst has no
way to tell an invented association from a real one. A missing link costs recall; a wrong
one manufactures evidence.

**Nothing here is a claim.** Recall returns context with a `source_ref` on every item, and
the report layer still requires a resolvable `evidence_id` for anything asserted. A
recalled fact is a lead to verify with a tool call, not a citation — memory is not
evidence, and treating it as evidence would put last month's understanding into this
week's report with nothing backing it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cortex.memory.entities import NodeLabel, RelType
from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import Hit, VectorStore
from cortex.tenancy.context import TenantContext

#: Metadata keys an ingest step uses to record "this text is about that node".
#:
#: Read rather than inferred: when the ingest already knows a Slack message referenced
#: PR 913, that is a fact, and re-deriving it from the prose would be strictly worse.
_LINK_KEYS = ("entity_label", "entity_key")

#: How far to walk from a linked node.
#:
#: One hop by default. Two hops from a PR reaches the deploy, the release, the feature,
#: the campaign that promoted it and every other PR in the release — which is most of a
#: small graph, and a context window full of weakly related nodes crowds out the one
#: that mattered.
DEFAULT_DEPTH = 1

#: Relationship types worth expanding for an investigation.
#:
#: Restricted on purpose: `ATTENDED` and `MENTIONS` fan out to everyone who was in a
#: meeting or ever named a feature, which is a popularity ranking rather than a causal
#: one.
CAUSAL_RELATIONS: tuple[RelType, ...] = (
    RelType.DELIVERED_BY,
    RelType.SHIPPED_IN,
    RelType.DEPLOYED_AS,
    RelType.AFFECTS,
    RelType.PROMOTES,
    RelType.MEASURED_BY,
    RelType.ABOUT,
    RelType.DECIDED,
)


@dataclass(frozen=True, slots=True)
class RecalledEntity:
    """A graph node reached from a semantic hit."""

    label: str | None
    key: str
    props: dict[str, Any] = field(default_factory=dict)
    #: Which hit led here, so a reader can see why this node is in the context.
    via_source_id: str = ""
    #: 0 for a directly linked node, 1+ for one reached by traversal.
    hops: int = 0


@dataclass(frozen=True, slots=True)
class Recall:
    """What memory offers about a question. Context, never conclusions."""

    query: str
    hits: list[Hit] = field(default_factory=list)
    entities: list[RecalledEntity] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.hits and not self.entities

    def render(self) -> str:
        """The context block handed to the analyst.

        Labelled as memory, with the disclaimer attached rather than left to the prompt
        author to remember. An analyst that cites a recalled Slack message as evidence
        has cited something with no evidence row behind it, and the gate would strip the
        claim — so the framing has to be unambiguous at the point the text is produced.
        """
        if self.is_empty:
            return (
                "MEMORY: nothing recalled for this question. Treat it as no prior "
                "context, not as an absence of history."
            )
        lines = [
            "MEMORY (prior context, NOT evidence). Every item here is a lead to confirm "
            "with a tool call. Do not cite it: only tool results carry evidence ids.",
        ]
        if self.hits:
            lines.append("")
            lines.append("Recalled text:")
            for hit in self.hits:
                where = f" [{hit.source_ref}]" if hit.source_ref else ""
                lines.append(f"  - ({hit.kind}, similarity {hit.score:.2f}){where} {hit.text}")
        if self.entities:
            lines.append("")
            lines.append("Related entities in the knowledge graph:")
            for entity in self.entities:
                summary = ", ".join(
                    f"{k}={v}" for k, v in sorted(entity.props.items()) if k != "key"
                )
                reach = "linked directly" if entity.hops == 0 else f"{entity.hops} hop away"
                lines.append(f"  - {entity.label or '?'} {entity.key} ({reach}) {summary}".rstrip())
        return "\n".join(lines)


class HybridRecall:
    def __init__(self, vectors: VectorStore, graph: GraphStore) -> None:
        self._vectors = vectors
        self._graph = graph

    async def recall(
        self,
        ctx: TenantContext,
        query: str,
        *,
        kinds: Sequence[str] | None = None,
        limit: int = 8,
        depth: int = DEFAULT_DEPTH,
        min_score: float = 0.0,
    ) -> Recall:
        """Semantic entry points, plus the graph structure around them."""
        hits = await self._vectors.search(ctx, query, kinds=kinds, limit=limit, min_score=min_score)
        if not hits:
            # No traversal without an entry point. Walking the graph from nothing would
            # mean picking a node by some proxy for importance, which returns whatever is
            # busiest rather than whatever is relevant.
            return Recall(query=query, hits=[], entities=[])

        entities: list[RecalledEntity] = []
        seen: set[tuple[str | None, str]] = set()

        for hit in hits:
            for label, key in self._links(hit):
                node = await self._graph.get_node(ctx, label, key)
                if node is None:
                    # The link named a node that is not in the graph. Skipped silently:
                    # an ingest can legitimately reference a PR before the graph sync
                    # that creates it, and inventing the node would put an entity into
                    # the context that nothing in the system knows anything about.
                    continue
                if (label.value, key) in seen:
                    continue
                seen.add((label.value, key))
                entities.append(
                    RecalledEntity(
                        label=node.get("_label") or label.value,
                        key=key,
                        props={k: v for k, v in node.items() if k != "_label"},
                        via_source_id=hit.source_id,
                        hops=0,
                    )
                )

                for neighbour in await self._graph.neighbors(
                    ctx, label, key, rel_types=CAUSAL_RELATIONS, depth=depth, limit=limit
                ):
                    n_label = neighbour.get("_label")
                    n_key = str(neighbour.get("key", ""))
                    if not n_key or (n_label, n_key) in seen:
                        continue
                    seen.add((n_label, n_key))
                    entities.append(
                        RecalledEntity(
                            label=n_label,
                            key=n_key,
                            props={k: v for k, v in neighbour.items() if k != "_label"},
                            via_source_id=hit.source_id,
                            hops=1,
                        )
                    )

        return Recall(query=query, hits=hits, entities=entities)

    @staticmethod
    def _links(hit: Hit) -> list[tuple[NodeLabel, str]]:
        """Graph nodes this hit is known to be about.

        Only what the ingest recorded. Guessing from the text was considered and
        rejected: a hit mentioning "onboarding" would match every onboarding PR ever
        merged, and pulling all of them in would present the analyst with several changes
        as if each were equally implicated in this one.
        """
        label_raw = hit.metadata.get(_LINK_KEYS[0])
        key_raw = hit.metadata.get(_LINK_KEYS[1])
        if not label_raw or not key_raw:
            return []
        try:
            label = NodeLabel(str(label_raw))
        except ValueError:
            # An unknown label means the ingest wrote a name outside the closed
            # vocabulary. Dropped rather than coerced: the enums exist so traversal
            # cannot silently stop matching.
            return []
        return [(label, str(key_raw))]
