"""What a syncer is, and what it is allowed to write.

A syncer turns one connector's reads into memory: graph nodes and edges, semantic
documents, metric points. It does **not** write evidence — evidence exists to be cited by a
report, and a sync produces no report. A nightly job minting thousands of Evidence rows
would fill the citation space with rows nothing ever cites while needing an investigation
id that does not exist.

**One syncer, several streams.** A GitHub sync pulls commits, deployments and issues, and
each advances its own watermark. That matters more than it sounds: with one cursor per
provider, a failed issues pull would rewind commits, or a successful commits pull would
mark issues current when it never ran. So a syncer declares its streams and each is
recorded separately.

**A stream that fails does not fail its siblings.** The runner records each independently,
because "GitHub is broken" and "GitHub's issues endpoint is broken" call for different
responses, and collapsing them loses the commits that did arrive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import CredentialProvider
from cortex.ingest.metrics import Point
from cortex.ingest.state import Window
from cortex.memory.entities import Edge, Node
from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import Document
from cortex.tenancy.context import TenantContext
from cortex.tools.base import ToolContext


@dataclass(slots=True)
class StreamResult:
    """What one stream produced.

    Returned as data rather than written by the syncer, so the runner owns every write and
    a syncer cannot accidentally commit half a stream. It also makes a syncer testable
    without a database, a graph or an embedding key.
    """

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    points: list[Point] = field(default_factory=list)

    #: Set when the data is genuinely **incomplete** — a truncated page, a stream the
    #: upstream refused, documents that could not be embedded. The watermark still advances,
    #: because what did arrive is real, but the gap surfaces in a report's data-quality
    #: section rather than being silently absorbed.
    partial: str | None = None

    #: Set when something is worth recording but the data is **not** incomplete: a project
    #: with no annotations, a repository with no issues that week.
    #:
    #: Separated from `partial` after seeing the two conflated on real data. All five real
    #: PostHog projects have zero annotations, and reporting that as "incomplete — treat
    #: counts as a floor" told a reader their change log might be missing entries when the
    #: truth is that nobody writes any. An emptiness that is real must not be dressed as a
    #: gap in our collection, or every disclosure stops being worth reading.
    note: str | None = None

    @property
    def item_count(self) -> int:
        return len(self.nodes) + len(self.edges) + len(self.documents) + len(self.points)


class Syncer(ABC):
    """One provider's ingest."""

    #: Which credential this syncer needs.
    provider: CredentialProvider
    #: The tool whose capabilities it calls, by registry name.
    tool_name: str

    @property
    @abstractmethod
    def streams(self) -> tuple[str, ...]:
        """The streams this syncer pulls, each with its own watermark."""

    @abstractmethod
    async def sync_stream(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        ctx: ToolContext,
        *,
        stream: str,
        window: Window,
        graph: GraphStore,
    ) -> StreamResult:
        """Read one stream for one window and return what it found.

        `session` and `graph` are for **reading**. A syncer must not write through either;
        the runner owns every write, so that a syncer cannot commit half a stream and so
        that ordering (nodes before edges) is enforced in one place.

        The graph is passed because some relationships can only be derived from what is
        already stored. A deployment record carries a sha, a pull request carries the same
        sha, and neither stream knows about the other — the edge between them exists only
        once both have landed, and finding it means reading the graph back.
        """
