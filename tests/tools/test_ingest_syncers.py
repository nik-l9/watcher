"""The syncers' mapping layer: upstream payload in, memory out.

Placed alongside the connector tests because it needs the same harness — the syncers call
the real connectors, so only the socket is replaced. What is being tested is the mapping,
which is where the silent bugs live: a payload shape that yields nothing produces an empty
graph, and an empty graph reads as "nothing happened" rather than as a bug.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from cortex.ingest.github import GitHubSyncer
from cortex.ingest.posthog import PostHogSyncer
from cortex.ingest.state import Window
from cortex.memory.entities import NodeLabel, RelType
from cortex.tenancy.context import TenantContext
from cortex.tools.base import InvalidParams, ToolContext

WINDOW = Window(
    since=datetime(2026, 7, 1, tzinfo=UTC),
    until=datetime(2026, 7, 30, tzinfo=UTC),
    first_sync=True,
)


class _Graph:
    """A graph that answers `nodes_by_label` from what a test seeded."""

    def __init__(self, nodes: dict[NodeLabel, list[dict[str, Any]]] | None = None) -> None:
        self._nodes = nodes or {}

    async def nodes_by_label(self, ctx, label, **kwargs):  # type: ignore[no-untyped-def]
        return self._nodes.get(label, [])


# ------------------------------------------------------------------------- github


@pytest.fixture
def github() -> GitHubSyncer:
    return GitHubSyncer()


@pytest.fixture
def gh_ctx(tenant: TenantContext) -> ToolContext:
    return ToolContext(
        tenant=tenant,
        credential="ghp_fake",
        credential_metadata={"repos": "acme/web"},
    )


COMMITS = [
    {
        "sha": "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a",
        "html_url": "https://github.com/acme/web/commit/91c3e4a",
        "commit": {
            "author": {"name": "Dana Whitfield", "date": "2026-07-14T10:02:00Z"},
            "message": "Rework mobile onboarding modal (#913)",
        },
    },
    {
        # No PR reference: a dependency bump. Deliberately present.
        "sha": "a41b8c92de3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b",
        "html_url": "https://github.com/acme/web/commit/a41b8c9",
        "commit": {
            "author": {"name": "dependabot[bot]", "date": "2026-07-14T09:41:00Z"},
            "message": "Bump tailwindcss from 4.1.2 to 4.1.3",
        },
    },
]


class TestGitHubCommits:
    async def test_a_commit_with_a_pr_becomes_a_pull_request_node(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        patch_client(github._tool, {"/commits": COMMITS})
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="commits", window=WINDOW, graph=_Graph()
        )

        prs = [n for n in result.nodes if n.label is NodeLabel.PR]
        assert [n.key for n in prs] == ["913"]
        assert prs[0].props["sha"].startswith("91c3e4a")
        assert prs[0].props["repo"] == "acme/web"

    async def test_a_commit_without_a_pr_adds_no_node(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """The vocabulary has no `Commit` label, and inventing one would add nodes nothing
        ever walks to. A dependency bump with no PR is skipped rather than stored as a
        change with no context."""
        patch_client(github._tool, {"/commits": COMMITS})
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="commits", window=WINDOW, graph=_Graph()
        )
        assert not any("tailwindcss" in str(n.props.get("title", "")) for n in result.nodes)

    async def test_the_author_is_linked_to_the_change(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """ "Who shipped this" is a question an investigation asks, and it is one hop."""
        patch_client(github._tool, {"/commits": COMMITS})
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="commits", window=WINDOW, graph=_Graph()
        )
        authored = [e for e in result.edges if e.rel is RelType.AUTHORED]
        assert ("Dana Whitfield", "913") in [(e.from_key, e.to_key) for e in authored]

    async def test_the_commit_subject_is_stored_as_searchable_text_linked_to_its_pr(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """This is the entity link recall refuses to guess: recorded here, where it is a
        fact rather than an inference from prose similarity."""
        patch_client(github._tool, {"/commits": COMMITS})
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="commits", window=WINDOW, graph=_Graph()
        )
        doc = result.documents[0]
        assert doc.metadata["entity_label"] == "PR"
        assert doc.metadata["entity_key"] == "913"
        assert "onboarding" in doc.text

    async def test_an_empty_repository_list_is_refused(
        self, github: GitHubSyncer, tenant: TenantContext
    ) -> None:
        """An allowlist, for the same reason PostHog's projects are: listing what the token
        can see would ingest a tenant's forks, dependencies and personal repositories."""
        ctx = ToolContext(tenant=tenant, credential="ghp_fake", credential_metadata={})
        with pytest.raises(InvalidParams, match="lists no repositories"):
            await github.sync_stream(
                None, tenant, ctx, stream="commits", window=WINDOW, graph=_Graph()
            )


class TestGitHubIssues:
    async def test_an_issue_becomes_a_ticket_with_its_reporter(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        patch_client(
            github._tool,
            {
                "/issues": [
                    {
                        "number": 921,
                        "title": "Cannot finish signup on iPhone",
                        "state": "open",
                        "created_at": "2026-07-15T07:12:00Z",
                        "user": {"login": "external-user-44"},
                        "labels": [{"name": "bug"}, {"name": "mobile"}],
                        "body": "The continue button never appears.",
                    }
                ]
            },
        )
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="issues", window=WINDOW, graph=_Graph()
        )

        tickets = [n for n in result.nodes if n.label is NodeLabel.SUPPORT_TICKET]
        assert [n.key for n in tickets] == ["acme/web#921"]
        assert tickets[0].props["labels"] == "bug,mobile"
        assert any(e.rel is RelType.RAISED for e in result.edges)
        # The body is searchable, so "did anyone report this" is answerable semantically.
        assert "continue button" in result.documents[0].text


class TestGitHubDeploymentsAndLinks:
    async def test_a_deployment_carries_its_state(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """A failed deploy explains a metric move completely differently from a successful
        one, so the state is stored rather than assumed."""
        patch_client(
            github._tool,
            {
                "/statuses": [{"state": "failure"}],
                "/deployments": [
                    {
                        "id": 1,
                        "sha": "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a",
                        "ref": "main",
                        "environment": "prod-web",
                        "created_at": "2026-07-14T11:04:00Z",
                    }
                ],
            },
        )
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="deployments", window=WINDOW, graph=_Graph()
        )
        deploy = result.nodes[0]
        assert deploy.label is NodeLabel.DEPLOY
        assert deploy.props["state"] == "failure"
        assert deploy.props["environment"] == "prod-web"

    async def test_links_join_a_deploy_to_its_pull_request_by_sha(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext
    ) -> None:
        """The edge that makes the graph worth having: metric -> deploy -> PR -> reviewer.
        Neither stream knows the other, so it is derived by reading both back."""
        sha = "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a"
        graph = _Graph(
            {
                NodeLabel.DEPLOY: [{"key": sha[:7], "sha": sha}],
                NodeLabel.PR: [{"key": "913", "sha": sha}],
            }
        )
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="links", window=WINDOW, graph=graph
        )

        assert len(result.edges) == 1
        edge = result.edges[0]
        assert (edge.rel, edge.from_key, edge.to_key) == (RelType.DEPLOYED_AS, "913", sha[:7])

    async def test_a_deploy_with_no_matching_pr_is_left_unlinked_and_reported(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext
    ) -> None:
        """The common case for a dependency bump or a CI change. Inventing a link would put
        an unrelated change into the causal chain for whatever the deploy is later blamed
        for — and silence would read as "no deploys happened"."""
        graph = _Graph({NodeLabel.DEPLOY: [{"key": "beef123", "sha": "beef123cafe"}]})
        result = await github.sync_stream(
            None, tenant, gh_ctx, stream="links", window=WINDOW, graph=graph
        )
        assert result.edges == []
        assert "matched no pull request" in (result.partial or "")


# ------------------------------------------------------------------------ posthog


@pytest.fixture
def posthog() -> PostHogSyncer:
    return PostHogSyncer()


@pytest.fixture
def ph_ctx(tenant: TenantContext) -> ToolContext:
    return ToolContext(
        tenant=tenant,
        credential="phx_fake",
        credential_metadata={"projects": "100001:saas,100002:oss"},
    )


def _definitions(*names: str) -> dict[str, Any]:
    return {
        "count": len(names),
        "results": [{"name": n, "last_seen_at": "2026-07-30T00:00:00Z"} for n in names],
    }


class TestPostHogEvents:
    async def test_a_trend_becomes_metric_points_segmented_by_project(
        self, posthog: PostHogSyncer, ph_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """The project is part of the segment, not the metric name: the same event exists in
        two products with different volumes, and merging them produces a baseline that
        describes neither."""
        patch_client(
            posthog._tool,
            {
                "/event_definitions/": _definitions("user signed up"),
                "/query/": {
                    "columns": ["bucket", "value"],
                    "results": [["2026-07-29T00:00:00", 42], ["2026-07-30T00:00:00", 51]],
                },
            },
            is_async_factory=True,
        )
        result = await posthog.sync_stream(
            None, tenant, ph_ctx, stream="events", window=WINDOW, graph=_Graph()
        )

        # Two projects are declared, so both are read.
        assert len(result.points) == 4
        assert {p.segment["project"] for p in result.points} == {"saas", "oss"}
        assert all(p.metric == "user signed up" for p in result.points)
        assert {p.value for p in result.points} == {42.0, 51.0}

    async def test_posthogs_own_events_are_excluded(
        self, posthog: PostHogSyncer, ph_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """A real run spent three of twelve slots per project on `$set`, `$identify` and
        `$groupidentify` — person-property bookkeeping, not metrics — crowding out real
        events under the cap."""
        patch_client(
            posthog._tool,
            {
                "/event_definitions/": _definitions("$set", "$identify", "user signed up"),
                "/query/": {"columns": ["bucket", "value"], "results": [["2026-07-30", 1]]},
            },
            is_async_factory=True,
        )
        result = await posthog.sync_stream(
            None, tenant, ph_ctx, stream="events", window=WINDOW, graph=_Graph()
        )
        assert {p.metric for p in result.points} == {"user signed up"}

    async def test_a_project_with_no_custom_events_is_a_note_not_a_gap(
        self, posthog: PostHogSyncer, ph_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """Staging and the CLI project both look like this. It is a real state, and calling
        it incomplete would tell a reader data is missing when none exists."""
        patch_client(
            posthog._tool,
            {"/event_definitions/": _definitions("$pageview")},
            is_async_factory=True,
        )
        result = await posthog.sync_stream(
            None, tenant, ph_ctx, stream="events", window=WINDOW, graph=_Graph()
        )
        assert result.points == []
        assert result.partial is None
        assert "no custom events" in (result.note or "")

    async def test_one_project_failing_does_not_lose_the_others(
        self, posthog: PostHogSyncer, ph_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """A tenant with three products should still get baselines for two of them."""
        import httpx

        calls = {"n": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            if "event_definitions" in str(request.url):
                return httpx.Response(200, json=_definitions("user signed up"))
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, json={"detail": "boom"})
            return httpx.Response(
                200, json={"columns": ["bucket", "value"], "results": [["2026-07-30", 7]]}
            )

        patch_client(posthog._tool, _handler, is_async_factory=True)
        result = await posthog.sync_stream(
            None, tenant, ph_ctx, stream="events", window=WINDOW, graph=_Graph()
        )

        assert result.points, "the healthy project's series must survive"
        assert "saas" in (result.partial or "")


class TestPostHogAnnotations:
    async def test_an_annotation_becomes_a_decision_distinguishing_person_from_deploy(
        self, posthog: PostHogSyncer, ph_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """One is testimony, the other a record of a ship. Conflating them would let a
        person's guess be cited as a deployment fact."""
        patch_client(
            posthog._tool,
            {
                "/annotations/": {
                    "count": 1,
                    "results": [
                        {
                            "content": "shipped onboarding rework",
                            "date_marker": "2026-07-14T11:04:00Z",
                            "creation_type": "GIT",
                        }
                    ],
                }
            },
            is_async_factory=True,
        )
        result = await posthog.sync_stream(
            None, tenant, ph_ctx, stream="annotations", window=WINDOW, graph=_Graph()
        )

        decisions = [n for n in result.nodes if n.label is NodeLabel.DECISION]
        assert decisions and decisions[0].props["source"] == "deployment"
        # And it is searchable, linked back to the node it describes.
        assert result.documents[0].metadata["entity_label"] == "Decision"

    async def test_no_annotations_is_a_note(
        self, posthog: PostHogSyncer, ph_ctx: ToolContext, tenant: TenantContext, patch_client: Any
    ) -> None:
        """Verified against all five real projects: they are empty."""
        patch_client(
            posthog._tool, {"/annotations/": {"count": 0, "results": []}}, is_async_factory=True
        )
        result = await posthog.sync_stream(
            None, tenant, ph_ctx, stream="annotations", window=WINDOW, graph=_Graph()
        )
        assert result.partial is None
        assert "no annotations" in (result.note or "")


class TestBothSyncersAgreeOnTheContract:
    @pytest.mark.parametrize("syncer_type", [GitHubSyncer, PostHogSyncer])
    def test_streams_are_declared_and_unique(self, syncer_type: type) -> None:
        syncer = syncer_type()
        assert syncer.streams
        # A duplicate would share a watermark with itself and advance it twice in one run.
        assert len(set(syncer.streams)) == len(syncer.streams), syncer.streams

    async def test_unknown_streams_are_refused(
        self, github: GitHubSyncer, gh_ctx: ToolContext, tenant: TenantContext
    ) -> None:
        """The runner only passes declared streams, but a typo in `streams` would otherwise
        sync nothing and report success."""
        with pytest.raises(ValueError, match="no stream"):
            await github.sync_stream(
                None, tenant, gh_ctx, stream="nope", window=WINDOW, graph=_Graph()
            )
