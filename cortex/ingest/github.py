"""GitHub ingest — the causal chain, made walkable.

This is the syncer that earns the graph. A GTM investigation's hardest question is "what
changed", and the answer is almost always a person merging a pull request that shipped in a
deployment. Live API calls can find any one of those; only a graph can walk from a metric
back through the deploy to the PR to the reviewer who warned about it.

Three streams, each with its own watermark:

  - `commits` — what landed. Not the same as what deployed, which is why both exist.
  - `deployments` — when it reached users, with the state, because a failed deploy is a
    different explanation from a successful one.
  - `issues` — reported breakage, which usually predates the metric noticing.

Pull requests come with the commits stream rather than as a fourth: a commit's subject
carries its PR number, and reading it there costs nothing where a separate pass would cost
another API round trip per window.

**Repositories come from credential metadata, not from discovery.** Listing everything the
token can see would ingest a tenant's dependencies, forks and personal repositories, which
is both noise and data nobody asked us to hold. `--meta repos=acme/web,acme/api`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import CredentialProvider
from cortex.ingest.base import StreamResult, Syncer
from cortex.ingest.state import Window
from cortex.memory.entities import Edge, Node, NodeLabel, RelType
from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import Document
from cortex.tenancy.context import TenantContext
from cortex.tools.base import InvalidParams, ToolContext
from cortex.tools.github import GitHubTool

#: A PR reference in a commit subject: "Rework onboarding modal (#913)".
_PR_IN_SUBJECT = re.compile(r"#(\d{1,7})\b")

#: Items per stream per window. A busy repository merges more than this in 90 days, and the
#: connector caps a page anyway; the cap is recorded as `partial` rather than hidden, so a
#: report can say the history is incomplete instead of implying it is whole.
_PER_STREAM = 100


class GitHubSyncer(Syncer):
    provider = CredentialProvider.GITHUB
    tool_name = "github"

    def __init__(self, tool: GitHubTool | None = None) -> None:
        self._tool = tool or GitHubTool()

    @property
    def streams(self) -> tuple[str, ...]:
        # `links` runs last, and depends on the three before it having landed: it joins
        # pull requests to deployments on the sha they share. Ordered rather than
        # concurrent for that reason -- a link pass that runs before its endpoints exist
        # finds nothing and reports success.
        return ("commits", "deployments", "issues", "links")

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
        del session
        repos = _repos(ctx)
        result = StreamResult()
        truncated: list[str] = []

        if stream == "links":
            # Not per-repository: the join is over what is already in the graph, and the
            # sha is unique across repositories anyway.
            return await self._link_deploys_to_prs(tenant, graph)

        for repo in repos:
            if stream == "commits":
                partial = await self._commits(ctx, repo, window, result)
            elif stream == "deployments":
                partial = await self._deployments(ctx, repo, window, result)
            elif stream == "issues":
                partial = await self._issues(ctx, repo, window, result)
            else:  # pragma: no cover - the runner only passes declared streams
                raise ValueError(f"github syncer has no stream {stream!r}")
            if partial:
                truncated.append(partial)

        if truncated:
            result.partial = "; ".join(truncated)
        return result

    # ------------------------------------------------------------------ streams

    async def _commits(
        self, ctx: ToolContext, repo: str, window: Window, out: StreamResult
    ) -> str | None:
        payload = (
            await self._tool.commits(
                ctx,
                repo=repo,
                since=window.since_date,
                until=window.until_date,
                limit=_PER_STREAM,
            )
        ).payload

        for commit in payload.get("commits", []):
            sha = str(commit.get("sha") or "")
            if not sha:
                continue
            short = sha[:7]
            subject = str(commit.get("subject") or "")

            # A commit is not a first-class label in the vocabulary; it is the thing that
            # *delivers* a PR, so it is recorded as the PR it references where one exists
            # and skipped where it does not. Inventing a `Commit` label was considered and
            # rejected: the closed enum exists so traversal cannot silently stop matching,
            # and a dependency-bump commit with no PR adds a node nothing ever walks to.
            match = _PR_IN_SUBJECT.search(subject)
            if not match:
                continue
            number = match.group(1)
            out.nodes.append(
                Node(
                    NodeLabel.PR,
                    number,
                    {
                        "title": subject[:300],
                        "sha": sha,
                        "short_sha": short,
                        "merged_at": commit.get("date"),
                        "url": commit.get("url"),
                        "repo": repo,
                    },
                )
            )
            author = str(commit.get("author") or "").strip()
            if author:
                out.nodes.append(Node(NodeLabel.PERSON, author, {"name": author}))
                out.edges.append(
                    Edge(RelType.AUTHORED, NodeLabel.PERSON, author, NodeLabel.PR, number)
                )
            # The subject as text, so "which change mentioned onboarding" is answerable
            # semantically and links back to the PR node by the metadata the recall path
            # reads. This is the entity link recall refuses to guess — recorded here,
            # where it is a fact rather than an inference.
            out.documents.append(
                Document(
                    kind="docs",
                    source_id=f"{repo}@{sha}",
                    text=f"{repo} commit {short}: {subject}",
                    source_ref=str(commit.get("url") or ""),
                    metadata={
                        "repo": repo,
                        "kind_detail": "commit",
                        "entity_label": NodeLabel.PR.value,
                        "entity_key": number,
                    },
                )
            )

        # The cap is reported, never hidden: a truncated history that reads as complete
        # would let a report say "nothing else shipped that week".
        if payload.get("count") == _PER_STREAM:
            return f"{repo} commits capped at {_PER_STREAM}"
        return None

    async def _deployments(
        self, ctx: ToolContext, repo: str, window: Window, out: StreamResult
    ) -> str | None:
        del window  # The connector returns most-recent-first; there is no since filter.
        payload = (await self._tool.deployment_history(ctx, repo=repo, limit=_PER_STREAM)).payload

        for deployment in payload.get("deployments", []):
            sha = str(deployment.get("sha") or "")
            if not sha:
                continue
            short = sha[:7]
            out.nodes.append(
                Node(
                    NodeLabel.DEPLOY,
                    short,
                    {
                        "sha": sha,
                        "environment": deployment.get("environment"),
                        "created_at": deployment.get("created_at"),
                        # Carried because a failed deploy explains a metric move
                        # completely differently from a successful one.
                        "state": deployment.get("state"),
                        "repo": repo,
                        "creator": deployment.get("creator"),
                    },
                )
            )

        # Reported for the same reason as the commits cap, and fixed after shipping it
        # inconsistently: the first real run hit exactly 100 deployments and said nothing,
        # while the commits stream would have disclosed the same truncation. A history
        # silently cut at 100 lets a report reason about "the deploys that week" from a
        # window that may not contain them.
        if payload.get("count") == _PER_STREAM:
            return f"{repo} deployments capped at {_PER_STREAM}"
        return None

    async def _issues(
        self, ctx: ToolContext, repo: str, window: Window, out: StreamResult
    ) -> str | None:
        payload = (
            await self._tool.issues(ctx, repo=repo, since=window.since_date, limit=_PER_STREAM)
        ).payload

        for issue in payload.get("issues", []):
            number = str(issue.get("number") or "")
            if not number:
                continue
            key = f"{repo}#{number}"
            out.nodes.append(
                Node(
                    NodeLabel.SUPPORT_TICKET,
                    key,
                    {
                        "title": str(issue.get("title") or "")[:300],
                        "state": issue.get("state"),
                        "created_at": issue.get("created_at"),
                        "labels": ",".join(issue.get("labels") or []),
                        "repo": repo,
                    },
                )
            )
            reporter = str(issue.get("author") or "").strip()
            if reporter:
                out.nodes.append(Node(NodeLabel.PERSON, reporter, {"name": reporter}))
                out.edges.append(
                    Edge(
                        RelType.RAISED,
                        NodeLabel.PERSON,
                        reporter,
                        NodeLabel.SUPPORT_TICKET,
                        key,
                    )
                )
            body = str(issue.get("body") or "")
            out.documents.append(
                Document(
                    kind="tickets",
                    source_id=key,
                    text=f"{issue.get('title')}\n\n{body}"[:4000],
                    source_ref=f"https://github.com/{repo}/issues/{number}",
                    metadata={
                        "repo": repo,
                        "entity_label": NodeLabel.SUPPORT_TICKET.value,
                        "entity_key": key,
                    },
                )
            )
        return None

    async def _link_deploys_to_prs(self, tenant: TenantContext, graph: GraphStore) -> StreamResult:
        """Join deployments to the pull requests they carried, on the shared sha.

        This edge is the reason the graph is worth having. Without it, "signups fell after
        a deploy" and "the onboarding rework caused it" are two separate observations a
        human has to connect; with it, an investigation can walk metric -> deploy -> PR ->
        the reviewer who warned about the change before it merged.

        Derived here rather than at write time because neither stream knows the other: a
        deployment record carries a sha and nothing about pull requests, and a commit
        carries a sha and nothing about whether it ever deployed. The edge exists only once
        both have landed, so it is computed by reading them back.

        A deploy with no matching PR is left unlinked. That is the common case for a
        dependency bump or a CI change, and inventing a link would put an unrelated change
        into the causal chain for whatever the deploy is later blamed for.
        """
        result = StreamResult()
        deploys = await graph.nodes_by_label(tenant, NodeLabel.DEPLOY, limit=1000)
        prs = await graph.nodes_by_label(tenant, NodeLabel.PR, limit=1000)

        by_sha: dict[str, str] = {}
        for pr in prs:
            sha = str(pr.get("sha") or "")
            key = str(pr.get("key") or "")
            if sha and key:
                by_sha[sha] = key

        for deploy in deploys:
            sha = str(deploy.get("sha") or "")
            deploy_key = str(deploy.get("key") or "")
            pr_key = by_sha.get(sha)
            if not (sha and deploy_key and pr_key):
                continue
            result.edges.append(
                Edge(
                    RelType.DEPLOYED_AS,
                    NodeLabel.PR,
                    pr_key,
                    NodeLabel.DEPLOY,
                    deploy_key,
                    {"sha": sha},
                )
            )

        # Reported so a graph that cannot be walked is visible as such. A tenant whose
        # deploys never match a PR has a broken chain, and silence would read as "no
        # deploys happened".
        if deploys and not result.edges:
            result.partial = (
                f"{len(deploys)} deployment(s) matched no pull request by sha; the "
                "deploy-to-PR chain is not walkable for them"
            )
        return result


def _repos(ctx: ToolContext) -> list[str]:
    """Repositories this tenant declared.

    An allowlist, for the same reason PostHog's projects are: listing what the token can
    see would ingest a tenant's forks, dependencies and personal repositories — noise, and
    data nobody asked us to hold.
    """
    raw = str(ctx.credential_metadata.get("repos", "")).strip()
    repos = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not repos:
        raise InvalidParams(
            "github ingest: this tenant's credential lists no repositories. Reconnect "
            "with --meta repos=owner/name,owner/other"
        )
    return repos


def latest_timestamp(values: list[Any], *, fallback: datetime) -> datetime:
    """The newest parseable timestamp in a batch, or the fallback.

    Used to set a watermark from the data rather than from the clock: if the newest commit
    in the window is from Tuesday, the cursor belongs on Tuesday, not on now. Setting it to
    `now` would skip anything that arrives with an earlier timestamp after the run — which
    is exactly what late-arriving data does.
    """
    best: datetime | None = None
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        if best is None or parsed > best:
            best = parsed
    return best or fallback
