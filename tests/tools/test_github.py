"""GitHub connector.

The capability that matters most is deployment_history: it supplies the causal
candidate an investigation tests its hypotheses against, and it is the one that
makes an extra request per deployment to establish whether the deploy actually
succeeded.
"""

from __future__ import annotations

import re

import httpx
import pytest

from cortex.tools.base import InvalidParams, RateLimited, ToolContext, UpstreamError
from cortex.tools.github import REPO_PATTERN, GitHubTool
from cortex.tools.http import AuthRejected

PRS = [
    {
        "number": 913,
        "title": "Rework onboarding modal",
        "user": {"login": "a-developer"},
        "state": "closed",
        "merged_at": "2026-07-20T10:00:00Z",
        "created_at": "2026-07-18T09:00:00Z",
        "html_url": "https://github.com/acme/web/pull/913",
        "labels": [{"name": "growth"}, {"name": "onboarding"}],
    }
]

DEPLOYMENTS = [
    {
        "id": 55501,
        "sha": "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a",
        "ref": "main",
        "environment": "production",
        "created_at": "2026-07-20T11:04:00Z",
        "creator": {"login": "a-developer"},
    }
]

STATUSES = [{"state": "success"}]


@pytest.fixture
def tool() -> GitHubTool:
    return GitHubTool()


@pytest.fixture
def ctx(tenant: object) -> ToolContext:
    return ToolContext(tenant=tenant, credential="ghp_fake_token")  # type: ignore[arg-type]


class TestRecentPrs:
    async def test_parses_pull_requests(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/pulls": PRS})
        result = await tool.recent_prs(ctx, repo="acme/web", limit=10)

        assert result.payload["count"] == 1
        pr = result.payload["pull_requests"][0]
        assert pr["number"] == 913
        assert pr["author"] == "a-developer"
        assert pr["labels"] == ["growth", "onboarding"]
        assert result.source_ref

    async def test_forwards_limit_and_sort(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The limit must reach GitHub, or a bounded capability silently isn't."""
        transport = patch_client(tool, {"/pulls": PRS})
        await tool.recent_prs(ctx, repo="acme/web", limit=7)

        params = transport.requests[0].url.params
        assert params["per_page"] == "7"
        assert params["sort"] == "updated"

    async def test_credential_never_appears_in_the_payload(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/pulls": PRS})
        result = await tool.recent_prs(ctx, repo="acme/web")
        assert "ghp_fake_token" not in str(result.payload)

    async def test_empty_result_is_not_an_error(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """No PRs is a legitimate observation, and must be citable as one."""
        patch_client(tool, {"/pulls": []})
        result = await tool.recent_prs(ctx, repo="acme/web")
        assert result.payload["count"] == 0
        assert result.payload["pull_requests"] == []


class TestDeploymentHistory:
    async def test_attaches_the_deployment_state(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """A failed deploy is a completely different explanation from a successful
        one, so state is fetched rather than assumed."""
        patch_client(tool, {"/statuses": STATUSES, "/deployments": DEPLOYMENTS})
        result = await tool.deployment_history(ctx, repo="acme/web", environment="production")

        deployment = result.payload["deployments"][0]
        assert deployment["sha"].startswith("91c3e")
        assert deployment["state"] == "success"
        assert deployment["environment"] == "production"

    async def test_missing_statuses_leave_state_unknown(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """State must be None, never guessed as success."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "/statuses" in str(request.url):
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=DEPLOYMENTS)

        patch_client(tool, handler)
        result = await tool.deployment_history(ctx, repo="acme/web")
        assert result.payload["deployments"][0]["state"] is None

    async def test_environment_filter_is_forwarded(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/statuses": STATUSES, "/deployments": DEPLOYMENTS})
        await tool.deployment_history(ctx, repo="acme/web", environment="production")
        assert transport.requests[0].url.params["environment"] == "production"

    async def test_environment_omitted_when_not_requested(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/statuses": STATUSES, "/deployments": DEPLOYMENTS})
        await tool.deployment_history(ctx, repo="acme/web")
        assert "environment" not in transport.requests[0].url.params


class TestCommitDiff:
    COMMIT = {
        "sha": "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a",
        "commit": {
            "message": "Rework onboarding modal",
            "author": {"name": "A Developer", "date": "2026-07-20T10:00:00Z"},
        },
        "stats": {"additions": 120, "deletions": 45},
        "files": [
            {
                "filename": "web/onboarding/Modal.tsx",
                "status": "modified",
                "additions": 100,
                "deletions": 40,
                "patch": "@@ -1 +1 @@\n-secret\n+other",
            }
        ],
    }

    async def test_summarises_the_commit(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/commits/": self.COMMIT})
        result = await tool.commit_diff(ctx, repo="acme/web", sha="91c3e4a")

        assert result.payload["files_changed"] == 1
        assert result.payload["additions"] == 120
        assert result.payload["files"][0]["filename"] == "web/onboarding/Modal.tsx"

    async def test_patch_bodies_are_excluded(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Patches are large and frequently contain secrets; the file list is what a
        GTM analyst actually needs."""
        patch_client(tool, {"/commits/": self.COMMIT})
        result = await tool.commit_diff(ctx, repo="acme/web", sha="91c3e4a")
        assert "patch" not in result.payload["files"][0]
        assert "-secret" not in str(result.payload)


class TestReleaseSummary:
    async def test_truncates_long_notes(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Release notes can be enormous; the loop only needs the gist."""
        patch_client(
            tool,
            {
                "/releases": [
                    {
                        "tag_name": "v2.4.0",
                        "name": "July release",
                        "published_at": "2026-07-20T12:00:00Z",
                        "body": "x" * 5000,
                        "draft": False,
                        "prerelease": False,
                    }
                ]
            },
        )
        result = await tool.release_summary(ctx, repo="acme/web")
        notes = result.payload["releases"][0]["notes"]
        assert len(notes) < 5000
        assert notes.endswith("…")


class TestFindFeature:
    async def test_parses_search_results(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/search/issues": {
                    "total_count": 3,
                    "items": [
                        {
                            "number": 913,
                            "title": "Rework onboarding modal",
                            "state": "closed",
                            "closed_at": "2026-07-20T10:00:00Z",
                            "html_url": "https://github.com/acme/web/pull/913",
                            "user": {"login": "a-developer"},
                        }
                    ],
                }
            },
        )
        result = await tool.find_feature(ctx, repo="acme/web", query="onboarding modal")
        assert result.payload["total"] == 3
        assert result.payload["count"] == 1

    async def test_query_is_scoped_to_the_repo(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Without repo scoping a search would return other organisations' code."""
        transport = patch_client(tool, {"/search/issues": {"total_count": 0, "items": []}})
        await tool.find_feature(ctx, repo="acme/web", query="onboarding")
        assert "repo:acme/web" in transport.requests[0].url.params["q"]


class TestFailureMapping:
    """The loop reacts differently to each of these: retry a 429, give up on a 401."""

    async def test_rate_limit_becomes_rate_limited(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {"/pulls": httpx.Response(429, headers={"Retry-After": "42"}, json={})},
        )
        with pytest.raises(RateLimited) as exc:
            await tool.recent_prs(ctx, repo="acme/web")
        assert exc.value.retry_after_seconds == 42.0

    async def test_bad_credential_becomes_auth_rejected(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/pulls": httpx.Response(401, json={"message": "Bad credentials"})})
        with pytest.raises(AuthRejected, match="reconnect"):
            await tool.recent_prs(ctx, repo="acme/web")

    async def test_server_error_becomes_upstream_error(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/pulls": httpx.Response(503, text="unavailable")})
        with pytest.raises(UpstreamError):
            await tool.recent_prs(ctx, repo="acme/web")

    async def test_error_messages_do_not_include_query_strings(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Query strings routinely carry tokens and customer identifiers."""
        patch_client(tool, {"/search/issues": httpx.Response(500, text="boom")})
        with pytest.raises(UpstreamError) as exc:
            await tool.find_feature(ctx, repo="acme/web", query="secret-project")
        assert "secret-project" not in str(exc.value)

    async def test_non_json_body_becomes_upstream_error(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/pulls": httpx.Response(200, text="<html>proxy error</html>")})
        with pytest.raises(UpstreamError, match="non-JSON"):
            await tool.recent_prs(ctx, repo="acme/web")


class TestSchemas:
    def test_schema_pattern_matches_the_code_check(self, tool: GitHubTool) -> None:
        """One pattern, used by both layers, so they cannot drift apart."""
        schema = tool.capability("recent_prs").params_schema
        assert schema["properties"]["repo"]["pattern"] == REPO_PATTERN

    @pytest.mark.parametrize("repo", ["acme/web", "acme/acme", "a-b/c_d.e", "x1/y2", "a/b"])
    def test_repo_pattern_accepts_real_repositories(self, repo: str) -> None:
        # Owners are alphanumeric plus hyphen; names may also contain _ and .
        assert re.match(REPO_PATTERN, repo), repo

    @pytest.mark.parametrize(
        "repo",
        [
            "acme",
            "acme/web/extra",
            "acme web",
            "",
            "/web",
            "acme/",
            # Path traversal: repo is interpolated into the API URL path.
            "../etc",
            "acme/../../users",
            "../..",
            "acme/./web",
            "-acme/web",
            "acme_corp/web",  # underscore is not legal in an owner
            "acme/web?x=1",
            "acme/web#frag",
        ],
    )
    def test_repo_pattern_rejects_malformed_and_hostile_names(self, repo: str) -> None:
        assert not re.match(REPO_PATTERN, repo), repo

    @pytest.mark.parametrize("repo", ["../etc", "acme/../../users", "acme_corp/web"])
    async def test_capabilities_reject_hostile_repo_in_code(
        self, tool: GitHubTool, ctx: ToolContext, repo: str
    ) -> None:
        """Defense in depth: a URL-path value must not rely on schema validation
        alone, since capabilities are directly callable."""
        with pytest.raises(InvalidParams, match="invalid repository"):
            await tool.recent_prs(ctx, repo=repo)
        with pytest.raises(InvalidParams, match="invalid repository"):
            await tool.commit_diff(ctx, repo=repo, sha="91c3e4a")

    def test_every_repo_capability_validates_in_code(self, tool: GitHubTool) -> None:
        """Guards against a new capability being added without the check."""
        import inspect

        checked = 0
        for name in tool.capability_names:
            handler = tool.capability(name).handler
            if "repo" not in inspect.signature(handler).parameters:
                continue
            assert "_check_repo(repo)" in inspect.getsource(handler), (
                f"{name} does not validate repo"
            )
            checked += 1
        # Derived rather than hardcoded: a magic number here means adding a capability
        # makes this test fail for the wrong reason, and the temptation is to bump the
        # number rather than check the new handler validates its input.
        expected = sum(
            1
            for name in tool.capability_names
            if "repo" in inspect.signature(tool.capability(name).handler).parameters
        )
        assert checked == expected and checked >= 5

    def test_sha_pattern_requires_hex(self, tool: GitHubTool) -> None:
        import re

        pattern = re.compile(
            tool.capability("commit_diff").params_schema["properties"]["sha"]["pattern"]
        )
        assert pattern.match("91c3e4a")
        assert not pattern.match("not-a-sha")
        assert not pattern.match("91c3")


COMMITS = [
    {
        "sha": "91c3e4a7bd2f1e0c9a8b7d6e5f4a3b2c1d0e9f8a",
        "html_url": "https://github.com/acme/web/commit/91c3e4a",
        "commit": {
            "author": {"name": "Dana Whitfield", "date": "2026-07-14T10:02:00Z"},
            "message": (
                "Rework mobile onboarding modal (#913)\n\n"
                "Longer body that should not\nappear in the subject."
            ),
        },
    }
]

ISSUES = [
    {
        "number": 921,
        "title": "Cannot finish signup on iPhone",
        "state": "open",
        "created_at": "2026-07-15T07:12:00Z",
        "user": {"login": "external-user-44"},
        "labels": [{"name": "bug"}, {"name": "mobile"}],
        "comments": 4,
        "body": "The continue button never appears.",
    },
    # GitHub returns pull requests from the issues endpoint. Present here on purpose.
    {
        "number": 913,
        "title": "Rework mobile onboarding modal",
        "state": "closed",
        "pull_request": {"url": "https://api.github.com/repos/acme/web/pulls/913"},
        "user": {"login": "dwhitfield"},
        "labels": [],
    },
]


class TestCommits:
    """What actually landed, which is not the same as what was deployed."""

    async def test_parses_commits_and_keeps_only_the_subject(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/commits": COMMITS})
        result = await tool.commits(ctx, repo="acme/web", since="2026-07-12")

        assert result.payload["count"] == 1
        commit = result.payload["commits"][0]
        assert commit["short_sha"] == "91c3e4a"
        assert commit["author"] == "Dana Whitfield"
        # First line only: a commit body can run to a page, and the subject is what
        # identifies the change.
        assert commit["subject"] == "Rework mobile onboarding modal (#913)"
        assert "Longer body" not in commit["subject"]

    async def test_a_top_level_array_is_not_read_as_empty(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """The regression that matters most in this file.

        `request_json` wraps a top-level JSON array as `{"data": [...]}`, and GitHub's
        commits endpoint returns a bare array. Treating the response as a list yields
        nothing — and "no commits shipped that day" is the most misleading answer this
        connector can give, because it reads as a real observation and silently removes
        the true cause from consideration.
        """
        patch_client(tool, {"/commits": COMMITS})
        result = await tool.commits(ctx, repo="acme/web", since="2026-07-12")
        assert result.payload["count"] == 1, "a bare array must not read as no commits"

    async def test_day_boundaries_are_explicit(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """ "Since a date" that silently excludes that date's own commits is the kind of
        off-by-one that moves a cause outside the window."""
        transport = patch_client(tool, {"/commits": COMMITS})
        await tool.commits(ctx, repo="acme/web", since="2026-07-12", until="2026-07-15")

        params = transport.requests[0].url.params
        assert params["since"] == "2026-07-12T00:00:00Z"
        assert params["until"] == "2026-07-15T23:59:59Z"

    async def test_path_filter_is_forwarded(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/commits": COMMITS})
        await tool.commits(ctx, repo="acme/web", since="2026-07-12", path="src/onboarding")
        assert transport.requests[0].url.params["path"] == "src/onboarding"

    async def test_no_commits_is_a_real_observation(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(tool, {"/commits": []})
        result = await tool.commits(ctx, repo="acme/web", since="2026-07-12")
        assert result.payload["count"] == 0
        assert result.payload["since"] == "2026-07-12"


class TestPullRequestActivity:
    """The human record around a change: what a metric cannot contain."""

    async def test_merges_the_pr_its_reviews_and_its_comments(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/pulls/913/reviews": [
                    {
                        "user": {"login": "sbeck"},
                        "state": "CHANGES_REQUESTED",
                        "submitted_at": "2026-07-13T16:22:00Z",
                        "body": "The continue button sits below the fold on 375px.",
                    }
                ],
                "/issues/913/comments": [
                    {
                        "user": {"login": "sbeck"},
                        "created_at": "2026-07-14T09:58:00Z",
                        "body": "Merging without the footer fix, noted.",
                    }
                ],
                "/pulls/913": {
                    "title": "Rework mobile onboarding modal",
                    "state": "closed",
                    "merged_at": "2026-07-14T10:00:00Z",
                    "user": {"login": "dwhitfield"},
                    "body": "Replaces the three-step modal.",
                },
            },
        )
        result = await tool.pull_request_activity(ctx, repo="acme/web", number=913)

        payload = result.payload
        assert payload["author"] == "dwhitfield"
        # Three endpoints because GitHub splits them; reading only one misses either the
        # verdict or the conversation.
        assert payload["reviews"][0]["verdict"] == "CHANGES_REQUESTED"
        assert "375px" in payload["reviews"][0]["body"]
        assert payload["comments"][0]["author"] == "sbeck"

    async def test_a_pr_with_no_discussion_is_not_an_error(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        patch_client(
            tool,
            {
                "/pulls/913/reviews": [],
                "/issues/913/comments": [],
                "/pulls/913": {"title": "Typo fix", "state": "closed", "user": {"login": "x"}},
            },
        )
        result = await tool.pull_request_activity(ctx, repo="acme/web", number=913)
        assert result.payload["reviews"] == []
        assert result.payload["comments"] == []


class TestIssues:
    async def test_pull_requests_are_excluded(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """GitHub returns PRs from the issues endpoint. "Twelve issues opened that day"
        meaning nine PRs would misread badly."""
        patch_client(tool, {"/issues": ISSUES})
        result = await tool.issues(ctx, repo="acme/web", since="2026-07-14")

        assert result.payload["count"] == 1
        assert result.payload["pull_requests_excluded"] is True
        assert [i["number"] for i in result.payload["issues"]] == [921]

    async def test_labels_are_flattened_and_forwarded(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        transport = patch_client(tool, {"/issues": ISSUES})
        result = await tool.issues(ctx, repo="acme/web", since="2026-07-14", labels="bug")

        assert result.payload["issues"][0]["labels"] == ["bug", "mobile"]
        assert transport.requests[0].url.params["labels"] == "bug"

    async def test_no_issues_is_a_real_observation(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """ "Nobody complained" is something an analyst should be able to cite, not a
        silence it has to interpret."""
        patch_client(tool, {"/issues": []})
        result = await tool.issues(ctx, repo="acme/web", since="2026-07-14")
        assert result.payload["count"] == 0
        assert result.payload["issues"] == []


class TestDeploymentStatusConcurrency:
    """Establishing whether 100 deploys succeeded is 100 requests, because GitHub has no
    bulk endpoint for it. Serially that exceeded the deadline twice on a real repository,
    and a live investigation had to infer deploy success from a bot comment instead."""

    async def test_states_are_fetched_concurrently_but_bounded(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Concurrent, so 100 deploys finish inside the deadline; bounded, because GitHub's
        secondary rate limit punishes bursts and a tool that trips it fails every later call
        in the same investigation.

        The handler is async and yields, so the peak measured here is real concurrency
        rather than an artefact of a synchronous stub returning instantly.
        """
        import asyncio

        from cortex.tools.github import _STATUS_CONCURRENCY

        in_flight = 0
        peak = 0
        deployments = [
            {"id": n, "sha": f"{n:040x}", "environment": "prod-web", "ref": "main"}
            for n in range(40)
        ]

        async def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            if "/statuses" not in str(request.url):
                return httpx.Response(200, json=deployments)
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                # Yields control, so other requests actually overlap this one.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return httpx.Response(200, json=[{"state": "success"}])
            finally:
                in_flight -= 1

        patch_client(tool, _handler)
        result = await tool.deployment_history(ctx, repo="acme/web", limit=40)

        assert len(result.payload["deployments"]) == 40
        assert all(d["state"] == "success" for d in result.payload["deployments"])
        assert peak > 1, "the whole point is that these do not run one at a time"
        assert peak <= _STATUS_CONCURRENCY, peak

    async def test_a_state_is_matched_to_its_own_deployment(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """Concurrency introduces the chance of mis-zipping a state onto the wrong
        deployment, which would report a failed deploy as successful — the worst possible
        wrong answer from this capability. The mapping is keyed by id for that reason."""
        deployments = [
            {"id": 1, "sha": "a" * 40, "environment": "prod-web"},
            {"id": 2, "sha": "b" * 40, "environment": "prod-web"},
        ]
        states = {"1": "failure", "2": "success"}

        def _handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/statuses" in url:
                identifier = url.split("/deployments/")[1].split("/")[0]
                return httpx.Response(200, json=[{"state": states[identifier]}])
            return httpx.Response(200, json=deployments)

        patch_client(tool, _handler)
        result = await tool.deployment_history(ctx, repo="acme/web")

        by_sha = {d["sha"][0]: d["state"] for d in result.payload["deployments"]}
        assert by_sha == {"a": "failure", "b": "success"}

    async def test_one_unreadable_status_does_not_lose_the_others(
        self, tool: GitHubTool, ctx: ToolContext, patch_client: object
    ) -> None:
        """`None` already means "state unknown", and the payload distinguishes it from
        "succeeded"."""
        deployments = [
            {"id": 1, "sha": "a" * 40, "environment": "prod-web"},
            {"id": 2, "sha": "b" * 40, "environment": "prod-web"},
        ]

        def _handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/deployments/1/statuses" in url:
                return httpx.Response(500, json={"message": "server error"})
            if "/statuses" in url:
                return httpx.Response(200, json=[{"state": "success"}])
            return httpx.Response(200, json=deployments)

        patch_client(tool, _handler)
        result = await tool.deployment_history(ctx, repo="acme/web")

        states = {d["sha"][0]: d["state"] for d in result.payload["deployments"]}
        assert states == {"a": None, "b": "success"}
