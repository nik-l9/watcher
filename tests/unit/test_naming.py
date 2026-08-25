"""Physical-name resolution.

These names are the isolation boundary — a slug that escapes its namespace is a
cross-tenant leak — so the validator gets adversarial input, not just happy paths.
"""

from __future__ import annotations

import uuid

import pytest

from cortex.memory.naming import (
    COLLECTION_PREFIX,
    GRAPH_PREFIX,
    InvalidTenantSlug,
    collection_name,
    graph_name_for_new_tenant,
    validate_slug,
)


class TestValidateSlug:
    @pytest.mark.parametrize(
        "slug",
        ["abc", "acme-corp", "a1b2", "acme", "x" * 63, "a-b-c-d", "123", "a0-9z"],
    )
    def test_accepts_valid(self, slug: str) -> None:
        assert validate_slug(slug) == slug

    @pytest.mark.parametrize(
        ("slug", "why"),
        [
            ("", "empty"),
            ("ab", "too short"),
            ("x" * 64, "too long"),
            ("Acme", "uppercase"),
            ("acme_corp", "underscore"),
            ("-acme", "leading hyphen"),
            ("acme-", "trailing hyphen"),
            ("acme corp", "space"),
            ("acme:corp", "colon separates Redis keyspaces"),
            ("acme*", "glob would match sibling graphs"),
            ("../etc", "path traversal"),
            ("acme/../other", "path traversal"),
            ("acme\ncorp", "newline"),
            ("acme%20", "url encoding"),
            ("acme.corp", "dot"),
            ("a" * 3 + "\x00", "null byte"),
        ],
    )
    def test_rejects_invalid(self, slug: str, why: str) -> None:
        with pytest.raises(InvalidTenantSlug):
            validate_slug(slug)

    @pytest.mark.parametrize("value", [None, 123, b"acme", ["acme"], {"slug": "acme"}])
    def test_rejects_non_strings(self, value: object) -> None:
        with pytest.raises(InvalidTenantSlug):
            validate_slug(value)  # type: ignore[arg-type]


class TestGraphName:
    def test_shape(self) -> None:
        tid = uuid.uuid4()
        name = graph_name_for_new_tenant("acme-corp", tid)
        assert name == f"{GRAPH_PREFIX}acme-corp_{tid.hex}"

    def test_is_deterministic_for_a_given_tenant(self) -> None:
        tid = uuid.uuid4()
        assert graph_name_for_new_tenant("acme-corp", tid) == graph_name_for_new_tenant(
            "acme-corp", tid
        )

    def test_recycled_slug_never_reuses_a_graph(self) -> None:
        """Offboarding then re-onboarding the same slug must not inherit old data."""
        a = graph_name_for_new_tenant("acme-corp", uuid.uuid4())
        b = graph_name_for_new_tenant("acme-corp", uuid.uuid4())
        assert a != b

    def test_different_tenants_never_collide(self) -> None:
        names = {graph_name_for_new_tenant("acme-corp", uuid.uuid4()) for _ in range(500)}
        assert len(names) == 500

    def test_validates_the_slug(self) -> None:
        with pytest.raises(InvalidTenantSlug):
            graph_name_for_new_tenant("../escape", uuid.uuid4())


class TestCollectionName:
    def test_derives_from_graph_name(self) -> None:
        tid = uuid.uuid4()
        graph = graph_name_for_new_tenant("acme-corp", tid)
        coll = collection_name(graph, "slack_threads")
        assert coll == f"{COLLECTION_PREFIX}acme-corp_{tid.hex}_slack_threads"

    def test_distinct_tenants_get_distinct_collections(self) -> None:
        g1 = graph_name_for_new_tenant("acme-corp", uuid.uuid4())
        g2 = graph_name_for_new_tenant("acme-corp", uuid.uuid4())
        assert collection_name(g1, "docs") != collection_name(g2, "docs")

    def test_distinct_kinds_get_distinct_collections(self) -> None:
        g = graph_name_for_new_tenant("acme-corp", uuid.uuid4())
        assert collection_name(g, "docs") != collection_name(g, "tickets")

    def test_rejects_foreign_graph_name(self) -> None:
        """A name not minted by this module must not be accepted."""
        with pytest.raises(InvalidTenantSlug):
            collection_name("some_other_graph", "docs")

    @pytest.mark.parametrize(
        "kind", ["", "Docs", "docs-threads", "docs threads", "../x", "x" * 33, "docs*"]
    )
    def test_rejects_invalid_kind(self, kind: str) -> None:
        g = graph_name_for_new_tenant("acme-corp", uuid.uuid4())
        with pytest.raises(InvalidTenantSlug):
            collection_name(g, kind)
