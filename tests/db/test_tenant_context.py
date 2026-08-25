"""Tenant context resolution.

This is the front door: everything downstream trusts the context it is handed, so
the interesting cases are the ones where a caller presents credentials that are
valid but belong somewhere else.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Tenant, User
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.tenancy.context import (
    TenantContext,
    TenantNotFound,
    UserNotInTenant,
    load_tenant_context,
    load_tenant_context_by_id,
)


async def _tenant(session: AsyncSession, slug: str, *, active: bool = True) -> Tenant:
    tid = uuid.uuid4()
    t = Tenant(
        id=tid,
        slug=slug,
        name=slug.title(),
        graph_name=graph_name_for_new_tenant(slug, tid),
        is_active=active,
    )
    session.add(t)
    await session.flush()
    return t


async def _user(
    session: AsyncSession, tenant: Tenant, external_id: str, role: str = "member"
) -> User:
    u = User(
        tenant_id=tenant.id,
        external_id=external_id,
        email=f"{external_id}@example.com",
        role=role,
    )
    session.add(u)
    await session.flush()
    return u


class TestResolution:
    async def test_resolves_tenant_without_a_user(self, session: AsyncSession) -> None:
        t = await _tenant(session, "acme-corp")
        ctx = await load_tenant_context(session, tenant_slug="acme-corp")
        assert ctx.tenant_id == t.id
        assert ctx.graph_name == t.graph_name
        assert ctx.user_id is None
        assert ctx.role == "member"

    async def test_resolves_tenant_with_a_user(self, session: AsyncSession) -> None:
        t = await _tenant(session, "acme-corp")
        u = await _user(session, t, "clerk_admin", role="admin")
        ctx = await load_tenant_context(
            session, tenant_slug="acme-corp", user_external_id="clerk_admin"
        )
        assert ctx.user_id == u.id
        assert ctx.role == "admin"
        assert ctx.is_admin is True

    async def test_graph_name_comes_from_the_row(self, session: AsyncSession) -> None:
        """The context must not re-derive the name — a recycled slug would then
        resolve to a different graph than the one holding the tenant's data."""
        t = await _tenant(session, "acme-corp")
        ctx = await load_tenant_context(session, tenant_slug="acme-corp")
        assert ctx.graph_name == t.graph_name


class TestRejections:
    """The shared module raises domain errors, not HTTP ones — a Celery task has
    no notion of a 403. The gateway translates them at its edge."""

    async def test_unknown_tenant(self, session: AsyncSession) -> None:
        with pytest.raises(TenantNotFound):
            await load_tenant_context(session, tenant_slug="does-not-exist")

    async def test_inactive_tenant(self, session: AsyncSession) -> None:
        """A suspended tenant must not be servable."""
        await _tenant(session, "suspended-corp", active=False)
        with pytest.raises(TenantNotFound):
            await load_tenant_context(session, tenant_slug="suspended-corp")

    async def test_user_from_another_tenant_is_refused(self, session: AsyncSession) -> None:
        """The core cross-tenant case: a real, valid user of tenant A presenting
        their identity against tenant B must be rejected, not silently admitted
        as an anonymous member of B."""
        a = await _tenant(session, "tenant-one")
        await _tenant(session, "tenant-two")
        await _user(session, a, "clerk_a", role="admin")

        with pytest.raises(UserNotInTenant):
            await load_tenant_context(session, tenant_slug="tenant-two", user_external_id="clerk_a")

    async def test_unknown_user_is_refused(self, session: AsyncSession) -> None:
        await _tenant(session, "acme-corp")
        with pytest.raises(UserNotInTenant):
            await load_tenant_context(
                session, tenant_slug="acme-corp", user_external_id="clerk_ghost"
            )


class TestResolutionById:
    """Workers receive tenant_id over the queue, never a graph name."""

    async def test_resolves_by_id(self, session: AsyncSession) -> None:
        t = await _tenant(session, "acme-corp")
        ctx = await load_tenant_context_by_id(session, t.id)
        assert ctx.tenant_id == t.id
        assert ctx.graph_name == t.graph_name

    async def test_unknown_id_is_refused(self, session: AsyncSession) -> None:
        with pytest.raises(TenantNotFound):
            await load_tenant_context_by_id(session, uuid.uuid4())

    async def test_inactive_tenant_is_refused(self, session: AsyncSession) -> None:
        """A queued job for a suspended tenant must not execute."""
        t = await _tenant(session, "suspended-corp", active=False)
        with pytest.raises(TenantNotFound):
            await load_tenant_context_by_id(session, t.id)

    async def test_worker_context_is_not_admin(self, session: AsyncSession) -> None:
        """A background job must not inherit admin privileges by default."""
        t = await _tenant(session, "acme-corp")
        ctx = await load_tenant_context_by_id(session, t.id)
        assert ctx.is_admin is False

    async def test_role_does_not_leak_across_tenants(self, session: AsyncSession) -> None:
        """Admin in tenant A must not imply admin anywhere else."""
        a = await _tenant(session, "tenant-one")
        b = await _tenant(session, "tenant-two")
        await _user(session, a, "clerk_shared", role="admin")
        await _user(session, b, "clerk_other", role="member")

        ctx = await load_tenant_context(
            session, tenant_slug="tenant-two", user_external_id="clerk_other"
        )
        assert ctx.is_admin is False


class TestContextImmutability:
    def test_context_is_frozen(self) -> None:
        """An ambient, mutable tenant is how cross-tenant leaks happen; the
        context must not be reassignable mid-request."""
        ctx = TenantContext(
            tenant_id=uuid.uuid4(), tenant_slug="acme-corp", graph_name="cortex_g_x_y"
        )
        with pytest.raises((AttributeError, TypeError)):
            ctx.graph_name = "cortex_g_someone_else"  # type: ignore[misc]

    def test_is_admin_only_for_admin_role(self) -> None:
        base = {"tenant_id": uuid.uuid4(), "tenant_slug": "acme-corp", "graph_name": "g"}
        assert TenantContext(**base, role="admin").is_admin is True
        for role in ("member", "viewer", "", "Admin", "superuser"):
            assert TenantContext(**base, role=role).is_admin is False, role
