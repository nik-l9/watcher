"""Tenant creation and teardown.

Creating a tenant is the only moment a physical graph name is assigned. Doing it
here — once, at creation, persisted on the row — is what allows every later code
path to read the name from the tenant context instead of deriving it.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Tenant
from cortex.memory.graph_store import GraphStore
from cortex.memory.naming import graph_name_for_new_tenant, validate_slug
from cortex.memory.vector_store import VectorStore
from cortex.tenancy.context import TenantContext


class TenantAlreadyExists(Exception):
    pass


async def create_tenant(
    session: AsyncSession,
    graph: GraphStore,
    *,
    slug: str,
    name: str,
    vectors: VectorStore | None = None,
) -> TenantContext:
    validate_slug(slug)

    existing = (
        await session.execute(select(Tenant.id).where(Tenant.slug == slug))
    ).scalar_one_or_none()
    if existing is not None:
        raise TenantAlreadyExists(slug)

    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        slug=slug,
        name=name,
        graph_name=graph_name_for_new_tenant(slug, tenant_id),
    )
    session.add(tenant)
    await session.flush()

    ctx = TenantContext(
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        graph_name=tenant.graph_name,
        role="admin",
    )
    # Provision after the row exists so an orphaned graph is impossible; a failed
    # provision rolls the transaction back with it.
    await graph.provision(ctx)
    # Semantic memory alongside the graph. Optional so a caller with no vector store
    # configured still creates tenants, but provisioned here when there is one: a tenant
    # whose collections are created lazily on first write has a window in which recall
    # returns nothing, and "no memory yet" and "the collection does not exist" are
    # indistinguishable from the caller's side.
    if vectors is not None:
        await vectors.provision(ctx)
    return ctx


async def delete_tenant(
    session: AsyncSession,
    graph: GraphStore,
    ctx: TenantContext,
    vectors: VectorStore | None = None,
) -> None:
    """Offboard a tenant: drop its stores, then its rows.

    Stores first — if the row disappeared first, the graph and collection names would be
    unrecoverable and the data would be orphaned rather than deleted. That is the
    difference between offboarding a customer and losing track of their data.
    """
    await graph.drop(ctx)
    if vectors is not None:
        await vectors.drop(ctx)
    tenant = await session.get(Tenant, ctx.tenant_id)
    if tenant is not None:
        await session.delete(tenant)
