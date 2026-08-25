"""Tenant context.

Every data access path in Cortex takes a TenantContext. It is deliberately not a
mutable global: an ambient tenant that can be forgotten or overwritten mid-request
is how cross-tenant leaks happen.

The context carries the resolved physical graph name so that no caller ever
constructs one. See cortex.memory.naming for the single resolver.

This module is shared by the gateway and both workers, so it raises its own
exceptions rather than HTTP ones — a Celery task has no notion of a 403. The
gateway translates them at its edge.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Tenant, User


class TenantNotFound(Exception):
    """Unknown or deactivated tenant."""


class UserNotInTenant(Exception):
    """A user that exists but does not belong to the requested tenant."""


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id: uuid.UUID
    tenant_slug: str
    graph_name: str
    user_id: uuid.UUID | None = None
    role: str = "member"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


async def load_tenant_context(
    session: AsyncSession,
    *,
    tenant_slug: str,
    user_external_id: str | None = None,
) -> TenantContext:
    tenant = (
        await session.execute(
            select(Tenant).where(Tenant.slug == tenant_slug, Tenant.is_active.is_(True))
        )
    ).scalar_one_or_none()
    if tenant is None:
        raise TenantNotFound(tenant_slug)

    user: User | None = None
    if user_external_id:
        user = (
            await session.execute(
                select(User).where(
                    User.external_id == user_external_id,
                    # Scoped to the tenant so a valid user of tenant A cannot be
                    # admitted into tenant B.
                    User.tenant_id == tenant.id,
                )
            )
        ).scalar_one_or_none()
        if user is None:
            raise UserNotInTenant(user_external_id)

    return TenantContext(
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        graph_name=tenant.graph_name,
        user_id=user.id if user else None,
        role=user.role if user else "member",
    )


async def load_tenant_context_by_id(session: AsyncSession, tenant_id: uuid.UUID) -> TenantContext:
    """Resolve a context from a tenant id.

    Workers receive tenant_id in queue messages rather than a slug, and must
    re-resolve rather than trust a denormalized graph name off the wire.
    """
    tenant = (
        await session.execute(
            select(Tenant).where(Tenant.id == tenant_id, Tenant.is_active.is_(True))
        )
    ).scalar_one_or_none()
    if tenant is None:
        raise TenantNotFound(str(tenant_id))
    return TenantContext(tenant_id=tenant.id, tenant_slug=tenant.slug, graph_name=tenant.graph_name)
