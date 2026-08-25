"""Gateway dependency wiring.

Resources live on app.state, created in the lifespan, so a test can stand up an
isolated app and a Celery worker can own its own copy. Nothing here is a module
global.

**How identity is established.** Two paths, and which one applies is decided by the
environment rather than by the request, so a caller cannot choose the weaker one:

| Environment | Accepted |
|---|---|
| `local`, `test` | a verified Clerk token when configured; `X-Cortex-*` headers otherwise |
| anything else | a verified Clerk token, and nothing else |

The headers were the original mechanism and they are unauthenticated: any caller could name
any tenant and any user, including one holding the `admin` role (F-02). That gap was made to
fail closed by refusing to serve outside development. This module now carries the other half —
real verification — so the closed door can be opened deliberately.

**A verified claim always beats a header.** When a token is present, the tenant comes from its
organisation claim. A request whose `X-Cortex-Tenant` names a different tenant is refused
rather than reconciled: silently preferring one would make the header meaningful again, and
silently preferring the other would be a bypass.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from fastapi import Depends, Header, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.db.models import Investigation, Tenant
from cortex.memory.graph_store import GraphStore
from cortex.runtime.resources import Resources
from cortex.security.clerk import ClerkAuthError, ClerkUnavailable, ClerkVerifier
from cortex.tenancy.context import (
    TenantContext,
    TenantNotFound,
    UserNotInTenant,
    load_tenant_context,
)


def get_resources(request: Request) -> Resources:
    return request.app.state.resources  # type: ignore[no-any-return]


def get_graph(resources: Resources = Depends(get_resources)) -> GraphStore:
    return resources.graph


async def get_session(
    resources: Resources = Depends(get_resources),
) -> AsyncIterator[AsyncSession]:
    async with resources.session() as session:
        yield session


class MisconfiguredAuth(Exception):
    """Raised when header-trust auth is reached outside a development environment."""


# Environments in which an unverified identity header is acceptable.
_HEADER_TRUST_ENVIRONMENTS = frozenset({"local", "test"})


def get_verifier(request: Request, resources: Resources = Depends(get_resources)):  # type: ignore[no-untyped-def]
    """The Clerk verifier for this process, or None when Clerk is not configured.

    Cached on app state rather than constructed per request, because the verifier holds the
    fetched JWKS. A fresh one per request would fetch Clerk's keys on every call — turning
    our own traffic into an amplifier pointed at Clerk.
    """
    if not resources.settings.clerk_configured:
        return None
    existing = getattr(request.app.state, "clerk_verifier", None)
    if existing is None:
        existing = ClerkVerifier(
            jwks_url=str(resources.settings.clerk_jwks_url),
            issuer=resources.settings.clerk_issuer,
            audience=resources.settings.clerk_audience,
        )
        request.app.state.clerk_verifier = existing
    return existing


async def require_tenant(
    resources: Resources = Depends(get_resources),
    session: AsyncSession = Depends(get_session),
    verifier: ClerkVerifier | None = Depends(get_verifier),
    x_cortex_tenant: str | None = Header(default=None, description="Tenant slug"),
    x_cortex_user: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    tenant: str | None = Query(
        default=None,
        description=(
            "Tenant slug, for local development only. A browser cannot send a header, so "
            "without this the report links Cortex posts to Slack cannot be opened at all."
        ),
    ),
) -> TenantContext:
    token = _bearer(authorization)
    development = resources.settings.env in _HEADER_TRUST_ENVIRONMENTS

    if token and verifier is not None:
        identity = await _verify(verifier, token)
        # The organisation claim is the tenant. Falling back to the header here would let a
        # caller with a valid token for one organisation read another's data by relabelling
        # the request, which is the whole bypass this replaces.
        slug = identity.org_slug
        if not slug:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "token carries no organisation; select an organisation before continuing",
            )
        if x_cortex_tenant and x_cortex_tenant != slug:
            # Refused rather than reconciled. Preferring the header would reopen the bypass;
            # preferring the claim silently would leave a caller convinced they were reading
            # a tenant they were not.
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "the tenant header does not match the tenant in the token",
            )
        return await _load(session, slug=slug, user_external_id=identity.subject)

    if not development:
        # Outside development a token is the only accepted identity. Two distinct failures,
        # deliberately: a missing token is the caller's problem (401), an unconfigured Clerk
        # is ours (500), and reporting the second as the first would send someone to debug
        # their login while the server was misconfigured.
        if verifier is None:
            raise MisconfiguredAuth(
                f"env={resources.settings.env!r} requires Clerk verification, but no JWKS "
                "url is configured; set CORTEX_CLERK_JWKS_URL before serving traffic"
            )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "a bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Development only, and only when Clerk is not configured: the headers are
    # unauthenticated, so this path exists to make local work possible and nothing else.
    #
    # The query parameter is here because a browser cannot send a header, and the report link
    # Cortex posts into Slack is meant to be *clicked*. Without it the whole report view was
    # unreachable outside curl -- discovered by trying to open one, after an ngrok outage had
    # been masking it. Restricted to the same development envs as the header for the same
    # reason: it is unauthenticated, and it must never be the way a real deployment identifies
    # a tenant. Before this is served anywhere non-local the link needs to carry a signed,
    # expiring token instead, so that possessing the URL proves something.
    slug = x_cortex_tenant or tenant
    if not slug:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "X-Cortex-Tenant header or ?tenant= is required in this environment",
        )
    return await _load(session, slug=slug, user_external_id=x_cortex_user)


async def tenant_for_investigation(
    investigation_id: uuid.UUID,
    resources: Resources = Depends(get_resources),
    session: AsyncSession = Depends(get_session),
    verifier: ClerkVerifier | None = Depends(get_verifier),
    x_cortex_tenant: str | None = Header(default=None, description="Tenant slug"),
    x_cortex_user: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    tenant: str | None = Query(default=None),
) -> TenantContext:
    """The tenant that owns this investigation, for the report page only.

    **Why this exists, stated plainly because it is a weakening and should be read as one.**
    The report page is reached by clicking a link Cortex posted into Slack. That link already
    contains the investigation's id, so whoever holds it holds the only secret involved. Adding
    `?tenant=` to it protected nothing -- the holder of the link had both halves -- while
    breaking every link that lost the query string on the way: pasted into a browser, forwarded,
    truncated by a chat client. The result was a link Cortex sent that Cortex's own users could
    not open, twice over.

    So for this one route the **investigation id is the credential**, which is what it already
    effectively was. A v4 uuid is 122 bits of entropy and is not enumerable; what this route no
    longer does is prove that the *reader* belongs to the owning tenant, and that is the honest
    cost. Everything else keeps `require_tenant`.

    An explicit identity still wins where one is supplied, so a real deployment behind Clerk
    behaves exactly as before and the cross-tenant isolation tests still exercise the strict
    path. This fallback is reached only when nobody said who they were.

    Before this page is served anywhere shared, the link needs a signed expiring token so that
    holding the URL proves something more than possession. That is recorded in
    `docs/rollout-plan.md` rather than left as a comment nobody will find.
    """
    if authorization or x_cortex_tenant or tenant:
        return await require_tenant(
            resources=resources,
            session=session,
            verifier=verifier,
            x_cortex_tenant=x_cortex_tenant,
            x_cortex_user=x_cortex_user,
            authorization=authorization,
            tenant=tenant,
        )

    slug = (
        await session.execute(
            select(Tenant.slug)
            .join(Investigation, Investigation.tenant_id == Tenant.id)
            .where(Investigation.id == investigation_id)
        )
    ).scalar_one_or_none()
    if slug is None:
        # One response for absent and foreign, as everywhere else, so ids cannot be enumerated
        # through the error either.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown investigation")
    return await _load(session, slug=slug, user_external_id=x_cortex_user)


async def _verify(verifier: ClerkVerifier, token: str):  # type: ignore[no-untyped-def]
    try:
        return await verifier.verify(token)
    except ClerkAuthError as exc:
        # The message is written to be safe to return: it never contains the token, and never
        # echoes a subject back to a caller who failed to prove they own it.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except ClerkUnavailable as exc:
        # 503, not 401. During a Clerk outage a valid token is still valid; telling the user
        # their login failed would be wrong and would hide the outage.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None


async def _load(session: AsyncSession, *, slug: str, user_external_id: str | None) -> TenantContext:
    try:
        return await load_tenant_context(
            session, tenant_slug=slug, user_external_id=user_external_id
        )
    except TenantNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown or inactive tenant") from None
    except UserNotInTenant:
        # Deliberately not "user not found" — that would confirm which tenant a
        # given identity belongs to.
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "user does not belong to this tenant"
        ) from None


def _bearer(header: str | None) -> str | None:
    """The token from an `Authorization` header, or None.

    Case-insensitive on the scheme, because clients disagree about it, and strict about the
    shape otherwise: anything that is not exactly a bearer credential is treated as absent
    rather than guessed at.
    """
    if not header:
        return None
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return credential.strip() or None
