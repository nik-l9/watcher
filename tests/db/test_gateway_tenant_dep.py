"""Gateway translation of tenancy errors into HTTP.

The domain layer raises TenantNotFound / UserNotInTenant so it stays usable from a
Celery task. The gateway is the only place those become status codes, so this is
where the mapping is pinned — including that the 403 message does not disclose
which tenant an identity actually belongs to.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cortex.config.settings import Settings
from cortex.db.models import Tenant, User
from cortex.memory.naming import graph_name_for_new_tenant
from cortex.runtime.resources import Resources
from cortex.tenancy.context import TenantContext
from cortex.tenancy.limits import NullRateLimiter
from services.gateway.deps import get_session, require_tenant


def _build_app(dsn: str, env: str = "test") -> FastAPI:
    """A minimal app mounting only the tenancy dependency under test.

    `env` matters: header-based tenant resolution is confined to development
    environments (F-02), so the dependency reads it from app state.
    """
    engine = create_async_engine(dsn)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    application = FastAPI()

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as s:
            yield s

    @application.get("/scoped")
    async def scoped(ctx: TenantContext = Depends(require_tenant)) -> dict[str, str]:
        return {
            "tenant": ctx.tenant_slug,
            "graph": ctx.graph_name,
            "role": ctx.role,
        }

    application.state.resources = Resources(
        settings=Settings(_env_file=None, env=env),  # type: ignore[arg-type]
        engine=engine,
        sessionmaker=maker,
        # The tenancy dependency never touches the graph or semantic memory.
        graph=None,  # type: ignore[arg-type]
        vectors=None,  # type: ignore[arg-type]
        embeddings=None,  # type: ignore[arg-type]
        # Permits everything, and reports that it checked nothing.
        limiter=NullRateLimiter(),
    )
    application.dependency_overrides[get_session] = _session
    return application


@pytest.fixture
def app(_test_database: str) -> Iterator[FastAPI]:
    yield _build_app(_test_database)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def seeded(session: AsyncSession) -> dict[str, str]:
    """Two tenants, one user in the first, committed so the request sees them."""
    rows = {}
    for slug in ("gw-tenant-one", "gw-tenant-two"):
        tid = uuid.uuid4()
        session.add(
            Tenant(id=tid, slug=slug, name=slug, graph_name=graph_name_for_new_tenant(slug, tid))
        )
        rows[slug] = str(tid)
    await session.flush()
    tenant_one = uuid.UUID(rows["gw-tenant-one"])
    session.add(
        User(
            tenant_id=tenant_one,
            external_id="gw_clerk_admin",
            email="admin@example.com",
            role="admin",
        )
    )
    session.add(
        Tenant(
            id=uuid.uuid4(),
            slug="gw-suspended",
            name="Suspended",
            graph_name=graph_name_for_new_tenant("gw-suspended", uuid.uuid4()),
            is_active=False,
        )
    )
    await session.commit()
    return rows


class TestAccepts:
    def test_tenant_only(self, client: TestClient, seeded: dict[str, str]) -> None:
        r = client.get("/scoped", headers={"X-Cortex-Tenant": "gw-tenant-one"})
        assert r.status_code == 200
        body = r.json()
        assert body["tenant"] == "gw-tenant-one"
        assert body["role"] == "member"

    def test_tenant_with_user(self, client: TestClient, seeded: dict[str, str]) -> None:
        r = client.get(
            "/scoped",
            headers={"X-Cortex-Tenant": "gw-tenant-one", "X-Cortex-User": "gw_clerk_admin"},
        )
        assert r.status_code == 200
        assert r.json()["role"] == "admin"


class TestRejects:
    def test_missing_tenant_header_is_422(self, client: TestClient) -> None:
        assert client.get("/scoped").status_code == 422

    def test_unknown_tenant_is_404(self, client: TestClient, seeded: dict[str, str]) -> None:
        r = client.get("/scoped", headers={"X-Cortex-Tenant": "no-such-tenant"})
        assert r.status_code == 404

    def test_suspended_tenant_is_404(self, client: TestClient, seeded: dict[str, str]) -> None:
        r = client.get("/scoped", headers={"X-Cortex-Tenant": "gw-suspended"})
        assert r.status_code == 404

    def test_cross_tenant_user_is_403(self, client: TestClient, seeded: dict[str, str]) -> None:
        """A real admin of tenant one, presented against tenant two."""
        r = client.get(
            "/scoped",
            headers={"X-Cortex-Tenant": "gw-tenant-two", "X-Cortex-User": "gw_clerk_admin"},
        )
        assert r.status_code == 403

    def test_403_does_not_disclose_the_users_real_tenant(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        r = client.get(
            "/scoped",
            headers={"X-Cortex-Tenant": "gw-tenant-two", "X-Cortex-User": "gw_clerk_admin"},
        )
        assert "gw-tenant-one" not in r.text
        assert "gw_clerk_admin" not in r.text

    def test_invalid_slug_is_not_a_server_error(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        """Hostile slugs must be refused as client errors, never a 500 or a lookup
        against a constructed graph name."""
        for hostile in ("../etc", "gw-tenant-one*", "gw:tenant", "GW-TENANT-ONE"):
            r = client.get("/scoped", headers={"X-Cortex-Tenant": hostile})
            assert r.status_code in (400, 403, 404, 422), (hostile, r.status_code)


class TestFailClosedOutsideLocal:
    """F-02. These headers are unauthenticated, so trusting them anywhere real would
    let any caller name any tenant and any user — including one with the admin role.

    Deferring real Clerk verification is a schedule decision. Deferring it
    fail-*open* would be an authentication bypass one route away from reachable.
    """

    @pytest.fixture
    def app_in_env(self, _test_database: str) -> object:
        """Build an app whose resources report a given environment."""

        def _build(env: str) -> FastAPI:
            return _build_app(_test_database, env=env)

        return _build

    @pytest.mark.parametrize("env", ["staging", "production"])
    def test_refuses_to_serve_outside_development(
        self, app_in_env: object, seeded: dict[str, str], env: str
    ) -> None:
        with TestClient(app_in_env(env), raise_server_exceptions=False) as client:  # type: ignore[operator]
            response = client.get("/scoped", headers={"X-Cortex-Tenant": "gw-tenant-one"})
        # A misconfiguration, surfaced as a server error — never a served request.
        assert response.status_code == 500

    @pytest.mark.parametrize("env", ["local", "test"])
    def test_permits_header_trust_in_development(
        self, app_in_env: object, seeded: dict[str, str], env: str
    ) -> None:
        with TestClient(app_in_env(env)) as client:  # type: ignore[operator]
            response = client.get("/scoped", headers={"X-Cortex-Tenant": "gw-tenant-one"})
        assert response.status_code == 200

    @pytest.mark.parametrize("env", ["staging", "production"])
    def test_admin_cannot_be_claimed_by_header_outside_development(
        self, app_in_env: object, seeded: dict[str, str], env: str
    ) -> None:
        """The escalation path specifically: naming a real admin subject."""
        with TestClient(app_in_env(env), raise_server_exceptions=False) as client:  # type: ignore[operator]
            response = client.get(
                "/scoped",
                headers={
                    "X-Cortex-Tenant": "gw-tenant-one",
                    "X-Cortex-User": "gw_clerk_admin",
                },
            )
        assert response.status_code == 500
        assert "gw-tenant-one" not in response.text

    def test_the_guard_names_the_remedy(self) -> None:
        """An operator hitting this needs to know what to do about it."""
        from services.gateway.deps import _HEADER_TRUST_ENVIRONMENTS, MisconfiguredAuth

        assert _HEADER_TRUST_ENVIRONMENTS == {"local", "test"}
        assert issubclass(MisconfiguredAuth, Exception)


# ------------------------------------------------------------------ verified tokens


class TestAVerifiedTokenBeatsAHeader:
    """Once a token is present, the tenant comes from its organisation claim.

    This is the half of F-02 that lets the closed door be opened: header trust exists only
    because there was nothing better. With a verified token, a header naming a different
    tenant is an attempt to read data the token does not cover.
    """

    class _Identity:
        def __init__(self, subject: str, org_slug: str | None) -> None:
            self.subject = subject
            self.org_slug = org_slug
            self.org_id = "org_1"
            self.org_role = "org:member"
            self.claims: dict[str, str] = {}

    class _Verifier:
        """Stands in for Clerk. Verification itself is covered in tests/tenancy."""

        def __init__(self, identity: object | None = None, error: Exception | None = None) -> None:
            self._identity = identity
            self._error = error

        async def verify(self, token: str) -> object:
            if self._error is not None:
                raise self._error
            return self._identity

    def _app_with(self, dsn: str, verifier: object, env: str = "test") -> FastAPI:
        from services.gateway.deps import get_verifier

        application = _build_app(dsn, env=env)
        application.dependency_overrides[get_verifier] = lambda: verifier
        return application

    async def _seed(self, session: AsyncSession, slug: str, external_id: str) -> None:
        tenant_id = uuid.uuid4()
        session.add(
            Tenant(
                id=tenant_id,
                slug=slug,
                name=slug,
                graph_name=graph_name_for_new_tenant(slug, tenant_id),
            )
        )
        await session.flush()
        session.add(User(tenant_id=tenant_id, external_id=external_id, email=f"{external_id}@x.io"))
        await session.commit()

    async def test_the_tenant_comes_from_the_token(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(_test_database, self._Verifier(self._Identity("user_2abc", "acme")))

        with TestClient(app) as client:
            response = client.get("/scoped", headers={"Authorization": "Bearer real.token.here"})

        assert response.status_code == 200
        assert response.json()["tenant"] == "acme"

    async def test_a_header_naming_another_tenant_is_refused(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        """Refused rather than reconciled. Preferring the header reopens the bypass;
        preferring the claim silently leaves a caller convinced they read a tenant they
        did not."""
        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(_test_database, self._Verifier(self._Identity("user_2abc", "acme")))

        with TestClient(app) as client:
            response = client.get(
                "/scoped",
                headers={"Authorization": "Bearer real.token", "X-Cortex-Tenant": "victim"},
            )

        assert response.status_code == 403
        assert "does not match" in response.json()["detail"]

    async def test_a_token_with_no_organisation_is_refused(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        """A personal Clerk session carries no org. There is no tenant to resolve, and
        falling back to the header here would be the bypass again."""
        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(_test_database, self._Verifier(self._Identity("user_2abc", None)))

        with TestClient(app) as client:
            response = client.get(
                "/scoped",
                headers={"Authorization": "Bearer real.token", "X-Cortex-Tenant": "acme"},
            )

        assert response.status_code == 403
        assert "organisation" in response.json()["detail"]

    async def test_a_rejected_token_is_401_and_says_nothing_useful(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        from cortex.security.clerk import ClerkAuthError

        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(
            _test_database,
            self._Verifier(error=ClerkAuthError("token signature could not be verified")),
        )

        with TestClient(app) as client:
            response = client.get("/scoped", headers={"Authorization": "Bearer forged"})

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        # No subject echoed back to a caller who failed to prove they own it.
        assert "user_" not in response.text

    async def test_a_clerk_outage_is_503_not_401(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        """During an outage a valid token is still valid. Reporting it as a login failure
        would be wrong and would hide the outage."""
        from cortex.security.clerk import ClerkUnavailable

        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(
            _test_database, self._Verifier(error=ClerkUnavailable("could not fetch Clerk keys"))
        )

        with TestClient(app) as client:
            response = client.get("/scoped", headers={"Authorization": "Bearer real.token"})

        assert response.status_code == 503

    async def test_production_without_a_token_is_401(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        """Outside development a token is the only accepted identity."""
        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(
            _test_database,
            self._Verifier(self._Identity("user_2abc", "acme")),
            env="production",
        )

        with TestClient(app) as client:
            response = client.get("/scoped", headers={"X-Cortex-Tenant": "acme"})

        assert response.status_code == 401

    async def test_production_without_clerk_configured_fails_loudly(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        """A missing token is the caller's problem; an unconfigured Clerk is ours. Reporting
        the second as the first would send someone to debug their login while the server was
        misconfigured."""
        from services.gateway.deps import MisconfiguredAuth, get_verifier

        await self._seed(session, "acme", "user_2abc")
        app = _build_app(_test_database, env="production")
        app.dependency_overrides[get_verifier] = lambda: None

        with TestClient(app, raise_server_exceptions=True) as client:
            with pytest.raises(MisconfiguredAuth):
                client.get("/scoped", headers={"X-Cortex-Tenant": "acme"})

    @pytest.mark.parametrize(
        "header", ["", "Basic abc", "Bearer", "Bearer   ", "bearer-token-without-space"]
    )
    async def test_a_non_bearer_authorization_header_is_treated_as_absent(
        self, session: AsyncSession, _test_database: str, header: str
    ) -> None:
        """Anything that is not exactly a bearer credential is absent rather than guessed
        at — and in development that means the header path, not a crash."""
        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(_test_database, self._Verifier(self._Identity("user_2abc", "acme")))

        with TestClient(app) as client:
            response = client.get(
                "/scoped",
                headers={"Authorization": header, "X-Cortex-Tenant": "acme"},
            )

        assert response.status_code == 200, response.text

    async def test_a_lowercase_bearer_scheme_is_accepted(
        self, session: AsyncSession, _test_database: str
    ) -> None:
        """Clients disagree about the scheme's case; the RFC does not."""
        await self._seed(session, "acme", "user_2abc")
        app = self._app_with(_test_database, self._Verifier(self._Identity("user_2abc", "acme")))

        with TestClient(app) as client:
            response = client.get("/scoped", headers={"Authorization": "bearer real.token"})

        assert response.status_code == 200
