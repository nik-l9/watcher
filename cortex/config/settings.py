"""Application settings.

All configuration arrives via environment (or a local .env) with the CORTEX_
prefix. Secrets are never given defaults — a missing vault key should fail loudly
at startup rather than silently fall back to something insecure.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CORTEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["local", "test", "staging", "production"] = "local"

    #: Lowest structlog level that prints. Defaults to `debug`, which is what every run has
    #: always done and what makes `llm.cache` attributable to a call.
    #:
    #: Raising it exists for one case: a run somebody else will read, where a per-request
    #: diagnostic interleaved with the report is noise the reader cannot tell from output.
    log_level: Literal["debug", "info", "warning", "error"] = "debug"

    #: Ask, once, whether anything reachable is still missing before the loop is allowed to
    #: finish. See cortex/agents/gaps.py.
    #:
    #: Off by default and read at construction rather than baked in, because the honest
    #: comparison is against the loop exactly as it was -- and because the last change of this
    #: shape, an unconditional reflection turn, measured worse than doing nothing.
    gap_recheck: bool = False

    postgres_dsn: str = "postgresql+asyncpg://cortex:cortex@localhost:5433/cortex"

    falkordb_host: str = "localhost"
    falkordb_port: int = 6381
    falkordb_password: str | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    redis_url: str = "redis://localhost:6380/0"

    # Read without the CORTEX_ prefix for the same reason as the Anthropic key: it is
    # the name Voyage's own tooling uses, and two names for one secret invites a
    # deployment where only one of them is set.
    voyage_api_key: str | None = Field(default=None, validation_alias="VOYAGE_API_KEY", repr=False)

    # Base64 urlsafe-encoded 32-byte key. Wraps per-tenant data keys.
    vault_master_key: str = Field(default="", repr=False)

    # Read from ANTHROPIC_API_KEY, without the CORTEX_ prefix: that is the name the
    # provider's own tooling uses, and a second name for the same secret invites a
    # deployment where one is set and the other is not. Resolved here rather than left
    # to the SDK's os.environ lookup, so a key in .env works for every entry point
    # (worker, eval harness, script) instead of only for shells that exported it.
    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY", repr=False
    )

    # The OpenAI-compatible provider, which is several providers: OpenAI itself, OpenRouter,
    # Groq, Together, vLLM, Ollama. Unset base URL means OpenAI's own endpoint, so someone with
    # only `OPENAI_API_KEY` needs no further configuration -- and someone pointing at a proxy
    # sets the URL and uses that proxy's key in the same variable, which is what every one of
    # those services documents.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY", repr=False)
    # Not a secret in itself, but a proxy base can carry a token in its path, so it is kept out
    # of reprs alongside the key rather than being treated as ordinary configuration.
    openai_base_url: str | None = Field(
        default=None, validation_alias="OPENAI_BASE_URL", repr=False
    )

    # Clerk. Absent in development, required before authenticated traffic is served: the
    # gateway refuses to start serving outside local/test without a JWKS url, so a
    # misconfigured deployment fails visibly instead of trusting headers.
    clerk_jwks_url: str | None = None
    clerk_issuer: str | None = None
    #: Optional. Set it when the Clerk template declares an audience; leaving it unset skips
    #: the audience check, which is correct for Clerk's default session tokens.
    clerk_audience: str | None = None

    @property
    def clerk_configured(self) -> bool:
        return bool(self.clerk_jwks_url)

    #: The Slack app's signing secret. Without it the Slack endpoint refuses every request:
    #: `verify_slack_request` fails closed on an empty secret rather than verifying an internet
    #: -facing webhook against an empty key.
    slack_signing_secret: str | None = None

    #: The Slack **app-level** token (`xapp-`), which opens a Socket Mode connection.
    #:
    #: Distinct from both the bot token and the user token, and it is not a per-tenant credential:
    #: it belongs to the Slack *app*, so it lives in the environment rather than the vault, like
    #: the signing secret. Unset means the Socket Mode service does not run, which is the correct
    #: default for a deployment using the webhook instead -- an app has one events transport and
    #: enabling Socket Mode disables the Request URL.
    slack_app_token: str | None = None

    #: Public base URL, used only to build the "full report" link in a delivered Slack message.
    #: Unset means no link rather than a localhost one -- a message telling a colleague to open
    #: localhost reads as a broken feature rather than an unconfigured one.
    public_base_url: str | None = None

    @property
    def slack_configured(self) -> bool:
        """Whether *some* inbound Slack transport is usable.

        Either satisfies it, because an app has exactly one events transport: the webhook needs a
        signing secret, Socket Mode needs an app token, and requiring both would report a
        correctly configured Socket Mode deployment as unconfigured.
        """
        return bool(self.slack_signing_secret or self.slack_app_token)

    # Per-tenant call ceilings, across investigations. The per-investigation budget bounds
    # one run; these bound a tenant, which is what a runaway loop or a hammered button
    # multiplies. Zero disables a window, which is how a local run opts out.
    tenant_calls_per_minute: int = 120
    tenant_calls_per_hour: int = 2000

    # Investigation budgets. A runaway loop is a cost incident, so these are
    # config, not constants buried in the agent.
    max_investigation_steps: int = 40
    max_investigation_seconds: int = 300

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
