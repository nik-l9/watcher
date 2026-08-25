# One multi-stage build for every Cortex service.
#
# A single file with per-service targets rather than a separately-tagged base
# image: BuildKit resolves a bare `FROM cortex-base:latest` against a registry,
# not the local daemon, so cross-image references fail unless the base is pushed.
# Multi-stage keeps the dependency layer built and cached exactly once while each
# service still produces its own image with its own entrypoint.
#
# Build a specific service:
#   docker build -f infra/docker/Dockerfile --target gateway .

# ----------------------------------------------------------------- base
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so editing source does not invalidate the install layer.
COPY pyproject.toml README.md ./
RUN mkdir -p cortex && touch cortex/__init__.py \
    && pip install --no-cache-dir hatchling \
    && pip install --no-cache-dir .

COPY cortex/ ./cortex/
COPY services/ ./services/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Run unprivileged: a compromised connector must not be root in its container.
RUN useradd --create-home --uid 10001 cortex && chown -R cortex:cortex /app
USER cortex

# ----------------------------------------------------------------- migrate
FROM base AS migrate
CMD ["alembic", "upgrade", "head"]

# ----------------------------------------------------------------- gateway
# The only externally-reachable service.
FROM base AS gateway
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "services.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------------------------------------------------------- investigation worker
# Consumes only cortex.investigation.
#
# Concurrency is deliberately low: each task is an LLM loop that is IO-bound but
# token-expensive, so throughput comes from adding replicas rather than from
# packing more tasks into one process.
FROM base AS investigation-worker
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD celery -A services.investigation_worker.worker.celery inspect ping || exit 1
CMD ["celery", "-A", "services.investigation_worker.worker.celery", "worker", \
     "--queues", "cortex.investigation", "--concurrency", "2", "--loglevel", "info"]

# ----------------------------------------------------------------- ingest worker
# Consumes only cortex.ingest.
FROM base AS ingest-worker
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD celery -A services.ingest_worker.worker.celery inspect ping || exit 1
CMD ["celery", "-A", "services.ingest_worker.worker.celery", "worker", \
     "--queues", "cortex.ingest", "--concurrency", "4", "--loglevel", "info"]

# ----------------------------------------------------------------- scheduler
# Beat. Exactly one replica must run: two would double-fire every nightly sync.
FROM base AS scheduler
CMD ["celery", "-A", "services.ingest_worker.worker.celery", "beat", "--loglevel", "info"]

# The Slack Socket Mode transport: an outbound WebSocket, so no inbound network path and no
# published port. Optional -- it exits cleanly when CORTEX_SLACK_APP_TOKEN is unset, which is the
# correct state for a deployment using the webhook instead.
FROM base AS slack-socket
CMD ["python", "-m", "services.slack_socket.main"]
