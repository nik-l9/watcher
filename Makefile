.PHONY: help setup up up-all down down-all logs migrate revision test test-cov eval ask \
	eval-harness test-tenancy \
	lint fmt gateway investigation-worker ingest-worker scheduler clean

PY := .venv/bin/python

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:  ## Create the venv, install deps, write .env if absent
	uv venv --python 3.12
	# Targeted explicitly: `uv pip install` honours an already-active VIRTUAL_ENV over the
	# venv just created, so with any other environment active this installs somewhere else
	# and reports success. The symptom is `make test` failing with "No module named pytest"
	# straight after a setup that printed the whole dependency list.
	uv pip install --python .venv/bin/python -e ".[dev]"
	@test -f .env || (cp .env.example .env && \
		$(PY) -c "import base64,os,pathlib;p=pathlib.Path('.env');p.write_text(p.read_text().replace('CORTEX_VAULT_MASTER_KEY=','CORTEX_VAULT_MASTER_KEY='+base64.urlsafe_b64encode(os.urandom(32)).decode()))" && \
		echo "wrote .env with a fresh vault master key")

up:  ## Start infrastructure only (Postgres, FalkorDB, Qdrant, Redis)
	docker compose up -d
	@docker compose ps --format '{{.Service}}\t{{.Status}}'

up-all:  ## Start infrastructure + all services (gateway, workers, scheduler)
	docker compose --profile app up -d --build
	@docker compose --profile app ps --format '{{.Service}}\t{{.Status}}'

down-all:  ## Stop everything
	docker compose --profile app down

down:  ## Stop the stack (volumes preserved)
	docker compose down

logs:
	docker compose logs -f --tail=100

migrate:  ## Apply migrations
	.venv/bin/alembic upgrade head

revision:  ## Autogenerate a migration: make revision m="add x"
	.venv/bin/alembic revision --autogenerate -m "$(m)"

test:  ## Full suite (requires `make up`)
	$(PY) -m pytest tests/ -q

test-cov:  ## Full suite with a coverage report
	$(PY) -m pytest tests/ -q --cov=cortex --cov=services --cov-report=term-missing

ask:  ## Ask the analyst a question and read the report. Q="why did signups fall?"
	$(PY) -m cortex.ask $(if $(Q),"$(Q)",) $(if $(DATASET),--dataset $(DATASET),) --show-truth

ingest:  ## Sync one connector now. TENANT=acme PROVIDER=github
	$(PY) -m cortex.ingest $(if $(TENANT),--tenant $(TENANT),) $(if $(PROVIDER),--provider $(PROVIDER),) $(if $(SINCE),--since $(SINCE),)

ingest-status:  ## Watermarks, health and staleness per stream. TENANT=acme
	$(PY) -m cortex.ingest --tenant $(TENANT) --status

ingest-targets:  ## What tonight's nightly run would dispatch
	$(PY) -m cortex.ingest --targets

eval:  ## Score the analyst against labeled fixtures. Gates the first real investigation.
	$(PY) -m cortex.eval

eval-harness:  ## Exercise the eval harness without spending tokens
	$(PY) -m pytest tests/db/test_eval.py -v

test-tenancy:  ## Isolation gate. Must pass before any real credential is entered.
	$(PY) -m pytest tests/tenancy/ -v

lint:
	.venv/bin/ruff check cortex services tests
	.venv/bin/ruff format --check cortex services tests

fmt:
	.venv/bin/ruff format cortex services tests
	.venv/bin/ruff check --fix cortex services tests

gateway:  ## Run the gateway locally against the compose infrastructure
	.venv/bin/uvicorn services.gateway.app:app --reload --port 8000

investigation-worker:  ## Run the investigation worker locally
	.venv/bin/celery -A services.investigation_worker.worker.celery worker \
		--queues cortex.investigation --concurrency 2 --loglevel info

ingest-worker:  ## Run the ingest worker locally
	.venv/bin/celery -A services.ingest_worker.worker.celery worker \
		--queues cortex.ingest --concurrency 4 --loglevel info

scheduler:  ## Run beat locally
	.venv/bin/celery -A services.ingest_worker.worker.celery beat --loglevel info

clean:
	docker compose --profile app down -v
	rm -rf .pytest_cache .ruff_cache
