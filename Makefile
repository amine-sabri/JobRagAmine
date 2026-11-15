# JobRAG — every number in the README is produced by one of these targets.
SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip
COMPOSE := docker compose --env-file .env -f docker/docker-compose.yml

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip

.PHONY: install
install: $(VENV)/bin/activate ## Create the venv and install pinned dependencies
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e . --no-deps
	$(VENV)/bin/pre-commit install

.PHONY: lock
lock: $(VENV)/bin/activate ## Recompile requirements*.txt from requirements*.in
	$(PIP) install --quiet pip-tools
	$(VENV)/bin/pip-compile --quiet --strip-extras requirements.in -o requirements.txt
	$(VENV)/bin/pip-compile --quiet --strip-extras requirements-dev.in -o requirements-dev.txt

.PHONY: up
up: ## Start the pgvector database and wait until it is healthy
	$(COMPOSE) up -d --wait

.PHONY: down
down: ## Stop the database (keeps the data volume)
	$(COMPOSE) down

.PHONY: logs
logs: ## Tail the database logs
	$(COMPOSE) logs -f

.PHONY: psql
psql: ## Open a psql shell on the JobRAG database
	$(COMPOSE) exec db psql -U jobrag -d jobrag

.PHONY: test
test: ## Run the unit tests (no database needed)
	$(PY) -m pytest

.PHONY: test-all
test-all: ## Run every test, including the ones needing the database (make up first)
	$(PY) -m pytest -m ""

.PHONY: lint
lint: ## Check formatting and lint rules
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

.PHONY: fmt
fmt: ## Autoformat and autofix
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

.PHONY: api
api: ## Run the API on http://127.0.0.1:8010
	$(VENV)/bin/uvicorn jobrag.api:app --reload --host 127.0.0.1 --port 8010

.PHONY: health
health: ## Curl the /health endpoint
	curl -s http://127.0.0.1:8010/health | $(PY) -m json.tool

.PHONY: clean
clean: ## Remove caches (keeps the venv and the data volume)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
