# Repository Guidelines

## Project Structure & Module Organization
Application code lives in `app/`. Use `app/main.py` for the FastAPI entrypoint and API routes, `app/agents/` for LangGraph orchestration and node logic, `app/indexer/` for eShop indexing, `app/symbolic/` for deterministic verification, `app/providers/` for Cosmos and LLM integrations, and `app/ledger/` for audit persistence. Top-level Markdown files capture product, architecture, and hackathon context. All tests are located in the `tests/` directory, modularized into `unit/`, `integration/`, `contracts/`, and `e2e/`, mirroring the `app/` package layout.

## Build, Test, and Development Commands
Install dependencies with `python -m pip install -r requirements.txt`. Run the API locally with `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`. Start the containerized stack with `docker compose up --build`. Rebuild the eShop index with `curl -X POST http://localhost:8000/index`. Smoke-test the service with `curl http://localhost:8000/health` and submit a sample incident with `curl -X POST http://localhost:8000/incident -F "report=HTTP 500 on /api/orders"`. Run the test suite with `pytest tests/`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints on public functions, and concise docstrings where behavior is not obvious. Keep modules and functions `snake_case`, classes `PascalCase`, and enum-style status values uppercase. Favor small, single-purpose functions and keep provider logic separated from agent-node logic. No formatter config is committed, so match the surrounding style and keep imports grouped logically.

## Testing Guidelines
Tests are modularized into `tests/unit`, `tests/integration`, `tests/contracts`, and `tests/e2e`. Place new files in their respective folders with names like `test_span_arbiter.py` or `test_incident_api.py`. Prioritize coverage for symbolic verification, FSM transitions, and API endpoint behavior. Run tests comprehensively with `pytest tests/`, or for specific suites use `pytest tests/unit/`. Add regression tests for each bug fix and verify coverage using `pytest --cov=app tests/`.

## Commit & Pull Request Guidelines
This workspace does not include `.git` history, so use clear conventional commits such as `feat: add ledger persistence fallback` or `fix: guard missing image uploads`. Keep PRs small and explain the operational impact, config changes, and verification steps. Link the relevant issue or task, include sample request/response payloads for API changes, and attach screenshots only when UI or observability output changes.

## Security & Configuration Tips
Configuration is loaded from `.env` via `app/config.py`. Never commit secrets such as Cosmos keys, Gemini API keys, or Langfuse credentials. Treat uploaded incident images as sensitive data and preserve the current pattern of excluding raw image payloads from persistence when extending storage logic.
