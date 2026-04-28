LangChain integration for the Tess AI API — exposes `ChatTessAI` plus tool-calling and file-upload helpers.

This project uses `uv` (with `UV_FROZEN=true`) driven through the `Makefile`. Prefer `make test`, `make lint`, `make type`, `make format`, and `make integration_tests` over invoking pytest, ruff, or mypy directly.

LangChain imports must come from `langchain_core` — `scripts/lint_imports.sh` rejects `from langchain.…` and `from langchain_experimental.…`. Relative imports are also banned (ruff `flake8-tidy-imports`); use absolute `langchain_tessai.…` paths.
