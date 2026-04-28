Unit tests under `unit_tests/` run with `pytest-socket --disable-socket --allow-unix-socket` (see the `Makefile`), so real network calls fail. Mock HTTP with `pytest-httpx` or `respx` instead of hitting the API.

Integration tests under `integration_tests/` exercise the live Tess AI API and require `TESSAI_API_KEY`, `TESSAI_AGENT_ID`, and `TESSAI_WORKSPACE_ID` in a `.env` file (auto-loaded by `conftest.py`). Gate them with the `has_tess_credentials()` helper from `tests/conftest.py` and run them via `make integration_tests`.
