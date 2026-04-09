"""Shared test configuration."""

import os

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def _load_env() -> None:
    load_dotenv()


def has_tess_credentials() -> bool:
    """Return True when all required Tess env vars are set."""
    api_key = os.environ.get("TESSAI_API_KEY", "")
    agent_id = os.environ.get("TESSAI_AGENT_ID", "")
    workspace_id = os.environ.get("TESSAI_WORKSPACE_ID", "")
    return bool(api_key and agent_id and workspace_id)
