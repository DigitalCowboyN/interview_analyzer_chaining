import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _load_env_file():
    """Load .env file into os.environ if keys aren't already set.

    This ensures tests that require API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY,
    etc.) can find them without requiring manual export. Only sets variables that
    aren't already in the environment.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in os.environ:
            os.environ[key] = value


_load_env_file()

from src.main import app  # noqa: E402  # Must import after loading .env


@pytest.fixture(scope="function")
def client():
    """
    Pytest fixture to create a TestClient instance for API testing.
    Scope is 'function' to reuse the client across tests in a module.
    """
    with TestClient(app) as c:
        yield c
