import pytest
from fastapi.testclient import TestClient
from src.main import app  # Assuming your FastAPI app instance is here


@pytest.fixture(scope="function")
def client():
    """
    Pytest fixture to create a TestClient instance for API testing.
    Scope is 'function' to reuse the client across tests in a module.
    """
    with TestClient(app) as c:
        yield c 