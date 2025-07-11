"""
Tests for the main FastAPI application (API endpoints).
"""

from fastapi.testclient import TestClient


def test_read_root(client: TestClient):
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
