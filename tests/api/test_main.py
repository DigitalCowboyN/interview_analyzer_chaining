"""
Tests for the main FastAPI application (e.g., health checks).
"""

from fastapi.testclient import TestClient
from src.main import app # Import the FastAPI app instance

client = TestClient(app)

def test_read_root():
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"} 