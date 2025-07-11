"""
Tests for the API endpoints defined in src/api/routers/analysis.py.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Import the FastAPI app to create a TestClient
from src.main import app


# FIX: Add the client fixture definition
@pytest.fixture(scope="module")
def client():
    """Provides a FastAPI TestClient instance for API tests."""
    with TestClient(app) as c:
        yield c


# Mock config data used by tests
MOCK_INPUT_DIR = "/mock/input/dir"

VALID_INPUT_FILENAME = "interview_transcript.txt"
NON_EXISTENT_FILENAME = "ghost.txt"
INVALID_FILENAME_FORMAT = "../sneaky_file.txt"

# Test functions will go here


@pytest.fixture
def mock_config_global():
    """Mocks the global config object."""
    # Mock config to provide necessary paths
    with patch("src.config.config", new_callable=dict) as mock_config:
        mock_config.update({
            "paths": {
                "input_dir": MOCK_INPUT_DIR,
                "output_dir": "/mock/output",
                "templates_dir": "/mock/templates"
            },
            "analysis_service": {
                "batch_size": 10,
                "max_workers": 4,
                "progress_interval": 10
            },
            "sentence_analyzer": {
                "model_name": "mock_model",
                "api_key": "mock_key"
            },
            "context_builder": {
                "context_window": 5
            }
        })
        yield mock_config


@pytest.mark.usefixtures("client", "mock_config_global")  # Ensure client and config mock are used
def test_trigger_analysis_success(client: TestClient):
    """Test POST /analysis/ success path (202 Accepted)."""
    # --- Setup ---
    filename = "test_success.txt"
    request_data = {"input_filename": filename}
    mock_task_id = "mock-uuid-1234"

    # --- Patching ---
    # Patch dependencies directly within the test
    with patch("src.pipeline.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline, \
         patch("src.api.routers.analysis.BackgroundTasks.add_task") as mock_add_task, \
         patch("uuid.uuid4", return_value=mock_task_id), \
         patch("pathlib.Path.is_file", return_value=True) as mock_is_file, \
         patch("src.config.config", {"paths": {"input_dir": "/mock"}}) as mock_cfg:
        # Provide a minimal config dict for the endpoint to read input_dir

        # --- Execute ---
        response = client.post("/analysis/", json=request_data)

        # --- Assertions ---
        # 1. Response Status and Body
        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()
        assert response_data["task_id"] == mock_task_id
        assert response_data["message"] == "Analysis task accepted and scheduled to run in background."

        # 2. Background Task Scheduling
        mock_add_task.assert_called_once()
        call_args, call_kwargs = mock_add_task.call_args
        assert call_args[0] == mock_run_pipeline
        assert call_kwargs.get("specific_file") == filename  # Check specific_file kwarg
        assert call_kwargs.get("task_id") == mock_task_id

        # 3. Ensure pipeline wasn't awaited in the request handler
        mock_run_pipeline.assert_not_awaited()

        # 4. Verify file existence check was performed
        mock_is_file.assert_called_once()

        # 5. Verify config was accessed for input directory
        assert mock_cfg["paths"]["input_dir"] == "/mock"


@pytest.mark.usefixtures("client", "mock_config_global")
def test_trigger_analysis_invalid_filename(client: TestClient):
    """Test triggering analysis with an invalid filename (e.g., containing path traversal)."""
    request_data = {"input_filename": "../invalid/path.txt"}
    response = client.post("/analysis/", json=request_data)
    assert response.status_code == 400  # Bad Request
    assert "Invalid input filename format." in response.json()["detail"]


@pytest.mark.usefixtures("client", "mock_config_global")
def test_trigger_analysis_missing_filename(client: TestClient):
    """Test triggering analysis with missing 'input_filename' in request body."""
    request_data = {}  # Missing input_filename
    response = client.post("/analysis/", json=request_data)
    assert response.status_code == 422  # Unprocessable Entity (validation error)


@pytest.mark.usefixtures("client")
def test_trigger_analysis_pipeline_error_still_accepts(client: TestClient):
    """Test POST /analysis/ returns 202 even if background task fails."""
    # --- Setup ---
    filename = "test_pipeline_fail.txt"
    request_data = {"input_filename": filename}
    mock_task_id = "mock-uuid-5678"
    pipeline_error = ValueError("Pipeline processing failed!")

    # --- Patching ---
    with patch("src.pipeline.run_pipeline", new_callable=AsyncMock, side_effect=pipeline_error) as mock_run_pipeline, \
         patch("src.api.routers.analysis.BackgroundTasks.add_task") as mock_add_task, \
         patch("uuid.uuid4", return_value=mock_task_id), \
         patch("pathlib.Path.is_file", return_value=True) as mock_is_file, \
         patch("src.config.config", {"paths": {"input_dir": "/mock"}}) as mock_cfg:

        # --- Execute ---
        response = client.post("/analysis/", json=request_data)

        # --- Assertions ---
        # 1. Response Status and Body (API should succeed)
        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()
        assert response_data["task_id"] == mock_task_id
        assert response_data["message"] == "Analysis task accepted and scheduled to run in background."

        # 2. Background Task Scheduling (Should still be called)
        mock_add_task.assert_called_once()
        call_args, call_kwargs = mock_add_task.call_args
        assert call_args[0] == mock_run_pipeline  # The mock that raises error
        assert call_kwargs.get("specific_file") == filename
        assert call_kwargs.get("task_id") == mock_task_id

        # 3. is_file check from endpoint
        mock_is_file.assert_called_once()

        # 4. Pipeline function itself was not awaited by the endpoint
        mock_run_pipeline.assert_not_awaited()

        # 5. Verify config was accessed for input directory
        assert mock_cfg["paths"]["input_dir"] == "/mock"


@pytest.mark.usefixtures("client")
def test_trigger_analysis_file_not_found_raises_404(client: TestClient):
    """Test POST /analysis/ returns 404 if input file not found."""
    # --- Setup ---
    filename = "ghost_file.txt"
    request_data = {"input_filename": filename}

    # --- Patching ---
    with patch("src.pipeline.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline, \
         patch("src.api.routers.analysis.BackgroundTasks.add_task") as mock_add_task, \
         patch("pathlib.Path.is_file", return_value=False) as mock_is_file, \
         patch("src.config.config", {"paths": {"input_dir": "/mock"}}) as mock_cfg:

        # --- Execute ---
        response = client.post("/analysis/", json=request_data)

        # --- Assertions ---
        # 1. Response Status and Body (API should return 404)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert response_data["detail"] == f"Input file not found: {filename}"

        # 2. is_file check from endpoint was called
        mock_is_file.assert_called_once()

        # 3. Background task should NOT have been scheduled
        mock_add_task.assert_not_called()
        mock_run_pipeline.assert_not_awaited()
        mock_run_pipeline.assert_not_called()

        # 4. Verify config was accessed for input directory
        assert mock_cfg["paths"]["input_dir"] == "/mock"

# Potential future tests:
# - Test validation of filename format (e.g., prevent path traversal)
# - Test behaviour when config cannot be loaded (though this might be better in config tests)
