"""
Tests for the API endpoints defined in src/api/routers/analysis.py.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
import unittest.mock

# Import the FastAPI app and dependencies to override
from src.main import app
# Import all dependency functions used
from src.api.routers.analysis import get_input_dir, get_output_dir, get_map_dir, get_config_dep
# Remove import of original config object if no longer needed elsewhere
# from src.config import config as app_config

client = TestClient(app)

# --- Mock Configuration --- (Can potentially be shared with test_files_api)
MOCK_INPUT_DIR = "/mock/input/dir"
MOCK_OUTPUT_DIR = "/mock/output/dir"
MOCK_MAP_DIR = "/mock/map/dir"
MOCK_CONFIG = {
    "paths": {
        "input_dir": MOCK_INPUT_DIR,
        "output_dir": MOCK_OUTPUT_DIR,
        "map_dir": MOCK_MAP_DIR,
        "analysis_suffix": "_analysis.jsonl" # Needed by other parts potentially
    }
}

# --- Global Dependency Overrides ---
# Remove override for the original config object
# if get_mock_config in app.dependency_overrides:
#     app.dependency_overrides.pop(get_mock_config) # Adjust if key was app_config
# Override the new config dependency function
app.dependency_overrides[get_config_dep] = lambda: MOCK_CONFIG

# --- Test Data ---
VALID_INPUT_FILENAME = "test_transcript.txt"
NON_EXISTENT_INPUT_FILENAME = "non_existent_transcript.txt"

# --- Tests ---

def test_trigger_analysis_success():
    """Test POST /analyze/ successfully triggers analysis for an existing file."""
    mock_input_dir_obj = MagicMock(spec=Path)
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_map_dir_obj = MagicMock(spec=Path)
    mock_input_file_path_obj = MagicMock(spec=Path)

    # Configure mocks
    mock_input_dir_obj.__truediv__.return_value = mock_input_file_path_obj
    mock_input_file_path_obj.is_file.return_value = True # File exists

    # Override path dependencies
    app.dependency_overrides[get_input_dir] = lambda: mock_input_dir_obj
    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj
    app.dependency_overrides[get_map_dir] = lambda: mock_map_dir_obj

    # Patch the .delay() method of the Celery task
    # Target where the task is imported and called from (the router)
    with patch("src.tasks.run_pipeline_for_file.delay") as mock_task_delay:
        # Mock the return value of .delay() to simulate a task ID if needed
        mock_task_delay.return_value = MagicMock(id="mock_task_123")
        
        response = client.post(
            "/analyze/", 
            json={"input_filename": VALID_INPUT_FILENAME}
        )

    assert response.status_code == 202
    assert response.json() == {
        "message": "Analysis task accepted and queued.", # Verify updated message
        "input_filename": VALID_INPUT_FILENAME
        # Optionally check for task_id if API returns it
    }

    # Assertions for mocks
    mock_input_dir_obj.__truediv__.assert_called_once_with(VALID_INPUT_FILENAME)
    mock_input_file_path_obj.is_file.assert_called_once()
    
    # Check that the task's .delay() method was called correctly
    mock_task_delay.assert_called_once_with(
        input_file_path_str=str(mock_input_dir_obj / VALID_INPUT_FILENAME), # Construct expected str path
        output_dir_str=str(mock_output_dir_obj),
        map_dir_str=str(mock_map_dir_obj),
        config_dict=MOCK_CONFIG # Check injected config was passed
    )

    # Clean up overrides
    app.dependency_overrides.pop(get_input_dir)
    app.dependency_overrides.pop(get_output_dir)
    app.dependency_overrides.pop(get_map_dir)

def test_trigger_analysis_file_not_found():
    """Test POST /analyze/ returns 404 if the input file does not exist."""
    mock_input_dir_obj = MagicMock()
    mock_input_file_path_obj = MagicMock()
    # Add mocks for other path dependencies, even if not used by main logic
    mock_output_dir_obj = MagicMock()
    mock_map_dir_obj = MagicMock()

    # Configure mocks for the input path check
    mock_input_dir_obj.__truediv__.return_value = mock_input_file_path_obj
    mock_input_file_path_obj.is_file.return_value = False # File does NOT exist

    # Override *all* path dependencies declared by the endpoint
    app.dependency_overrides[get_input_dir] = lambda: mock_input_dir_obj
    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj
    app.dependency_overrides[get_map_dir] = lambda: mock_map_dir_obj

    # Patch the .delay() method
    with patch("src.tasks.run_pipeline_for_file.delay") as mock_task_delay:
        response = client.post(
            "/analyze/",
            json={"input_filename": NON_EXISTENT_INPUT_FILENAME}
        )

    # Assertions (remain the same)
    assert response.status_code == 404
    assert "Input file not found" in response.json()["detail"]
    mock_input_dir_obj.__truediv__.assert_called_once_with(NON_EXISTENT_INPUT_FILENAME)
    mock_input_file_path_obj.is_file.assert_called_once()
    mock_task_delay.assert_not_called() # Ensure task wasn't sent

    # Clean up *all* overrides added in this test
    app.dependency_overrides.pop(get_input_dir)
    app.dependency_overrides.pop(get_output_dir)
    app.dependency_overrides.pop(get_map_dir) 