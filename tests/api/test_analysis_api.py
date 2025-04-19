"""
Tests for the API endpoints defined in src/api/routers/analysis.py.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status, BackgroundTasks
from pathlib import Path

# Import the FastAPI app to create a TestClient
from src.main import app 
# Import schemas used in requests/responses
from src.api.schemas import AnalysisTriggerRequest, AnalysisTriggerResponse

client = TestClient(app)

# Mock config data (similar to test_files_api.py)
MOCK_INPUT_DIR = "/mock/input/dir"
MOCK_CONFIG = {
    "paths": {
        "input_dir": MOCK_INPUT_DIR,
        # Add other paths if needed by run_pipeline or its callees
        "output_dir": "/mock/output/dir", 
        "map_dir": "/mock/map/dir",
        "analysis_suffix": "_analysis.jsonl"
    }
}

VALID_INPUT_FILENAME = "interview_transcript.txt"
NON_EXISTENT_FILENAME = "ghost.txt"
INVALID_FILENAME_FORMAT = "../sneaky_file.txt"

# Test functions will go here

def test_trigger_analysis_success():
    """Test POST /analysis/ successfully schedules analysis."""
    # Patch config, Path, run_pipeline (which is now called by BackgroundTasks)
    # AND BackgroundTasks itself to check add_task
    with patch.dict("src.config.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.analysis.Path") as MockPath, \
         patch("src.pipeline.run_pipeline") as mock_run_pipeline, \
         patch("fastapi.BackgroundTasks.add_task") as mock_add_task: # Patch add_task
        
        # Configure Path mocks
        mock_input_dir_obj = MagicMock(spec=Path)
        mock_input_file_path_obj = MagicMock(spec=Path)
        MockPath.return_value = mock_input_dir_obj
        mock_input_dir_obj.__truediv__.return_value = mock_input_file_path_obj
        mock_input_file_path_obj.is_file.return_value = True

        # --- Act --- 
        response = client.post(
            "/analysis/",
            json={"input_filename": VALID_INPUT_FILENAME}
        )

        # --- Assert --- 
        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json() == {
            "message": "Analysis task accepted and scheduled to run in background.",
            "input_filename": VALID_INPUT_FILENAME
        }
        
        # Check Path calls
        MockPath.assert_called_once_with(MOCK_INPUT_DIR)
        mock_input_dir_obj.__truediv__.assert_called_once_with(VALID_INPUT_FILENAME)
        mock_input_file_path_obj.is_file.assert_called_once()
        
        # Check that BackgroundTasks.add_task was called correctly
        # Note: run_pipeline mock is checked implicitly via add_task args
        mock_add_task.assert_called_once_with(
            mock_run_pipeline, # Check the function itself was passed
            input_dir=MOCK_INPUT_DIR, 
            specific_file=VALID_INPUT_FILENAME
        )
        # Ensure the pipeline itself wasn't awaited directly
        mock_run_pipeline.assert_not_awaited()
        mock_run_pipeline.assert_not_called() # add_task holds the ref, doesn't call it here

def test_trigger_analysis_invalid_filename():
    """Test POST /analysis/ with invalid filename format (400)."""
    # Only patch config internal dict
    with patch.dict("src.config.config.config", MOCK_CONFIG, clear=True):
        response = client.post(
            "/analysis/",
            json={"input_filename": INVALID_FILENAME_FORMAT}
        )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid input filename format" in response.json()["detail"]

def test_trigger_analysis_file_not_found():
    """Test POST /analysis/ when input file does not exist (404)."""
    # Patch config, Path, run_pipeline (not called), and add_task (not called)
    with patch.dict("src.config.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.analysis.Path") as MockPath, \
         patch("src.pipeline.run_pipeline") as mock_run_pipeline, \
         patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        
        # Configure Path mocks for file not found
        mock_input_dir_obj = MagicMock(spec=Path)
        mock_input_file_path_obj = MagicMock(spec=Path)
        MockPath.return_value = mock_input_dir_obj
        mock_input_dir_obj.__truediv__.return_value = mock_input_file_path_obj
        mock_input_file_path_obj.is_file.return_value = False

        response = client.post(
            "/analysis/",
            json={"input_filename": NON_EXISTENT_FILENAME}
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Input file not found" in response.json()["detail"]
        
        # Check Path calls
        MockPath.assert_called_once_with(MOCK_INPUT_DIR)
        mock_input_dir_obj.__truediv__.assert_called_once_with(NON_EXISTENT_FILENAME)
        mock_input_file_path_obj.is_file.assert_called_once()
        
        # Ensure add_task and run_pipeline were NOT called
        mock_run_pipeline.assert_not_called()
        mock_add_task.assert_not_called()

def test_trigger_analysis_pipeline_error_still_accepts():
    """Test POST /analysis/ still returns 202 even if background task would fail."""
    # Test that the API call itself succeeds (202) even if run_pipeline would error.
    # The error happens in the background, not affecting the initial response.
    pipeline_error_message = "Something broke in the pipeline!"
    # Patch config, Path, run_pipeline (to configure side_effect for add_task) and add_task
    with patch.dict("src.config.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.analysis.Path") as MockPath, \
         patch("src.pipeline.run_pipeline") as mock_run_pipeline, \
         patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        
        # Configure Path mocks for file exists
        mock_input_dir_obj = MagicMock(spec=Path)
        mock_input_file_path_obj = MagicMock(spec=Path)
        MockPath.return_value = mock_input_dir_obj
        mock_input_dir_obj.__truediv__.return_value = mock_input_file_path_obj
        mock_input_file_path_obj.is_file.return_value = True
        
        # Configure run_pipeline side effect (won't be called directly)
        # This error would happen when the background task runner executes it.
        mock_run_pipeline.side_effect = Exception(pipeline_error_message)

        response = client.post(
            "/analysis/",
            json={"input_filename": VALID_INPUT_FILENAME}
        )

        # API should still accept the request
        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json() == {
            "message": "Analysis task accepted and scheduled to run in background.",
            "input_filename": VALID_INPUT_FILENAME
        }

        # Check Path calls were made
        MockPath.assert_called_once_with(MOCK_INPUT_DIR)
        mock_input_dir_obj.__truediv__.assert_called_once_with(VALID_INPUT_FILENAME)
        mock_input_file_path_obj.is_file.assert_called_once()
        
        # Check add_task was called correctly, even though run_pipeline would fail later
        mock_add_task.assert_called_once_with(
            mock_run_pipeline, # The function that *would* raise error
            input_dir=MOCK_INPUT_DIR, 
            specific_file=VALID_INPUT_FILENAME
        )
        # Ensure run_pipeline wasn't called or awaited directly by the endpoint
        mock_run_pipeline.assert_not_awaited()
        mock_run_pipeline.assert_not_called() 