"""
Tests for the API endpoints defined in src/api/routers/files.py.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient
from pathlib import Path
import unittest.mock
import json

# Import the FastAPI app to create a TestClient
from src.main import app 
# Import the specific dependency functions to override
# from src.api.routers.files import get_output_dir, get_analysis_suffix 
# Keep config import for MOCK_CONFIG definition (or define paths directly)
# from src.config import config as app_config 
from src.api.schemas import FileContentResponse, AnalysisResult, FileListResponse # Import response models

client = TestClient(app)

# --- Mock Configuration Definition (remains the same) ---
MOCK_OUTPUT_DIR = "/mock/output/dir"
MOCK_ANALYSIS_SUFFIX = "_analysis.jsonl"
MOCK_CONFIG = {
    "paths": {
        "output_dir": MOCK_OUTPUT_DIR,
        "analysis_suffix": MOCK_ANALYSIS_SUFFIX
    }
}

# Override the app_config dependency (remains the same)
# def get_mock_config():
#     return MOCK_CONFIG
# app.dependency_overrides[app_config] = get_mock_config

# Override the get_analysis_suffix dependency for all tests
# app.dependency_overrides[get_analysis_suffix] = lambda: MOCK_ANALYSIS_SUFFIX

# --- Tests Refactored for get_output_dir Dependency Override ---

def test_list_analysis_files_empty():
    """Test GET /files/ when the output directory is empty."""
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath: # Patch constructor

        # Configure the mock instance returned for Path(MOCK_OUTPUT_DIR)
        mock_dir_path_instance = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path_instance # Simplest case: assume only Path(dir) is called

        mock_dir_path_instance.is_dir.return_value = True
        mock_dir_path_instance.glob.return_value = [] # Simulate empty directory

        response = client.get("/files/")

        assert response.status_code == 200
        assert response.json() == {"filenames": []}
        
        # Assert Path was called with the correct dir string
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        # Assert methods were called on the instance returned by Path()
        mock_dir_path_instance.is_dir.assert_called_once()
        mock_dir_path_instance.glob.assert_called_once_with(f"*{MOCK_ANALYSIS_SUFFIX}")

def test_list_analysis_files_success():
    """Test GET /files/ with multiple analysis files present."""
    # Mock files to be returned by glob
    mock_file1 = MagicMock(spec=Path)
    mock_file1.name = f"file_a{MOCK_ANALYSIS_SUFFIX}"
    mock_file1.is_file.return_value = True
    mock_file2 = MagicMock(spec=Path)
    mock_file2.name = f"file_c{MOCK_ANALYSIS_SUFFIX}"
    mock_file2.is_file.return_value = True

    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:

        mock_dir_path_instance = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path_instance

        mock_dir_path_instance.is_dir.return_value = True
        # Return the mock file objects from glob
        mock_dir_path_instance.glob.return_value = [mock_file2, mock_file1] # Unsorted

        response = client.get("/files/")

        assert response.status_code == 200
        # Expect sorted results from the endpoint
        assert response.json() == {"filenames": [
            f"file_a{MOCK_ANALYSIS_SUFFIX}", 
            f"file_c{MOCK_ANALYSIS_SUFFIX}"
        ]}
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path_instance.is_dir.assert_called_once()
        mock_dir_path_instance.glob.assert_called_once_with(f"*{MOCK_ANALYSIS_SUFFIX}")

def test_list_analysis_files_dir_not_found():
    """Test GET /files/ when the output directory does not exist."""
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:
        
        mock_dir_path_instance = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path_instance
        
        # Simulate directory not found
        mock_dir_path_instance.is_dir.return_value = False
        
        response = client.get("/files/")

        assert response.status_code == 200
        assert response.json() == {"filenames": []}
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path_instance.is_dir.assert_called_once()
        # Glob should not be called if is_dir is false
        mock_dir_path_instance.glob.assert_not_called()

def test_list_files_internal_error():
    """Test GET /files/ when Path.glob raises an OS error."""
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:

        mock_dir_path_instance = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path_instance
        
        mock_dir_path_instance.is_dir.return_value = True
        # Simulate OS error during glob
        mock_dir_path_instance.glob.side_effect = OSError("Disk read error")

        response = client.get("/files/")

        assert response.status_code == 500
        assert "Internal server error listing analysis files" in response.json().get("detail", "")
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path_instance.is_dir.assert_called_once()
        mock_dir_path_instance.glob.assert_called_once_with(f"*{MOCK_ANALYSIS_SUFFIX}")


# --- Tests for GET /files/{filename} ---

VALID_FILENAME = f"test_data{MOCK_ANALYSIS_SUFFIX}"
# INVALID_SUFFIX_FILENAME = "test_data.txt" # Removed
NON_EXISTENT_FILENAME = f"non_existent{MOCK_ANALYSIS_SUFFIX}"

# Sample JSONL content
SAMPLE_CONTENT_LINE_1 = {"sentence_id": 0, "sequence_order": 0, "sentence": "First.", "analysis": "A"}
SAMPLE_CONTENT_LINE_2 = {"sentence_id": 1, "sequence_order": 1, "sentence": "Second.", "analysis": "B"}
SAMPLE_JSONL_CONTENT = f"{json.dumps(SAMPLE_CONTENT_LINE_1)}\n{json.dumps(SAMPLE_CONTENT_LINE_2)}\n"

INVALID_JSONL_CONTENT = f"{json.dumps(SAMPLE_CONTENT_LINE_1)}\nthis is not json\n{json.dumps(SAMPLE_CONTENT_LINE_2)}\n"

# Update expected content structure based on FileContentResponse
# Add sequence_order and all optional None fields to the expected result
EXPECTED_CONTENT_RESULT = [
    {
        "sentence_id": 0, "sequence_order": 0, "sentence": "First.", "analysis": "A",
        "function_type": None, "structure_type": None, "purpose": None,
        "topic_level_1": None, "topic_level_3": None,
        "overall_keywords": None, "domain_keywords": None
    },
    {
        "sentence_id": 1, "sequence_order": 1, "sentence": "Second.", "analysis": "B",
        "function_type": None, "structure_type": None, "purpose": None,
        "topic_level_1": None, "topic_level_3": None,
        "overall_keywords": None, "domain_keywords": None
    }
]

def test_get_file_content_success():
    """Test GET /files/{filename} successfully retrieves content."""
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath: # Patch constructor
        
        # Create mock instances for dir and file paths
        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)

        # Configure Path(MOCK_OUTPUT_DIR) to return mock_dir_path
        MockPath.return_value = mock_dir_path 

        # Configure mock_dir_path / filename to return mock_file_path
        mock_dir_path.__truediv__.return_value = mock_file_path

        # Configure mock_file_path for success scenario
        mock_file_path.is_file.return_value = True
        # Use mock_open on the specific mock_file_path instance
        mock_file_path.open = mock_open(read_data=SAMPLE_JSONL_CONTENT)
        
        response = client.get(f"/files/{VALID_FILENAME}")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["filename"] == VALID_FILENAME
        assert response_data["results"] == EXPECTED_CONTENT_RESULT

        # Assertions
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(VALID_FILENAME)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8')

def test_get_file_content_not_found():
    """Test GET /files/{filename} when the file does not exist."""
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:

        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path

        # Simulate file not found
        mock_file_path.is_file.return_value = False
        # Ensure open is mockable even if not called
        mock_file_path.open = MagicMock()

        response = client.get(f"/files/{NON_EXISTENT_FILENAME}")

        assert response.status_code == 404
        assert "Analysis file not found" in response.json()["detail"]
        assert NON_EXISTENT_FILENAME in response.json()["detail"]
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(NON_EXISTENT_FILENAME)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_not_called() # Assert open wasn't called

def test_get_file_content_read_error():
    """Test GET /files/{filename} when an IOError occurs during reading."""
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:

        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path

        mock_file_path.is_file.return_value = True
        # Configure open to raise OSError when iterated
        mock_file_handle = mock_open()()
        mock_file_handle.__iter__.side_effect = OSError("Permission denied")
        mock_file_path.open.return_value = mock_file_handle
        
        response = client.get(f"/files/{VALID_FILENAME}")

        assert response.status_code == 500
        assert "Error reading file" in response.json()["detail"]
        assert VALID_FILENAME in response.json()["detail"]
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(VALID_FILENAME)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8')

def test_get_file_content_malformed_json():
    """Test GET /files/{filename} when a file contains invalid JSON lines."""
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath, \
         patch("src.api.routers.files.logger") as mock_logger:

        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path

        mock_file_path.is_file.return_value = True
        # Use mock_open directly on the instance
        mock_file_path.open = mock_open(read_data=INVALID_JSONL_CONTENT)

        response = client.get(f"/files/{VALID_FILENAME}")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["filename"] == VALID_FILENAME
        # Should skip the bad line and return only the valid ones
        assert response_data["results"] == EXPECTED_CONTENT_RESULT

        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(VALID_FILENAME)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8')
        # Check warning log
        mock_logger.warning.assert_called()
        assert any(f"Skipping malformed JSON line" in call.args[0] and "this is not json" in call.args[0] for call in mock_logger.warning.call_args_list)

# === Tests for GET /files/{filename}/sentences/{sentence_id} ===

def test_get_specific_sentence_success():
    """Tests successfully retrieving a specific sentence analysis."""
    test_filename = "target_analysis.jsonl"
    target_id = 1
    line0 = {"sentence_id": 0, "sequence_order": 0, "sentence": "First.", "analysis": "A"}
    line1 = {"sentence_id": target_id, "sequence_order": 1, "sentence": "Second target.", "analysis": "B", "topic_level_1": "Target Topic"}
    line2 = {"sentence_id": 2, "sequence_order": 2, "sentence": "Third.", "analysis": "C"}
    mock_file_content = f"{json.dumps(line0)}\n{json.dumps(line1)}\n{json.dumps(line2)}\n"

    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:
        
        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path

        mock_file_path.is_file.return_value = True
        mock_file_path.open = mock_open(read_data=mock_file_content)

        response = client.get(f"/files/{test_filename}/sentences/{target_id}")

        assert response.status_code == 200
        response_data = response.json()
        # Check specific fields of the returned AnalysisResult
        assert response_data["sentence_id"] == target_id
        assert response_data["sequence_order"] == 1
        assert response_data["sentence"] == "Second target."
        assert response_data["analysis"] == "B"
        assert response_data["topic_level_1"] == "Target Topic"

        # Verify mocks
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(test_filename)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8')

def test_get_specific_sentence_file_not_found():
    """Tests 404 when the analysis file itself is not found."""
    test_filename = "no_such_file.jsonl"
    target_id = 0
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:
        
        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path
        
        mock_file_path.is_file.return_value = False # Simulate file not found
        mock_file_path.open = MagicMock() # Ensure open is mockable
        
        response = client.get(f"/files/{test_filename}/sentences/{target_id}")

        assert response.status_code == 404
        assert "Analysis file not found" in response.json()["detail"]
        assert test_filename in response.json()["detail"]
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(test_filename)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_not_called()

def test_get_specific_sentence_id_not_found():
    """Tests 404 when the sentence ID is not found within the file."""
    test_filename = "analysis_exists.jsonl"
    target_id = 99 # ID not in the file
    line0 = {"sentence_id": 0, "sequence_order": 0, "sentence": "First.", "analysis": "A"}
    line1 = {"sentence_id": 1, "sequence_order": 1, "sentence": "Second.", "analysis": "B"}
    mock_file_content = f"{json.dumps(line0)}\n{json.dumps(line1)}\n"

    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:
        
        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path
        
        mock_file_path.is_file.return_value = True
        mock_file_path.open = mock_open(read_data=mock_file_content)

        response = client.get(f"/files/{test_filename}/sentences/{target_id}")

        assert response.status_code == 404
        response_data = response.json()
        assert "Sentence ID" in response_data["detail"]
        assert str(target_id) in response_data["detail"]
        assert "not found in file" in response_data["detail"]
        assert test_filename in response_data["detail"]
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(test_filename)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8')

def test_get_specific_sentence_read_error():
    """Tests 500 response when an OS error occurs during file read."""
    test_filename = "read_error_file.jsonl"
    target_id = 0
    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:
        
        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path
        
        mock_file_path.is_file.return_value = True
        # Configure open to raise OSError when iterated
        mock_file_handle = mock_open()()
        mock_file_handle.__iter__.side_effect = OSError("Disk read failed")
        mock_file_path.open.return_value = mock_file_handle

        response = client.get(f"/files/{test_filename}/sentences/{target_id}")

        assert response.status_code == 500
        assert "Error reading file" in response.json()["detail"]
        assert test_filename in response.json()["detail"]
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(test_filename)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8')

def test_get_specific_sentence_skips_malformed():
    """Tests that malformed JSON lines are skipped when searching for an ID."""
    test_filename = "mixed_quality_specific.jsonl"
    target_id = 2
    line0 = {"sentence_id": 0, "sequence_order": 0, "sentence": "First.", "analysis": "A"}
    bad_line1 = "not json { an error"
    line2 = {"sentence_id": target_id, "sequence_order": 1, "sentence": "The target one.", "analysis": "B"}
    bad_line2 = "{\\\"incomplete json\\\"}"
    mock_file_content = f"{json.dumps(line0)}\n{bad_line1}\n{json.dumps(line2)}\n{bad_line2}\n"

    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath, \
         patch("src.api.routers.files.logger") as mock_logger:

        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path
        
        mock_file_path.is_file.return_value = True
        mock_file_path.open = mock_open(read_data=mock_file_content)

        response = client.get(f"/files/{test_filename}/sentences/{target_id}")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["sentence_id"] == target_id
        assert response_data["sequence_order"] == 1
        assert response_data["sentence"] == "The target one."
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(test_filename)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8')
        # Check that *a* warning about skipping a malformed line was logged
        assert any("Skipping malformed JSON line" in call.args[0] for call in mock_logger.warning.call_args_list)

def test_get_specific_sentence_validation_error():
    """Tests 500 if the found sentence data fails Pydantic validation."""
    test_filename = "bad_data.jsonl"
    target_id = 1
    line0 = {"sentence_id": 0, "sequence_order": 0, "sentence": "First.", "analysis": "A"}
    bad_line1 = {"sentence_id": target_id, "sequence_order": 1, "sentence": 12345, "analysis": "B"} # sentence should be str
    mock_file_content = f"{json.dumps(line0)}\n{json.dumps(bad_line1)}\n"

    with patch.dict("src.config.config", MOCK_CONFIG, clear=True), \
         patch("src.api.routers.files.Path") as MockPath:

        mock_dir_path = MagicMock(spec=Path)
        mock_file_path = MagicMock(spec=Path)
        MockPath.return_value = mock_dir_path
        mock_dir_path.__truediv__.return_value = mock_file_path
        
        mock_file_path.is_file.return_value = True
        mock_file_path.open = mock_open(read_data=mock_file_content)

        response = client.get(f"/files/{test_filename}/sentences/{target_id}")

        assert response.status_code == 500
        assert "Data validation error" in response.json()["detail"]
        assert str(target_id) in response.json()["detail"]
        
        MockPath.assert_called_once_with(MOCK_OUTPUT_DIR)
        mock_dir_path.__truediv__.assert_called_once_with(test_filename)
        mock_file_path.is_file.assert_called_once()
        mock_file_path.open.assert_called_once_with('r', encoding='utf-8') 