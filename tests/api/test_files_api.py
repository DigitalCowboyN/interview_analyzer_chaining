"""
Tests for the API endpoints defined in src/api/routers/files.py.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
import unittest.mock

# Import the FastAPI app to create a TestClient
from src.main import app 
# Import the specific dependency functions to override
from src.api.routers.files import get_output_dir, get_analysis_suffix 
# Keep config import for MOCK_CONFIG definition (or define paths directly)
from src.config import config as app_config 

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
def get_mock_config():
    return MOCK_CONFIG
app.dependency_overrides[app_config] = get_mock_config

# Override the get_analysis_suffix dependency for all tests
app.dependency_overrides[get_analysis_suffix] = lambda: MOCK_ANALYSIS_SUFFIX

# --- Tests Refactored for get_output_dir Dependency Override ---

def test_list_analysis_files_empty():
    """Test GET /files/ when the output directory is empty or does not exist."""
    # Mock the Path object that will be injected by get_output_dir override
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_output_dir_obj.is_dir.return_value = True
    mock_output_dir_obj.glob.return_value = [] # Simulate empty directory

    # Override the dependency to return our mock Path object
    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj

    response = client.get("/files/")

    assert response.status_code == 200
    assert response.json() == {"filenames": []}
    mock_output_dir_obj.is_dir.assert_called_once()
    mock_output_dir_obj.glob.assert_called_once_with(f"*{MOCK_ANALYSIS_SUFFIX}")

    # Clean up only the output dir override
    app.dependency_overrides.pop(get_output_dir)

def test_list_analysis_files_success():
    """Test GET /files/ with multiple analysis files present."""
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_output_dir_obj.is_dir.return_value = True

    # Mock the files returned by glob
    mock_file1 = MagicMock(spec=Path)
    mock_file1.name = f"file_a{MOCK_ANALYSIS_SUFFIX}"
    mock_file1.is_file.return_value = True
    mock_file2 = MagicMock(spec=Path)
    mock_file2.name = f"file_c{MOCK_ANALYSIS_SUFFIX}"
    mock_file2.is_file.return_value = True
    mock_output_dir_obj.glob.return_value = [mock_file2, mock_file1]

    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj

    response = client.get("/files/")

    assert response.status_code == 200
    assert response.json() == {"filenames": [
        f"file_a{MOCK_ANALYSIS_SUFFIX}", 
        f"file_c{MOCK_ANALYSIS_SUFFIX}"
    ]}
    mock_output_dir_obj.is_dir.assert_called_once()
    mock_output_dir_obj.glob.assert_called_once_with(f"*{MOCK_ANALYSIS_SUFFIX}")

    app.dependency_overrides.pop(get_output_dir)

def test_list_analysis_files_dir_not_found():
    """Test GET /files/ when the output directory does not exist."""
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_output_dir_obj.is_dir.return_value = False # Simulate directory not found

    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj

    response = client.get("/files/")

    assert response.status_code == 200
    assert response.json() == {"filenames": []}
    mock_output_dir_obj.is_dir.assert_called_once()
    mock_output_dir_obj.glob.assert_not_called()

    app.dependency_overrides.pop(get_output_dir)


# --- Tests for GET /files/{filename} ---

VALID_FILENAME = f"test_data{MOCK_ANALYSIS_SUFFIX}"
INVALID_SUFFIX_FILENAME = "test_data.txt"
NON_EXISTENT_FILENAME = f"non_existent{MOCK_ANALYSIS_SUFFIX}"

# Sample JSONL content
SAMPLE_CONTENT_LINE_1 = '{"id": 1, "text": "line one"}'
SAMPLE_CONTENT_LINE_2 = '{"id": 2, "text": "line two"}'
SAMPLE_JSONL_CONTENT = f"{SAMPLE_CONTENT_LINE_1}\n{SAMPLE_CONTENT_LINE_2}\n"

INVALID_JSONL_CONTENT = f"{SAMPLE_CONTENT_LINE_1}\nthis is not json\n{SAMPLE_CONTENT_LINE_2}\n"

EXPECTED_CONTENT = [
    {"id": 1, "text": "line one"},
    {"id": 2, "text": "line two"}
]

# REMOVED @patch("pathlib.Path")
def test_get_file_content_success():
    """Test GET /files/{filename} successfully retrieves content."""
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_file_path_obj = MagicMock(spec=Path) # Mock for the specific file path

    # Configure the injected output dir mock
    mock_output_dir_obj.__truediv__.return_value = mock_file_path_obj 

    # Configure the file path mock methods
    mock_file_path_obj.is_file.return_value = True
    mock_file_path_obj.open.return_value = unittest.mock.mock_open(read_data=SAMPLE_JSONL_CONTENT)()

    # Override dependency
    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj

    response = client.get(f"/files/{VALID_FILENAME}")

    assert response.status_code == 200
    assert response.json() == {"content": EXPECTED_CONTENT}

    # Assertions for mocks
    mock_output_dir_obj.__truediv__.assert_called_once_with(VALID_FILENAME)
    mock_file_path_obj.is_file.assert_called_once_with()
    mock_file_path_obj.open.assert_called_once_with('r', encoding='utf-8')

    # Clean up only the output dir override
    app.dependency_overrides.pop(get_output_dir)

# REMOVED @patch("pathlib.Path")
def test_get_file_content_not_found():
    """Test GET /files/{filename} when the file does not exist."""
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_file_path_obj = MagicMock(spec=Path)
    mock_output_dir_obj.__truediv__.return_value = mock_file_path_obj
    mock_file_path_obj.is_file.return_value = False # Simulate file not found

    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj

    response = client.get(f"/files/{NON_EXISTENT_FILENAME}")

    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]

    # Assertions for mocks
    mock_output_dir_obj.__truediv__.assert_called_once_with(NON_EXISTENT_FILENAME)
    mock_file_path_obj.is_file.assert_called_once_with()
    mock_file_path_obj.open.assert_not_called()

    app.dependency_overrides.pop(get_output_dir)

def test_get_file_content_invalid_suffix():
    """Test GET /files/{filename} with an invalid filename suffix."""
    # No get_output_dir override needed, uses global get_analysis_suffix override
    if get_output_dir in app.dependency_overrides:
        app.dependency_overrides.pop(get_output_dir) # Cleanup just in case
    response = client.get(f"/files/{INVALID_SUFFIX_FILENAME}")

    assert response.status_code == 400
    assert "Invalid filename suffix" in response.json()["detail"]

# REMOVED @patch("pathlib.Path")
def test_get_file_content_io_error():
    """Test GET /files/{filename} when an IOError occurs during reading."""
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_file_path_obj = MagicMock(spec=Path)
    mock_output_dir_obj.__truediv__.return_value = mock_file_path_obj
    mock_file_path_obj.is_file.return_value = True
    mock_file_path_obj.open.side_effect = IOError("Disk read error") # Simulate IOError

    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj

    response = client.get(f"/files/{VALID_FILENAME}")

    assert response.status_code == 500
    assert "Error reading file" in response.json()["detail"]

    # Assertions for mocks
    mock_output_dir_obj.__truediv__.assert_called_once_with(VALID_FILENAME)
    mock_file_path_obj.is_file.assert_called_once_with()
    mock_file_path_obj.open.assert_called_once_with('r', encoding='utf-8')

    app.dependency_overrides.pop(get_output_dir)

# REMOVED @patch("pathlib.Path")
def test_get_file_content_invalid_json_line():
    """Test GET /files/{filename} when a file contains invalid JSON lines."""
    mock_output_dir_obj = MagicMock(spec=Path)
    mock_file_path_obj = MagicMock(spec=Path)
    mock_output_dir_obj.__truediv__.return_value = mock_file_path_obj
    mock_file_path_obj.is_file.return_value = True
    # Simulate open with invalid content
    mock_file_path_obj.open.return_value = unittest.mock.mock_open(read_data=INVALID_JSONL_CONTENT)()

    app.dependency_overrides[get_output_dir] = lambda: mock_output_dir_obj

    # Patch logger within this test's scope
    with patch("src.api.routers.files.logger") as mock_logger:
        response = client.get(f"/files/{VALID_FILENAME}")

    assert response.status_code == 200
    assert response.json() == {"content": EXPECTED_CONTENT}

    # Assertions for mocks
    mock_output_dir_obj.__truediv__.assert_called_once_with(VALID_FILENAME)
    mock_file_path_obj.is_file.assert_called_once_with()
    mock_file_path_obj.open.assert_called_once_with('r', encoding='utf-8')
    mock_logger.warning.assert_called_once_with(
        f"Skipping invalid JSON line in {VALID_FILENAME}: this is not json"
    )

    app.dependency_overrides.pop(get_output_dir)

# Clean up both overrides at the end of tests where get_output_dir was overridden
def cleanup_dependencies():
    if get_output_dir in app.dependency_overrides:
        app.dependency_overrides.pop(get_output_dir)
    # No need to pop get_analysis_suffix if it's set globally for the module

    if get_analysis_suffix in app.dependency_overrides:
        app.dependency_overrides.pop(get_analysis_suffix) 