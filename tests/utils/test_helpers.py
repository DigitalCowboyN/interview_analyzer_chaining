"""
tests/utils/test_helpers.py

This module contains unit tests for utility functions defined in `src/utils/helpers.py`,
currently focusing on the `append_json_line` function.

Tests verify the behavior of `append_json_line` under various conditions,
including writing to new and existing files, handling different dictionary contents,
and managing non-serializable data, using pytest's `tmp_path` fixture for file operations.
"""

import pytest
import json
from pathlib import Path
from src.utils.helpers import append_json_line # Assuming this function will exist

# --- Tests for append_json_line ---

def test_append_json_line_new_file(tmp_path):
    """
    Test appending a JSON line to a non-existent file.

    Verifies that `append_json_line` creates the file and writes the
    correct JSON data as the first line when the target file does not initially exist.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    output_file = tmp_path / "output.jsonl"
    data_to_append = {"id": 1, "message": "hello"}
    
    # Ensure file doesn't exist initially
    assert not output_file.exists()

    append_json_line(data_to_append, output_file)

    # Check file exists and content is correct
    assert output_file.exists()
    lines = output_file.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) == 1
    assert json.loads(lines[0]) == data_to_append

def test_append_json_line_existing_file(tmp_path):
    """
    Test appending multiple JSON lines to an existing file.

    Verifies that `append_json_line` correctly appends new JSON data
    as subsequent lines to a file that already contains JSON lines.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    output_file = tmp_path / "output.jsonl"
    data1 = {"id": 1, "message": "first line"}
    data2 = {"id": 2, "message": "second line"}
    
    # Create the file with the first line
    append_json_line(data1, output_file)
    
    # Append the second line
    append_json_line(data2, output_file)

    # Check file content
    assert output_file.exists()
    lines = output_file.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) == 2
    assert json.loads(lines[0]) == data1
    assert json.loads(lines[1]) == data2

def test_append_json_line_empty_dict(tmp_path):
    """
    Test appending an empty dictionary as a JSON line.

    Verifies that an empty dictionary is correctly serialized to '{}' and
    written to the file.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    output_file = tmp_path / "output.jsonl"
    data_to_append = {}
    
    append_json_line(data_to_append, output_file)

    assert output_file.exists()
    lines = output_file.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) == 1
    assert json.loads(lines[0]) == data_to_append

def test_append_json_line_complex_dict(tmp_path):
    """
    Test appending a dictionary with nested structures and various data types.

    Verifies that complex but valid JSON structures (nested dicts, lists, booleans)
    are correctly serialized and written.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    output_file = tmp_path / "output.jsonl"
    data_to_append = {
        "id": 10, 
        "data": {"values": [1, 2, 3], "flag": True}, 
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    append_json_line(data_to_append, output_file)

    assert output_file.exists()
    lines = output_file.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) == 1
    assert json.loads(lines[0]) == data_to_append

# Potential edge case: What if data is not JSON serializable?
# The json.dumps call inside append_json_line should raise a TypeError.
def test_append_json_line_non_serializable(tmp_path):
    """
    Test that attempting to append non-JSON serializable data raises TypeError.

    Verifies that `append_json_line` correctly propagates the `TypeError` from
    `json.dumps` when provided with data containing non-serializable types (like sets).

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    output_file = tmp_path / "output.jsonl"
    # Sets are not directly JSON serializable
    data_to_append = {"id": 1, "value": {1, 2, 3}} 
    
    with pytest.raises(TypeError):
        append_json_line(data_to_append, output_file)
    
    # File should ideally not be created or be empty if error happens before write
    # Depending on implementation detail (e.g., if opened in 'a' mode first)
    # assert not output_file.exists() or output_file.read_text() == "" 