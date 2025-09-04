"""
tests/utils/test_helpers.py

This module contains comprehensive unit tests for utility functions defined in `src/utils/helpers.py`.

Tests verify the behavior of JSON, YAML, and DataFrame utility functions under various conditions,
including file I/O operations, error handling, and data integrity validation.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Check if openpyxl is available for Excel tests
try:
    import openpyxl

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from src.utils.helpers import (
    append_json_line,
    load_dataframe,
    load_json,
    load_yaml,
    save_dataframe,
    save_json,
    save_yaml,
)

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
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
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
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
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
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
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
        "timestamp": "2024-01-01T12:00:00Z",
    }

    append_json_line(data_to_append, output_file)

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
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


# --- Tests for save_json ---


class TestSaveJson:
    """Test the save_json function."""

    def test_save_json_basic_dict(self, tmp_path):
        """Test saving a basic dictionary to JSON."""
        test_data = {"name": "test", "value": 42, "active": True}
        file_path = tmp_path / "test.json"

        save_json(test_data, file_path)

        # Verify file exists and content is correct
        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_save_json_list_data(self, tmp_path):
        """Test saving a list to JSON."""
        test_data = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
        file_path = tmp_path / "list.json"

        save_json(test_data, file_path)

        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_save_json_nested_structure(self, tmp_path):
        """Test saving nested data structures."""
        test_data = {
            "users": [
                {"id": 1, "profile": {"name": "Alice", "settings": {"theme": "dark"}}},
                {"id": 2, "profile": {"name": "Bob", "settings": {"theme": "light"}}},
            ],
            "metadata": {"version": "1.0", "created": "2024-01-01"},
        }
        file_path = tmp_path / "nested.json"

        save_json(test_data, file_path)

        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_save_json_creates_directory(self, tmp_path):
        """Test that save_json creates parent directories."""
        file_path = tmp_path / "subdir" / "deep" / "test.json"
        test_data = {"test": "data"}

        # Directory should not exist initially
        assert not file_path.parent.exists()

        save_json(test_data, file_path)

        # Directory and file should now exist
        assert file_path.parent.exists()
        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_save_json_custom_indent(self, tmp_path):
        """Test save_json with custom indentation."""
        test_data = {"key": "value", "nested": {"inner": "data"}}
        file_path = tmp_path / "indented.json"

        save_json(test_data, file_path, indent=2)

        # Read raw content to check indentation
        content = file_path.read_text(encoding="utf-8")
        # With indent=2, nested objects should have 2-space indentation
        assert '  "key":' in content
        assert '    "inner":' in content

    def test_save_json_path_object(self, tmp_path):
        """Test save_json accepts Path objects."""
        test_data = {"path": "test"}
        file_path = Path(tmp_path) / "path_test.json"

        save_json(test_data, file_path)

        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_save_json_unicode_content(self, tmp_path):
        """Test save_json handles Unicode content correctly."""
        test_data = {"message": "Hello 世界", "emoji": "🚀", "accents": "café"}
        file_path = tmp_path / "unicode.json"

        save_json(test_data, file_path)

        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data


# --- Tests for load_json ---


class TestLoadJson:
    """Test the load_json function."""

    def test_load_json_basic_dict(self, tmp_path):
        """Test loading a basic JSON dictionary."""
        test_data = {"name": "test", "value": 42, "active": True}
        file_path = tmp_path / "test.json"

        # Create test file
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(test_data, f)

        loaded_data = load_json(file_path)
        assert loaded_data == test_data

    def test_load_json_list_data(self, tmp_path):
        """Test loading JSON list data."""
        test_data = [1, 2, 3, {"nested": "value"}]
        file_path = tmp_path / "list.json"

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(test_data, f)

        loaded_data = load_json(file_path)
        assert loaded_data == test_data

    def test_load_json_file_not_found(self, tmp_path):
        """Test load_json raises FileNotFoundError for non-existent files."""
        file_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_json(file_path)

    def test_load_json_invalid_json(self, tmp_path):
        """Test load_json raises JSONDecodeError for invalid JSON."""
        file_path = tmp_path / "invalid.json"

        # Write invalid JSON content
        file_path.write_text("{ invalid json content", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_json(file_path)

    def test_load_json_empty_file(self, tmp_path):
        """Test load_json handles empty files."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_json(file_path)

    def test_load_json_path_object(self, tmp_path):
        """Test load_json accepts Path objects."""
        test_data = {"path": "test"}
        file_path = Path(tmp_path) / "path_test.json"

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(test_data, f)

        loaded_data = load_json(file_path)
        assert loaded_data == test_data

    def test_load_json_unicode_content(self, tmp_path):
        """Test load_json handles Unicode content correctly."""
        test_data = {"message": "Hello 世界", "emoji": "🚀", "accents": "café"}
        file_path = tmp_path / "unicode.json"

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)

        loaded_data = load_json(file_path)
        assert loaded_data == test_data


# --- Tests for save_yaml ---


class TestSaveYaml:
    """Test the save_yaml function."""

    def test_save_yaml_basic_dict(self, tmp_path):
        """Test saving a basic dictionary to YAML."""
        test_data = {"name": "test", "value": 42, "active": True}
        file_path = tmp_path / "test.yaml"

        save_yaml(test_data, file_path)

        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == test_data

    def test_save_yaml_nested_structure(self, tmp_path):
        """Test saving nested YAML structures."""
        test_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"user": "admin", "password": "secret"},
            },
            "features": ["auth", "logging", "monitoring"],
        }
        file_path = tmp_path / "config.yaml"

        save_yaml(test_data, file_path)

        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == test_data

    def test_save_yaml_creates_directory(self, tmp_path):
        """Test that save_yaml creates parent directories."""
        file_path = tmp_path / "config" / "app.yaml"
        test_data = {"app": "test"}

        assert not file_path.parent.exists()

        save_yaml(test_data, file_path)

        assert file_path.parent.exists()
        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == test_data

    def test_save_yaml_path_object(self, tmp_path):
        """Test save_yaml accepts Path objects."""
        test_data = {"path": "test"}
        file_path = Path(tmp_path) / "path_test.yaml"

        save_yaml(test_data, file_path)

        assert file_path.exists()
        with file_path.open("r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == test_data


# --- Tests for load_yaml ---


class TestLoadYaml:
    """Test the load_yaml function."""

    def test_load_yaml_basic_dict(self, tmp_path):
        """Test loading a basic YAML dictionary."""
        test_data = {"name": "test", "value": 42, "active": True}
        file_path = tmp_path / "test.yaml"

        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(test_data, f)

        loaded_data = load_yaml(file_path)
        assert loaded_data == test_data

    def test_load_yaml_complex_structure(self, tmp_path):
        """Test loading complex YAML structures."""
        test_data = {
            "services": {
                "web": {"image": "nginx", "ports": [80, 443]},
                "db": {"image": "postgres", "environment": {"POSTGRES_DB": "app"}},
            }
        }
        file_path = tmp_path / "docker.yaml"

        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(test_data, f)

        loaded_data = load_yaml(file_path)
        assert loaded_data == test_data

    def test_load_yaml_file_not_found(self, tmp_path):
        """Test load_yaml raises FileNotFoundError for non-existent files."""
        file_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_yaml(file_path)

    def test_load_yaml_invalid_yaml(self, tmp_path):
        """Test load_yaml raises YAMLError for invalid YAML."""
        file_path = tmp_path / "invalid.yaml"

        # Write invalid YAML content
        file_path.write_text("key: value\n  invalid: indentation", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            load_yaml(file_path)

    def test_load_yaml_empty_file(self, tmp_path):
        """Test load_yaml handles empty files."""
        file_path = tmp_path / "empty.yaml"
        file_path.write_text("", encoding="utf-8")

        loaded_data = load_yaml(file_path)
        assert loaded_data is None

    def test_load_yaml_path_object(self, tmp_path):
        """Test load_yaml accepts Path objects."""
        test_data = {"path": "test"}
        file_path = Path(tmp_path) / "path_test.yaml"

        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(test_data, f)

        loaded_data = load_yaml(file_path)
        assert loaded_data == test_data


# --- Tests for save_dataframe ---


class TestSaveDataframe:
    """Test the save_dataframe function."""

    def test_save_dataframe_csv_basic(self, tmp_path):
        """Test saving a basic DataFrame to CSV."""
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95.5, 87.2, 92.1],
            }
        )
        file_path = tmp_path / "test.csv"

        save_dataframe(test_df, file_path, "csv")

        assert file_path.exists()
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(loaded_df, test_df)

    @pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
    def test_save_dataframe_excel_basic(self, tmp_path):
        """Test saving a basic DataFrame to Excel."""
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [10.5, 20.3, 15.7],
            }
        )
        file_path = tmp_path / "test.xlsx"

        save_dataframe(test_df, file_path, "excel")

        assert file_path.exists()
        loaded_df = pd.read_excel(file_path)
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_save_dataframe_csv_default(self, tmp_path):
        """Test save_dataframe defaults to CSV when file_type not specified."""
        test_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        file_path = tmp_path / "default.csv"

        save_dataframe(test_df, file_path)  # No file_type specified

        assert file_path.exists()
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_save_dataframe_creates_directory(self, tmp_path):
        """Test that save_dataframe creates parent directories."""
        test_df = pd.DataFrame({"data": [1, 2, 3]})
        file_path = tmp_path / "data" / "output" / "test.csv"

        assert not file_path.parent.exists()

        save_dataframe(test_df, file_path, "csv")

        assert file_path.parent.exists()
        assert file_path.exists()
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_save_dataframe_invalid_file_type(self, tmp_path):
        """Test save_dataframe raises ValueError for invalid file types."""
        test_df = pd.DataFrame({"data": [1, 2, 3]})
        file_path = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="Unsupported file_type. Choose 'csv' or 'excel'"):
            save_dataframe(test_df, file_path, "txt")

    def test_save_dataframe_case_insensitive_file_type(self, tmp_path):
        """Test save_dataframe handles case-insensitive file types."""
        test_df = pd.DataFrame({"data": [1, 2, 3]})

        # Test uppercase CSV
        csv_path = tmp_path / "upper.csv"
        save_dataframe(test_df, csv_path, "CSV")
        assert csv_path.exists()

        # Test uppercase Excel (only if openpyxl is available)
        if OPENPYXL_AVAILABLE:
            excel_path = tmp_path / "upper.xlsx"
            save_dataframe(test_df, excel_path, "EXCEL")
            assert excel_path.exists()

    def test_save_dataframe_path_object(self, tmp_path):
        """Test save_dataframe accepts Path objects."""
        test_df = pd.DataFrame({"path": [1, 2, 3]})
        file_path = Path(tmp_path) / "path_test.csv"

        save_dataframe(test_df, file_path, "csv")

        assert file_path.exists()
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_save_dataframe_empty_dataframe(self, tmp_path):
        """Test saving an empty DataFrame."""
        test_df = pd.DataFrame()
        file_path = tmp_path / "empty.csv"

        save_dataframe(test_df, file_path, "csv")

        assert file_path.exists()
        # Empty DataFrames may cause EmptyDataError when reading
        try:
            loaded_df = pd.read_csv(file_path)
            assert loaded_df.empty
        except pd.errors.EmptyDataError:
            # This is acceptable behavior for completely empty DataFrames
            pass


# --- Tests for load_dataframe ---


class TestLoadDataframe:
    """Test the load_dataframe function."""

    def test_load_dataframe_csv_basic(self, tmp_path):
        """Test loading a basic CSV DataFrame."""
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95.5, 87.2, 92.1],
            }
        )
        file_path = tmp_path / "test.csv"
        test_df.to_csv(file_path, index=False)

        loaded_df = load_dataframe(file_path, "csv")
        pd.testing.assert_frame_equal(loaded_df, test_df)

    @pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
    def test_load_dataframe_excel_basic(self, tmp_path):
        """Test loading a basic Excel DataFrame."""
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [10.5, 20.3, 15.7],
            }
        )
        file_path = tmp_path / "test.xlsx"
        test_df.to_excel(file_path, index=False)

        loaded_df = load_dataframe(file_path, "excel")
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_load_dataframe_csv_default(self, tmp_path):
        """Test load_dataframe defaults to CSV when file_type not specified."""
        test_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        file_path = tmp_path / "default.csv"
        test_df.to_csv(file_path, index=False)

        loaded_df = load_dataframe(file_path)  # No file_type specified
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_load_dataframe_file_not_found(self, tmp_path):
        """Test load_dataframe raises FileNotFoundError for non-existent files."""
        file_path = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            load_dataframe(file_path, "csv")

    def test_load_dataframe_invalid_file_type(self, tmp_path):
        """Test load_dataframe raises ValueError for invalid file types."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("data", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file_type. Choose 'csv' or 'excel'"):
            load_dataframe(file_path, "txt")

    def test_load_dataframe_case_insensitive_file_type(self, tmp_path):
        """Test load_dataframe handles case-insensitive file types."""
        test_df = pd.DataFrame({"data": [1, 2, 3]})

        # Test uppercase CSV
        csv_path = tmp_path / "upper.csv"
        test_df.to_csv(csv_path, index=False)
        loaded_df = load_dataframe(csv_path, "CSV")
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_load_dataframe_path_object(self, tmp_path):
        """Test load_dataframe accepts Path objects."""
        test_df = pd.DataFrame({"path": [1, 2, 3]})
        file_path = Path(tmp_path) / "path_test.csv"
        test_df.to_csv(file_path, index=False)

        loaded_df = load_dataframe(file_path, "csv")
        pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_load_dataframe_empty_csv(self, tmp_path):
        """Test loading an empty CSV file."""
        file_path = tmp_path / "empty.csv"
        file_path.write_text("", encoding="utf-8")

        # pandas should handle empty CSV files, but behavior may vary
        # This tests the actual behavior rather than hardcoding expectations
        try:
            loaded_df = load_dataframe(file_path, "csv")
            assert isinstance(loaded_df, pd.DataFrame)
        except pd.errors.EmptyDataError:
            # This is also acceptable behavior
            pass

    def test_load_dataframe_malformed_csv(self, tmp_path):
        """Test load_dataframe handling of malformed CSV data."""
        file_path = tmp_path / "malformed.csv"
        # Create a malformed CSV with inconsistent columns
        file_path.write_text("col1,col2\nvalue1,value2,extra_value\nvalue3", encoding="utf-8")

        # Should still load but may have NaN values or raise pandas-specific exceptions
        # Test that it either loads or raises a pandas-related exception
        try:
            loaded_df = load_dataframe(file_path, "csv")
            assert isinstance(loaded_df, pd.DataFrame)
        except (pd.errors.ParserError, ValueError):
            # This is acceptable behavior for malformed data
            pass
