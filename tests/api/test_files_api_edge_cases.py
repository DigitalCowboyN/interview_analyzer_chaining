"""
tests/api/test_files_api_edge_cases.py

Edge case and error path tests for src/api/routers/files.py
Tests configuration errors, OS errors, and processing edge cases.

These tests follow the cardinal rules:
1. Test actual functionality, not mock interactions
2. Use realistic data and scenarios, not hardcoded values
3. Create real files and test actual I/O operations
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


class TestFilesAPIEdgeCases:
    """Edge case tests for comprehensive error path coverage."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_config_missing_get_method(self):
        """Test when config object doesn't have a 'get' method."""

        class BadConfig:
            def __getitem__(self, key):
                return {"output_dir": "/tmp"}

            # Missing 'get' method

        with patch("src.api.routers.files.config", BadConfig()):
            client = TestClient(app)
            response = client.get("/files/")
            assert response.status_code == 500
            assert "Internal server error listing analysis files" in response.json()["detail"]

    def test_config_get_raises_exception(self):
        """Test when config.get() raises an unexpected exception."""

        class BadConfig:
            def get(self, key, default=None):
                raise RuntimeError("Config access error")

        with patch("src.api.routers.files.config", BadConfig()):
            client = TestClient(app)
            response = client.get("/files/")
            assert response.status_code == 500
            assert "Internal server error listing analysis files" in response.json()["detail"]

    def test_list_files_unexpected_error(self, temp_output_dir):
        """Test list_analysis_files when an unexpected error occurs during file processing."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        # Create a file that will cause issues during processing
        problem_file = temp_output_dir / "test_analysis.jsonl"
        problem_file.write_text("valid content")

        # Patch Path.glob to raise an unexpected error
        with patch("src.api.routers.files.config", test_config):
            with patch("pathlib.Path.glob", side_effect=RuntimeError("Unexpected filesystem error")):
                client = TestClient(app)
                response = client.get("/files/")
                assert response.status_code == 500
                assert "Internal server error listing analysis files" in response.json()["detail"]

    def test_get_file_content_json_decode_error(self, temp_output_dir):
        """Test get_file_content when file contains invalid JSON that causes JSONDecodeError."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        # Create a file with content that looks like JSON but isn't valid
        filename = "broken_json_analysis.jsonl"
        file_path = temp_output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('{"sentence_id": 1, "incomplete": \n')  # Broken JSON

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get(f"/files/{filename}")
            # The router skips malformed JSON lines and continues, so should return 200 with empty results
            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == filename
            assert data["results"] == []  # Should be empty since all lines were skipped

    def test_get_file_content_pydantic_validation_error(self, temp_output_dir):
        """Test get_file_content when valid JSON fails Pydantic validation."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        # Create a file with JSON that doesn't match AnalysisResult schema
        filename = "invalid_schema_analysis.jsonl"
        file_path = temp_output_dir / filename
        invalid_data = {
            "sentence_id": "not_an_integer",  # Should be int
            "sentence": 123,  # Should be string
            # Missing required fields
        }
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(invalid_data) + "\n")

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get(f"/files/{filename}")
            # The router skips validation errors and continues, so should return 200 with empty results
            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == filename
            assert data["results"] == []  # Should be empty since validation failed

    def test_get_file_content_os_error_via_permissions(self, temp_output_dir):
        """Test get_file_content when file permissions prevent reading."""
        import platform

        if platform.system() == "Windows":
            pytest.skip("Permission testing not reliable on Windows")

        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "access_error_analysis.jsonl"
        file_path = temp_output_dir / filename
        # Create file with valid data
        valid_data = {"sentence_id": 1, "sequence_order": 0, "sentence": "test sentence", "analysis": "test analysis"}
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_data) + "\n")

        # Remove read permissions to trigger OSError
        try:
            file_path.chmod(0o000)  # No permissions

            with patch("src.api.routers.files.config", test_config):
                client = TestClient(app)
                response = client.get(f"/files/{filename}")

                # Check if we actually get a permission error or if system ignores it
                if response.status_code == 500:
                    assert "Error reading file" in response.json()["detail"]
                else:
                    # If permissions aren't enforced, skip this test
                    pytest.skip("File permissions not enforced in this environment")

        finally:
            # Restore permissions for cleanup
            try:
                file_path.chmod(0o644)
            except (OSError, FileNotFoundError):
                pass

    def test_get_file_content_handles_processing_errors_gracefully(self, temp_output_dir):
        """Test that get_file_content handles various processing errors by skipping problematic lines."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "mixed_content_analysis.jsonl"
        file_path = temp_output_dir / filename

        # Create file with mixed content: valid, invalid JSON, and validation errors
        with open(file_path, "w", encoding="utf-8") as f:
            # Valid line
            f.write(
                json.dumps(
                    {"sentence_id": 1, "sequence_order": 0, "sentence": "Valid sentence", "analysis": "Valid analysis"}
                )
                + "\n"
            )

            # Invalid JSON line
            f.write('{"sentence_id": 2, "broken": json\n')

            # Valid JSON but fails validation (missing required fields)
            f.write(json.dumps({"sentence_id": 3, "sentence": "Missing sequence_order"}) + "\n")

            # Another valid line
            f.write(
                json.dumps(
                    {
                        "sentence_id": 4,
                        "sequence_order": 1,
                        "sentence": "Another valid sentence",
                        "analysis": "Another valid analysis",
                    }
                )
                + "\n"
            )

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get(f"/files/{filename}")

            # Should succeed but only return valid entries
            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == filename
            assert len(data["results"]) == 2  # Only the 2 valid entries
            assert data["results"][0]["sentence_id"] == 1
            assert data["results"][1]["sentence_id"] == 4

    def test_get_specific_sentence_json_decode_error(self, temp_output_dir):
        """Test get_specific_sentence when file contains invalid JSON."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "broken_json_analysis.jsonl"
        file_path = temp_output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('{"sentence_id": 1, "incomplete": \n')  # Broken JSON

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get(f"/files/{filename}/sentences/1")
            # The router skips malformed JSON lines, so sentence won't be found
            assert response.status_code == 404
            assert "not found in file" in response.json()["detail"]

    def test_get_specific_sentence_pydantic_validation_error_for_target(self, temp_output_dir):
        """Test get_specific_sentence when target sentence has validation error."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "invalid_target_analysis.jsonl"
        file_path = temp_output_dir / filename
        # Create a valid sentence first, then invalid target sentence
        valid_data = {"sentence_id": 0, "sentence": "Valid sentence", "analysis": "Valid analysis"}
        invalid_data = {
            "sentence_id": 1,
            "sentence": 123,
            "analysis": "Invalid sentence type",
        }  # sentence should be string
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_data) + "\n")
            f.write(json.dumps(invalid_data) + "\n")

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get(f"/files/{filename}/sentences/1")
            # When the target sentence has validation error, it should return 500
            assert response.status_code == 500
            assert "Data validation error for sentence 1" in response.json()["detail"]

    def test_get_specific_sentence_os_error_via_permissions(self, temp_output_dir):
        """Test get_specific_sentence when file permissions prevent reading."""
        import platform

        if platform.system() == "Windows":
            pytest.skip("Permission testing not reliable on Windows")

        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "access_error_analysis.jsonl"
        file_path = temp_output_dir / filename
        # Create file with valid data
        valid_data = {"sentence_id": 1, "sequence_order": 0, "sentence": "test sentence", "analysis": "test analysis"}
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_data) + "\n")

        # Remove read permissions to trigger OSError
        try:
            file_path.chmod(0o000)  # No permissions

            with patch("src.api.routers.files.config", test_config):
                client = TestClient(app)
                response = client.get(f"/files/{filename}/sentences/1")

                # Check if we actually get a permission error or if system ignores it
                if response.status_code == 500:
                    assert "Error reading file" in response.json()["detail"]
                else:
                    # If permissions aren't enforced, skip this test
                    pytest.skip("File permissions not enforced in this environment")

        finally:
            # Restore permissions for cleanup
            try:
                file_path.chmod(0o644)
            except (OSError, FileNotFoundError):
                pass

    def test_get_specific_sentence_handles_processing_errors_gracefully(self, temp_output_dir):
        """Test that get_specific_sentence handles various processing errors by skipping problematic lines."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "mixed_content_analysis.jsonl"
        file_path = temp_output_dir / filename

        # Create file with mixed content where target sentence comes after problematic lines
        with open(file_path, "w", encoding="utf-8") as f:
            # Invalid JSON line
            f.write('{"sentence_id": 1, "broken": json\n')

            # Valid JSON but fails validation (missing required fields)
            f.write(json.dumps({"sentence_id": 2, "sentence": "Missing sequence_order"}) + "\n")

            # Target sentence (valid)
            f.write(
                json.dumps(
                    {
                        "sentence_id": 3,
                        "sequence_order": 2,
                        "sentence": "Target sentence",
                        "analysis": "Target analysis",
                    }
                )
                + "\n"
            )

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get(f"/files/{filename}/sentences/3")

            # Should find the valid sentence despite problematic lines before it
            assert response.status_code == 200
            data = response.json()
            assert data["sentence_id"] == 3
            assert data["sentence"] == "Target sentence"
            assert data["analysis"] == "Target analysis"
