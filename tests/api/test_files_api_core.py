"""
tests/api/test_files_api_core.py

Core integration tests for src/api/routers/files.py that test real functionality.

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


class TestFilesAPI:
    """Core integration tests for the files API router with real file operations."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_analysis_data(self):
        """Provide realistic analysis data for testing."""
        return [
            {
                "sentence_id": 0,
                "sequence_order": 0,
                "sentence": "This is the first sentence of the interview.",
                "analysis": "Opening statement providing context",
                "function_type": "introduction",
                "structure_type": "declarative",
                "purpose": "context_setting",
                "topic_level_1": "interview_opening",
                "topic_level_3": "contextual_introduction",
                "overall_keywords": ["interview", "first", "context"],
                "domain_keywords": ["interview_process"],
            },
            {
                "sentence_id": 1,
                "sequence_order": 1,
                "sentence": "Can you tell me about your experience with Python?",
                "analysis": "Technical question about programming experience",
                "function_type": "question",
                "structure_type": "interrogative",
                "purpose": "information_gathering",
                "topic_level_1": "technical_skills",
                "topic_level_3": "programming_languages",
                "overall_keywords": ["experience", "Python", "programming"],
                "domain_keywords": ["python", "technical_skills"],
            },
            {
                "sentence_id": 2,
                "sequence_order": 2,
                "sentence": "I have been working with Python for over 5 years.",
                "analysis": "Response providing quantified experience",
                "function_type": "response",
                "structure_type": "declarative",
                "purpose": "information_provision",
                "topic_level_1": "experience_description",
                "topic_level_3": "quantified_experience",
                "overall_keywords": ["working", "Python", "years"],
                "domain_keywords": ["python", "experience_level"],
            },
        ]

    @pytest.fixture
    def create_test_files(self, temp_output_dir, sample_analysis_data):
        """Create realistic test analysis files."""
        # Create multiple analysis files with different content
        files_created = {}

        # File 1: Complete analysis file
        file1_name = "interview_001_analysis.jsonl"
        file1_path = temp_output_dir / file1_name
        with open(file1_path, "w", encoding="utf-8") as f:
            for item in sample_analysis_data:
                f.write(json.dumps(item) + "\n")
        files_created[file1_name] = sample_analysis_data

        # File 2: Single sentence file
        file2_name = "interview_002_analysis.jsonl"
        file2_path = temp_output_dir / file2_name
        single_item = {
            "sentence_id": 10,
            "sequence_order": 0,
            "sentence": "Thank you for your time today.",
            "analysis": "Closing statement expressing gratitude",
            "function_type": "closing",
            "structure_type": "declarative",
            "purpose": "relationship_maintenance",
        }
        with open(file2_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(single_item) + "\n")
        files_created[file2_name] = [single_item]

        # File 3: Mixed quality file (some malformed JSON)
        file3_name = "interview_003_analysis.jsonl"
        file3_path = temp_output_dir / file3_name
        valid_item = {
            "sentence_id": 20,
            "sequence_order": 0,
            "sentence": "What are your career goals?",
            "analysis": "Future-oriented question",
            "function_type": "question",
        }
        with open(file3_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_item) + "\n")
            f.write("invalid json line here\n")
            f.write(
                json.dumps(
                    {
                        "sentence_id": 21,
                        "sequence_order": 1,
                        "sentence": "I want to become a senior developer.",
                        "analysis": "Career aspiration response",
                        "function_type": "response",
                    }
                )
                + "\n"
            )
        files_created[file3_name] = [
            valid_item,
            {
                "sentence_id": 21,
                "sequence_order": 1,
                "sentence": "I want to become a senior developer.",
                "analysis": "Career aspiration response",
                "function_type": "response",
            },
        ]

        # Create a non-analysis file (should be ignored)
        non_analysis_file = temp_output_dir / "readme.txt"
        with open(non_analysis_file, "w") as f:
            f.write("This is not an analysis file")

        return files_created

    @pytest.fixture
    def configured_client(self, temp_output_dir):
        """Create a test client with real configuration pointing to temp directory."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        # Patch the config import in the files router
        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            yield client

    def test_list_analysis_files_success(self, configured_client, create_test_files):
        """Test GET /files/ successfully lists analysis files."""
        response = configured_client.get("/files/")

        assert response.status_code == 200
        data = response.json()

        # Should return sorted list of analysis files
        expected_files = sorted(
            ["interview_001_analysis.jsonl", "interview_002_analysis.jsonl", "interview_003_analysis.jsonl"]
        )
        assert data["filenames"] == expected_files

    def test_list_analysis_files_empty_directory(self, temp_output_dir):
        """Test GET /files/ with empty directory."""
        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get("/files/")
            assert response.status_code == 200
            assert response.json() == {"filenames": []}

    def test_list_analysis_files_nonexistent_directory(self, temp_output_dir):
        """Test GET /files/ when output directory doesn't exist."""
        nonexistent_dir = temp_output_dir / "does_not_exist"

        test_config = {"paths": {"output_dir": str(nonexistent_dir), "analysis_suffix": "_analysis.jsonl"}}

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get("/files/")
            assert response.status_code == 200
            assert response.json() == {"filenames": []}

    def test_get_file_content_success(self, configured_client, create_test_files, sample_analysis_data):
        """Test GET /files/{filename} successfully retrieves file content."""
        filename = "interview_001_analysis.jsonl"
        response = configured_client.get(f"/files/{filename}")

        assert response.status_code == 200
        data = response.json()

        assert data["filename"] == filename
        assert len(data["results"]) == 3

        # Verify the content matches our sample data (with proper Pydantic defaults)
        for i, result in enumerate(data["results"]):
            expected = sample_analysis_data[i]
            assert result["sentence_id"] == expected["sentence_id"]
            assert result["sentence"] == expected["sentence"]
            assert result["analysis"] == expected["analysis"]
            # Check that optional fields are handled correctly
            assert result.get("function_type") == expected.get("function_type")

    def test_get_file_content_with_malformed_json(self, configured_client, create_test_files):
        """Test GET /files/{filename} skips malformed JSON lines."""
        filename = "interview_003_analysis.jsonl"
        response = configured_client.get(f"/files/{filename}")

        assert response.status_code == 200
        data = response.json()

        assert data["filename"] == filename
        # Should skip the malformed line and return only valid entries
        assert len(data["results"]) == 2
        assert data["results"][0]["sentence_id"] == 20
        assert data["results"][1]["sentence_id"] == 21

    def test_get_file_content_not_found(self, configured_client):
        """Test GET /files/{filename} returns 404 for nonexistent file."""
        response = configured_client.get("/files/nonexistent_analysis.jsonl")

        assert response.status_code == 404
        assert "Analysis file not found" in response.json()["detail"]

    def test_get_specific_sentence_success(self, configured_client, create_test_files):
        """Test GET /files/{filename}/sentences/{sentence_id} successfully retrieves specific sentence."""
        filename = "interview_001_analysis.jsonl"
        sentence_id = 1

        response = configured_client.get(f"/files/{filename}/sentences/{sentence_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["sentence_id"] == sentence_id
        assert data["sentence"] == "Can you tell me about your experience with Python?"
        assert data["analysis"] == "Technical question about programming experience"
        assert data["function_type"] == "question"

    def test_get_specific_sentence_file_not_found(self, configured_client):
        """Test GET /files/{filename}/sentences/{sentence_id} returns 404 for nonexistent file."""
        response = configured_client.get("/files/nonexistent.jsonl/sentences/1")

        assert response.status_code == 404
        assert "Analysis file not found" in response.json()["detail"]

    def test_get_specific_sentence_id_not_found(self, configured_client, create_test_files):
        """Test GET /files/{filename}/sentences/{sentence_id} returns 404 for nonexistent sentence ID."""
        filename = "interview_001_analysis.jsonl"
        nonexistent_id = 999

        response = configured_client.get(f"/files/{filename}/sentences/{nonexistent_id}")

        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "Sentence ID" in detail
        assert str(nonexistent_id) in detail
        assert "not found in file" in detail

    def test_get_specific_sentence_skips_malformed_json(self, configured_client, create_test_files):
        """Test GET /files/{filename}/sentences/{sentence_id} skips malformed lines while searching."""
        filename = "interview_003_analysis.jsonl"
        sentence_id = 21  # This comes after the malformed line

        response = configured_client.get(f"/files/{filename}/sentences/{sentence_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["sentence_id"] == sentence_id
        assert data["sentence"] == "I want to become a senior developer."
        assert data["analysis"] == "Career aspiration response"

    def test_unicode_content_handling(self, temp_output_dir):
        """Test that the API properly handles Unicode content in analysis files."""
        # Create file with Unicode content
        unicode_data = {
            "sentence_id": 100,
            "sequence_order": 0,
            "sentence": "¿Cómo está usted? 你好世界! 🌍",
            "analysis": "Multi-language greeting with emoji",
            "function_type": "greeting",
            "overall_keywords": ["greeting", "multilingual", "emoji"],
        }

        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "unicode_test_analysis.jsonl"
        file_path = temp_output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(unicode_data, ensure_ascii=False) + "\n")

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)

            # Test file listing
            response = client.get("/files/")
            assert response.status_code == 200
            assert filename in response.json()["filenames"]

            # Test file content retrieval
            response = client.get(f"/files/{filename}")
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 1
            assert data["results"][0]["sentence"] == "¿Cómo está usted? 你好世界! 🌍"

            # Test specific sentence retrieval
            response = client.get(f"/files/{filename}/sentences/100")
            assert response.status_code == 200
            data = response.json()
            assert data["sentence"] == "¿Cómo está usted? 你好世界! 🌍"

    def test_large_file_handling(self, temp_output_dir):
        """Test API performance with larger files."""
        # Create a larger file with many entries
        large_file_data = []
        for i in range(100):
            large_file_data.append(
                {
                    "sentence_id": i,
                    "sequence_order": i,
                    "sentence": f"This is sentence number {i} in a large analysis file.",
                    "analysis": f"Analysis for sentence {i}",
                    "function_type": "statement" if i % 2 == 0 else "question",
                }
            )

        test_config = {"paths": {"output_dir": str(temp_output_dir), "analysis_suffix": "_analysis.jsonl"}}

        filename = "large_file_analysis.jsonl"
        file_path = temp_output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for item in large_file_data:
                f.write(json.dumps(item) + "\n")

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)

            # Test file content retrieval
            response = client.get(f"/files/{filename}")
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 100

            # Test specific sentence retrieval from middle of file
            response = client.get(f"/files/{filename}/sentences/50")
            assert response.status_code == 200
            data = response.json()
            assert data["sentence_id"] == 50
            assert "sentence number 50" in data["sentence"]
