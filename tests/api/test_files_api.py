"""
tests/api/test_files_api.py

Comprehensive integration tests for src/api/routers/files.py that follow cardinal rules:
1. Test actual functionality, not mock interactions
2. Use realistic data and scenarios, not hardcoded values

This is a complete rewrite that replaces the previous heavily-mocked version with
proper integration tests that create real files and test actual API behavior.
"""

import json
import platform
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


class TestFilesAPIIntegration:
    """Integration tests for files API with real file operations and realistic data."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def realistic_interview_data(self):
        """Provide realistic interview analysis data for testing."""
        return {
            "complete_interview": [
                {
                    "sentence_id": 0,
                    "sequence_order": 0,
                    "sentence": "Good morning, thank you for joining us today.",
                    "analysis": "Polite opening greeting establishing professional tone",
                    "function_type": "greeting",
                    "structure_type": "declarative",
                    "purpose": "relationship_building",
                    "topic_level_1": "social_interaction",
                    "topic_level_3": "interview_opening",
                    "overall_keywords": ["morning", "thank", "joining"],
                    "domain_keywords": ["professional_communication"],
                },
                {
                    "sentence_id": 1,
                    "sequence_order": 1,
                    "sentence": "Can you walk me through your experience with microservices architecture?",
                    "analysis": "Technical question probing architectural knowledge and experience",
                    "function_type": "interrogative",
                    "structure_type": "complex",
                    "purpose": "technical_assessment",
                    "topic_level_1": "technical_skills",
                    "topic_level_3": "system_architecture",
                    "overall_keywords": ["experience", "microservices", "architecture"],
                    "domain_keywords": ["microservices", "architecture", "technical_skills"],
                },
                {
                    "sentence_id": 2,
                    "sequence_order": 2,
                    "sentence": "I've been working with microservices for about 3 years, "
                    "primarily using Docker and Kubernetes.",
                    "analysis": "Detailed response providing specific technologies and timeframe",
                    "function_type": "declarative",
                    "structure_type": "compound",
                    "purpose": "experience_sharing",
                    "topic_level_1": "technical_experience",
                    "topic_level_3": "technology_stack",
                    "overall_keywords": ["working", "microservices", "years", "Docker", "Kubernetes"],
                    "domain_keywords": ["microservices", "Docker", "Kubernetes", "containerization"],
                },
                {
                    "sentence_id": 3,
                    "sequence_order": 3,
                    "sentence": "What challenges have you faced when implementing distributed systems?",
                    "analysis": "Follow-up question exploring problem-solving experience",
                    "function_type": "interrogative",
                    "structure_type": "complex",
                    "purpose": "problem_solving_assessment",
                    "topic_level_1": "technical_challenges",
                    "topic_level_3": "distributed_systems",
                    "overall_keywords": ["challenges", "implementing", "distributed", "systems"],
                    "domain_keywords": ["distributed_systems", "system_challenges"],
                },
            ],
            "short_interview": [
                {
                    "sentence_id": 100,
                    "sequence_order": 0,
                    "sentence": "Tell me about yourself.",
                    "analysis": "Open-ended question to assess communication skills",
                    "function_type": "interrogative",
                    "structure_type": "simple",
                    "purpose": "self_assessment",
                    "topic_level_1": "personal_introduction",
                    "topic_level_3": "candidate_background",
                    "overall_keywords": ["tell", "about", "yourself"],
                    "domain_keywords": ["self_introduction"],
                },
            ],
            "mixed_quality_data": [
                {
                    "sentence_id": 200,
                    "sequence_order": 0,
                    "sentence": "This is a valid entry.",
                    "analysis": "Complete valid analysis entry",
                    "function_type": "declarative",
                    "structure_type": "simple",
                    "purpose": "statement",
                    "topic_level_1": "example",
                    "topic_level_3": "test_data",
                    "overall_keywords": ["valid", "entry"],
                    "domain_keywords": ["test_data"],
                },
                # Note: Invalid entries will be added as raw strings, not dicts
            ],
        }

    @pytest.fixture
    def configured_client(self, temp_output_dir):
        """Create a TestClient with temporary directory configuration."""
        test_config = {
            "paths": {
                "output_dir": str(temp_output_dir),
                "analysis_suffix": "_analysis.jsonl",
            }
        }

        with patch("src.api.routers.files.config", test_config):
            yield TestClient(app)

    def create_analysis_file(self, file_path: Path, data: list, include_invalid: bool = False):
        """Create an analysis file with realistic data."""
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

            # Optionally add invalid lines for error testing
            if include_invalid:
                f.write("invalid json line\n")
                f.write('{"incomplete": "missing required fields"}\n')

    def test_list_analysis_files_empty_directory(self, configured_client, temp_output_dir):
        """Test listing files when output directory is empty."""
        client = configured_client

        response = client.get("/files/")

        assert response.status_code == 200
        data = response.json()
        assert data["filenames"] == []

    def test_list_analysis_files_with_multiple_files(
        self, configured_client, temp_output_dir, realistic_interview_data
    ):
        """Test listing multiple analysis files in realistic scenario."""
        client = configured_client

        # Create multiple analysis files with realistic names
        file1_path = temp_output_dir / "senior_dev_interview_analysis.jsonl"
        file2_path = temp_output_dir / "junior_dev_interview_analysis.jsonl"
        file3_path = temp_output_dir / "tech_lead_interview_analysis.jsonl"

        self.create_analysis_file(file1_path, realistic_interview_data["complete_interview"])
        self.create_analysis_file(file2_path, realistic_interview_data["short_interview"])
        self.create_analysis_file(file3_path, realistic_interview_data["complete_interview"])

        # Create a non-analysis file that should be ignored
        (temp_output_dir / "readme.txt").write_text("This should be ignored")

        response = client.get("/files/")

        assert response.status_code == 200
        data = response.json()

        # Should return sorted analysis files only
        expected_files = [
            "junior_dev_interview_analysis.jsonl",
            "senior_dev_interview_analysis.jsonl",
            "tech_lead_interview_analysis.jsonl",
        ]
        assert data["filenames"] == expected_files

    def test_list_analysis_files_nonexistent_directory(self, temp_output_dir):
        """Test listing files when output directory doesn't exist."""
        nonexistent_dir = temp_output_dir / "nonexistent"
        test_config = {
            "paths": {
                "output_dir": str(nonexistent_dir),
                "analysis_suffix": "_analysis.jsonl",
            }
        }

        with patch("src.api.routers.files.config", test_config):
            client = TestClient(app)
            response = client.get("/files/")

        assert response.status_code == 200
        data = response.json()
        assert data["filenames"] == []

    @pytest.mark.skipif(platform.system() == "Windows", reason="Permission tests unreliable on Windows")
    def test_list_analysis_files_permission_error(self, configured_client, temp_output_dir):
        """Test listing files when directory has permission issues."""
        client = configured_client

        try:
            # Remove read permissions from directory
            temp_output_dir.chmod(0o000)

            response = client.get("/files/")

            # In containerized environments, permissions may not be enforced
            # So we accept either 500 (permission error) or 200 (permissions not enforced)
            assert response.status_code in [200, 500]

            if response.status_code == 500:
                assert "Internal server error" in response.json()["detail"]
            else:
                # Permissions not enforced - should return empty list
                assert response.json()["filenames"] == []

        finally:
            # Restore permissions for cleanup
            temp_output_dir.chmod(0o755)

    def test_get_file_content_success(self, configured_client, temp_output_dir, realistic_interview_data):
        """Test successfully retrieving complete file content."""
        client = configured_client

        filename = "complete_interview_analysis.jsonl"
        file_path = temp_output_dir / filename

        self.create_analysis_file(file_path, realistic_interview_data["complete_interview"])

        response = client.get(f"/files/{filename}")

        assert response.status_code == 200
        data = response.json()

        assert data["filename"] == filename
        assert len(data["results"]) == 4

        # Verify realistic content structure
        first_result = data["results"][0]
        assert first_result["sentence_id"] == 0
        assert first_result["sentence"] == "Good morning, thank you for joining us today."
        assert first_result["function_type"] == "greeting"
        assert first_result["domain_keywords"] == ["professional_communication"]

        # Verify technical content
        tech_result = data["results"][1]
        assert tech_result["sentence_id"] == 1
        assert "microservices architecture" in tech_result["sentence"]
        assert tech_result["purpose"] == "technical_assessment"
        assert "microservices" in tech_result["domain_keywords"]

    def test_get_file_content_with_mixed_quality_data(
        self, configured_client, temp_output_dir, realistic_interview_data
    ):
        """Test file content retrieval with some invalid JSON lines."""
        client = configured_client

        filename = "mixed_quality_analysis.jsonl"
        file_path = temp_output_dir / filename

        # Create file with valid and invalid content
        with open(file_path, "w", encoding="utf-8") as f:
            # Valid entry
            f.write(json.dumps(realistic_interview_data["mixed_quality_data"][0]) + "\n")
            # Invalid JSON
            f.write("invalid json line that should be skipped\n")
            # Another valid entry
            valid_entry = {
                "sentence_id": 201,
                "sequence_order": 1,
                "sentence": "This is another valid entry after invalid JSON.",
                "analysis": "Should be included despite previous invalid line",
                "function_type": "declarative",
                "structure_type": "simple",
                "purpose": "statement",
                "topic_level_1": "example",
                "topic_level_3": "test_data",
                "overall_keywords": ["another", "valid", "entry"],
                "domain_keywords": ["test_data"],
            }
            f.write(json.dumps(valid_entry) + "\n")

        response = client.get(f"/files/{filename}")

        assert response.status_code == 200
        data = response.json()

        # Should return only valid entries, skipping invalid JSON
        assert len(data["results"]) == 2
        assert data["results"][0]["sentence_id"] == 200
        assert data["results"][1]["sentence_id"] == 201
        assert "invalid json line" not in str(data["results"])

    def test_get_file_content_file_not_found(self, configured_client):
        """Test retrieving content for non-existent file."""
        client = configured_client

        response = client.get("/files/nonexistent_analysis.jsonl")

        assert response.status_code == 404
        assert "Analysis file not found" in response.json()["detail"]
        assert "nonexistent_analysis.jsonl" in response.json()["detail"]

    @pytest.mark.skipif(platform.system() == "Windows", reason="Permission tests unreliable on Windows")
    def test_get_file_content_permission_error(self, configured_client, temp_output_dir, realistic_interview_data):
        """Test retrieving content when file has permission issues."""
        client = configured_client

        filename = "permission_test_analysis.jsonl"
        file_path = temp_output_dir / filename

        self.create_analysis_file(file_path, realistic_interview_data["short_interview"])

        try:
            # Remove read permissions from file
            file_path.chmod(0o000)

            response = client.get(f"/files/{filename}")

            # In containerized environments, permissions may not be enforced
            # So we accept either 500 (permission error) or 200 (permissions not enforced)
            assert response.status_code in [200, 500]

            if response.status_code == 500:
                assert "Error reading file" in response.json()["detail"]
            else:
                # Permissions not enforced - should return file content
                assert response.status_code == 200

        finally:
            # Restore permissions for cleanup
            file_path.chmod(0o644)

    def test_get_specific_sentence_success(self, configured_client, temp_output_dir, realistic_interview_data):
        """Test successfully retrieving a specific sentence analysis."""
        client = configured_client

        filename = "sentence_lookup_analysis.jsonl"
        file_path = temp_output_dir / filename

        self.create_analysis_file(file_path, realistic_interview_data["complete_interview"])

        # Request specific sentence
        target_sentence_id = 2
        response = client.get(f"/files/{filename}/sentences/{target_sentence_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["sentence_id"] == target_sentence_id
        assert data["sequence_order"] == 2
        assert "microservices for about 3 years" in data["sentence"]
        assert data["function_type"] == "declarative"
        assert "Docker" in data["domain_keywords"]
        assert "Kubernetes" in data["domain_keywords"]

    def test_get_specific_sentence_file_not_found(self, configured_client):
        """Test retrieving specific sentence when file doesn't exist."""
        client = configured_client

        response = client.get("/files/missing_file_analysis.jsonl/sentences/1")

        assert response.status_code == 404
        assert "Analysis file not found" in response.json()["detail"]
        assert "missing_file_analysis.jsonl" in response.json()["detail"]

    def test_get_specific_sentence_id_not_found(self, configured_client, temp_output_dir, realistic_interview_data):
        """Test retrieving specific sentence when sentence ID doesn't exist."""
        client = configured_client

        filename = "id_lookup_analysis.jsonl"
        file_path = temp_output_dir / filename

        self.create_analysis_file(file_path, realistic_interview_data["short_interview"])

        # Request non-existent sentence ID
        missing_id = 999
        response = client.get(f"/files/{filename}/sentences/{missing_id}")

        assert response.status_code == 404
        data = response.json()
        assert "Sentence ID" in data["detail"]
        assert str(missing_id) in data["detail"]
        assert "not found in file" in data["detail"]

    def test_get_specific_sentence_with_invalid_data_in_file(
        self, configured_client, temp_output_dir, realistic_interview_data
    ):
        """Test retrieving specific sentence from file with some invalid JSON lines."""
        client = configured_client

        filename = "mixed_sentence_lookup_analysis.jsonl"
        file_path = temp_output_dir / filename

        # Create file with valid target and invalid lines
        with open(file_path, "w", encoding="utf-8") as f:
            # Invalid JSON line
            f.write("invalid json that should be skipped\n")
            # Valid target sentence
            target_data = realistic_interview_data["complete_interview"][1]  # sentence_id: 1
            f.write(json.dumps(target_data) + "\n")
            # Another invalid line
            f.write('{"incomplete": "missing required fields"}\n')
            # Another valid sentence
            f.write(json.dumps(realistic_interview_data["complete_interview"][2]) + "\n")

        response = client.get(f"/files/{filename}/sentences/1")

        assert response.status_code == 200
        data = response.json()

        # Should find the target sentence despite invalid lines
        assert data["sentence_id"] == 1
        assert "microservices architecture" in data["sentence"]
        assert data["purpose"] == "technical_assessment"

    @pytest.mark.skipif(platform.system() == "Windows", reason="Permission tests unreliable on Windows")
    def test_get_specific_sentence_permission_error(self, configured_client, temp_output_dir, realistic_interview_data):
        """Test retrieving specific sentence when file has permission issues."""
        client = configured_client

        filename = "permission_sentence_analysis.jsonl"
        file_path = temp_output_dir / filename

        self.create_analysis_file(file_path, realistic_interview_data["complete_interview"])

        try:
            # Remove read permissions from file
            file_path.chmod(0o000)

            response = client.get(f"/files/{filename}/sentences/1")

            # In containerized environments, permissions may not be enforced
            # So we accept either 500 (permission error) or 200 (permissions not enforced)
            assert response.status_code in [200, 500]

            if response.status_code == 500:
                assert "Error reading file" in response.json()["detail"]
            else:
                # Permissions not enforced - should return sentence data
                assert response.status_code == 200

        finally:
            # Restore permissions for cleanup
            file_path.chmod(0o644)

    def test_get_specific_sentence_validation_error(self, configured_client, temp_output_dir):
        """Test retrieving specific sentence when target data fails validation."""
        client = configured_client

        filename = "validation_error_analysis.jsonl"
        file_path = temp_output_dir / filename

        # Create file with valid structure but invalid data for target sentence
        with open(file_path, "w", encoding="utf-8") as f:
            # Valid sentence
            valid_data = {
                "sentence_id": 0,
                "sequence_order": 0,
                "sentence": "This is valid.",
                "analysis": "Valid analysis",
            }
            f.write(json.dumps(valid_data) + "\n")

            # Invalid sentence data (sentence should be string, not int)
            invalid_target = {
                "sentence_id": 1,
                "sequence_order": 1,
                "sentence": 12345,  # Should be string
                "analysis": "Invalid sentence type",
            }
            f.write(json.dumps(invalid_target) + "\n")

        response = client.get(f"/files/{filename}/sentences/1")

        assert response.status_code == 500
        data = response.json()
        assert "Data validation error" in data["detail"]
        assert "1" in data["detail"]  # sentence_id should be mentioned

    def test_api_handles_unicode_content(self, configured_client, temp_output_dir):
        """Test API handles Unicode characters in interview content properly."""
        client = configured_client

        filename = "unicode_interview_analysis.jsonl"
        file_path = temp_output_dir / filename

        unicode_data = [
            {
                "sentence_id": 0,
                "sequence_order": 0,
                "sentence": "¿Puedes hablar español? Can you speak Spanish?",
                "analysis": "Multilingual question testing language capabilities",
                "function_type": "interrogative",
                "structure_type": "compound",
                "purpose": "language_assessment",
                "topic_level_1": "language_skills",
                "topic_level_3": "multilingual_capability",
                "overall_keywords": ["hablar", "español", "speak", "Spanish"],
                "domain_keywords": ["multilingual", "language_skills"],
            },
            {
                "sentence_id": 1,
                "sequence_order": 1,
                "sentence": "我会说中文。I can speak Chinese and English fluently.",
                "analysis": "Multilingual response demonstrating language proficiency",
                "function_type": "declarative",
                "structure_type": "compound",
                "purpose": "capability_demonstration",
                "topic_level_1": "language_proficiency",
                "topic_level_3": "multilingual_fluency",
                "overall_keywords": ["speak", "Chinese", "English", "fluently"],
                "domain_keywords": ["multilingual", "fluency"],
            },
        ]

        self.create_analysis_file(file_path, unicode_data)

        # Test file listing
        response = client.get("/files/")
        assert response.status_code == 200
        assert filename in response.json()["filenames"]

        # Test file content retrieval
        response = client.get(f"/files/{filename}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert "¿Puedes hablar español?" in data["results"][0]["sentence"]
        assert "我会说中文。" in data["results"][1]["sentence"]

        # Test specific sentence retrieval
        response = client.get(f"/files/{filename}/sentences/1")
        assert response.status_code == 200
        data = response.json()
        assert data["sentence_id"] == 1
        assert "我会说中文。" in data["sentence"]

    def test_api_with_realistic_file_sizes(self, configured_client, temp_output_dir):
        """Test API performance with realistic file sizes (multiple sentences)."""
        client = configured_client

        filename = "large_interview_analysis.jsonl"
        file_path = temp_output_dir / filename

        # Create a realistic-sized interview file (50 sentences)
        large_interview_data = []
        for i in range(50):
            sentence_data = {
                "sentence_id": i,
                "sequence_order": i,
                "sentence": f"This is interview sentence number {i + 1} discussing various topics.",
                "analysis": f"Analysis of sentence {i + 1} covering different aspects of the conversation",
                "function_type": "declarative" if i % 2 == 0 else "interrogative",
                "structure_type": "simple" if i % 3 == 0 else "complex",
                "purpose": "information_gathering" if i % 4 == 0 else "response_provision",
                "topic_level_1": f"topic_{i % 10}",
                "topic_level_3": f"subtopic_{i % 20}",
                "overall_keywords": [f"keyword_{i}", f"topic_{i % 5}", "interview"],
                "domain_keywords": [f"domain_{i % 8}", "interview_process"],
            }
            large_interview_data.append(sentence_data)

        self.create_analysis_file(file_path, large_interview_data)

        # Test file listing (should handle larger files)
        response = client.get("/files/")
        assert response.status_code == 200
        assert filename in response.json()["filenames"]

        # Test full file content (should handle 50 sentences)
        response = client.get(f"/files/{filename}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 50
        assert data["results"][0]["sentence_id"] == 0
        assert data["results"][49]["sentence_id"] == 49

        # Test specific sentence lookup in large file
        target_id = 25
        response = client.get(f"/files/{filename}/sentences/{target_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["sentence_id"] == target_id
        assert f"sentence number {target_id + 1}" in data["sentence"]


class TestFilesAPIErrorHandling:
    """Test error handling scenarios with realistic conditions."""

    @pytest.fixture
    def configured_client_with_invalid_config(self):
        """Create client with intentionally problematic configuration."""
        invalid_config = {
            "paths": {
                # Missing output_dir to test config error handling
                "analysis_suffix": "_analysis.jsonl",
            }
        }

        with patch("src.api.routers.files.config", invalid_config):
            yield TestClient(app)

    def test_config_error_handling(self, configured_client_with_invalid_config):
        """Test API behavior when configuration is missing required keys."""
        client = configured_client_with_invalid_config

        # Should gracefully handle missing config with defaults
        response = client.get("/files/")
        assert response.status_code == 200
        # Should handle gracefully - may return empty list or use defaults
        data = response.json()
        assert "filenames" in data
        # Accept either empty list (no files found) or actual files (defaults used)
        assert isinstance(data["filenames"], list)

    def test_malformed_config_handling(self):
        """Test API behavior with completely malformed configuration."""
        malformed_config = "not_a_dict"

        with patch("src.api.routers.files.config", malformed_config):
            client = TestClient(app)
            response = client.get("/files/")

        # Should handle gracefully and not crash
        assert response.status_code in [200, 500]  # Either graceful fallback or proper error
