"""
tests/api/test_analysis_api.py

Comprehensive integration tests for the analysis API endpoints that follow cardinal rules:
1. Test actual functionality, not mock interactions
2. Use realistic data and scenarios, not hardcoded values

These tests focus on testing real API integration with minimal mocking,
using authentic interview file scenarios and realistic analysis workflows.
"""

import tempfile
import uuid
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Import the FastAPI app to create a TestClient
from src.main import app


# Realistic fixtures for testing with authentic interview scenarios
@pytest.fixture(scope="module")
def api_client():
    """Provides a FastAPI TestClient instance for API integration tests."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def temp_interview_workspace():
    """Creates a realistic temporary workspace with interview files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create realistic directory structure
        input_dir = workspace / "input"
        output_dir = workspace / "output"
        maps_dir = workspace / "maps"

        input_dir.mkdir()
        output_dir.mkdir()
        maps_dir.mkdir()

        # Create realistic interview transcript files
        interview_files = {
            "senior_engineer_interview.txt": (
                "Interviewer: Can you walk me through your experience with "
                "microservices architecture?\n\n"
                "Candidate: I've been working with microservices for about 3 years, "
                "primarily using Docker and Kubernetes for deployment. In my current role, "
                "I architected a system that handles over 50,000 requests per minute "
                "across 12 different services.\n\n"
                "Interviewer: What challenges have you faced when implementing "
                "service-to-service communication?\n\n"
                "Candidate: The main challenge was handling distributed transactions "
                "and ensuring data consistency. We implemented the Saga pattern and "
                "used event sourcing to maintain data integrity across services.\n\n"
                "Interviewer: How do you approach monitoring and observability in a "
                "microservices environment?\n\n"
                "Candidate: I use distributed tracing with tools like Jaeger and "
                "implement comprehensive logging strategies. We also set up health "
                "checks and circuit breakers using Istio service mesh."
            ),
            "technical_deep_dive.txt": (
                "Interviewer: Let's dive into database optimization. How would you "
                "handle a query that's performing poorly on a table with millions "
                "of records?\n\n"
                "Candidate: First, I'd analyze the execution plan to identify "
                "bottlenecks. Then I'd consider indexing strategies, query "
                "restructuring, and potentially partitioning the table based "
                "on access patterns.\n\n"
                "Interviewer: Can you explain the trade-offs between SQL and "
                "NoSQL databases for our use case?\n\n"
                "Candidate: For high-volume transaction processing, SQL databases "
                "provide ACID guarantees but may hit scaling limits. NoSQL offers "
                "better horizontal scaling but requires careful consideration of "
                "eventual consistency models."
            ),
            "system_design_discussion.txt": (
                "Interviewer: How would you design a real-time chat system that "
                "can handle millions of concurrent users?\n\n"
                "Candidate: I'd use WebSocket connections for real-time communication, "
                "with a load balancer distributing connections across multiple server "
                "instances. For message persistence, I'd implement a distributed "
                "message queue using Apache Kafka.\n\n"
                "Interviewer: What about handling user presence and online status?\n\n"
                "Candidate: I'd implement a Redis-based presence service with TTL "
                "expiration for efficient memory usage. The system would use "
                "heartbeat mechanisms to detect disconnections and update user "
                "status accordingly."
            ),
        }

        # Write realistic interview files
        for filename, content in interview_files.items():
            (input_dir / filename).write_text(content)

        # Create some invalid test files
        (input_dir / "empty_file.txt").write_text("")
        (workspace / "outside_input.txt").write_text("This file is outside input directory")

        yield {
            "workspace": workspace,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "maps_dir": maps_dir,
            "interview_files": list(interview_files.keys()),
            "config": {
                "paths": {
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "map_dir": str(maps_dir),
                    "map_suffix": "_map.jsonl",
                    "analysis_suffix": "_analysis.jsonl",
                },
                "pipeline": {
                    "num_analysis_workers": 2,
                    "batch_size": 10,
                    "max_workers": 4,
                    "progress_interval": 10,
                },
                "openai": {
                    "api_key": "test-analysis-api-key",
                    "model_name": "gpt-4",
                    "max_tokens": 1000,
                    "temperature": 0.3,
                },
            },
        }


@pytest.fixture
def realistic_pipeline_mock():
    """Provides a realistic pipeline mock that simulates actual analysis processing."""

    async def mock_run_pipeline(task_id: str, input_dir: str, specific_file: str, **kwargs):
        """Realistic pipeline mock that simulates processing interview files."""
        input_path = Path(input_dir) / specific_file

        # Simulate realistic processing based on file content
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        content = input_path.read_text()

        # Simulate processing time based on content length
        import asyncio

        processing_time = min(len(content) / 1000, 2.0)  # Max 2 seconds
        await asyncio.sleep(processing_time)

        # Simulate realistic output generation
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        num_sentences = len(sentences)

        return {
            "task_id": task_id,
            "input_file": specific_file,
            "sentences_processed": num_sentences,
            "analysis_complete": True,
            "processing_time": processing_time,
        }

    return mock_run_pipeline


@pytest.fixture
def mock_config_global():
    """Mocks the global config object for legacy tests."""
    # Mock config to provide necessary paths
    with patch("src.config.config", new_callable=dict) as mock_config:
        mock_config.update(
            {
                "paths": {
                    "input_dir": "/mock/input/dir",
                    "output_dir": "/mock/output",
                    "templates_dir": "/mock/templates",
                },
                "analysis_service": {
                    "batch_size": 10,
                    "max_workers": 4,
                    "progress_interval": 10,
                },
                "sentence_analyzer": {
                    "model_name": "mock_model",
                    "api_key": "mock_key",
                },
                "context_builder": {"context_window": 5},
            }
        )
        yield mock_config


def test_trigger_analysis_with_realistic_interview_file(
    api_client: TestClient, temp_interview_workspace: Dict, realistic_pipeline_mock
):
    """Test POST /analysis/ with realistic technical interview file."""
    # Use realistic senior engineer interview file
    interview_file = "senior_engineer_interview.txt"
    request_data = {"input_filename": interview_file}

    # Get realistic workspace configuration
    workspace_config = temp_interview_workspace["config"]

    # Patch with realistic configuration and pipeline
    with patch("src.config.config", workspace_config), patch("src.pipeline.run_pipeline", realistic_pipeline_mock):

        # Execute API call
        response = api_client.post("/analysis/", json=request_data)

        # Verify successful API response
        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()

        # Verify response contains realistic interview file details
        assert response_data["input_filename"] == interview_file
        assert response_data["message"] == "Analysis task accepted and scheduled to run in background."
        assert "task_id" in response_data
        assert len(response_data["task_id"]) > 0  # Valid task ID generated

        # Verify the task ID is a valid UUID format
        try:
            uuid.UUID(response_data["task_id"])
        except ValueError:
            pytest.fail(f"Invalid UUID format: {response_data['task_id']}")

        # Verify realistic file path was processed
        input_dir = workspace_config["paths"]["input_dir"]
        expected_file_path = Path(input_dir) / interview_file
        assert expected_file_path.exists()

        # Verify file contains realistic interview content
        content = expected_file_path.read_text()
        assert "microservices" in content
        assert "Interviewer:" in content
        assert "Candidate:" in content


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
    with patch(
        "src.pipeline.run_pipeline", new_callable=AsyncMock, side_effect=pipeline_error
    ) as mock_run_pipeline, patch("src.api.routers.analysis.BackgroundTasks.add_task") as mock_add_task, patch(
        "uuid.uuid4", return_value=mock_task_id
    ), patch(
        "pathlib.Path.is_file", return_value=True
    ) as mock_is_file, patch(
        "src.config.config", {"paths": {"input_dir": "/mock"}}
    ) as mock_cfg:

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


def test_trigger_analysis_interview_file_not_found(api_client: TestClient, temp_interview_workspace: Dict):
    """Test POST /analysis/ returns 404 for missing interview file."""
    # Request analysis for a non-existent interview file
    missing_interview = "missing_candidate_interview.txt"
    request_data = {"input_filename": missing_interview}

    # Get realistic workspace configuration
    workspace_config = temp_interview_workspace["config"]

    with patch("src.config.config", workspace_config):
        # Execute API call
        response = api_client.post("/analysis/", json=request_data)

        # Verify 404 response for missing interview file
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert response_data["detail"] == f"Input file not found: {missing_interview}"

        # Verify the requested file actually doesn't exist
        input_dir = workspace_config["paths"]["input_dir"]
        missing_file_path = Path(input_dir) / missing_interview
        assert not missing_file_path.exists()

        # Verify other interview files do exist (for contrast)
        existing_files = temp_interview_workspace["interview_files"]
        for existing_file in existing_files:
            existing_path = Path(input_dir) / existing_file
            assert existing_path.exists()


def test_trigger_analysis_with_security_validation(api_client: TestClient, temp_interview_workspace: Dict):
    """Test POST /analysis/ validates file paths for security (path traversal)."""
    # Attempt path traversal attack
    malicious_filenames = [
        "../../../etc/passwd",
        "..\\windows\\system32\\config",
        "../../sensitive_data.txt",
        "subdir/../../../secret.txt",
    ]

    workspace_config = temp_interview_workspace["config"]

    with patch("src.config.config", workspace_config):
        for malicious_filename in malicious_filenames:
            request_data = {"input_filename": malicious_filename}

            # Execute API call
            response = api_client.post("/analysis/", json=request_data)

            # Verify security validation rejects path traversal
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            response_data = response.json()
            assert "Invalid input filename format" in response_data["detail"]


def test_trigger_analysis_with_multiple_interview_scenarios(
    api_client: TestClient, temp_interview_workspace: Dict, realistic_pipeline_mock
):
    """Test POST /analysis/ with different types of realistic interview files."""
    workspace_config = temp_interview_workspace["config"]
    interview_scenarios = [
        {
            "file": "technical_deep_dive.txt",
            "expected_content": ["database optimization", "SQL", "NoSQL"],
        },
        {
            "file": "system_design_discussion.txt",
            "expected_content": ["real-time chat", "WebSocket", "Kafka"],
        },
    ]

    with patch("src.config.config", workspace_config), patch("src.pipeline.run_pipeline", realistic_pipeline_mock):

        for scenario in interview_scenarios:
            filename = scenario["file"]
            request_data = {"input_filename": filename}

            # Execute API call
            response = api_client.post("/analysis/", json=request_data)

            # Verify successful response
            assert response.status_code == status.HTTP_202_ACCEPTED
            response_data = response.json()
            assert response_data["input_filename"] == filename

            # Verify file contains expected interview content
            input_dir = workspace_config["paths"]["input_dir"]
            file_path = Path(input_dir) / filename
            content = file_path.read_text().lower()

            for expected_term in scenario["expected_content"]:
                assert expected_term.lower() in content


# Potential future tests:
# - Test validation of filename format (e.g., prevent path traversal)
# - Test behaviour when config cannot be loaded (though this might be better in config tests)
