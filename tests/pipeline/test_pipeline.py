import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call, mock_open
from pathlib import Path
import asyncio
import json
import logging
import copy
from typing import List, Dict, Any, Tuple
import re

# Import necessary components from the source code
from src.pipeline import (
    PipelineOrchestrator, # NEW: Import orchestrator
    run_pipeline,
    analyze_specific_sentences,
    # segment_text is likely not needed directly if patched via utils
)
# from src.config import Config # Config likely not needed directly in tests anymore
from src.utils.metrics import MetricsTracker
from src.services.analysis_service import AnalysisService
# from src.utils.text_processing import segment_text # Import locally in test
from src.utils.path_helpers import generate_pipeline_paths
from src.io.local_storage import LocalJsonlMapStorage
from src.io.protocols import TextDataSource, ConversationMapStorage, SentenceAnalysisWriter

# Define the missing helper function
def create_mock_analysis(sentence_id, sequence_order, sentence_text):
    """
    Helper function to generate a consistent mock analysis result dictionary.

    Used within pipeline tests to simulate the output of `SentenceAnalyzer`.

    Args:
        sentence_id (int): The ID of the sentence.
        sequence_order (int): The sequence order of the sentence.
        sentence_text (str): The text content of the sentence.

    Returns:
        dict: A dictionary containing mock analysis fields.
    """
    return {
        "sentence_id": sentence_id,
        "sequence_order": sequence_order,
        "sentence": sentence_text,
        "function_type": "mock_declarative",
        "structure_type": "mock_simple",
        # ... other analysis fields ...
    }

# Define missing fixtures

@pytest.fixture
def sample_text_file(tmp_path):
    """
    Pytest fixture creating a temporary sample text file.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        Path: Path object pointing to the created temp file.
    """
    file_content = "First sentence. Second sentence."
    test_file = tmp_path / "test_input.txt"
    test_file.write_text(file_content)
    return test_file

@pytest.fixture
def mock_config(tmp_path):
    """
    Pytest fixture providing a mock configuration dictionary.

    Includes nested structures for paths, pipeline settings, preprocessing,
    classification, and domain keywords used by various components.
    Uses tmp_path to provide realistic, test-specific absolute paths.
    """
    map_dir = tmp_path / "mock_maps"
    output_dir = tmp_path / "mock_output"
    logs_dir = tmp_path / "mock_logs"
    return {
        "paths": {
            "output_dir": str(output_dir),
            "map_dir": str(map_dir),
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl",
            "logs_dir": str(logs_dir)
        },
        "pipeline": {
            "num_analysis_workers": 2,
            "num_concurrent_files": 1 # Add this if orchestrator uses it
        },
        # Add nested structure expected by dependencies
        "preprocessing": {
            "context_windows": {
                "immediate": 1,
                "broader": 3,
                "observer": 5
            }
        },
        "classification": {
            "local": {
                "prompt_files": {
                    "no_context": "mock/path/to/prompts.yaml"
                }
            }
        },
        "domain_keywords": ["mock_keyword"]
    }

@pytest.fixture
def mock_analysis_service() -> MagicMock:
    """
    Provides a mock `AnalysisService` instance with mocked async/sync methods
    and attributes like context_builder.
    """
    mock = MagicMock(spec=AnalysisService)

    # Mock the build_contexts method directly on the service mock
    # This aligns with the call `analysis_service.build_contexts(sentences)` in process_file
    mock.build_contexts = MagicMock(return_value={
        0: {"ctx": "mock_ctx_0"},
        1: {"ctx": "mock_ctx_1"}
        # Tests can override this return value if needed
    })

    # Also mock the context_builder attribute and its method for analyze_specific_sentences
    mock.context_builder = MagicMock()
    mock.context_builder.build_all_contexts = MagicMock(return_value={
        0: {"ctx": "all_ctx_0"}, 1: {"ctx": "all_ctx_1"}, 2: {"ctx": "all_ctx_2"}, 3: {"ctx": "all_ctx_3"}
        # Use a potentially different return value for easier differentiation if needed
    })

    # Configure analyze_sentences directly on the main mock
    mock.analyze_sentences = AsyncMock(return_value=[{"result": "mock"}]) # Async method
    # Ensure metrics_tracker attribute exists if needed by tests (though pipeline passes it explicitly now)
    mock.metrics_tracker = MagicMock(spec=MetricsTracker)
    return mock

# --- ADDED from tests/test_pipeline.py START ---
# These tests ideally belong in tests/utils/test_text_processing.py
def test_segment_text():
    """Tests `segment_text` for basic sentence splitting (uses default spaCy model)."""
    # Assuming segment_text is available, potentially needs import or patch adjustment
    from src.utils.text_processing import segment_text # Adjust import if needed
    test_text = "Hello world. How are you today? This pipeline is running well!"
    sentences = segment_text(test_text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you today?"
    assert sentences[2] == "This pipeline is running well!"

def test_segment_text_empty():
    """Tests `segment_text` returns an empty list for empty input."""
    from src.utils.text_processing import segment_text # Adjust import if needed
    sentences = segment_text("")
    assert sentences == []
# --- ADDED from tests/test_pipeline.py END ---

# --- Test analyze_specific_sentences (Refactored) ---
# These tests check the orchestration logic of analyze_specific_sentences,
# mocking the internal refactored helpers.

@pytest.mark.asyncio
async def test_analyze_specific_sentences_success(
    tmp_path: Path,
    mock_analysis_service: MagicMock # Config no longer needed here
):
    """Tests analyze_specific_sentences successfully orchestrates analysis."""
    # --- Setup ---
    map_file_path = tmp_path / "mock_map_specific_success.jsonl"
    task_id = "task-specific-ok-new"
    sentence_ids_to_analyze = [1, 3]

    # Mock map storage (don't need real file for this orchestration test)
    mock_map_storage = AsyncMock(spec=ConversationMapStorage)
    mock_map_storage.get_identifier.return_value = str(map_file_path)

    # Expected intermediate values from helpers
    all_sentences_text = ["S0.", "S1.", "S2.", "S3."]
    target_indices_sorted = [1, 3]
    target_sentences_sorted = ["S1.", "S3."]
    target_contexts_sorted = [{"ctx": "target_ctx_1"}, {"ctx": "target_ctx_3"}]
    mock_service_results = [
        {"analysis": "for_s1"},
        {"analysis": "for_s3"}
    ]
    expected_final_results = [
        {"sentence_id": 1, "sequence_order": 1, "analysis": "for_s1"},
        {"sentence_id": 3, "sequence_order": 3, "analysis": "for_s3"}
    ]

    # Configure mock service return value
    mock_analysis_service.analyze_sentences = AsyncMock(return_value=mock_service_results)

    # Patch the internal helpers
    with patch("src.pipeline._prepare_data_for_specific_analysis", new_callable=AsyncMock) as mock_prepare, \
         patch("src.pipeline._build_contexts_for_specific_analysis") as mock_build_ctx, \
         patch("src.pipeline._post_process_specific_results") as mock_post_process, \
         patch("src.pipeline.logger") as mock_logger:

        # Configure mock returns for helpers
        mock_prepare.return_value = (target_sentences_sorted, target_indices_sorted, all_sentences_text)
        mock_build_ctx.return_value = target_contexts_sorted
        mock_post_process.return_value = expected_final_results # Simulate post-processing

        # --- Execute --- 
        final_results = await analyze_specific_sentences(
            map_storage=mock_map_storage,
            sentence_ids=sentence_ids_to_analyze,
            analysis_service=mock_analysis_service,
            task_id=task_id
        )

        # --- Assertions ---
        # Assert helpers were called correctly
        mock_prepare.assert_awaited_once_with(mock_map_storage, sentence_ids_to_analyze, f"[Task {task_id}] ")
        mock_build_ctx.assert_called_once_with(all_sentences_text, target_indices_sorted, mock_analysis_service, f"[Task {task_id}] ")

        # Assert analysis service was called correctly
        mock_analysis_service.analyze_sentences.assert_awaited_once_with(
            target_sentences_sorted, target_contexts_sorted, task_id=task_id
        )

        # Assert post-processing was called correctly
        mock_post_process.assert_called_once_with(mock_service_results, target_indices_sorted, f"[Task {task_id}] ")

        # Final result check
        assert final_results == expected_final_results

@pytest.mark.asyncio
async def test_analyze_specific_sentences_prepare_error(
    tmp_path: Path,
    mock_analysis_service: MagicMock
):
    """Tests analyze_specific_sentences handles errors from _prepare_data helper."""
    mock_map_storage = AsyncMock(spec=ConversationMapStorage)
    sentence_ids = [1, 5] # ID 5 will be missing
    task_id = "task-specific-prep-err"
    prepare_error = ValueError("Sentence IDs not found")

    with patch("src.pipeline._prepare_data_for_specific_analysis", new_callable=AsyncMock, side_effect=prepare_error) as mock_prepare, \
         patch("src.pipeline._build_contexts_for_specific_analysis") as mock_build_ctx, \
         patch("src.pipeline._post_process_specific_results") as mock_post_process, \
         patch("src.pipeline.logger") as mock_logger:

        with pytest.raises(ValueError, match="Sentence IDs not found"):
            await analyze_specific_sentences(
                map_storage=mock_map_storage,
                sentence_ids=sentence_ids,
                analysis_service=mock_analysis_service,
                task_id=task_id
            )

        mock_prepare.assert_awaited_once()
        mock_build_ctx.assert_not_called() # Should not be called if prepare fails
        mock_analysis_service.analyze_sentences.assert_not_awaited()
        mock_post_process.assert_not_called()
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "Specific analysis failed" in log_message
        assert "Sentence IDs not found" in log_message

# === Unit tests for run_pipeline ===
@pytest.mark.asyncio
async def test_run_pipeline_instantiates_and_executes_orchestrator(mock_config):
    """Tests that run_pipeline instantiates PipelineOrchestrator and calls execute."""
    input_dir = "fake/input"
    output_dir = "fake/output"
    map_dir = "fake/map"
    specific_file = "file.txt"
    task_id = "run-task-123"

    # Patch the orchestrator itself
    with patch("src.pipeline.PipelineOrchestrator", autospec=True) as MockPipelineOrchestrator:
        # Get the mock instance that __init__ returns
        mock_orchestrator_instance = MockPipelineOrchestrator.return_value
        # Make execute an AsyncMock
        mock_orchestrator_instance.execute = AsyncMock()

        await run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            specific_file=specific_file,
            config_dict=mock_config,
            task_id=task_id
        )

        # Assert orchestrator was instantiated correctly
        MockPipelineOrchestrator.assert_called_once_with(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=mock_config,
            task_id=task_id
        )
        # Assert execute was called on the instance
        mock_orchestrator_instance.execute.assert_awaited_once_with(specific_file=specific_file)

@pytest.mark.asyncio
async def test_run_pipeline_handles_orchestrator_init_error(mock_config):
    """Tests run_pipeline handles exceptions during orchestrator initialization."""
    input_dir = "fake/input"
    init_error = ValueError("Bad config")

    with patch("src.pipeline.PipelineOrchestrator", side_effect=init_error), \
         patch("src.pipeline.logger") as mock_logger:

        with pytest.raises(ValueError, match="Bad config"):
            await run_pipeline(input_dir=input_dir, config_dict=mock_config)

        mock_logger.critical.assert_called_once()
        log_message = mock_logger.critical.call_args[0][0]
        assert "Pipeline setup failed" in log_message
        assert "Bad config" in log_message
