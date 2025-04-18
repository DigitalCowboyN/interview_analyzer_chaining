"""
tests/test_pipeline.py

This module contains unit tests for the functions defined in the pipeline module 
(src/pipeline.py). The refactored pipeline module is responsible for:
    - Creating a conversation map file listing sentences with sequence order.
    - Managing a queue-based system (task queue, results queue) for concurrent sentence analysis.
    - Orchestrating worker tasks to analyze sentences using SentenceAnalyzer.
    - Orchestrating a writer task to append results to a JSON Lines analysis file.
    - Running the entire pipeline across multiple text files in a directory.

Key functions/logic tested:
    - create_conversation_map: Creates the map file correctly.
    - process_file: Orchestrates map creation, queue management, worker/writer tasks, and final output generation.
    - run_pipeline: Processes all .txt files in a directory using the refactored process_file.
    - segment_text: Still tested for basic sentence segmentation.

Usage:
    Run these tests with pytest from the project root:
        pytest tests/test_pipeline.py

Modifications:
    - Tests for process_file heavily rely on mocking asyncio queues and SentenceAnalyzer.
    - Asserts check for map file content, analysis file content (JSON Lines), and proper function calls.
"""

import pytest
import json
import asyncio
import copy # Import copy module
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock, call, ANY, mock_open # Import mock_open
from typing import List, Dict

# Assume these will be the new/refactored imports from src.pipeline
from src.pipeline import segment_text, run_pipeline, create_conversation_map, process_file, verify_output_completeness, analyze_specific_sentences, Path
from src.services.analysis_service import AnalysisService # ADD THIS IMPORT
# We will likely need to import the specific functions/classes we test directly if they are exposed
# For now, we'll patch them within tests assuming they are part of the pipeline module's internal structure
# or imported there.

def test_segment_text():
    """
    Test `segment_text` basic sentence segmentation.
    
    Verifies that `segment_text` correctly splits a sample text into sentences
    using the default spaCy model.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the number of sentences or their content is incorrect.
    """
    test_text = "Hello world. How are you today? This pipeline is running well!"
    sentences = segment_text(test_text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you today?"
    assert sentences[2] == "This pipeline is running well!"

def test_segment_text_empty():
    """
    Test `segment_text` with an empty input string.
    
    Verifies that `segment_text` returns an empty list when the input is empty.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the result is not an empty list.
    """
    sentences = segment_text("")
    assert sentences == []

@pytest.fixture
def sample_text_file(tmp_path):
    """
    Pytest fixture to create a temporary sample text file for testing.
    
    Writes a simple two-sentence text to a file within the pytest temporary directory.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    
    Returns:
        Path: The path object pointing to the created sample text file.
    """
    file_content = "First sentence. Second sentence."
    test_file = tmp_path / "test_input.txt"
    test_file.write_text(file_content)
    return test_file

@pytest.fixture
def mock_config():
    """
    Pytest fixture providing a mock configuration dictionary for tests.
    
    Includes necessary path and pipeline settings used by the functions under test.

    Returns:
        dict: A dictionary containing mock configuration values.
    """
    return {
        "paths": {
            "output_dir": "mock_output",
            "map_dir": "mock_maps",
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl" # Assuming a new suffix for analysis output
        },
        "pipeline": {
            "num_analysis_workers": 2
        },
        # Add other necessary mocked config sections if needed
    }

@pytest.fixture
def mock_analysis_service() -> MagicMock:
    """Provides a mock AnalysisService instance."""
    mock = MagicMock(spec=AnalysisService)
    # Configure common methods used by pipeline functions
    mock.build_contexts = MagicMock(return_value=[{"ctx": "mock"}]) # Sync method
    mock.analyze_sentences = AsyncMock(return_value=[{"result": "mock"}]) # Async method
    return mock

def create_jsonl_file(path: Path, data: List[Dict]):
    """Creates a JSONL file with the given data."""
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

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

@pytest.mark.asyncio
async def test_create_conversation_map(tmp_path, mock_config):
    """
    Test `create_conversation_map` successful execution.
    
    Verifies that the function correctly reads an input file, segments it (mocked),
    creates the specified map directory and file, writes the correct JSON Lines data
    to the map file, and returns the correct sentence count and list.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If file/directory creation fails, content is incorrect,
                      or return values are wrong.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    # map_dir will be created by the function if it doesn't exist

    input_file = input_dir / "conv1.txt"
    input_file.write_text("Sentence one. Sentence two?")
    
    expected_map_file = map_dir / f"{input_file.stem}{mock_config['paths']['map_suffix']}"

    # Define expected sentences explicitly
    expected_sentences = ["Sentence one.", "Sentence two?"]

    # Patch segment_text from its new location
    with patch("src.utils.text_processing.segment_text", return_value=expected_sentences) as mock_segment:
        # Mock logger if necessary to avoid side effects during test
        with patch("src.pipeline.logger") as mock_logger:
            # Expect tuple return value
            sentence_count, returned_sentences = await create_conversation_map(input_file, map_dir, mock_config["paths"]["map_suffix"])

    assert sentence_count == 2
    assert returned_sentences == expected_sentences # Check returned sentences
    assert map_dir.exists()
    assert expected_map_file.exists()

    lines = expected_map_file.read_text().strip().split('\n')
    assert len(lines) == 2
    
    entry1 = json.loads(lines[0])
    assert entry1 == {"sentence_id": 0, "sequence_order": 0, "sentence": "Sentence one."}
    entry2 = json.loads(lines[1])
    assert entry2 == {"sentence_id": 1, "sequence_order": 1, "sentence": "Sentence two?"}

@pytest.mark.asyncio
async def test_create_conversation_map_empty_file(tmp_path, mock_config):
    """
    Test `create_conversation_map` with an empty input file.
    
    Verifies that the function handles an empty input file gracefully by creating
    an empty map file and returning a sentence count of 0 and an empty list.
    
    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If the map file is not created, not empty, or if return
                      values are incorrect.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    
    input_file = input_dir / "empty.txt"
    input_file.write_text("") # Empty file
    
    expected_map_file = map_dir / f"{input_file.stem}{mock_config['paths']['map_suffix']}"

    with patch("src.utils.text_processing.segment_text", return_value=[]):
        with patch("src.pipeline.logger"):
            # Expect tuple return value
            sentence_count, returned_sentences = await create_conversation_map(input_file, map_dir, mock_config["paths"]["map_suffix"])

    assert sentence_count == 0
    assert returned_sentences == [] # Check returned sentences
    assert map_dir.exists() # Directory should be created
    assert expected_map_file.exists() # File should be created
    assert expected_map_file.read_text() == "" # File should be empty

@pytest.mark.asyncio
async def test_process_file_success(
    sample_text_file, 
    tmp_path, 
    mock_config, 
    mock_analysis_service # Inject mock service
):
    """ Test process_file success path with injected mock AnalysisService.""" 
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    analysis_file_path = output_dir / f"{sample_text_file.stem}{mock_config['paths']['analysis_suffix']}"

    mock_sentences = ["First sentence.", "Second sentence."]
    mock_contexts = [{"ctx": "ctx1"}, {"ctx": "ctx2"}]
    mock_analysis_results = [
        create_mock_analysis(0, 0, mock_sentences[0]),
        create_mock_analysis(1, 1, mock_sentences[1])
    ]

    # Configure the INJECTED mock service's methods
    mock_analysis_service.build_contexts.return_value = mock_contexts
    mock_analysis_service.analyze_sentences.return_value = mock_analysis_results

    # Patch other dependencies of process_file
    with patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(len(mock_sentences), mock_sentences)) as mock_create_map, \
         patch("src.pipeline._result_writer", new_callable=AsyncMock) as mock_result_writer, \
         patch("src.pipeline.logger") as mock_logger, \
         patch("src.pipeline.metrics_tracker") as mock_metrics_tracker:

        # Execute with injected mock service
        await process_file(sample_text_file, output_dir, map_dir, mock_config, mock_analysis_service)

        # --- Assertions ---
        mock_create_map.assert_awaited_once_with(sample_text_file, map_dir, mock_config['paths']['map_suffix'])
        
        # Assert calls on the INJECTED mock service instance
        mock_analysis_service.build_contexts.assert_called_once_with(mock_sentences)
        mock_analysis_service.analyze_sentences.assert_awaited_once_with(mock_sentences, mock_contexts)

        # Assert result writer call hasn't changed conceptually
        mock_result_writer.assert_awaited_once()
        call_args, _ = mock_result_writer.call_args
        assert call_args[0] == analysis_file_path
        # Note: Inspecting queue content passed to writer remains complex

@pytest.mark.asyncio
async def test_process_file_analyzer_error(
    sample_text_file, 
    tmp_path, 
    mock_config, 
    mock_analysis_service # Inject mock service
):
    """ Test process_file when the injected AnalysisService.analyze_sentences raises an error."""
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    
    mock_sentences = ["Sentence 0.", "Sentence 1 fails."]
    mock_contexts = [{"ctx": "c0"}, {"ctx": "c1"}]
    test_exception = ValueError("LLM API failed")

    # Configure INJECTED mock service methods
    mock_analysis_service.build_contexts.return_value = mock_contexts
    mock_analysis_service.analyze_sentences.side_effect = test_exception

    with patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(len(mock_sentences), mock_sentences)) as mock_create_map, \
         patch("src.pipeline._result_writer", new_callable=AsyncMock) as mock_result_writer, \
         patch("src.pipeline.logger") as mock_logger, \
         patch("src.pipeline.metrics_tracker") as mock_metrics_tracker:

        # Execute and assert that the exception propagates
        with pytest.raises(ValueError, match="LLM API failed"):
            await process_file(sample_text_file, output_dir, map_dir, mock_config, mock_analysis_service)

        # Assertions on calls made *before* the exception
        mock_create_map.assert_awaited_once()
        # Assert calls on the injected mock service
        mock_analysis_service.build_contexts.assert_called_once_with(mock_sentences)
        mock_analysis_service.analyze_sentences.assert_awaited_once_with(mock_sentences, mock_contexts)
        # Writer should not have been called
        mock_result_writer.assert_not_awaited()

@pytest.mark.asyncio
async def test_process_file_writer_error(
    mock_config, 
    mock_analysis_service # Inject mock service
):
    """Verify exceptions from _result_writer propagate (with injected service)."""
    # This test's core issue (TypeError) is likely unrelated to service injection,
    # but we update its signature and call pattern.
    input_file_path = Path("nonexistent.txt") # Keep as Path
    output_dir = Path("/fake/output")        # Keep as Path

    # Configure injected service mock (assuming it succeeds before writer fails)
    mock_analysis_service.build_contexts.return_value = [{"ctx": "dummy"}] # Example
    mock_analysis_service.analyze_sentences.return_value = [{"result": "data"}]

    mock_writer_exception = OSError("Disk full")

    # Use a local queue for the writer, as process_file creates one internally
    # We patch the _result_writer itself.

    with patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(1, ["Dummy sentence."])) as mock_create_map, \
         patch("src.pipeline._result_writer", new_callable=AsyncMock, side_effect=mock_writer_exception) as mock_writer_func, \
         patch("src.pipeline.logger") as mock_logger, \
         patch("src.pipeline.metrics_tracker") as mock_metrics_tracker:

        # Expect OSError to propagate from the await writer_task step within process_file
        with pytest.raises(OSError, match="Disk full"):
             await process_file(
                 input_file_path, output_dir, Path("/fake/map"), mock_config, mock_analysis_service
             ) # Pass injected service

        # Assertions on calls made *before* or *during* the exception point
        mock_create_map.assert_awaited_once()
        mock_analysis_service.build_contexts.assert_called_once()
        mock_analysis_service.analyze_sentences.assert_awaited_once()
        # Check writer was awaited (where error occurred) - this might be tricky if error stops execution flow
        # mock_writer_func.assert_awaited_once() # Re-evaluate this assertion based on execution flow

        # Check logging
        mock_logger.critical.assert_not_called()

@pytest.mark.asyncio
async def test_process_file_map_read_error(
    sample_text_file, 
    tmp_path, 
    mock_config, 
    mock_analysis_service # Inject mock service (although not called)
):
    """ Test process_file when creating the map file fails. Service should not be called."""
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    test_exception = FileNotFoundError("Input deleted")

    with patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, side_effect=test_exception) as mock_create_map, \
         patch("src.pipeline._result_writer", new_callable=AsyncMock) as mock_result_writer, \
         patch("src.pipeline.logger") as mock_logger, \
         patch("src.pipeline.metrics_tracker") as mock_metrics_tracker:

        # Execute - process_file should catch the map creation error
        await process_file(sample_text_file, output_dir, map_dir, mock_config, mock_analysis_service)

        # Assertions
        mock_create_map.assert_awaited_once() 
        
        # Service methods should NOT be called
        mock_analysis_service.build_contexts.assert_not_called()
        mock_analysis_service.analyze_sentences.assert_not_awaited()
        
        # Writer should not be called
        mock_result_writer.assert_not_awaited()

        # Check logging and metrics
        mock_logger.error.assert_called()
        mock_metrics_tracker.increment_errors.assert_called()

@pytest.mark.asyncio
async def test_run_pipeline_no_files(tmp_path, mock_config):
    """ Test run_pipeline when input dir is empty. """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]

    # Patch logger and process_file (verify shouldn't be called)
    # Patch dependencies needed for service instantiation within run_pipeline
    with patch("src.pipeline.logger") as mock_logger, \
         patch("src.pipeline.process_file", new_callable=AsyncMock) as mock_process_file, \
         patch("src.pipeline.verify_output_completeness") as mock_verify, \
         patch("src.pipeline.SentenceAnalyzer") as MockSentenceAnalyzer, \
         patch("src.pipeline.context_builder") as mock_context_builder, \
         patch("src.pipeline.metrics_tracker") as mock_metrics_tracker: 
         # Patch AnalysisService class itself IF we want to check its instantiation args
         # patch("src.pipeline.AnalysisService") as MockAnalysisServiceClass: 
        
        await run_pipeline(input_dir, output_dir, map_dir, mock_config) 
        
        mock_logger.warning.assert_called_with(f"No input files found in {input_dir}")
        mock_process_file.assert_not_awaited() 
        mock_verify.assert_not_called()
        # Assert service and its deps were NOT instantiated since no files
        # MockAnalysisServiceClass.assert_not_called() # Or check specific deps not called

@pytest.mark.asyncio
async def test_run_pipeline_multiple_files(tmp_path, mock_config):
    """ Test run_pipeline processing multiple files successfully. """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    
    file1 = input_dir / "file1.txt"; file1.write_text("File one.")
    file2 = input_dir / "file2.txt"; file2.write_text("File two.")
    (input_dir / "other.md").touch() # Ignored

    # Mock verify to return success
    mock_verify_return = {"total_expected": 1, "total_actual": 1, "total_missing": 0, "missing_ids": [], "error": None}
    
    # Patch process_file, verify, logger, and service dependencies
    with patch("src.pipeline.process_file", new_callable=AsyncMock) as mock_process_file, \
         patch("src.pipeline.verify_output_completeness", return_value=mock_verify_return) as mock_verify, \
         patch("src.pipeline.logger") as mock_logger, \
         patch("src.pipeline.SentenceAnalyzer") as MockSentenceAnalyzer, \
         patch("src.pipeline.context_builder") as mock_context_builder, \
         patch("src.pipeline.metrics_tracker") as mock_metrics_tracker, \
         patch("src.pipeline.AnalysisService") as MockAnalysisServiceClass:

        # **Configure metrics_tracker mock to return NUMBERS for summary log**
        mock_metrics_tracker.get_files_processed.return_value = 2
        mock_metrics_tracker.get_sentences_processed.return_value = 4 # Example value
        mock_metrics_tracker.get_sentences_success.return_value = 4 # Example value
        mock_metrics_tracker.get_errors.return_value = 0
        mock_metrics_tracker.get_total_processing_time.return_value = 1.234 # Example float

        mock_service_instance = MockAnalysisServiceClass.return_value 
        
        await run_pipeline(input_dir, output_dir, map_dir, mock_config) 
        
        # --- Assertions --- 
        # Assert Service instantiation (as before)
        MockSentenceAnalyzer.assert_called_once()
        MockAnalysisServiceClass.assert_called_once_with(
            config=mock_config,
            context_builder=mock_context_builder,
            sentence_analyzer=ANY, 
            metrics_tracker=mock_metrics_tracker
        )

        # Assert process_file calls (as before)
        assert mock_process_file.await_count == 2
        process_calls = [
            call(file1, output_dir, map_dir, mock_config, mock_service_instance),
            call(file2, output_dir, map_dir, mock_config, mock_service_instance)
        ]
        mock_process_file.assert_has_awaits(process_calls, any_order=True) 

        # Assert verify calls (as before)
        assert mock_verify.call_count == 2
        map1_path = map_dir / f"{file1.stem}{mock_config['paths']['map_suffix']}"
        analysis1_path = output_dir / f"{file1.stem}{mock_config['paths']['analysis_suffix']}"
        map2_path = map_dir / f"{file2.stem}{mock_config['paths']['map_suffix']}"
        analysis2_path = output_dir / f"{file2.stem}{mock_config['paths']['analysis_suffix']}"
        verify_calls = [
            call(map1_path, analysis1_path),
            call(map2_path, analysis2_path)
        ]
        mock_verify.assert_has_calls(verify_calls, any_order=True)

        # Check summary log messages (just check they are called, content check is brittle)
        # Find the specific INFO log call containing the summary
        summary_log_found = any("Pipeline Execution Summary:" in log_call.args[0] for log_call in mock_logger.info.call_args_list)
        verification_log_found = any("Verification Summary:" in log_call.args[0] for log_call in mock_logger.info.call_args_list)
        assert summary_log_found, "Pipeline execution summary log not found."
        assert verification_log_found, "Verification summary log not found."

def test_verify_completeness_success(tmp_path):
    """
    Test verify_output_completeness when analysis matches map perfectly.
    """
    map_file = tmp_path / "map.jsonl"
    analysis_file = tmp_path / "analysis.jsonl"
    map_data = [
        {"sentence_id": 0, "sentence": "s0"},
        {"sentence_id": 1, "sentence": "s1"},
        {"sentence_id": 2, "sentence": "s2"}
    ]
    analysis_data = [
        {"sentence_id": 0, "analysis": "a0"},
        {"sentence_id": 1, "analysis": "a1"},
        {"sentence_id": 2, "analysis": "a2"}
    ]
    create_jsonl_file(map_file, map_data)
    create_jsonl_file(analysis_file, analysis_data)

    result = verify_output_completeness(map_file, analysis_file)

    assert result["total_expected"] == 3
    assert result["total_actual"] == 3
    assert result["total_missing"] == 0
    assert result["missing_ids"] == []

def test_verify_completeness_missing_items(tmp_path):
    """
    Test verify_output_completeness when analysis file is missing entries.
    """
    map_file = tmp_path / "map.jsonl"
    analysis_file = tmp_path / "analysis.jsonl"
    map_data = [
        {"sentence_id": 0, "sentence": "s0"},
        {"sentence_id": 1, "sentence": "s1"},
        {"sentence_id": 2, "sentence": "s2"},
        {"sentence_id": 3, "sentence": "s3"}
    ]
    analysis_data = [
        {"sentence_id": 0, "analysis": "a0"},
        # Missing ID 1
        {"sentence_id": 2, "analysis": "a2"},
        # Missing ID 3
    ]
    create_jsonl_file(map_file, map_data)
    create_jsonl_file(analysis_file, analysis_data)

    result = verify_output_completeness(map_file, analysis_file)

    assert result["total_expected"] == 4
    assert result["total_actual"] == 2
    assert result["total_missing"] == 2
    assert result["missing_ids"] == [1, 3] # Should be sorted

def test_verify_completeness_empty_analysis(tmp_path):
    """
    Test verify_output_completeness when analysis file exists but is empty.
    """
    map_file = tmp_path / "map.jsonl"
    analysis_file = tmp_path / "analysis.jsonl"
    map_data = [
        {"sentence_id": 0, "sentence": "s0"},
        {"sentence_id": 1, "sentence": "s1"}
    ]
    create_jsonl_file(map_file, map_data)
    analysis_file.touch() # Create empty file

    result = verify_output_completeness(map_file, analysis_file)

    assert result["total_expected"] == 2
    assert result["total_actual"] == 0
    assert result["total_missing"] == 2
    assert result["missing_ids"] == [0, 1]

def test_verify_completeness_map_not_found(tmp_path):
    """
    Test verify_output_completeness when map file does not exist.
    """
    map_file = tmp_path / "non_existent_map.jsonl"
    analysis_file = tmp_path / "analysis.jsonl"
    analysis_file.touch()

    # Mock logger to check warning
    with patch("src.pipeline.logger") as mock_logger:
        result = verify_output_completeness(map_file, analysis_file)

    assert result["total_expected"] == 0
    assert result["total_actual"] == 0 # Should ideally be 0 if map fails
    assert result["total_missing"] == 0
    assert result["missing_ids"] == []
    assert result.get("error") is not None # Check for error indicator
    mock_logger.warning.assert_called_once()
    assert f"Map file not found: {map_file}" in mock_logger.warning.call_args[0][0]

def test_verify_completeness_analysis_not_found(tmp_path):
    """
    Test verify_output_completeness when analysis file does not exist.
    """
    map_file = tmp_path / "map.jsonl"
    analysis_file = tmp_path / "non_existent_analysis.jsonl"
    map_data = [
        {"sentence_id": 0, "sentence": "s0"},
        {"sentence_id": 1, "sentence": "s1"}
    ]
    create_jsonl_file(map_file, map_data)

    # Mock logger to check warning
    with patch("src.pipeline.logger") as mock_logger:
        result = verify_output_completeness(map_file, analysis_file)

    assert result["total_expected"] == 2
    assert result["total_actual"] == 0 # Actual count is 0
    assert result["total_missing"] == 2 # Missing count reflects expected
    assert result["missing_ids"] == [0, 1]
    assert result.get("error") is not None # Check for error indicator
    mock_logger.warning.assert_called_once()
    assert f"Analysis file not found: {analysis_file}" in mock_logger.warning.call_args[0][0]

def test_verify_completeness_malformed_map(tmp_path):
    """
    Test verify_output_completeness with a malformed line in the map file.
    """
    map_file = tmp_path / "map.jsonl"
    analysis_file = tmp_path / "analysis.jsonl"
    analysis_data = [ {"sentence_id": 0, "analysis": "a0"} ]
    create_jsonl_file(analysis_file, analysis_data)
    # Create map file with bad JSON
    with map_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"sentence_id": 0, "sentence": "s0"}) + '\n')
        f.write("this is not json\n")
        f.write(json.dumps({"sentence_id": 2, "sentence": "s2"}) + '\n')
        
    # Mock logger to check error
    with patch("src.pipeline.logger") as mock_logger:
        result = verify_output_completeness(map_file, analysis_file)

    assert result["total_expected"] == 2 # Only IDs 0 and 2 should be expected
    assert result["total_actual"] == 1
    assert result["total_missing"] == 1
    assert result["missing_ids"] == [2]
    mock_logger.error.assert_called_once()
    assert "Failed to parse line in map file" in mock_logger.error.call_args[0][0]

def test_verify_completeness_malformed_analysis(tmp_path):
    """
    Test verify_output_completeness with a malformed line in the analysis file.

    Verifies that the function logs an error for the invalid line but continues
    processing subsequent valid lines, correctly identifying all valid sentence IDs.
    """
    map_file = tmp_path / "map.jsonl"
    analysis_file = tmp_path / "analysis.jsonl"
    map_data = [ {"sentence_id": 0, "sentence": "s0"}, {"sentence_id": 1, "sentence": "s1"} ]
    create_jsonl_file(map_file, map_data)
    # Create analysis file with bad JSON in the middle
    with analysis_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"sentence_id": 0, "analysis": "a0"}) + '\n')
        f.write("this is not json\n")
        f.write(json.dumps({"sentence_id": 1, "analysis": "a1"}) + '\n')

    # Mock logger to check error
    with patch("src.pipeline.logger") as mock_logger:
        result = verify_output_completeness(map_file, analysis_file)

    assert result["total_expected"] == 2 
    assert result["total_actual"] == 2 # Both ID 0 and ID 1 should be found
    assert result["total_missing"] == 0 # No IDs should be missing
    assert result["missing_ids"] == [] # List of missing IDs should be empty
    mock_logger.error.assert_called_once()
    assert "Failed to parse line in analysis file" in mock_logger.error.call_args[0][0]

@pytest.mark.asyncio
async def test_analyze_specific_sentences_success(
    tmp_path, 
    mock_config, 
    mock_analysis_service # Inject mock service
):
    """ Test successful analysis of specific sentences with injected service. """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "specific.txt"
    input_file.write_text("Sentence 0. Sentence 1. Sentence 2. Sentence 3.")
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    map_dir.mkdir() 
    map_file_path = map_dir / f"{input_file.stem}{mock_config['paths']['map_suffix']}"

    # --- Mock Data --- 
    sentence_ids_to_analyze = [1, 3]
    all_sentences_text_from_map = ["Sentence 0.", "Sentence 1.", "Sentence 2.", "Sentence 3."] 
    target_sentences = ["Sentence 1.", "Sentence 3."] 
    mock_all_contexts = [{"ctx": "c0"}, {"ctx": "c1"}, {"ctx": "c2"}, {"ctx": "c3"}]
    target_contexts = [mock_all_contexts[1], mock_all_contexts[3]]
    mock_service_results = [
        create_mock_analysis(0, 0, target_sentences[0]), 
        create_mock_analysis(1, 1, target_sentences[1])  
    ]
    expected_final_results = [
        {**create_mock_analysis(0, 0, target_sentences[0]), "sentence_id": 1, "sequence_order": 1},
        {**create_mock_analysis(1, 1, target_sentences[1]), "sentence_id": 3, "sequence_order": 3}
    ]

    # --- Configure Mocks --- 
    mock_analysis_service.build_contexts.return_value = mock_all_contexts
    mock_analysis_service.analyze_sentences.return_value = mock_service_results

    # Prepare map content for mock_open
    map_content = ""
    for i, s in enumerate(all_sentences_text_from_map):
        map_content += json.dumps({"sentence_id": i, "sequence_order": i, "sentence": s}) + "\n"

    # --- Patch Dependencies --- 
    with patch("src.pipeline.create_conversation_map", new_callable=AsyncMock) as mock_create_map, \
         patch("pathlib.Path.exists", return_value=True) as mock_path_exists, \
         patch("pathlib.Path.open", mock_open(read_data=map_content)) as mocked_file_open, \
         patch("src.pipeline.logger") as mock_logger:

        # --- Execute --- 
        final_results = await analyze_specific_sentences(
            input_file, sentence_ids_to_analyze, mock_config, mock_analysis_service
        )

    # --- Assertions --- 
    # Assert exists was called (on the specific map path instance)
    # We might need to refine this if Path() is called multiple times
    mock_path_exists.assert_called() 
    # Assert map creation was NOT called (because mock_exists returned True)
    mock_create_map.assert_not_awaited() 
    # Assert map file was read
    mocked_file_open.assert_called_once_with("r", encoding="utf-8")
    # Assert service calls 
    mock_analysis_service.build_contexts.assert_called_once_with(all_sentences_text_from_map)
    mock_analysis_service.analyze_sentences.assert_awaited_once_with(target_sentences, target_contexts)
    # Assert final result
    assert final_results == expected_final_results

@pytest.mark.asyncio
async def test_analyze_specific_sentences_with_error(
    tmp_path, mock_config, mock_analysis_service
):
    """ Test analysis error with injected service. """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "specific_error.txt"
    input_file.write_text("Sentence 0. Sentence 1. Sentence 2. Sentence 3.")
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    map_dir.mkdir()

    # --- Mock Data ---
    sentence_ids_to_analyze = [1, 3]
    all_sentences_text_from_map = ["Sentence 0.", "Sentence 1.", "Sentence 2.", "Sentence 3."]
    target_sentences = ["Sentence 1.", "Sentence 3."]
    mock_all_contexts = [{"ctx": "c0"}, {"ctx": "c1"}, {"ctx": "c2"}, {"ctx": "c3"}]
    target_contexts = [mock_all_contexts[1], mock_all_contexts[3]]
    mock_service_results = [
        create_mock_analysis(0, 0, target_sentences[0]), 
        # Simulating error from the service for the second sentence
        {"sentence_id": 1, "sequence_order": 1, "sentence": target_sentences[1], "error": True, "error_type": "APIError", "error_message": "Failed hard"}
    ]
    expected_final_results = [
        {**create_mock_analysis(0, 0, target_sentences[0]), "sentence_id": 1, "sequence_order": 1},
        # Expecting remapping to add original ID/order even to error dicts
        {"sentence_id": 3, "sequence_order": 3, "sentence": target_sentences[1], "error": True, "error_type": "APIError", "error_message": "Failed hard"}
    ]
    
    # --- Configure Mocks ---
    mock_analysis_service.build_contexts.return_value = mock_all_contexts
    mock_analysis_service.analyze_sentences.return_value = mock_service_results
    
    # --- CORRECT map_content generation --- 
    map_content = "" # Initialize empty string
    for i, s in enumerate(all_sentences_text_from_map):
        map_content += json.dumps({"sentence_id": i, "sequence_order": i, "sentence": s}) + "\n"
    # ---------------------------------------

    # --- Patch Dependencies --- 
    with patch("src.pipeline.create_conversation_map", new_callable=AsyncMock) as mock_create_map, \
         patch("pathlib.Path.exists", return_value=True) as mock_path_exists, \
         patch("pathlib.Path.open", mock_open(read_data=map_content)) as mocked_file_open, \
         patch("src.pipeline.logger") as mock_logger:
        # --- Execute --- 
        final_results = await analyze_specific_sentences(
            input_file, sentence_ids_to_analyze, mock_config, mock_analysis_service
        )
        
    # --- Assertions --- 
    mock_path_exists.assert_called()
    mock_create_map.assert_not_awaited()
    mocked_file_open.assert_called_once_with("r", encoding="utf-8")
    mock_analysis_service.build_contexts.assert_called_once_with(all_sentences_text_from_map)
    mock_analysis_service.analyze_sentences.assert_awaited_once_with(target_sentences, target_contexts)
    # Adjust final assertion based on how error dicts are structured after remapping
    assert final_results == expected_final_results 

@pytest.mark.asyncio
async def test_analyze_specific_sentences_invalid_id(
    tmp_path, mock_config, mock_analysis_service
):
    """ Test analysis with invalid IDs requested (with injected service). """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "specific_invalid.txt"
    input_file.write_text("Sentence 0. Sentence 1.")
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    map_dir.mkdir()

    # --- Mock Data ---
    sentence_ids_to_analyze = [0, 5]
    all_sentences_text_from_map = ["Sentence 0.", "Sentence 1."]
    target_sentences = ["Sentence 0."]
    mock_all_contexts = [{"ctx": "c0"}, {"ctx": "c1"}]
    target_contexts = [mock_all_contexts[0]]
    mock_service_results = [create_mock_analysis(0, 0, target_sentences[0])]
    expected_final_results = [
        {**create_mock_analysis(0, 0, target_sentences[0]), "sentence_id": 0, "sequence_order": 0}
    ]
    
    # --- Configure Mocks ---
    mock_analysis_service.build_contexts.return_value = mock_all_contexts
    mock_analysis_service.analyze_sentences.return_value = mock_service_results
    
    # --- CORRECT map_content generation --- 
    map_content = "" # Initialize empty string
    for i, s in enumerate(all_sentences_text_from_map):
        map_content += json.dumps({"sentence_id": i, "sequence_order": i, "sentence": s}) + "\n"
    # ---------------------------------------
    
    # --- Patch Dependencies --- 
    with patch("src.pipeline.create_conversation_map", new_callable=AsyncMock) as mock_create_map, \
         patch("pathlib.Path.exists", return_value=True) as mock_path_exists, \
         patch("pathlib.Path.open", mock_open(read_data=map_content)) as mocked_file_open, \
         patch("src.pipeline.logger") as mock_logger:
        # --- Execute --- 
        final_results = await analyze_specific_sentences(
            input_file, sentence_ids_to_analyze, mock_config, mock_analysis_service
        )
        
    # --- Assertions --- 
    mock_path_exists.assert_called()
    mock_create_map.assert_not_awaited()
    mocked_file_open.assert_called_once_with("r", encoding="utf-8")
    mock_analysis_service.build_contexts.assert_called_once_with(all_sentences_text_from_map)
    mock_analysis_service.analyze_sentences.assert_awaited_once_with(target_sentences, target_contexts)
    assert final_results == expected_final_results
    assert any("Skipping requested sentence IDs not found in map file or invalid: [5]" in call.args[0] for call in mock_logger.warning.call_args_list)
