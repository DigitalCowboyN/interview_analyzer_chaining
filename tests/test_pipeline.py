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
from unittest.mock import patch, AsyncMock, MagicMock, call, ANY # Import ANY for flexible mocking
from typing import List, Dict

# Assume these will be the new/refactored imports from src.pipeline
from src.pipeline import segment_text, run_pipeline, create_conversation_map, process_file, verify_output_completeness, analyze_specific_sentences
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

    # Patch segment_text if it's called internally
    with patch("src.pipeline.segment_text", return_value=expected_sentences):
        with patch("src.pipeline.logger"):
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

    with patch("src.pipeline.segment_text", return_value=[]):
        with patch("src.pipeline.logger"):
            # Expect tuple return value
            sentence_count, returned_sentences = await create_conversation_map(input_file, map_dir, mock_config["paths"]["map_suffix"])

    assert sentence_count == 0
    assert returned_sentences == [] # Check returned sentences
    assert map_dir.exists() # Directory should be created
    assert expected_map_file.exists() # File should be created
    assert expected_map_file.read_text() == "" # File should be empty

# --- Tests for process_file (Refactored Logic) ---

# Mock Analysis Result Structure (add sentence_id, sequence_order)
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
async def test_process_file_success(sample_text_file, tmp_path, mock_config):
    """ 
    Test `process_file` successful execution with mocked dependencies.

    Verifies the main success path of `process_file`, ensuring:
    - Map creation is called correctly.
    - Contexts are built for the sentences.
    - Tasks are loaded based on the map.
    - Sentence analysis (mocked) is performed for each sentence with correct context.
    - Results are written to the output file.
    - Queues are managed, and workers/writer are shut down correctly.

    Uses extensive mocking for `create_conversation_map`, `SentenceAnalyzer`,
    `context_builder`, `_load_tasks`, and `append_json_line`.

    Args:
        sample_text_file: Fixture providing a path to a sample input file.
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If mocks are not called as expected, output file content
                      is incorrect, or context passing is wrong.
    """ 
    from src.pipeline import process_file

    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    expected_analysis_file = output_dir / f"{sample_text_file.stem}{mock_config['paths']['analysis_suffix']}"

    # --- Mock Contexts ---
    # Define specific mock contexts to be returned by ContextBuilder
    mock_contexts_list = [
        {"immediate_context": "ctx0", "observer_context": "obs0", "broader_context": "broad0"}, # Context for sentence 0
        {"immediate_context": "ctx1", "observer_context": "obs1", "broader_context": "broad1"}, # Context for sentence 1
    ]
    mock_context_builder_instance = MagicMock()
    mock_context_builder_instance.build_all_contexts.return_value = mock_contexts_list

    # --- Mock Analyzer ---
    mock_analyzer_instance = MagicMock()
    # Store calls to verify context later
    classify_calls = []
    async def mock_classify_side_effect_track_context(s, c):
        sentence_id = 0 if "First" in s else 1
        result = create_mock_analysis(sentence_id, sentence_id, s)
        classify_calls.append({'sentence': s, 'context': c, 'result': result})
        return result
        
    mock_analyzer_instance.classify_sentence = AsyncMock(side_effect=mock_classify_side_effect_track_context)

    # --- Patching ---
    # Define a side effect for the loader mock to put items on the queue
    async def mock_load_tasks_side_effect(map_file, contexts, task_queue):
        # Simulate loading tasks based on mock_sentences_list
        for i, sentence in enumerate(mock_sentences_list):
            task_item = (i, i, sentence, contexts[i]) # (id, order, sentence, context)
            await task_queue.put(task_item)
        print(f"Mock loader finished putting {len(mock_sentences_list)} items on queue.") # Debug print

    mock_sentences_list = ["First sentence.", "Second sentence."]
    with (patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(2, mock_sentences_list)) as mock_create_map, 
          patch("src.pipeline.SentenceAnalyzer", return_value=mock_analyzer_instance) as mock_analyzer_class, 
          patch("src.pipeline.context_builder", mock_context_builder_instance) as mock_patched_context_builder, 
          patch("src.pipeline._load_tasks", side_effect=mock_load_tasks_side_effect) as mock_load_tasks,
          patch("src.pipeline.append_json_line") as mock_append_json, 
          patch("src.pipeline.logger") as mock_logger, 
          patch("src.pipeline.metrics_tracker") as mock_metrics): # Removed patch for asyncio.Queue

        # --- Execute ---
        # No longer need to patch segment_text here as process_file uses sentences from create_map
        await process_file(sample_text_file, output_dir, map_dir, mock_config)

        # --- Asserts ---
        # 1. Map creation called
        mock_create_map.assert_awaited_once_with(sample_text_file, map_dir, mock_config['paths']['map_suffix'])

        # 2. Directories created (implicitly tested if files are written, or mock mkdir)
        # We assume process_file ensures output_dir exists
        # mock_output_dir_path = Path(output_dir)
        # mock_output_dir_path.mkdir.assert_called_once_with(parents=True, exist_ok=True) 

        # 3. Context Builder called with correct segments (if patching segment_text)
        mock_patched_context_builder.build_all_contexts.assert_called_once_with(["First sentence.", "Second sentence."])

        # 4. Sentence Analyzer used
        mock_analyzer_class.assert_called_once() # Instantiated once
        # classify_sentence should be called twice (once per sentence) by workers
        assert mock_analyzer_instance.classify_sentence.await_count == 2

        # 5. Results written via append_json_line
        assert mock_append_json.call_count == 2
        # Check calls (order might vary due to concurrency)
        call_args_list = [call[0][0] for call in mock_append_json.call_args_list] # Get the 'data' arg from each call
        # Recreate expected results based on the input sentences
        expected_result1 = classify_calls[0]['result']
        expected_result2 = classify_calls[1]['result']

        # Check if both expected results were arguments to append_json_line, regardless of order
        assert expected_result1 in call_args_list
        assert expected_result2 in call_args_list
        # Check the file path argument for append_json_line
        mock_append_json.assert_called_with(ANY, expected_analysis_file) # ANY matches the data dict

        # 6. Logging and Metrics (Basic checks)
        mock_logger.info.assert_called() # Check if info logs happened
        # Check specific metric calls if needed, e.g., success count
        # mock_metrics.increment_success.assert_called()

        # 7. Correct Context Passed to Analyzer
        assert len(classify_calls) == 2
        # Find the call for the first sentence
        call1 = next((call for call in classify_calls if call['sentence'] == "First sentence."), None)
        assert call1 is not None
        assert call1['context'] == mock_contexts_list[0] # Check context matches mock for sentence 0
        # Find the call for the second sentence
        call2 = next((call for call in classify_calls if call['sentence'] == "Second sentence."), None)
        assert call2 is not None
        assert call2['context'] == mock_contexts_list[1] # Check context matches mock for sentence 1

@pytest.mark.asyncio
async def test_process_file_analyzer_error(sample_text_file, tmp_path, mock_config):
    """
    Test `process_file` handling an error during sentence analysis.

    Mocks `SentenceAnalyzer.classify_sentence` to raise an exception for one sentence.
    Verifies that:
    - The pipeline continues processing other sentences.
    - The error is logged (via mocked logger).
    - The final output file contains entries for *both* the successfully analyzed
      sentence and the failed one (marked with an error flag).
    - Metrics tracker correctly counts the error.

    Args:
        sample_text_file: Fixture providing a path to a sample input file.
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If the output file does not contain both success and error
                      results, the error is not logged, or metrics are not updated.
    """
    from src.pipeline import process_file
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    expected_analysis_file = output_dir / f"{sample_text_file.stem}{mock_config['paths']['analysis_suffix']}"

    # Mock SentenceAnalyzer to fail on the second sentence
    mock_analyzer_instance = MagicMock()
    fail_msg = "Mock analysis failed!"
    async def mock_classify_side_effect_error(sentence, context):
        if "Second" in sentence:
            raise ValueError(fail_msg)
        else:
            # For the successful case, return the dict directly
            return create_mock_analysis(0, 0, sentence)

    mock_analyzer_instance.classify_sentence = AsyncMock(side_effect=mock_classify_side_effect_error)

    # Define mock loader side effect (same as above)
    async def mock_load_tasks_side_effect(map_file, contexts, task_queue):
        mock_sentences_list = ["First sentence.", "Second sentence."]
        for i, sentence in enumerate(mock_sentences_list):
            task_item = (i, i, sentence, contexts[i])
            await task_queue.put(task_item)

    mock_sentences_list = ["First sentence.", "Second sentence."] # Needed for side effect scope
    with (patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(2, mock_sentences_list)),
          patch("src.pipeline.SentenceAnalyzer", return_value=mock_analyzer_instance),
          patch("src.pipeline.context_builder") as mock_patched_context_builder,
          patch("src.pipeline._load_tasks", side_effect=mock_load_tasks_side_effect) as mock_load_tasks,
          patch("src.pipeline.append_json_line") as mock_append_json,
          patch("src.pipeline.logger") as mock_logger,
          patch("src.pipeline.metrics_tracker") as mock_metrics): # Removed patch for asyncio.Queue

        # Configure the mock context builder instance returned by the patch
        mock_patched_context_builder.build_all_contexts.return_value = [{}, {}]

        await process_file(sample_text_file, output_dir, map_dir, mock_config)

        # Asserts
        # 1. Analyzer called twice (attempted for both sentences)
        assert mock_analyzer_instance.classify_sentence.await_count == 2

        # 2. append_json_line called TWICE (once for success, once for error)
        assert mock_append_json.call_count == 2
        
        # 3. Check the content written
        call_args_list = [call[0][0] for call in mock_append_json.call_args_list] # Get the 'data' arg from each call
        
        # Find the successful result (sentence_id 0)
        success_result = next((res for res in call_args_list if res.get("sentence_id") == 0), None)
        assert success_result is not None
        assert success_result.get("error") is not True # Ensure error flag is not set
        assert success_result.get("sentence") == "First sentence."
        assert success_result.get("function_type") == "mock_declarative" # Check analysis field

        # Find the error result (sentence_id 1)
        error_result = next((res for res in call_args_list if res.get("sentence_id") == 1), None)
        assert error_result is not None
        assert error_result.get("error") is True
        assert error_result.get("sentence") == "Second sentence."
        assert error_result.get("error_type") == "ValueError"
        assert error_result.get("error_message") == fail_msg
        # Check that analysis fields are NOT present in error result
        assert "function_type" not in error_result 
        
        # Check the file path argument for append_json_line
        # (assert_called_with checks the *last* call, use call_args_list for all)
        assert all(call[0][1] == expected_analysis_file for call in mock_append_json.call_args_list)

        # 4. Error logged by worker
        mock_logger.error.assert_called_once()
        # Check log message content for the specific error and sentence id
        error_log_call = mock_logger.error.call_args
        log_message = error_log_call[0][0]
        assert "failed analyzing sentence_id 1" in log_message # Check sentence ID part
        assert f"ValueError: {fail_msg}" in log_message # Check error type and message part
        
        # 5. Metrics updated for error
        mock_metrics.increment_errors.assert_called_once()

@pytest.mark.asyncio
async def test_process_file_writer_error(sample_text_file, tmp_path, mock_config):
    """
    Test `process_file` handling an error during result writing.

    Mocks `append_json_line` to raise an exception when writing one of the results.
    Verifies that:
    - The pipeline attempts to write all results.
    - The error during writing is logged (via mocked logger).
    - Metrics tracker correctly counts the error.
    - The output file may contain results written before the error occurred.

    Args:
        sample_text_file: Fixture providing a path to a sample input file.
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If the write error is not logged or metrics not updated.
    """
    from src.pipeline import process_file
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]

    # Mock append_json_line to raise an error
    mock_append_json = MagicMock(side_effect=IOError("Disk full!"))

    # Mock analyzer to return successfully
    mock_analyzer_instance = MagicMock()
    async def mock_classify_side_effect_success(s, c):
        sentence_id = 0 if "First" in s else 1
        return create_mock_analysis(sentence_id, sentence_id, s)
    mock_analyzer_instance.classify_sentence = AsyncMock(side_effect=mock_classify_side_effect_success)

    # Define mock loader side effect (same as above)
    async def mock_load_tasks_side_effect(map_file, contexts, task_queue):
        mock_sentences_list = ["First sentence.", "Second sentence."]
        for i, sentence in enumerate(mock_sentences_list):
            task_item = (i, i, sentence, contexts[i])
            await task_queue.put(task_item)

    mock_sentences_list = ["First sentence.", "Second sentence."] # Needed for side effect scope
    with (patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(2, mock_sentences_list)), 
          patch("src.pipeline.SentenceAnalyzer", return_value=mock_analyzer_instance), 
          patch("src.pipeline.context_builder") as mock_patched_context_builder, 
          patch("src.pipeline._load_tasks", side_effect=mock_load_tasks_side_effect) as mock_load_tasks, 
          patch("src.pipeline.append_json_line", mock_append_json), # Use our failing append mock
          patch("src.pipeline.logger") as mock_logger, 
          patch("src.pipeline.metrics_tracker") as mock_metrics): # Removed patch for asyncio.Queue

        # Configure the mock context builder instance returned by the patch
        mock_patched_context_builder.build_all_contexts.return_value = [{}, {}]

        await process_file(sample_text_file, output_dir, map_dir, mock_config)

        # Asserts
        # 1. Analyzer still called for both sentences
        assert mock_analyzer_instance.classify_sentence.await_count == 2
        
        # 2. append_json_line was attempted (likely twice, depending on queue/writer interaction)
        assert mock_append_json.call_count > 0 
        
        # 3. Critical Error logged due to write failure
        mock_logger.error.assert_called() # Or critical, depending on implementation
        # Check that the log message indicates a write error
        write_error_logged = any("Disk full!" in call.args[0] for call in mock_logger.error.call_args_list)
        assert write_error_logged
        
        # 4. Metrics updated for write error (assuming a specific metric exists)
        # mock_metrics.increment_write_errors.assert_called() 
        # Or just general errors incremented
        mock_metrics.increment_errors.assert_called()

@pytest.mark.asyncio
async def test_process_file_map_read_error(sample_text_file, tmp_path, mock_config):
    """
    Test `process_file` handling an error during map file reading/task loading.

    Mocks `_load_tasks` (or simulates an error during map file access within it)
    to raise an exception (e.g., FileNotFoundError or JSONDecodeError).
    Verifies that:
    - The error during task loading is logged.
    - Metrics tracker correctly counts the error.
    - The pipeline stops processing for this file, and no analysis results are written.

    Args:
        sample_text_file: Fixture providing a path to a sample input file.
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If the loading error is not logged, metrics not updated, or if
                      analysis results are unexpectedly written.
    """
    from src.pipeline import process_file
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    
    # Expect an OSError from process_file when the loader raises it
    mock_map_read_error = OSError("Cannot read map file!")

    mock_sentences_list = ["Sentence 1.", "Sentence 2."] # Needed for map mock return

    with (patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(2, mock_sentences_list)) as mock_create_map, 
          patch("src.pipeline.SentenceAnalyzer") as mock_analyzer_class, # Mock analyzer, won't be used
          patch("src.pipeline.context_builder") as mock_patched_context_builder,
          patch("src.pipeline._load_tasks", new_callable=AsyncMock, side_effect=mock_map_read_error) as mock_load_tasks, # Mock loader to raise error
          patch("src.pipeline._result_writer", new_callable=AsyncMock) as mock_result_writer, # Mock writer, won't be used
          patch("src.pipeline.logger") as mock_logger, 
          patch("src.pipeline.metrics_tracker") as mock_metrics):

        mock_patched_context_builder.build_all_contexts.return_value = [{}, {}]

        # Expect the specific OSError to propagate from process_file
        with pytest.raises(OSError, match="Cannot read map file!"):
            await process_file(sample_text_file, output_dir, map_dir, mock_config)

        # Asserts
        # 1. Map creation was attempted
        mock_create_map.assert_awaited_once()

        # 2. Loader task was awaited and raised the error
        mock_load_tasks.assert_awaited_once() 

        # 3. Context builder was called (happens before loader starts)
        mock_patched_context_builder.build_all_contexts.assert_called_once()

        # 4. Analyzer and Writer tasks should NOT have been called significantly
        #    (They might be created but cancelled quickly)
        #    We don't need strict asserts here, focus on error propagation and logs
        # mock_analyzer_instance.classify_sentence.assert_not_called()
        # mock_result_writer.assert_not_awaited() # Or assert_not_called if structure changes

        # 5. Error logged by process_file's handler
        #    The logs from within _load_tasks won't happen because the mock raises instantly.
        # assert any("Map file not found during task loading" in call.args[0] for call in mock_logger.error.call_args_list) or \
        #       any("Unexpected error loading tasks" in call.args[0] for call in mock_logger.error.call_args_list)
        
        # Check for the error logged by the process_file exception handler
        assert any("Exception during task coordination" in call.args[0] for call in mock_logger.error.call_args_list)

        # 6. Metrics updated for error (potentially twice: once in loader, once in process_file, but loader mock skips internal increment)
        #    Therefore, expect only the increment from the process_file handler.
        mock_metrics.increment_errors.assert_called_once() 

@pytest.mark.asyncio
async def test_process_file_zero_workers(sample_text_file, tmp_path, mock_config):
    """
    Test `process_file` handling configuration with zero analysis workers.

    Modifies the mock config to set `num_analysis_workers` to 0.
    Verifies that:
    - An error is logged indicating zero workers is invalid.
    - Metrics tracker counts the error.
    - No analysis is attempted, and the output file remains empty.
    - `create_conversation_map` might still be called initially.

    Args:
        sample_text_file: Fixture providing a path to a sample input file.
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration (modified in test).

    Returns:
        None

    Raises:
        AssertionError: If the zero worker error is not logged, metrics not updated,
                      or if analysis/writing is unexpectedly performed.
    """
    from src.pipeline import process_file
    output_dir = tmp_path / mock_config["paths"]["output_dir"]

    # Create a deep copy of the config for this test to avoid side effects
    test_specific_config = copy.deepcopy(mock_config)
    # Override worker count in the copied config
    test_specific_config["pipeline"]["num_analysis_workers"] = 0

    map_dir = tmp_path / test_specific_config["paths"]["map_dir"]
    
    mock_sentences_list = ["First sentence.", "Second sentence."]
    with (patch("src.pipeline.create_conversation_map", new_callable=AsyncMock, return_value=(2, mock_sentences_list)), 
          patch("src.pipeline.SentenceAnalyzer") as mock_analyzer_class, 
          patch("src.pipeline.context_builder") as mock_patched_context_builder, 
          patch("src.pipeline.append_json_line") as mock_append_json, 
          patch("src.pipeline.logger") as mock_logger, 
          patch("src.pipeline.metrics_tracker") as mock_metrics,
          patch("asyncio.create_task") as mock_create_task): 

        mock_patched_context_builder.build_all_contexts.return_value = [{}, {}]

        # Pass the test-specific config
        await process_file(sample_text_file, output_dir, map_dir, test_specific_config)

        # Asserts
        # 1. Should log an error about zero workers
        assert mock_logger.error.called # Expect error, not warning
        # Use the value from the config passed to the function
        expected_log_substring = f"num_analysis_workers set to {test_specific_config['pipeline']['num_analysis_workers']}"
        correct_log_found = any(expected_log_substring in call.args[0]
                              for call in mock_logger.error.call_args_list)
        assert correct_log_found, f"Did not find expected log substring: '{expected_log_substring}' in error logs: {mock_logger.error.call_args_list}"

        # 2. No tasks should be created when workers is 0
        mock_create_task.assert_not_called()

        # 3. Analysis should not proceed
        mock_analyzer_class.return_value.classify_sentence.assert_not_called()

        # 4. No results should be written
        mock_append_json.assert_not_called()
        
        # 5. Metrics updated for the error
        mock_metrics.increment_errors.assert_called_once()

@pytest.mark.asyncio
async def test_run_pipeline_no_files(tmp_path, mock_config):
    """
    Test `run_pipeline` when the input directory contains no .txt files.

    Sets up an empty input directory and calls `run_pipeline`.
    Verifies that:
    - A warning is logged about no files being found.
    - No processing functions (like `process_file`) are called.
    - `verify_output_completeness` is not called.
    - The output/map directories might be created but remain empty.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If the warning is not logged or if processing is attempted.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]

    # Patch logger, process_file, and verify_output_completeness
    with (patch("src.pipeline.logger") as mock_logger,
          patch("src.pipeline.process_file", new_callable=AsyncMock) as mock_process_file,
          patch("src.pipeline.verify_output_completeness") as mock_verify):
        
        await run_pipeline(input_dir, output_dir, map_dir, mock_config) 
        
        mock_logger.warning.assert_called_with(f"No input files found in {input_dir}")
        mock_process_file.assert_not_awaited() # process_file should not be called
        mock_verify.assert_not_called() # verify should not be called

@pytest.mark.asyncio
async def test_run_pipeline_multiple_files(tmp_path, mock_config):
    """
    Test `run_pipeline` processing multiple files successfully.

    Sets up an input directory with multiple .txt files (including one empty).
    Mocks `process_file` and `verify_output_completeness`.
    Verifies that:
    - `process_file` is called once for each .txt file found.
    - `verify_output_completeness` is called for each processed file.
    - The final summary log includes completeness metrics.
    - The overall pipeline metrics (timer) are managed.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If mocks are not called the correct number of times or the
                      summary log is missing.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    
    # Create dummy files
    file1 = input_dir / "file1.txt"
    file2 = input_dir / "file2.txt"
    file3 = input_dir / "other.md" # Should be ignored
    file1.write_text("File one.")
    file2.write_text("File two.")
    file3.write_text("Markdown file.")

    # Patch process_file and verify_output_completeness
    # Mock verify to return success
    mock_verify_return = {"total_expected": 1, "total_actual": 1, "total_missing": 0, "missing_ids": [], "error": None}
    with (patch("src.pipeline.process_file", new_callable=AsyncMock) as mock_process_file,
          patch("src.pipeline.verify_output_completeness", return_value=mock_verify_return) as mock_verify,
          patch("src.pipeline.logger") as mock_logger):
        
        await run_pipeline(input_dir, output_dir, map_dir, mock_config) # Pass map_dir and config
        
        # Assert process_file called twice (once for each .txt file)
        assert mock_process_file.await_count == 2
        
        # Check calls more specifically for process_file
        process_calls = [
            call(file1, output_dir, map_dir, mock_config),
            call(file2, output_dir, map_dir, mock_config)
        ]
        mock_process_file.assert_has_awaits(process_calls, any_order=True) # Order might vary

        # Assert verify_output_completeness called twice
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

        # Check that the completeness summary was logged correctly
        summary_log_found = False
        verification_log_found = False
        for log_call in mock_logger.info.call_args_list:
            log_msg = log_call[0][0]
            if "Pipeline Execution Summary:" in log_msg:
                summary_log_found = True
            if "Verification Summary:" in log_msg:
                verification_log_found = True
                assert "Checked 2 files" in log_msg
                assert "Total Expected Sentences: 2" in log_msg # 1 expected per file from mock
                assert "Total Actual Entries: 2" in log_msg    # 1 actual per file from mock
                assert "Total Missing Sentences: 0" in log_msg # 0 missing per file from mock
                assert "Verification Errors: 0" in log_msg
                
        assert summary_log_found, "Metrics summary log message not found"
        assert verification_log_found, "Verification summary log message not found"

@pytest.mark.asyncio
async def test_run_pipeline_with_missing(tmp_path, mock_config):
    """
    Test `run_pipeline` reporting missing sentences in the summary log.

    Mocks `verify_output_completeness` to return data indicating missing sentences
    for one of the processed files.
    Verifies that the final summary log message correctly reflects the aggregated
    count of missing sentences and files with missing entries.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        mock_config: Fixture providing mock configuration.

    Returns:
        None

    Raises:
        AssertionError: If the summary log does not report the missing sentences correctly.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / mock_config["paths"]["output_dir"]
    map_dir = tmp_path / mock_config["paths"]["map_dir"]
    
    file1 = input_dir / "file1.txt"
    file2 = input_dir / "file2.txt"
    file1.write_text("File one.")
    file2.write_text("File two.")

    # Mock verify results: file1 OK, file2 has 1 missing
    mock_verify_results = [
        {"total_expected": 2, "total_actual": 2, "total_missing": 0, "missing_ids": [], "error": None}, # file1
        {"total_expected": 3, "total_actual": 2, "total_missing": 1, "missing_ids": [1], "error": None}  # file2
    ]

    with (patch("src.pipeline.process_file", new_callable=AsyncMock) as mock_process_file,
          patch("src.pipeline.verify_output_completeness", side_effect=mock_verify_results) as mock_verify,
          patch("src.pipeline.logger") as mock_logger):
        
        await run_pipeline(input_dir, output_dir, map_dir, mock_config)
        
        # Assert process_file called twice
        assert mock_process_file.await_count == 2
        # Assert verify called twice
        assert mock_verify.call_count == 2

        # Check the final summary log message for aggregated missing counts
        verification_log_found = False
        for log_call in mock_logger.warning.call_args_list: # Missing sentences trigger WARNING level
             log_msg = log_call[0][0]
             if "Verification Summary:" in log_msg:
                verification_log_found = True
                assert "Checked 2 files" in log_msg
                assert "Total Expected Sentences: 5" in log_msg # 2 + 3
                assert "Total Actual Entries: 4" in log_msg    # 2 + 2
                assert "Total Missing Sentences: 1" in log_msg # 0 + 1
                assert "across 1 files" in log_msg # Only file2 had missing
                assert "Verification Errors: 0" in log_msg
                break 
        
        assert verification_log_found, "Verification summary log message (warning level) not found or incorrect"

# --- Tests for verify_output_completeness ---

# Helper to create dummy JSONL files
def create_jsonl_file(path: Path, data: List[Dict]):
    """Creates a JSONL file with the given data."""
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

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
async def test_analyze_specific_sentences_success(tmp_path, mock_config):
    """
    Test `analyze_specific_sentences` successfully analyzing a subset.
    """
    from src.pipeline import analyze_specific_sentences # Import the new function
    
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sentence 0. Sentence 1 is target. Sentence 2 is also target. Sentence 3.")
    sentence_ids_to_analyze = [1, 2] # Analyze only the middle two
    
    # Mock dependencies
    mock_analyzer_instance = MagicMock()
    # Store calls to verify context/sentence passing if needed
    classify_calls = [] 
    async def mock_classify(sentence, context):
        # Determine sentence_id based on text for simplicity in mock
        sentence_id = 1 if " 1 " in sentence else (2 if " 2 " in sentence else -1)
        result = create_mock_analysis(sentence_id, sentence_id, sentence)
        classify_calls.append({'sentence': sentence, 'context': context, 'result': result})
        return result
    mock_analyzer_instance.classify_sentence = AsyncMock(side_effect=mock_classify)
    
    mock_context_builder_instance = MagicMock()
    # Mock context builder to return plausible dicts for each sentence index
    mock_context_builder_instance.build_all_contexts.return_value = {
        0: {"ctx": "c0"}, 1: {"ctx": "c1"}, 2: {"ctx": "c2"}, 3: {"ctx": "c3"}
    }
    
    with (patch("src.pipeline.segment_text", return_value=["Sentence 0.", "Sentence 1 is target.", "Sentence 2 is also target.", "Sentence 3."]),
          patch("src.pipeline.context_builder", mock_context_builder_instance),
          patch("src.pipeline.SentenceAnalyzer", return_value=mock_analyzer_instance),
          patch("src.pipeline.logger")):
              
        results = await analyze_specific_sentences(input_file, sentence_ids_to_analyze, mock_config)

    # Assertions
    assert len(results) == 2 # Should only get results for IDs 1 and 2
    # Check content (order might vary due to gather)
    results_by_id = {res["sentence_id"]: res for res in results}
    assert 1 in results_by_id
    assert 2 in results_by_id
    assert results_by_id[1]["sentence"] == "Sentence 1 is target."
    assert results_by_id[1].get("error") is not True
    assert results_by_id[2]["sentence"] == "Sentence 2 is also target."
    assert results_by_id[2].get("error") is not True
    assert mock_analyzer_instance.classify_sentence.await_count == 2
    # Check context passing (optional but good)
    call1_context = next(call['context'] for call in classify_calls if call['sentence'] == "Sentence 1 is target.")
    call2_context = next(call['context'] for call in classify_calls if call['sentence'] == "Sentence 2 is also target.")
    assert call1_context == {"ctx": "c1"}
    assert call2_context == {"ctx": "c2"}

@pytest.mark.asyncio
async def test_analyze_specific_sentences_with_error(tmp_path, mock_config):
    """
    Test `analyze_specific_sentences` when one analysis fails.
    """
    from src.pipeline import analyze_specific_sentences
    
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sentence 0. Sentence 1 fails. Sentence 2 works.")
    sentence_ids_to_analyze = [1, 2]
    fail_msg = "Analysis failed for sentence 1"

    # Mock dependencies
    mock_analyzer_instance = MagicMock()
    async def mock_classify_error(sentence, context):
        if " 1 " in sentence:
            raise ValueError(fail_msg)
        else:
            return create_mock_analysis(2, 2, sentence)
    mock_analyzer_instance.classify_sentence = AsyncMock(side_effect=mock_classify_error)
    
    mock_context_builder_instance = MagicMock()
    mock_context_builder_instance.build_all_contexts.return_value = {
        0: {"ctx": "c0"}, 1: {"ctx": "c1"}, 2: {"ctx": "c2"}
    }
    
    with (patch("src.pipeline.segment_text", return_value=["Sentence 0.", "Sentence 1 fails.", "Sentence 2 works."]),
          patch("src.pipeline.context_builder", mock_context_builder_instance),
          patch("src.pipeline.SentenceAnalyzer", return_value=mock_analyzer_instance),
          patch("src.pipeline.logger")):
              
        results = await analyze_specific_sentences(input_file, sentence_ids_to_analyze, mock_config)

    # Assertions
    assert len(results) == 2
    results_by_id = {res["sentence_id"]: res for res in results}
    assert 1 in results_by_id
    assert 2 in results_by_id
    
    # Check error result for ID 1
    assert results_by_id[1].get("error") is True
    assert results_by_id[1]["sentence"] == "Sentence 1 fails."
    assert results_by_id[1]["error_type"] == "ValueError"
    assert results_by_id[1]["error_message"] == fail_msg

    # Check success result for ID 2
    assert results_by_id[2].get("error") is not True
    assert results_by_id[2]["sentence"] == "Sentence 2 works."
    assert mock_analyzer_instance.classify_sentence.await_count == 2

@pytest.mark.asyncio
async def test_analyze_specific_sentences_invalid_id(tmp_path, mock_config):
    """
    Test `analyze_specific_sentences` handling invalid sentence IDs.
    """
    from src.pipeline import analyze_specific_sentences
    
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sentence 0. Sentence 1.") # Only 2 sentences (IDs 0, 1)
    sentence_ids_to_analyze = [0, 2] # ID 2 is out of bounds
    
    # Mock dependencies (analyzer won't be called for invalid ID)
    mock_analyzer_instance = MagicMock()
    async def mock_classify(sentence, context): 
         return create_mock_analysis(0, 0, sentence) # Only called for ID 0
    mock_analyzer_instance.classify_sentence = AsyncMock(side_effect=mock_classify)
    
    mock_context_builder_instance = MagicMock()
    mock_context_builder_instance.build_all_contexts.return_value = {0: {"ctx": "c0"}, 1: {"ctx": "c1"}}
    
    with (patch("src.pipeline.segment_text", return_value=["Sentence 0.", "Sentence 1."]),
          patch("src.pipeline.context_builder", mock_context_builder_instance),
          patch("src.pipeline.SentenceAnalyzer", return_value=mock_analyzer_instance),
          patch("src.pipeline.logger") as mock_logger):
              
        results = await analyze_specific_sentences(input_file, sentence_ids_to_analyze, mock_config)

    # Assertions
    assert len(results) == 1 # Only the result for ID 0 should be returned
    assert results[0]["sentence_id"] == 0
    assert results[0].get("error") is not True
    mock_analyzer_instance.classify_sentence.assert_awaited_once() # Only called for ID 0
    # Check warning log for invalid ID
    mock_logger.warning.assert_called_once()
    assert "Invalid sentence_id requested: 2" in mock_logger.warning.call_args[0][0]

@pytest.mark.asyncio
async def test_analyze_specific_sentences_empty_list(tmp_path, mock_config):
    """
    Test `analyze_specific_sentences` with an empty list of IDs.
    """
    from src.pipeline import analyze_specific_sentences
    
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sentence 0. Sentence 1.") 
    sentence_ids_to_analyze = [] # Empty list
    
    # Mock dependencies (should not be called)
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance.classify_sentence = AsyncMock()
    mock_context_builder_instance = MagicMock()
    mock_context_builder_instance.build_all_contexts.return_value = {0: {}, 1: {}}

    with (patch("src.pipeline.segment_text", return_value=["Sentence 0.", "Sentence 1."]),
          patch("src.pipeline.context_builder", mock_context_builder_instance),
          patch("src.pipeline.SentenceAnalyzer", return_value=mock_analyzer_instance),
          patch("src.pipeline.logger")):
              
        results = await analyze_specific_sentences(input_file, sentence_ids_to_analyze, mock_config)

    # Assertions
    assert results == [] # Expect empty list back
    mock_analyzer_instance.classify_sentence.assert_not_awaited() # Analyzer should not be called
