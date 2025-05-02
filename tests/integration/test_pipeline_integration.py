"""
tests/integration/test_pipeline_integration.py

Contains end-to-end integration tests for the main processing pipeline defined
in `src.pipeline.run_pipeline`.

These tests verify the overall pipeline flow, including:
- Processing of text files from an input directory.
- Creation of intermediate map files.
- Creation of final analysis output files.
- Handling of successful runs, partial failures (mocked API errors), and empty files.
- Correct aggregation of results.

External dependencies, specifically the OpenAI API calls made via
`src.agents.agent.OpenAIAgent.call_model`, are mocked to ensure tests are
repeatable and do not rely on external services.
"""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, ANY, MagicMock
from typing import Dict, Optional, Any

# Assuming main entry point or run_pipeline can be invoked
from src.pipeline import run_pipeline, PipelineOrchestrator # Updated Import
from src.config import config # Use the actual config
from src.utils.neo4j_driver import Neo4jConnectionManager # Import for type hint

# --- Fixtures ---

@pytest.fixture(scope="function") # Use function scope for isolation
def integration_dirs(tmp_path_factory):
    """
    Pytest fixture to create temporary input, output, and map directories.

    Uses pytest's `tmp_path_factory` to create session-scoped temporary
    directories, ensuring test isolation.

    Args:
        tmp_path_factory: Pytest fixture for creating temporary directories.

    Returns:
        tuple[Path, Path, Path]: A tuple containing the paths to the created
                                 input, output, and map directories.
    """
    base_dir = tmp_path_factory.mktemp("pipeline_integration")
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    map_dir = base_dir / "maps"
    input_dir.mkdir()
    output_dir.mkdir()
    map_dir.mkdir()
    return input_dir, output_dir, map_dir

@pytest.fixture(scope="function")
def setup_input_files(integration_dirs):
    """
    Pytest fixture to create sample input files in the temporary input directory.

    Sets up different scenarios (e.g., a file with content, an empty file)
    for testing various pipeline behaviors.

    Args:
        integration_dirs (tuple): The tuple returned by the `integration_dirs` fixture,
                                containing paths to input, output, and map directories.

    Returns:
        dict[str, Path]: A dictionary mapping logical file names (e.g., "file1", "empty")
                         to their actual Path objects in the temporary input directory.
    """
    input_dir, _, _ = integration_dirs
    
    # File with content
    file1 = input_dir / "test1.txt"
    file1.write_text("This is the first sentence. Followed by the second.")

    # Empty file
    file_empty = input_dir / "empty.txt"
    file_empty.write_text("")
    
    return {
        "file1": file1,
        "empty": file_empty
    }

@pytest.fixture(scope="function")
def setup_single_empty_file(integration_dirs):
    """Fixture to create ONLY an empty input file."""
    input_dir, _, _ = integration_dirs
    file_empty = input_dir / "empty_only.txt"
    file_empty.write_text("")
    return {"empty": file_empty}

# --- Helper to load JSON Lines file ---
def load_jsonl(file_path: Path) -> list:
    """
    Load data from a JSON Lines (.jsonl) file.

    Reads a file line by line, parsing each valid line as a JSON object.

    Args:
        file_path (Path): The path to the .jsonl file.

    Returns:
        list: A list of dictionaries, where each dictionary is loaded from a line
              in the file. Returns an empty list if the file does not exist or is empty.
    """
    if not file_path.exists():
        return []
    lines = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines

# --- Helper for Graph Assertions (MOVED TO MODULE LEVEL) ---
async def _assert_graph_state(
    db_manager: Neo4jConnectionManager, 
    expected_result_success: Dict[str, Any], 
    expected_result_fail: Optional[Dict[str, Any]] = None
):
    """Asserts the state of the Neo4j graph based on expected results."""
    session_cm = await db_manager.get_session()
    async with session_cm as session:
        filename_base = expected_result_success['filename'] # Get basename from result
        sentence_id_success = expected_result_success['sentence_id']
        text_success = expected_result_success['sentence']
        
        # Check SourceFile
        result_file = await session.run("MATCH (f:SourceFile {filename: $fname}) RETURN count(f)", fname=filename_base)
        count_file = await result_file.single()
        assert count_file and count_file[0] == 1, f"SourceFile node not created or duplicated for {filename_base}"
        
        # Check successful sentence node and relationships
        query_success = """\
        MATCH (f:SourceFile {filename: $fname})<-[:PART_OF_FILE]-(s:Sentence {sentence_id: $sid, text: $text})
        // Check for at least one key relationship as proof of successful persistence
        MATCH (s)-[:HAS_FUNCTION_TYPE]->()
        MATCH (s)-[:HAS_TOPIC]->()
        MATCH (s)-[:MENTIONS_OVERALL_KEYWORD]->()
        RETURN s IS NOT NULL as exists
        """
        result_success_node = await session.run(query_success, fname=filename_base, sid=sentence_id_success, text=text_success)
        record_success_node = await result_success_node.single()
        assert record_success_node and record_success_node["exists"], f"Successful sentence node ({sentence_id_success}) with relationships not found for {filename_base}"
        
        # Check failed sentence node (if applicable)
        if expected_result_fail:
            sentence_id_fail = expected_result_fail['sentence_id']
            text_fail = expected_result_fail['sentence']
            
            # This block now asserts the *second successful* sentence in the success test case,
            # or the *failed* sentence node in the partial failure case.
            # For the success case, we expect relationships. For failure, we expect absence.
            # The logic needs refinement, let's assume this helper is mainly for asserting 
            # the *expected state* based on the input dicts.
            # We will assert *presence* of relationships here, assuming the input dict implies success.
            # The partial failure test calls this with expected_result_fail=None, skipping this block.
            
            query_second_sentence = """\
            MATCH (f:SourceFile {filename: $fname})<-[:PART_OF_FILE]-(s:Sentence {sentence_id: $sid, text: $text})
            // Ensure it *DOES* have the analysis relationships (for success test)
            MATCH (s)-[:HAS_FUNCTION_TYPE]->() 
            MATCH (s)-[:HAS_TOPIC]->() 
            MATCH (s)-[:MENTIONS_OVERALL_KEYWORD]->()
            RETURN s IS NOT NULL as exists
            """
            result_second_node = await session.run(query_second_sentence, fname=filename_base, sid=sentence_id_fail, text=text_fail)
            record_second_node = await result_second_node.single()
            assert record_second_node and record_second_node["exists"], f"Second sentence node ({sentence_id_fail}) *with* analysis relationships not found for {filename_base}"

            # Check FOLLOWS relationship between success and second sentence
            query_follows = """\
            MATCH (s1:Sentence {filename: $fname, sentence_id: $sid1})-[r:FOLLOWS]->(s2:Sentence {filename: $fname, sentence_id: $sid2})
            RETURN count(r) as count
            """
            result_follows = await session.run(query_follows, fname=filename_base, sid1=sentence_id_success, sid2=sentence_id_fail)
            record_follows = await result_follows.single()
            assert record_follows and record_follows["count"] == 1, f":FOLLOWS relationship missing between {sentence_id_success} and {sentence_id_fail} for {filename_base}"
        else:
             # If only one sentence expected (partial failure case), check FOLLOWS doesn't exist from it
             query_no_follows = """\
             MATCH (s1:Sentence {filename: $fname, sentence_id: $sid1})-[r:FOLLOWS]->()
             RETURN count(r) as count
             """
             result_no_follows = await session.run(query_no_follows, fname=filename_base, sid1=sentence_id_success)
             record_no_follows = await result_no_follows.single()
             assert record_no_follows and record_no_follows["count"] == 0, f"Sentence {sentence_id_success} should not have an outgoing FOLLOWS relationship in this test case for {filename_base}"

# --- Mock Analysis Data ---
def generate_mock_analysis(sentence_id, sequence_order, sentence, filename):
    """
    Generate a simple, consistent mock analysis result dictionary, including filename.

    Used in tests to provide predictable return values when mocking the actual
    sentence analysis process.

    Args:
        sentence_id (int): The ID of the sentence.
        sequence_order (int): The original sequence order of the sentence.
        sentence (str): The text of the sentence.
        filename (str): The source filename.

    Returns:
        dict: A dictionary representing a mock analysis result, including the input
              IDs/text/filename and placeholder values for analysis fields.
    """
    # Simple mock analysis result
    return {
        "filename": filename,
        "sentence_id": sentence_id,
        "sequence_order": sequence_order,
        "sentence": sentence,
        "function_type": "mock_func",
        "structure_type": "mock_struct",
        "purpose": "mock_purpose",
        "topic_level_1": "mock_topic1",
        "topic_level_3": "mock_topic3",
        "overall_keywords": ["mock"],
        "domain_keywords": ["mock"]
    }

# --- Integration Tests ---

# Restore autouse fixture for metrics
@pytest.fixture(autouse=True)
def mock_pipeline_metrics(monkeypatch):
    """Mocks the singleton metrics_tracker instance imported in src.pipeline."""
    mock_tracker = create_mock_metrics_tracker()
    # Patch the name 'metrics_tracker' within the src.pipeline module 
    # where it is imported and used by the orchestrator.
    monkeypatch.setattr("src.pipeline.metrics_tracker", mock_tracker)
    return mock_tracker

def create_mock_metrics_tracker():
    """Creates a MagicMock for MetricsTracker with necessary methods."""
    mock_tracker = MagicMock()
    mock_tracker.increment_files_processed = MagicMock()
    mock_tracker.increment_files_failed = MagicMock()
    mock_tracker.increment_sentences_processed = MagicMock() # Method called by map creation
    mock_tracker.increment_sentences_success = MagicMock() # Method called by worker on success
    mock_tracker.increment_errors = MagicMock() # Method called by worker on error
    mock_tracker.add_processing_time = MagicMock() # Method called by worker on success
    mock_tracker.add_file_processing_time = MagicMock()
    # Add any other methods that might be called by the pipeline
    mock_tracker.get_summary = MagicMock(return_value={})
    mock_tracker.start_file_timer = MagicMock()
    mock_tracker.stop_file_timer = MagicMock()
    mock_tracker.increment_results_processed = MagicMock()
    return mock_tracker

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_empty_file(
    integration_dirs, 
    setup_single_empty_file, 
    mock_pipeline_metrics,
    clear_test_db
):
    """
    Test the pipeline correctly handles an empty input file using Local IO.

    Verifies that:
    - An empty map file is created via LocalJsonlMapStorage.
    - No analysis file is created.
    - No errors are raised during processing.
    """
    input_dir, output_dir, map_dir = integration_dirs
    input_files = setup_single_empty_file # Use the fixture providing only the empty file
    target_file = input_files["empty"]
    target_stem = target_file.stem
    task_id = "int-test-empty" # Added Task ID

    # Patch the API call, AnalysisService, and logger
    with patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock) as mock_call_model, \
         patch("src.pipeline.AnalysisService") as MockAnalysisService, \
         patch("src.pipeline.logger") as mock_pipeline_logger: 
            
        # Configure mocks (metrics mock comes from fixture)
        # mock_metrics_instance = create_mock_metrics_tracker()
        # mock_metrics_singleton.return_value = mock_metrics_instance

        # Run the pipeline on the directory containing ONLY the empty file
        await run_pipeline(
            input_dir=input_dir, 
            output_dir=output_dir, 
            map_dir=map_dir, 
            config_dict=config, # Pass config as dict
            specific_file=target_file.name, # Process only the specific empty file
            task_id=task_id # Added Task ID
        )

        # Assertions
        map_file = map_dir / f"{target_stem}{config['paths']['map_suffix']}"
        analysis_file = output_dir / f"{target_stem}{config['paths']['analysis_suffix']}"

        # Check map file exists and is empty (created by LocalJsonlMapStorage.initialize/finalize)
        assert map_file.exists(), "Map file was not created for empty input"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 0, "Map file for empty input should be empty"
        
        # Analysis file SHOULD NOT be created
        assert not analysis_file.exists(), "Analysis file SHOULD NOT be created for empty input"

        # API should not have been called
        mock_call_model.assert_not_called() 
        # Ensure analysis service methods were not called for processing
        if MockAnalysisService.called:
            instance = MockAnalysisService.return_value
            instance.build_contexts.assert_not_called()
            instance.analyze_sentences.assert_not_called()

        # Assert metrics using the fixture mock
        mock_pipeline_metrics.start_file_timer.assert_called_once_with(target_file.name)
        mock_pipeline_metrics.stop_file_timer.assert_called_once_with(target_file.name)
        mock_pipeline_metrics.increment_files_processed.assert_called_once_with(1)
        mock_pipeline_metrics.increment_results_processed.assert_not_called()
        mock_pipeline_metrics.increment_errors.assert_not_called()
        mock_pipeline_metrics.increment_files_failed.assert_not_called()

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_success_rewritten(
    integration_dirs, 
    setup_input_files, 
    mock_pipeline_metrics,
    clear_test_db, 
    test_db_manager: Neo4jConnectionManager # Inject DB manager for verification
):
    """Test rewritten: Full pipeline success with mocked API calls using Local IO."""
    input_dir, output_dir, map_dir = integration_dirs
    input_files = setup_input_files # Use the standard fixture
    target_file = input_files["file1"] # Use the file with content
    target_stem = target_file.stem
    task_id = "int-test-success" # Added Task ID

    # Mock Config (remains largely the same, using integration_dirs)
    mock_config_dict = {
        "paths": {
            "output_dir": str(output_dir),
            "map_dir": str(map_dir),
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl",
            "logs_dir": str(integration_dirs[0].parent / "logs")
        },
        "pipeline": { "num_concurrent_files": 1 },
        "preprocessing": {"context_windows": {"immediate": 1, "broader": 3, "observer": 5}},
        "classification": {"local": {"prompt_files": {"no_context": "dummy"}}},
        "domain_keywords": []
    }

    # --- Mock API Call Logic (applied to classify_sentence) ---
    sentence_alpha = "This is the first sentence."
    sentence_beta = "Followed by the second."
    filename_path = target_file # Use the full Path object
    filename_str = str(filename_path) # Use the full path string for generation/assertion
    async def mock_classify_sentence(*args, **kwargs):
        sentence = args[0] 
        if sentence == sentence_alpha:
             # Pass full path string to generator
             result = generate_mock_analysis(0, 0, sentence_alpha, filename_str)
             return {k: v for k, v in result.items() if k not in ["sentence_id", "sequence_order", "sentence", "filename"]}
        elif sentence == sentence_beta:
             # Pass full path string to generator
             result = generate_mock_analysis(1, 1, sentence_beta, filename_str)
             return {k: v for k, v in result.items() if k not in ["sentence_id", "sequence_order", "sentence", "filename"]}
        else:
             return {"mock_fallback": []}

    # --- Patching ---
    # Patch the core dependency: sentence classification, AnalysisService, logger
    with patch("src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock, side_effect=mock_classify_sentence) as mock_classify, \
         patch("src.pipeline.AnalysisService") as MockAnalysisService, \
         patch("src.pipeline.logger") as mock_pipeline_logger: 
            
        # Configure mocks (metrics mock comes from fixture)
        # mock_metrics_instance = create_mock_metrics_tracker()
        # mock_metrics_singleton.return_value = mock_metrics_instance
        # Configure the mock AnalysisService instance
        mock_analysis_service_instance = MockAnalysisService.return_value
        # Mock build_contexts (return value might not be critical for this test)
        mock_analysis_service_instance.build_contexts = MagicMock(return_value={
            0: {'ctx': 'mock_ctx_alpha'}, 
            1: {'ctx': 'mock_ctx_beta'}
        })
        # Mock analyze_sentences to return the expected structured results
        # Ensure filename uses the basename, matching what the orchestrator saves
        expected_results = [
            generate_mock_analysis(0, 0, sentence_alpha, filename_path.name), # Use BASENAME
            generate_mock_analysis(1, 1, sentence_beta, filename_path.name)  # Use BASENAME
        ]
        mock_analysis_service_instance.analyze_sentences = AsyncMock(return_value=expected_results)

        # --- Execute ---
        await run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=mock_config_dict,  # Pass the test-specific config
            specific_file=filename_path.name, # Pass basename here for discovery
            task_id=task_id # Added Task ID
        )

        # --- Assertions ---
        map_file = map_dir / f"{target_stem}{mock_config_dict['paths']['map_suffix']}"
        analysis_file = output_dir / f"{target_stem}{mock_config_dict['paths']['analysis_suffix']}"

        # 1. Check map file content
        assert map_file.exists(), f"Map file was not created at {map_file}"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 2
        assert map_data[0] == {"sentence_id": 0, "sequence_order": 0, "sentence": sentence_alpha}
        assert map_data[1] == {"sentence_id": 1, "sequence_order": 1, "sentence": sentence_beta}

        # 2. Check analysis file content
        assert analysis_file.exists(), f"Analysis file was not created at {analysis_file}"
        analysis_data = load_jsonl(analysis_file)
        assert len(analysis_data) == 2
        
        # Check data content (flexible order)
        expected_result_0 = generate_mock_analysis(0, 0, sentence_alpha, filename_path.name) # Use BASENAME
        expected_result_1 = generate_mock_analysis(1, 1, sentence_beta, filename_path.name) # Use BASENAME
        
        # Helper function to make dict values hashable (convert lists to tuples)
        def make_hashable(d):
            return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items()))
            
        # Convert list of dicts to set of hashable tuples for order-independent comparison
        analysis_data_set = {make_hashable(d) for d in analysis_data}
        expected_set = {make_hashable(expected_result_0), make_hashable(expected_result_1)}
        assert analysis_data_set == expected_set

        # 3. Check mocked AnalysisService was called
        mock_analysis_service_instance.analyze_sentences.assert_awaited_once()

        # 4. Assert Graph Database Content
        # Call the refactored helper
        await _assert_graph_state(test_db_manager, expected_result_0, expected_result_1)

        # 5. Assert metrics via the fixture mock
        mock_pipeline_metrics.start_file_timer.assert_called_once_with(filename_path.name)
        mock_pipeline_metrics.stop_file_timer.assert_called_once_with(filename_path.name)
        mock_pipeline_metrics.increment_files_processed.assert_called_once_with(1)
        mock_pipeline_metrics.increment_results_processed.assert_called_with(filename_path.name)
        assert mock_pipeline_metrics.increment_results_processed.call_count == 2
        mock_pipeline_metrics.increment_errors.assert_not_called()
        mock_pipeline_metrics.increment_files_failed.assert_not_called()

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_partial_failure_rewritten(
    integration_dirs, 
    mock_pipeline_metrics,
    clear_test_db, 
    test_db_manager: Neo4jConnectionManager # Inject DB manager
):
    """Test rewritten: Pipeline handles partial failure (one sentence error) using Local IO."""
    input_dir, output_dir, map_dir = integration_dirs

    # --- Setup Input File ---
    input_file_name_base = "integ_partial.txt"
    input_file_path = input_dir / input_file_name_base # Use full Path object
    input_filename_str = str(input_file_path) # Use full path string for generation/assertion
    sentence_ok = "This sentence is fine."
    sentence_fail = "This sentence will fail."
    input_file_path.write_text(f"{sentence_ok} {sentence_fail}")
    input_file_stem = input_file_path.stem
    task_id = "int-test-partial-fail" # Added Task ID

    # --- Mock Config ---
    mock_config_dict = {
        "paths": {
            "output_dir": str(output_dir),
            "map_dir": str(map_dir),
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl",
            "logs_dir": str(integration_dirs[0].parent / "logs")
        },
        "pipeline": {"num_concurrent_files": 1},
        "preprocessing": {"context_windows": {"immediate": 1, "broader": 3, "observer": 5}},
        "classification": {"local": {"prompt_files": {"no_context": "dummy"}}},
        "domain_keywords": []
    }

    # --- Mock classify_sentence Logic ---
    fail_error = ValueError("Simulated classify error")
    async def mock_classify_fail_one(*args, **kwargs):
        sentence = args[0]
        if sentence == sentence_ok:
             # Pass full path string to generator
             result = generate_mock_analysis(0, 0, sentence_ok, input_filename_str)
             return {k: v for k, v in result.items() if k not in ["sentence_id", "sequence_order", "sentence", "filename"]}
        elif sentence == sentence_fail:
             raise fail_error
        else:
             return {"mock_fallback": []}

    # --- Patching ---
    # Patch dependencies: classify, loggers. Let AnalysisService run.
    with patch("src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock, side_effect=mock_classify_fail_one) as mock_classify, \
         patch("src.pipeline.logger") as mock_pipeline_logger, \
         patch("src.services.analysis_service.logger") as mock_service_logger:

        # --- Execute ---
        await run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=mock_config_dict,
            specific_file=input_file_name_base, # Pass basename for discovery
            task_id=task_id # Added Task ID
        )

        # --- Assertions ---
        map_file = map_dir / f"{input_file_stem}{mock_config_dict['paths']['map_suffix']}"
        analysis_file = output_dir / f"{input_file_stem}{mock_config_dict['paths']['analysis_suffix']}"

        # 1. Check map file content
        assert map_file.exists(), f"Map file was not created at {map_file}"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 2
        assert map_data[0]["sentence"] == sentence_ok
        assert map_data[1]["sentence"] == sentence_fail

        # 2. Check analysis file content
        assert analysis_file.exists(), f"Analysis file was not created at {analysis_file}"
        analysis_data = load_jsonl(analysis_file)
        assert len(analysis_data) == 1
        
        # Check data content
        expected_result_ok = generate_mock_analysis(0, 0, sentence_ok, input_file_path.name) # Use BASENAME
        
        # ADDED Definition for make_hashable
        def make_hashable(d):
            return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items()))

        # Convert list of dicts to set of hashable tuples for order-independent comparison
        analysis_data_set = {make_hashable(d) for d in analysis_data}
        expected_set = {make_hashable(expected_result_ok)} # Only expect the OK one
        assert analysis_data_set == expected_set, "Analysis file content mismatch for successful sentence"

        # 3. Check mocked AnalysisService was called
        # Check that the underlying mock was called twice (once per sentence attempt)
        assert mock_classify.await_count == 2
        # mock_classify.assert_awaited_once() # Original assertion was wrong

        # 4. Check graph database state (only successful sentence should be there)
        await _assert_graph_state(test_db_manager, expected_result_ok, None) # Pass None for expected_result_fail

        # 5. Assert metrics via the fixture mock
        mock_pipeline_metrics.start_file_timer.assert_called_once_with(input_file_path.name)
        mock_pipeline_metrics.stop_file_timer.assert_called_once_with(input_file_path.name)
        mock_pipeline_metrics.increment_files_processed.assert_called_once_with(1)
        mock_pipeline_metrics.increment_results_processed.assert_called_once_with(input_file_path.name)
        # Error should be logged by AnalysisService now, not during saving by orchestrator
        # mock_pipeline_metrics.increment_errors.assert_called_with(f"{input_file_path.name}_result_save_failure")
        # Check if *any* error was incremented (likely by AnalysisService)
        mock_pipeline_metrics.increment_errors.assert_called() # Check if called at least once
        mock_pipeline_metrics.increment_files_failed.assert_not_called()

        # 6. Check Log (optional - ensure specific error related to analysis failure is logged)
        error_logged = False
        # Error is now raised inside classify_sentence, caught & logged by AnalysisService
        expected_log_fragment = f"Error analyzing sentence {sentence_fail}: Simulated classify error" # Adjust based on actual service log
        all_logs = mock_service_logger.error.call_args_list # Check service logger
        for call in all_logs:
            # Check if the expected fragment is IN the logged message
            if "Error analyzing sentence" in call.args[0] and "Simulated classify error" in call.args[0]:
                 error_logged = True
                 break
        assert error_logged, f"Expected analysis error log containing '{expected_log_fragment}' not found in service logger. Logs: {all_logs}"

# ... (Rest of integration tests) ... 