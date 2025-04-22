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
from unittest.mock import patch, AsyncMock

# Assuming main entry point or run_pipeline can be invoked
from src.pipeline import run_pipeline
from src.config import config # Use the actual config

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

# --- Mock Analysis Data ---
def generate_mock_analysis(sentence_id, sequence_order, sentence):
    """
    Generate a simple, consistent mock analysis result dictionary.

    Used in tests to provide predictable return values when mocking the actual
    sentence analysis process.

    Args:
        sentence_id (int): The ID of the sentence.
        sequence_order (int): The original sequence order of the sentence.
        sentence (str): The text of the sentence.

    Returns:
        dict: A dictionary representing a mock analysis result, including the input
              IDs/text and placeholder values for analysis fields.
    """
    # Simple mock analysis result
    return {
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

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_empty_file(integration_dirs, setup_single_empty_file):
    """
    Test the pipeline correctly handles an empty input file.

    Verifies that:
    - An empty map file is created.
    - No analysis file is created.
    - No errors are raised during processing.
    """
    input_dir, output_dir, map_dir = integration_dirs
    # FIX: Use the input file from the new fixture
    input_files = setup_single_empty_file
    target_file = input_files["empty"]
    target_stem = target_file.stem

    # Patch the API call - it shouldn't be called for an empty file
    with patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock) as mock_call_model, \
         patch("src.pipeline.logger") as mock_pipeline_logger: # Monitor logs

        # Run the pipeline on the directory containing ONLY the empty file
        await run_pipeline(
            input_dir=input_dir, 
            output_dir=output_dir, 
            map_dir=map_dir, 
            config=config.config, 
            specific_file=None
        )

        # Assertions
        map_file = map_dir / f"{target_stem}{config['paths']['map_suffix']}"
        analysis_file = output_dir / f"{target_stem}{config['paths']['analysis_suffix']}"

        # Check map file exists and is empty 
        assert map_file.exists(), "Map file was not created for empty input"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 0, "Map file for empty input should be empty"
        assert any(f"contains no sentences after segmentation. Map file will be empty." in call.args[0] for call in mock_pipeline_logger.warning.call_args_list)

        # Analysis file SHOULD NOT be created
        assert not analysis_file.exists(), "Analysis file SHOULD NOT be created for empty input"

        # API should not have been called
        mock_call_model.assert_not_called() 

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_success_rewritten(integration_dirs):
    """Test rewritten: Full pipeline success with mocked API calls."""
    input_dir, output_dir, map_dir = integration_dirs

    # --- Setup Input File ---
    input_file_name = "integ_success.txt"
    input_file = input_dir / input_file_name
    input_file.write_text("Sentence Alpha. Sentence Beta.")
    input_file_stem = input_file.stem

    # --- Mock Config (Scoped to this test) ---
    # Use specific paths within the temp directories provided by integration_dirs
    mock_config_dict = {
        "paths": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "map_dir": str(map_dir),
            "map_suffix": "_map.jsonl",
            "analysis_suffix": "_analysis.jsonl",
            "logs_dir": str(integration_dirs[0].parent / "logs") # Place logs nearby
        },
        "pipeline": {
            "num_concurrent_files": 1 # Keep simple for integration test
        },
        # Add other minimal required config sections if validation needs them
        "preprocessing": {"context_windows": {"immediate": 1, "broader": 3, "observer": 5}},
        "classification": {"local": {"prompt_files": {"no_context": "dummy"}}},
        "domain_keywords": []
    }

    # --- Mock API Call Logic (Now applied to classify_sentence) ---
    sentence_alpha = "Sentence Alpha."
    sentence_beta = "Sentence Beta."
    async def mock_classify_sentence(*args, **kwargs):
        # The actual sentence is likely the first arg to classify_sentence
        sentence = args[0] 
        if sentence == sentence_alpha:
             # Note: classify_sentence returns only the analysis part,
             # the worker adds sentence_id, sequence_order etc.
             result = generate_mock_analysis(0, 0, sentence_alpha)
             return {k: v for k, v in result.items() if k not in ["sentence_id", "sequence_order", "sentence"]}
        elif sentence == sentence_beta:
             result = generate_mock_analysis(1, 1, sentence_beta)
             return {k: v for k, v in result.items() if k not in ["sentence_id", "sequence_order", "sentence"]}
        else:
             return {"mock_fallback": []}

    # --- Patching ---
    # Patch classify_sentence directly, remove call_model patch
    with patch("src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock, side_effect=mock_classify_sentence) as mock_classify, \
         patch("src.config.config", mock_config_dict) as mock_global_config, \
         patch("src.pipeline.append_json_line") as mock_append_jsonl:

        # --- Execute ---
        await run_pipeline(
            input_dir=input_dir,      # Use temp dir
            output_dir=output_dir,    # Use temp dir
            map_dir=map_dir,          # Use temp dir
            config=mock_config_dict,  # Pass the test-specific config
            specific_file=input_file_name # Process only this file
        )

        # --- Assertions ---
        map_file = map_dir / f"{input_file_stem}{mock_config_dict['paths']['map_suffix']}"
        analysis_file = output_dir / f"{input_file_stem}{mock_config_dict['paths']['analysis_suffix']}"

        # 1. Check map file
        assert map_file.exists(), f"Map file was not created at {map_file}"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 2
        assert map_data[0] == {"sentence_id": 0, "sequence_order": 0, "sentence": sentence_alpha}
        assert map_data[1] == {"sentence_id": 1, "sequence_order": 1, "sentence": sentence_beta}

        # 2. Check that append_json_line was called correctly (since file writing is mocked)
        analysis_file_path = output_dir / f"{input_file_stem}{mock_config_dict['paths']['analysis_suffix']}"
        assert mock_append_jsonl.call_count == 2, "append_json_line should have been called twice"

        # Verify calls (order might not be guaranteed, check args based on content)
        call_args_list = mock_append_jsonl.call_args_list
        call_data = [c.args[0] for c in call_args_list] # Get the first arg (the data dict) from each call
        call_paths = [c.args[1] for c in call_args_list] # Get the second arg (the path) from each call

        # Check paths
        assert all(p == analysis_file_path for p in call_paths), "append_json_line called with incorrect file path"

        # Check data content (flexible order)
        expected_result_0 = generate_mock_analysis(0, 0, sentence_alpha)
        expected_result_1 = generate_mock_analysis(1, 1, sentence_beta)
        assert expected_result_0 in call_data
        assert expected_result_1 in call_data

        # 3. Check API mock was called
        assert mock_classify.called

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_partial_failure_rewritten(integration_dirs):
    """Test rewritten: Pipeline handles partial failure (one sentence error)."""
    input_dir, output_dir, map_dir = integration_dirs

    # --- Setup Input File ---
    input_file_name = "integ_partial.txt"
    input_file = input_dir / input_file_name
    sentence_ok = "This sentence is fine."
    sentence_fail = "This sentence will fail."
    input_file.write_text(f"{sentence_ok} {sentence_fail}")
    input_file_stem = input_file.stem

    # --- Mock Config ---
    mock_config_dict = {
        "paths": {
            "input_dir": str(input_dir),
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
             result = generate_mock_analysis(0, 0, sentence_ok)
             return {k: v for k, v in result.items() if k not in ["sentence_id", "sequence_order", "sentence"]}
        elif sentence == sentence_fail:
             raise fail_error
        else:
             return {"mock_fallback": []}

    # --- Patching ---
    with patch("src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock, side_effect=mock_classify_fail_one) as mock_classify, \
         patch("src.config.config", mock_config_dict) as mock_global_config, \
         patch("src.pipeline.append_json_line") as mock_append_jsonl, \
         patch("src.pipeline.logger") as mock_pipeline_logger, \
         patch("src.services.analysis_service.logger") as mock_service_logger: # Patch service logger too

        # --- Execute ---
        await run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config=mock_config_dict,
            specific_file=input_file_name
        )

        # --- Assertions ---
        map_file = map_dir / f"{input_file_stem}{mock_config_dict['paths']['map_suffix']}"
        analysis_file_path = output_dir / f"{input_file_stem}{mock_config_dict['paths']['analysis_suffix']}"

        # 1. Check map file
        assert map_file.exists(), f"Map file was not created at {map_file}"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 2
        assert map_data[0]["sentence"] == sentence_ok
        assert map_data[1]["sentence"] == sentence_fail

        # 2. Check append_json_line calls
        assert mock_append_jsonl.call_count == 2
        call_args_list = mock_append_jsonl.call_args_list
        call_data = {c.args[0]["sentence_id"]: c.args[0] for c in call_args_list} # Data dicts by ID
        call_paths = [c.args[1] for c in call_args_list]

        # Check paths
        assert all(p == analysis_file_path for p in call_paths)

        # Check successful data (ID 0)
        assert 0 in call_data
        assert not call_data[0].get("error")
        assert call_data[0]["sentence"] == sentence_ok
        assert call_data[0]["function_type"] == "mock_func" # From generate_mock_analysis

        # Check error data (ID 1)
        assert 1 in call_data
        assert call_data[1].get("error") is True
        assert call_data[1]["sentence"] == sentence_fail
        assert call_data[1]["error_type"] == "ValueError"
        assert call_data[1]["error_message"] == "Simulated classify error"

        # 3. Check classify mock was called for both
        assert mock_classify.call_count == 2
        # Could add more detailed call arg checks if needed
        
        # 4. Check Log (optional)
        error_logged = False
        expected_log_fragment = f"Worker 0 failed analyzing sentence_id 1: {fail_error}"
        all_logs = mock_pipeline_logger.error.call_args_list + mock_service_logger.error.call_args_list
        for call in all_logs:
            if expected_log_fragment in call.args[0]:
                 error_logged = True
                 break
        assert error_logged, f"Expected worker error log not found. Logs: {all_logs}"

# ... (Rest of integration tests) ... 