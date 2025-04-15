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
@pytest.mark.integration # Optional marker
async def test_pipeline_integration_success(integration_dirs, setup_input_files):
    """
    Test the full pipeline successfully processing a file with content.

    Verifies that `run_pipeline` correctly processes a sample text file, creates
    the expected map and analysis files with the correct content, and that the
    mocked API call function was invoked.

    Args:
        integration_dirs: Fixture providing temporary directories.
        setup_input_files: Fixture providing sample input files.
    """
    input_dir, output_dir, map_dir = integration_dirs
    input_files = setup_input_files
    target_file = input_files["file1"]
    target_stem = target_file.stem

    # Mock the lowest level API call within the SentenceAnalyzer's agent
    async def mock_api_call(*args, **kwargs):
        # Find the sentence in the prompt (this is fragile, depends on prompt structure)
        prompt_str = args[0]
        sentence = "Unknown Sentence"
        if "Sentence one." in prompt_str:
             sentence = "Sentence one."
        elif "Sentence two?" in prompt_str: # Based on actual segmentation of test1.txt
             sentence = "Sentence two?"
        elif "first sentence." in prompt_str: # Handle segmentation variations
             sentence = "This is the first sentence."
        elif "Followed by the second." in prompt_str:
             sentence = "Followed by the second."
        
        # Simulate successful analysis based on which sentence it is (crude mapping)
        # A more robust mock might inspect kwargs or prompt structure better
        if "first" in sentence.lower():
            return generate_mock_analysis(0, 0, sentence)
        else:
            return generate_mock_analysis(1, 1, sentence)

    # Adjust patch target based on where the actual external API call is made
    # Assuming it's in src.agents.agent.OpenAIAgent.call_model
    with patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock, side_effect=mock_api_call) as mock_call_model:
        
        # Run the pipeline (using the real config, which includes paths/suffixes)
        await run_pipeline(input_dir, output_dir, map_dir, config)

        # Assertions
        map_file = map_dir / f"{target_stem}{config['paths']['map_suffix']}"
        analysis_file = output_dir / f"{target_stem}{config['paths']['analysis_suffix']}"

        # Check map file
        assert map_file.exists(), "Map file was not created"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 2
        assert map_data[0] == {"sentence_id": 0, "sequence_order": 0, "sentence": "This is the first sentence."}
        assert map_data[1] == {"sentence_id": 1, "sequence_order": 1, "sentence": "Followed by the second."}

        # Check analysis file
        assert analysis_file.exists(), "Analysis file was not created"
        analysis_data = load_jsonl(analysis_file)
        assert len(analysis_data) == 2, "Incorrect number of analysis results written"

        # Check content of analysis results (order might vary)
        results_by_id = {res["sentence_id"]: res for res in analysis_data}
        assert 0 in results_by_id
        assert 1 in results_by_id
        assert results_by_id[0]["sentence"] == "This is the first sentence."
        assert results_by_id[0]["function_type"] == "mock_func"
        assert results_by_id[1]["sentence"] == "Followed by the second."
        assert results_by_id[1]["structure_type"] == "mock_struct"
        assert results_by_id[0]["sequence_order"] == 0
        assert results_by_id[1]["sequence_order"] == 1
        
        # Check API call mock was used
        assert mock_call_model.call_count > 0 # Should be called multiple times per sentence

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_partial_failure(integration_dirs, setup_input_files):
    """
    Test the pipeline handling a failure during the analysis of one sentence.

    Mocks the API call to raise an exception for one of the sentences in the input file.
    Verifies that:
    - The map file is still created correctly for all sentences.
    - The analysis file contains results only for the successfully processed sentences.
    - An error message is logged for the failed sentence analysis.

    Args:
        integration_dirs: Fixture providing temporary directories.
        setup_input_files: Fixture providing sample input files.
    """
    input_dir, output_dir, map_dir = integration_dirs
    input_files = setup_input_files
    target_file = input_files["file1"]
    target_stem = target_file.stem

    # Mock the API call to fail for the second sentence
    async def mock_api_call_fail_second(*args, **kwargs):
        prompt_str = args[0]
        sentence = "Unknown Sentence"
        if "first sentence." in prompt_str:
             sentence = "This is the first sentence."
             return generate_mock_analysis(0, 0, sentence)
        elif "Followed by the second." in prompt_str:
             # Raise an exception for the second sentence analysis call
             # Note: This mocks the API call level. If classify_sentence catches this,
             # the test might need adjustment or mock higher up.
             raise ValueError("Mock API Error for second sentence")
        else:
            # Fallback for other potential calls (e.g., keyword extraction)
            return {"mock_fallback": []} 

    # Use parentheses for multi-line with statement
    with (patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock, side_effect=mock_api_call_fail_second),
          patch("src.pipeline.logger") as mock_pipeline_logger): # Mock logger to check error logs
        
        await run_pipeline(input_dir, output_dir, map_dir, config)

        # Assertions
        map_file = map_dir / f"{target_stem}{config['paths']['map_suffix']}"
        analysis_file = output_dir / f"{target_stem}{config['paths']['analysis_suffix']}"

        # Check map file (should still be complete)
        assert map_file.exists(), "Map file was not created"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 2

        # Check analysis file (should only contain the first sentence)
        assert analysis_file.exists(), "Analysis file was not created"
        analysis_data = load_jsonl(analysis_file)
        assert len(analysis_data) == 1, "Expected only one successful analysis result"
        assert analysis_data[0]["sentence_id"] == 0
        assert analysis_data[0]["sentence"] == "This is the first sentence."
        
        # Check that the error was logged (by the worker)
        error_logged = any("failed analyzing sentence_id 1" in call.args[0] 
                           for call in mock_pipeline_logger.error.call_args_list)
        assert error_logged, "Worker error for sentence 1 was not logged"

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_integration_empty_file(integration_dirs, setup_input_files):
    """
    Test the pipeline correctly handling an empty input file.

    Verifies that `run_pipeline` processes an empty .txt file without errors,
    creates empty (but existing) map and analysis files, and logs an appropriate
    warning message.

    Args:
        integration_dirs: Fixture providing temporary directories.
        setup_input_files: Fixture providing sample input files.
    """
    input_dir, output_dir, map_dir = integration_dirs
    input_files = setup_input_files
    target_file = input_files["empty"]
    target_stem = target_file.stem

    # No API mocking needed as no analysis should happen
    with patch("src.pipeline.logger") as mock_pipeline_logger: # Check logs
        await run_pipeline(input_dir, output_dir, map_dir, config)

        # Assertions
        map_file = map_dir / f"{target_stem}{config['paths']['map_suffix']}"
        analysis_file = output_dir / f"{target_stem}{config['paths']['analysis_suffix']}"

        # Check map file (exists but is empty)
        assert map_file.exists(), "Map file was not created for empty input"
        map_data = load_jsonl(map_file)
        assert len(map_data) == 0

        # Check analysis file (exists but is empty)
        assert analysis_file.exists(), "Analysis file was not created for empty input"
        analysis_data = load_jsonl(analysis_file)
        assert len(analysis_data) == 0
        
        # Check for appropriate warning log
        warning_logged = any("contains 0 processable sentences" in call.args[0] 
                             for call in mock_pipeline_logger.warning.call_args_list)
        assert warning_logged, "Warning for empty file was not logged" 