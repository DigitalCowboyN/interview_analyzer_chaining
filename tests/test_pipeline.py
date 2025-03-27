"""
tests/test_pipeline.py

This module contains unit tests for the functions defined in the pipeline module 
(src/pipeline.py). The pipeline module is responsible for:
    - Segmenting text into sentences (using spaCy).
    - Processing text files by segmenting their content, analyzing each sentence, 
      and saving the results to a JSON file.
    - Running the entire pipeline across multiple text files in a directory.

Key functions tested:
    - segment_text: Segments a string into sentences.
    - process_file: Reads a text file, segments the content, analyzes each sentence,
      and saves the analysis as JSON.
    - run_pipeline: Processes all .txt files in a directory.

Usage:
    Run these tests with pytest from the project root:
        pytest tests/test_pipeline.py

Modifications:
    - If the segmentation logic in segment_text changes, update the expected outputs
      in test_segment_text and test_segment_text_empty.
    - If the file processing logic in process_file changes (e.g. additional keys in the
      analysis results), update the fake analysis in test_process_file accordingly.
    - If new behavior is added to run_pipeline, add new tests or update existing ones.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock
from src.pipeline import segment_text, process_file, run_pipeline

def test_segment_text():
    """
    Test the sentence segmentation functionality.
    
    This test verifies that the segment_text function correctly segments a given text 
    into individual sentences using spaCy. It checks both the number of sentences and 
    the content of each sentence.
    
    Asserts:
        - The returned list has the expected number of sentences.
        - Each sentence in the list matches the expected text.
    """
    test_text = "Hello world. How are you today? This pipeline is running well!"
    sentences = segment_text(test_text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you today?"
    assert sentences[2] == "This pipeline is running well!"

def test_segment_text_empty():
    """
    Test the segmentation of an empty string.
    
    This test checks that segment_text returns an empty list when given an empty string.
    Depending on desired behavior, this could be modified to raise a ValueError instead.
    
    Asserts:
        - An empty list is returned for empty input.
    """
    sentences = segment_text("")
    assert sentences == []

@pytest.fixture
def sample_text_file(tmp_path):
    """
    Fixture to create a temporary sample text file.
    
    This fixture writes a small text with two sentences to a temporary file and returns 
    the file path. This file is used for testing the process_file function.
    
    Returns:
        Path: The path to the created sample text file.
    """
    file_content = "This is a test. Ensure proper segmentation."
    test_file = tmp_path / "test_file.txt"
    test_file.write_text(file_content)
    return test_file

@pytest.mark.asyncio
async def test_process_file(sample_text_file, tmp_path):
    """
    Test the file processing functionality.
    
    This test verifies that the process_file function correctly:
      - Reads a text file.
      - Segments the text into sentences.
      - Analyzes each sentence using SentenceAnalyzer (patched here to return fake results).
      - Saves the analysis results to a JSON file.
      
    We patch SentenceAnalyzer.analyze_sentences to return two fake analysis results,
    one for each segmented sentence.
    
    Asserts:
        - The output JSON file is created.
        - The JSON file contains a list of analysis results.
        - The analysis result for the first sentence matches the expected data.
    """
    # Create fake analysis results for each sentence.
    fake_analysis = [
        {
            "sentence": "This is a test.",
            "function_type": "declarative",
            "structure_type": "simple sentence",
            "purpose": "informational",
            "topic_level_1": "unit testing",
            "topic_level_3": "sentence segmentation",
            "overall_keywords": ["test", "segmentation"],
            "domain_keywords": ["unit test"]
        },
        {
            "sentence": "Ensure proper segmentation.",
            "function_type": "informational",
            "structure_type": "simple sentence",
            "purpose": "informational",
            "topic_level_1": "unit testing",
            "topic_level_3": "sentence segmentation",
            "overall_keywords": ["test", "segmentation"],
            "domain_keywords": ["unit test"]
        }
    ]

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Patch the analyze_sentences method to return our fake analysis results.
    with patch("src.agents.sentence_analyzer.SentenceAnalyzer.analyze_sentences", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = fake_analysis
        await process_file(sample_text_file, output_dir)

    output_file = output_dir / f"{sample_text_file.stem}_analysis.json"
    assert output_file.exists(), "Output JSON file was not created."

    data = json.loads(output_file.read_text())
    assert isinstance(data, list), "Output data is not a list."
    assert len(data) == 2, "Unexpected number of analysis results."
    assert data[0]["sentence"] == "This is a test."
    assert data[0]["function_type"] == "declarative"

@pytest.mark.asyncio
async def test_process_file_nonexistent(tmp_path):
    """
    Test that process_file raises an error when the input file does not exist.
    
    This test attempts to process a file that does not exist and expects an exception.
    
    Asserts:
        - An exception is raised when a non-existent file is processed.
    """
    non_existent_file = tmp_path / "no_file.txt"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(Exception):
        await process_file(non_existent_file, output_dir)

@pytest.mark.asyncio
async def test_run_pipeline_no_files(tmp_path):
    """
    Test that run_pipeline behaves correctly when there are no text files in the input directory.
    
    This test creates an empty input directory and checks that run_pipeline logs a warning 
    and exits gracefully without processing any files.
    
    Asserts:
        - A warning is logged indicating no input files were found.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    with patch("src.pipeline.logger") as mock_logger:
        await run_pipeline(input_dir, output_dir)
        mock_logger.warning.assert_called_with(f"No input files found in {input_dir}")

@pytest.mark.asyncio
async def test_run_pipeline_multiple_files(tmp_path):
    """
    Test that run_pipeline processes all text files in the input directory.
    
    This test creates an input directory with two text files and patches process_file 
    to avoid actual file processing. It then verifies that process_file is called for 
    each file.
    
    Asserts:
        - process_file is called the expected number of times (once for each text file).
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    file1 = input_dir / "file1.txt"
    file2 = input_dir / "file2.txt"
    file1.write_text("File one text.")
    file2.write_text("File two text.")
    
    with patch("src.pipeline.process_file", new_callable=AsyncMock) as mock_process:
        await run_pipeline(input_dir, output_dir)
        assert mock_process.call_count == 2, "Expected process_file to be called for each input file."
