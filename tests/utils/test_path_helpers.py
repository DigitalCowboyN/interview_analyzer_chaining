# tests/utils/test_path_helpers.py
"""
Tests for utility functions in src.utils.path_helpers.
"""

import logging  # Import the logging module
from pathlib import Path

import pytest

# Import the function and dataclass to test
from src.utils.path_helpers import PipelinePaths, generate_pipeline_paths


def test_generate_pipeline_paths_success(tmp_path):
    """Tests successful generation of PipelinePaths with standard inputs."""
    input_dir = tmp_path / "input"
    map_dir = tmp_path / "maps"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    map_dir.mkdir()
    output_dir.mkdir()

    input_file = input_dir / "test_doc.txt"
    input_file.touch()  # Create the dummy file

    map_suffix = "_map.jsonl"
    analysis_suffix = "_analysis.jsonl"

    expected_map_path = map_dir / "test_doc_map.jsonl"
    expected_analysis_path = output_dir / "test_doc_analysis.jsonl"

    # Call the function
    result_paths = generate_pipeline_paths(
        input_file=input_file,
        map_dir=map_dir,
        output_dir=output_dir,
        map_suffix=map_suffix,
        analysis_suffix=analysis_suffix,
    )

    # Assertions
    assert isinstance(result_paths, PipelinePaths)
    assert result_paths.map_file == expected_map_path
    assert result_paths.analysis_file == expected_analysis_path


def test_generate_pipeline_paths_different_suffixes(tmp_path):
    """Tests successful generation with non-default suffixes."""
    input_dir = tmp_path / "input"
    map_dir = tmp_path / "maps_custom"
    output_dir = tmp_path / "output_custom"
    input_dir.mkdir()
    map_dir.mkdir()
    output_dir.mkdir()

    input_file = input_dir / "another.file.with.dots.txt"
    input_file.touch()

    map_suffix = ".sentence_map.json"
    analysis_suffix = ".results.json"

    # pathlib's stem handles multiple dots correctly: "another.file.with.dots"
    expected_map_path = map_dir / "another.file.with.dots.sentence_map.json"
    expected_analysis_path = output_dir / "another.file.with.dots.results.json"

    result_paths = generate_pipeline_paths(
        input_file=input_file,
        map_dir=map_dir,
        output_dir=output_dir,
        map_suffix=map_suffix,
        analysis_suffix=analysis_suffix,
    )

    assert result_paths.map_file == expected_map_path
    assert result_paths.analysis_file == expected_analysis_path


@pytest.mark.parametrize(
    "invalid_input_path",
    [
        Path("."),
        Path("/"),
        # Path("no_suffix") # This actually has stem "no_suffix", name "no_suffix"
        # Path(".hiddenfile") # This has stem ".hiddenfile", name ".hiddenfile"
    ],
    ids=["dot", "root"],
)
def test_generate_pipeline_paths_invalid_input_stem(tmp_path, invalid_input_path):
    """Tests that ValueError is raised for input paths without a valid stem."""
    map_dir = tmp_path / "maps"
    output_dir = tmp_path / "output"
    map_suffix = "_map.jsonl"
    analysis_suffix = "_analysis.jsonl"

    with pytest.raises(ValueError, match="must have a valid filename stem"):
        generate_pipeline_paths(
            input_file=invalid_input_path,
            map_dir=map_dir,
            output_dir=output_dir,
            map_suffix=map_suffix,
            analysis_suffix=analysis_suffix,
        )


# Optional: Test with task_id for logging coverage if needed, though it's minor.
def test_generate_pipeline_paths_with_task_id(tmp_path, caplog):
    """Tests that task_id is included in log messages."""
    caplog.set_level(logging.DEBUG)
    input_dir = tmp_path / "input"
    map_dir = tmp_path / "maps"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    map_dir.mkdir()
    output_dir.mkdir()
    input_file = input_dir / "log_test.txt"
    input_file.touch()
    task_id = "test-123"

    generate_pipeline_paths(
        input_file=input_file,
        map_dir=map_dir,
        output_dir=output_dir,
        map_suffix="_m.jsonl",
        analysis_suffix="_a.jsonl",
        task_id=task_id,
    )

    assert f"[Task {task_id}]" in caplog.text
    assert "Generating pipeline paths" in caplog.text
    assert "Generated paths: Map=" in caplog.text
