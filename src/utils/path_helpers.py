"""
Utility functions for generating and manipulating file paths specific to the pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional  # Added Optional for task_id consistency

# Initialize logger for this module if needed, or rely on root logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelinePaths:
    """Holds the derived paths for map and analysis files related to an input file."""

    map_file: Path
    analysis_file: Path
    # Add other related paths here if needed in the future (e.g., temp_dir)


def generate_pipeline_paths(
    input_file: Path,
    map_dir: Path,
    output_dir: Path,
    map_suffix: str,
    analysis_suffix: str,
    task_id: Optional[str] = None,  # Add task_id for consistent logging
) -> PipelinePaths:
    """
    Generates the full paths for map and analysis files based on an input file.

    Args:
        input_file: Path to the source text file.
        map_dir: Directory for storing map files.
        output_dir: Directory for storing analysis output files.
        map_suffix: Suffix for map files (e.g., "_map.jsonl").
        analysis_suffix: Suffix for analysis files (e.g., "_analysis.jsonl").
        task_id: Optional task identifier for logging context.

    Returns:
        A PipelinePaths object containing the map_file and analysis_file paths.

    Raises:
        ValueError: If input_file does not have a stem (e.g., is root dir or invalid).
    """
    prefix = f"[Task {task_id}] " if task_id else ""
    logger.debug(f"{prefix}Generating pipeline paths for input: {input_file}")

    if not input_file.name or not input_file.stem:
        logger.error(
            f"{prefix}Input file path '{input_file}' lacks a valid filename/stem "
            f"for generating output filenames."
        )
        raise ValueError(
            f"Input file path '{input_file}' must have a valid filename stem."
        )

    map_file_path = map_dir / f"{input_file.stem}{map_suffix}"
    analysis_file_path = output_dir / f"{input_file.stem}{analysis_suffix}"

    logger.debug(
        f"{prefix}Generated paths: Map='{map_file_path}', Analysis='{analysis_file_path}'"
    )
    return PipelinePaths(map_file=map_file_path, analysis_file=analysis_file_path)
