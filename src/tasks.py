"""
src/tasks.py

Celery task definitions for background processing.
"""

import asyncio
from pathlib import Path

from src.celery_app import celery_app
from src.pipeline import run_pipeline
from src.utils.logger import get_logger

logger = get_logger()


def _run_pipeline_for_file_core(
    input_file_path_str: str,
    output_dir_str: str,
    map_dir_str: str,
    config_dict: dict,
    task_id: str = "unknown",
):
    """
    Core logic for running pipeline for a single file.
    Extracted for easier testing.

    Args:
        input_file_path_str (str): Path to the input file as a string.
        output_dir_str (str): Path to the output directory as a string.
        map_dir_str (str): Path to the map directory as a string.
        config_dict (dict): Configuration dictionary.
        task_id (str): Task ID for logging.

    Returns:
        dict: Status result

    Raises:
        FileNotFoundError: If input file doesn't exist
        Exception: Other pipeline errors
    """
    logger.info(f"[Task {task_id}] Received task for file: {input_file_path_str}")

    try:
        input_file = Path(input_file_path_str)
        output_dir = Path(output_dir_str)
        map_dir = Path(map_dir_str)

        # --- Run Core Pipeline Logic ---
        logger.info(f"[Task {task_id}] Starting run_pipeline for {input_file}")

        asyncio.run(
            run_pipeline(
                input_dir=input_file.parent,  # Use the directory containing the file
                output_dir=output_dir,
                map_dir=map_dir,
                specific_file=input_file.name,  # Pass the filename
                config_dict=config_dict,
                task_id=task_id,
            )
        )

        logger.info(f"[Task {task_id}] Successfully processed: {input_file_path_str}")
        return {"status": "Success", "file": input_file_path_str}

    except FileNotFoundError:
        logger.error(
            f"[Task {task_id}] Input file not found during processing: {input_file_path_str}",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(
            f"[Task {task_id}] Error processing file {input_file_path_str}: {e}",
            exc_info=True,
        )
        raise  # Re-raise the exception so Celery marks the task as FAILED


@celery_app.task(bind=True)
def run_pipeline_for_file(
    self,
    input_file_path_str: str,
    output_dir_str: str,
    map_dir_str: str,
    config_dict: dict,
):
    """
    Celery task to run the analysis pipeline for a single input file.

    Instantiates necessary services and calls the core run_pipeline function.

    Args:
        self: The task instance (available via bind=True).
        input_file_path_str (str): Path to the input file as a string.
        output_dir_str (str): Path to the output directory as a string.
        map_dir_str (str): Path to the map directory as a string.
        config_dict (dict): Configuration dictionary (passed from API).
    """
    task_id = self.request.id
    return _run_pipeline_for_file_core(
        input_file_path_str=input_file_path_str,
        output_dir_str=output_dir_str,
        map_dir_str=map_dir_str,
        config_dict=config_dict,
        task_id=task_id,
    )
