"""
src/tasks.py

Celery task definitions for background processing.

The task name (`run_pipeline_for_file`) is preserved for queue compatibility;
its body now runs Layer 1 ingestion followed by Layer 2 enrichment.
"""

import asyncio
from pathlib import Path

from src.celery_app import celery_app
from src.utils.logger import get_logger

logger = get_logger()


async def _ingest_and_enrich(
    input_file_path_str: str, map_dir_str: str, project_id: str, task_id: str,
    config_dict: dict = None,
):
    """Layer 1 ingestion then Layer 2 enrichment for a single file."""
    from src.enrichment.orchestrator import EnrichmentOrchestrator
    from src.ingestion.orchestrator import IngestionOrchestrator

    ingest = IngestionOrchestrator(project_id=project_id, map_dir=Path(map_dir_str))
    result = await ingest.ingest_file(Path(input_file_path_str))
    logger.info(f"[Task {task_id}] Ingested interview {result.interview_id}")
    # Empty dict falls back to global config; a populated dict is honored.
    enrich = EnrichmentOrchestrator(config_dict=config_dict or None)
    enrich_result = await enrich.enrich_interview(result.interview_id)
    logger.info(
        f"[Task {task_id}] Enriched {result.interview_id}: "
        f"{enrich_result.fragments_enriched} fragments"
    )
    return result.interview_id


def _run_pipeline_for_file_core(
    input_file_path_str: str,
    output_dir_str: str,
    map_dir_str: str,
    config_dict: dict,
    task_id: str = "unknown",
):
    """Core logic: ingest + enrich a single file. Extracted for testing.

    output_dir_str is retained in the signature for queue/back-compat but is no
    longer used (the event-sourced path writes no local output files).
    """
    logger.info(f"[Task {task_id}] Received task for file: {input_file_path_str}")

    try:
        project_id = config_dict.get("project_id", "default-project")
        interview_id = asyncio.run(
            _ingest_and_enrich(
                input_file_path_str=input_file_path_str,
                map_dir_str=map_dir_str,
                project_id=project_id,
                task_id=task_id,
                config_dict=config_dict,
            )
        )
        logger.info(f"[Task {task_id}] Successfully processed: {input_file_path_str}")
        return {"status": "Success", "file": input_file_path_str, "interview_id": interview_id}

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
        raise  # Re-raise so Celery marks the task FAILED


@celery_app.task(bind=True)
def run_pipeline_for_file(
    self,
    input_file_path_str: str,
    output_dir_str: str,
    map_dir_str: str,
    config_dict: dict,
):
    """Celery task: ingest + enrich a single input file (name kept for queue compat)."""
    task_id = self.request.id
    return _run_pipeline_for_file_core(
        input_file_path_str=input_file_path_str,
        output_dir_str=output_dir_str,
        map_dir_str=map_dir_str,
        config_dict=config_dict,
        task_id=task_id,
    )
