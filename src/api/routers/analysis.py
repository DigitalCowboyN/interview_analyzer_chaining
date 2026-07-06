"""
src/api/routers/analysis.py

API router for triggering and managing analysis tasks.
"""

import uuid  # Import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

# Import necessary schemas
from src.api.schemas import AnalysisTriggerRequest, AnalysisTriggerResponse
from src.config import config
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.ingestion.orchestrator import IngestionOrchestrator
from src.utils.logger import get_logger

router = APIRouter(prefix="/analysis", tags=["Analysis"])

logger = get_logger()


async def _ingest_and_enrich(file_path: Path, map_dir: str, project_id: str, task_id: str) -> None:
    """Background job: Layer 1 ingestion then Layer 2 enrichment."""
    try:
        ingest = IngestionOrchestrator(project_id=project_id, map_dir=Path(map_dir))
        result = await ingest.ingest_file(file_path)
        logger.info(f"[Task {task_id}] Ingested interview {result.interview_id}")
        enrich_result = await EnrichmentOrchestrator().enrich_interview(result.interview_id)
        logger.info(
            f"[Task {task_id}] Enriched interview {result.interview_id}: "
            f"{enrich_result.fragments_enriched} fragments"
        )
    except Exception as e:  # background task: log, do not crash the worker
        logger.error(f"[Task {task_id}] ingest+enrich failed: {e}", exc_info=True)


@router.post(
    "/",
    response_model=AnalysisTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Analysis Pipeline",
)
async def trigger_analysis(
    request: AnalysisTriggerRequest,
    background_tasks: BackgroundTasks,  # Inject BackgroundTasks
):
    """
    Triggers the analysis pipeline for a specific input file to run in the background.

    - Validates the input filename format.
    - Checks if the specified file exists in the configured input directory.
    - Schedules the `run_pipeline` function using background tasks.

    Returns an immediate `202 Accepted` response indicating the task has been scheduled.
    Errors during the background task execution itself are logged but do not affect this response.

    Args:
        request (AnalysisTriggerRequest): Request body containing the `input_filename`.
        background_tasks (BackgroundTasks): FastAPI mechanism for running background tasks.

    Returns:
        AnalysisTriggerResponse: Confirmation message and the input filename.

    Raises:
        HTTPException(400): If the `input_filename` has an invalid format (e.g., contains `/` or `..`).
        HTTPException(404): If the specified `input_filename` is not found in the input directory.
        HTTPException(500): If there is an unexpected internal server error during request
        validation or task scheduling.
    """
    task_id = str(uuid.uuid4())  # Generate unique task ID
    logger.info(
        f"[Task {task_id}] Received analysis trigger request for: {request.input_filename}"
    )
    try:
        config_dict: Dict[str, Any] = config
        input_dir_str = config_dict.get("paths", {}).get("input_dir", "./data/input")
        map_dir_str = config_dict.get("paths", {}).get("map_dir", "./data/maps")
        input_dir = Path(input_dir_str)
        input_file_path = input_dir / request.input_filename

        # --- Validation ---
        # Basic filename check (prevent path traversal)
        if "/" in request.input_filename or ".." in request.input_filename:
            logger.warning(
                f"[Task {task_id}] Invalid characters in requested input filename: "
                f"{request.input_filename}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input filename format.",
            )

        # Check if input file exists (run_pipeline might also do this, but good to check early)
        if not input_file_path.is_file():
            logger.warning(
                f"[Task {task_id}] Requested input file for analysis not found: {input_file_path}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Input file not found: {request.input_filename}",
            )

        # --- Schedule ingest + enrich in Background ---
        logger.info(
            f"[Task {task_id}] Scheduling background ingest+enrich task for {input_file_path}..."
        )
        background_tasks.add_task(
            _ingest_and_enrich,
            file_path=input_file_path,
            map_dir=map_dir_str,
            project_id=config_dict.get("project_id", "default-project"),
            task_id=task_id,
        )

        return AnalysisTriggerResponse(
            message="Analysis task accepted and scheduled to run in background.",
            input_filename=request.input_filename,
            task_id=task_id,  # Include task_id in response
        )

    except FileNotFoundError as fnf_error:
        # Catch specific error if run_pipeline raises it for the input file
        logger.warning(
            f"[Task {task_id}] Analysis failed: Input file not found during pipeline execution: "
            f"{fnf_error}"
        )
        # The file existence check above should prevent this, but handle defensively.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Input file not found during analysis: {request.input_filename}",
        )
    except HTTPException as http_exc:
        raise http_exc  # Re-raise specific HTTP exceptions
    except Exception as e:
        logger.error(
            f"[Task {task_id}] Error triggering analysis for {request.input_filename}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during analysis trigger.",
        )
