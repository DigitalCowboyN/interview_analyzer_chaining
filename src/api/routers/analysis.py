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

# Import the core pipeline function (adjust path if necessary)
# from src.pipeline import run_pipeline
# Import logger
from src.utils.logger import get_logger

router = APIRouter(prefix="/analysis", tags=["Analysis"])

logger = get_logger()


# Endpoint implementation will go here


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
        # Import locally to avoid potential circular dependencies at module level
        from src.config import config
        from src.pipeline import run_pipeline

        config_dict: Dict[str, Any] = config
        input_dir_str = config_dict.get("paths", {}).get("input_dir", "./data/input")
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

        # --- Schedule Pipeline in Background ---
        logger.info(
            f"[Task {task_id}] Scheduling background analysis task for {input_file_path}..."
        )
        background_tasks.add_task(
            run_pipeline,
            task_id=task_id,  # Pass the task_id
            input_dir=input_dir_str,  # Pass arguments needed by run_pipeline
            specific_file=request.input_filename,
            # Pass other necessary args if run_pipeline's signature changes
            # output_dir=config.get("paths",{}).get("output_dir"),
            # map_dir=config.get("paths",{}).get("map_dir"),
            # config=config.config # Pass the actual config dict if needed
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
