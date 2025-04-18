"""
src/api/routers/analysis.py

API router for triggering and managing analysis tasks.
"""

from fastapi import APIRouter, HTTPException, Depends # Removed BackgroundTasks
from pathlib import Path

from src.api.schemas import AnalysisRequest, AnalysisResponse
# Removed direct config import as it's injected
# from src.config import config 
from src.utils.logger import get_logger
# Import the Celery task
from src.tasks import run_pipeline_for_file 
# Keep dependency function imports from *other* files
from src.api.routers.files import get_output_dir 
# Remove the incorrect relative import for locally defined functions
# from . import get_input_dir, get_map_dir, get_config_dep 

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"]
)

logger = get_logger()

# --- Dependency Functions (similar to files.py, adapt if needed) ---
def get_input_dir() -> Path:
    """Dependency function to get the input directory Path object from config."""
    from src.config import config # Import locally
    try:
        return Path(config['paths']['input_dir'])
    except KeyError as e:
        logger.critical(f"Config missing 'paths.input_dir': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server configuration error for input path.")

# Reuse get_output_dir from files router (or define here if preferred)
# Add get_map_dir dependency
def get_map_dir() -> Path:
    """Dependency function to get the map directory Path object from config."""
    from src.config import config # Import locally
    try:
        # Use .get for the optional map_dir key
        map_dir_str = config['paths'].get("map_dir", "data/maps") 
        return Path(map_dir_str)
    except KeyError as e:
        # This would only happen if 'paths' itself is missing
        logger.critical(f"Config missing 'paths' key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server configuration error for paths.")
    except Exception as e:
        logger.critical(f"Error creating map_dir Path: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server configuration error for map path.")

# Add config dependency function
def get_config_dep() -> dict:
    """Dependency function to return the underlying config dictionary."""
    from src.config import config # Import locally
    if config is None or not hasattr(config, 'config'):
        logger.critical("Global config object or its data not loaded!")
        raise HTTPException(status_code=500, detail="Server configuration not loaded.")
    # Return the actual dictionary attribute, not the Config object
    return config.config

@router.post("/", response_model=AnalysisResponse, status_code=202)
async def trigger_analysis(
    analysis_request: AnalysisRequest,
    # background_tasks: BackgroundTasks, # Removed
    input_dir: Path = Depends(get_input_dir),
    output_dir: Path = Depends(get_output_dir),
    map_dir: Path = Depends(get_map_dir),
    config_dep: dict = Depends(get_config_dep) 
):
    """Triggers the analysis pipeline task for a specified input file via Celery."""
    input_filename = analysis_request.input_filename
    # Construct the full path for validation
    # NOTE: input_dir here is the *directory* from the dependency
    input_file_path = input_dir / input_filename
    
    logger.info(f"Received analysis request for: {input_filename}")

    # --- Validation (Input file exists) --- 
    if not input_file_path.is_file():
        logger.warning(f"Requested input file not found: {input_file_path}")
        raise HTTPException(status_code=404, detail=f"Input file not found: {input_filename}")

    # --- Trigger Celery Task --- 
    logger.info(f"Sending analysis task for {input_file_path} to Celery queue.")
    
    # Convert Path objects to strings for Celery task arguments
    input_file_path_str = str(input_file_path)
    output_dir_str = str(output_dir)
    map_dir_str = str(map_dir)
    
    # Call .delay() on the task to send it to the queue
    try:
        task = run_pipeline_for_file.delay(
            input_file_path_str=input_file_path_str, 
            output_dir_str=output_dir_str,
            map_dir_str=map_dir_str,
            config_dict=config_dep 
        )
        logger.info(f"Task {task.id} sent to queue for file {input_filename}")
    except Exception as e:
        logger.error(f"Failed to send task to Celery queue: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to schedule analysis task.")

    # --- Return Response --- 
    return AnalysisResponse(
        message="Analysis task accepted and queued.", # Updated message
        input_filename=input_filename
        # task_id=task.id # Optionally return task id
    ) 