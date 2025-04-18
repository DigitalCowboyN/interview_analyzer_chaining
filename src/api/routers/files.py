"""
src/api/routers/files.py

API router for file-related operations (listing, retrieving).
"""

from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
from typing import List, Dict, Any
import json # Needed for loading json lines

from src.config import config # Assuming config is accessible
from src.api.schemas import FileListResponse, FileContentResponse # Import the response schema
from src.utils.logger import get_logger

router = APIRouter(
    prefix="/files",
    tags=["Files"]
)

logger = get_logger()

# --- Dependency Functions --- 
def get_output_dir() -> Path:
    """Dependency function to get the output directory Path object from config."""
    from src.config import config # Import locally
    try:
        return Path(config['paths']['output_dir'])
    except KeyError as e:
        logger.critical(f"Config missing 'paths.output_dir': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server configuration error for paths.")

def get_analysis_suffix() -> str:
    """Dependency function to get the analysis file suffix from config."""
    from src.config import config # Import locally
    try:
        # Use .get for flexibility, provide a default
        suffix = config['paths'].get('analysis_suffix', '_analysis.jsonl') 
        if not isinstance(suffix, str):
            logger.error(f"Config 'paths.analysis_suffix' is not a string: {suffix}")
            raise HTTPException(status_code=500, detail="Server configuration error for suffix type.")
        return suffix
    except KeyError as e: # Should not happen with .get unless 'paths' is missing
        logger.critical(f"Config missing 'paths' key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server configuration error for paths.")
    except Exception as e: # Catch other unexpected errors
        logger.critical(f"Error retrieving analysis_suffix from config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server configuration error accessing suffix.")

# --- Updated Endpoints --- 

@router.get("/", response_model=FileListResponse)
async def list_analysis_files(output_dir: Path = Depends(get_output_dir),
                              analysis_suffix: str = Depends(get_analysis_suffix)):
    """Lists available analysis result files (.jsonl) in the output directory."""
    try:
        # Dependencies injected
        logger.info(f"Scanning for analysis files in: {output_dir} with suffix: {analysis_suffix}")

        if not output_dir.is_dir():
            logger.warning(f"Output directory not found: {output_dir}")
            return FileListResponse(filenames=[]) 

        analysis_files = [f.name for f in output_dir.glob(f"*{analysis_suffix}") if f.is_file()]
        
        logger.info(f"Found {len(analysis_files)} analysis files.")
        return FileListResponse(filenames=sorted(analysis_files))
        
    except Exception as e: # Simplified catch block as config errors handled in deps
        logger.error(f"Error listing analysis files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{filename}", response_model=FileContentResponse)
async def get_analysis_file_content(filename: str, 
                                  output_dir: Path = Depends(get_output_dir),
                                  analysis_suffix: str = Depends(get_analysis_suffix)):
    """Retrieves the content of a specific analysis result file (.jsonl)."""
    try:
        # Dependencies injected

        # Basic validation: check suffix
        if not filename.endswith(analysis_suffix):
            logger.warning(f"Requested filename '{filename}' does not have expected suffix '{analysis_suffix}'.")
            raise HTTPException(status_code=400, detail="Invalid filename suffix.")

        # Use the injected output_dir object
        file_path = output_dir / filename
        logger.info(f"Attempting to read file: {file_path}")

        if not file_path.is_file():
            logger.warning(f"Requested file not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found.")

        content = []
        try:
            # Use the derived file_path object (which originated from the injected mock)
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        content.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {filename}: {line.strip()}")
            logger.info(f"Successfully read {len(content)} lines from {filename}.")
            return FileContentResponse(content=content)
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error reading file.")

    except HTTPException: # Re-raise HTTP exceptions directly
        raise
    except IOError as e: # Specific handling for file read errors
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error reading file.")
    except Exception as e:
        # Catch other potential errors
        logger.error(f"Error retrieving file content for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") 