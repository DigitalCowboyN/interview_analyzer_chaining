"""
src/api/routers/files.py

API router for file-related operations (listing, retrieving).
"""

from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
from typing import List, Dict, Any
import json # Needed for loading json lines
import pydantic

from src.config import config # Assuming config is accessible
from src.api.schemas import FileListResponse, FileContentResponse, AnalysisResult # Import the response schema and new models
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

@router.get(
    "/", 
    response_model=FileListResponse,
    summary="List Available Analysis Files"
)
async def list_analysis_files():
    """
    Retrieves a list of completed analysis filenames.
    
    Scans the configured output directory for files ending with the analysis suffix.
    Returns an empty list if the directory doesn't exist or no files are found.
    Raises HTTPException 500 on internal server errors during listing.
    """
    try:
        output_dir_str = config.get("paths", {}).get("output_dir", "./data/output")
        analysis_suffix = config.get("paths", {}).get("analysis_suffix", "_analysis.jsonl")
        output_dir = Path(output_dir_str)
        
        if not output_dir.is_dir():
            logger.warning(f"Output directory not found: {output_dir}")
            # Return empty list if dir doesn't exist
            return FileListResponse(filenames=[]) 
            
        # Use glob to find files ending with the suffix
        analysis_files = [f.name for f in output_dir.glob(f"*{analysis_suffix}") if f.is_file()]
        
        return FileListResponse(filenames=sorted(analysis_files))
        
    except Exception as e:
        logger.error(f"Error listing analysis files: {e}", exc_info=True)
        # Return an internal server error response
        raise HTTPException(status_code=500, detail="Internal server error listing analysis files.")

@router.get(
    "/{filename}", 
    response_model=FileContentResponse,
    summary="Get Analysis File Content"
)
async def get_analysis_file_content(filename: str):
    """
    Retrieves the content of a specific analysis file.

    Reads the specified .jsonl file from the output directory, parses each line
    as JSON, and returns the results. Malformed JSON lines are skipped and logged.

    Args:
        filename (str): The name of the analysis file (e.g., "interview1_analysis.jsonl").

    Returns:
        FileContentResponse: The content of the file including filename and results.

    Raises:
        HTTPException(404): If the specified file does not exist.
        HTTPException(500): If there's an OS error reading the file or another internal error.
    """
    try:
        output_dir_str = config.get("paths", {}).get("output_dir", "./data/output")
        output_dir = Path(output_dir_str)
        file_path = output_dir / filename

        if not file_path.is_file():
            logger.warning(f"Requested analysis file not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"Analysis file not found: {filename}")

        results = []
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            # Parse JSON line into a dictionary
                            data = json.loads(line)
                            # Validate with Pydantic model (optional but good practice)
                            # analysis_item = AnalysisResult(**data) 
                            # results.append(analysis_item)
                            results.append(data) # For now, just append the dict
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed JSON line in {filename}: {line[:100]}...")
                            # Decide whether to skip or raise 500 - skipping for now
                        except Exception as pydantic_err: # Catch potential Pydantic validation error
                            logger.warning(f"Skipping line due to validation/data error in {filename}: {pydantic_err}. Line: {line[:100]}...")
                            # results.append({"error": "validation failed", "line": line}) # Option to include errors
        except OSError as e:
            logger.error(f"Error reading analysis file {filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error reading file: {filename}")

        # Return the results wrapped in the response model
        return FileContentResponse(filename=filename, results=results)

    except HTTPException as http_exc:
        raise http_exc # Re-raise specific HTTP exceptions (like 404)
    except Exception as e:
        logger.error(f"Unexpected error retrieving content for file {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving file content.")

@router.get(
    "/{filename}/sentences/{sentence_id}", 
    response_model=AnalysisResult,
    summary="Get Specific Sentence Analysis"
)
async def get_specific_sentence_analysis(filename: str, sentence_id: int):
    """
    Retrieves the analysis result for a specific sentence within a file.

    Reads the specified .jsonl analysis file line by line, searching for the
    entry matching the given `sentence_id`. Skips malformed lines.

    Args:
        filename (str): The name of the analysis file.
        sentence_id (int): The ID of the sentence to retrieve.

    Returns:
        AnalysisResult: The analysis data for the specified sentence.

    Raises:
        HTTPException(404): If the file is not found, or the sentence_id is not found 
                            within the file (after skipping malformed lines).
        HTTPException(500): If there's an OS error reading the file, a validation error 
                            for the target sentence's data, or another internal error.
    """
    logger.debug(f"Request received for sentence {sentence_id} in file {filename}")
    try:
        output_dir_str = config.get("paths", {}).get("output_dir", "./data/output")
        output_dir = Path(output_dir_str)
        file_path = output_dir / filename

        if not file_path.is_file():
            logger.warning(f"Analysis file not found for specific sentence request: {file_path}")
            raise HTTPException(status_code=404, detail=f"Analysis file not found: {filename}")

        found_result = None
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Check if 'sentence_id' exists and matches the requested ID
                        if data.get("sentence_id") == sentence_id:
                            logger.info(f"Found sentence {sentence_id} in {filename} at line {line_num + 1}")
                            # Validate data against the response model
                            try:
                                found_result = AnalysisResult(**data)
                                break # Exit loop once found
                            except pydantic.ValidationError as validation_err:
                                logger.error(f"Data validation failed for sentence {sentence_id} in {filename}: {validation_err}. Data: {data}")
                                # Raise 500 as this indicates bad data was written for the specific requested sentence
                                raise HTTPException(status_code=500, detail=f"Data validation error for sentence {sentence_id}")
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line {line_num + 1} in {filename} while searching for sentence {sentence_id}")
                        continue # Keep searching
                    # Catch the specific HTTPException from validation and re-raise it
                    except HTTPException as http_exc:
                        raise http_exc
                    # Catch other unexpected errors during line processing (excluding JSONDecodeError and ValidationError handled above)
                    except Exception as inner_e:
                         logger.warning(f"Skipping line {line_num + 1} in {filename} due to unexpected error processing line data (excluding JSON/Validation): {inner_e}. Line: {line[:100]}...")
                         continue # Keep searching

        except OSError as e:
            logger.error(f"Error reading analysis file {filename} while searching for sentence {sentence_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error reading file: {filename}")

        if found_result:
            return found_result
        else:
            logger.warning(f"Sentence ID {sentence_id} not found in file {filename}")
            raise HTTPException(status_code=404, detail=f"Sentence ID {sentence_id} not found in file {filename}")

    except HTTPException as http_exc:
        raise http_exc # Re-raise specific HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error retrieving sentence {sentence_id} for file {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving sentence analysis.") 