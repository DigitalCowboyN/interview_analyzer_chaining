"""
src/api/schemas.py

Defines Pydantic models used for API request validation and response serialization.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AnalysisResult(BaseModel):
    """Schema for a single sentence analysis result."""
    sentence_id: int
    sequence_order: int
    sentence: str
    
    # Core analysis fields (adjust based on actual SentenceAnalyzer output)
    function_type: Optional[str] = None 
    structure_type: Optional[str] = None
    purpose: Optional[str] = None
    topic_level_1: Optional[str] = None
    topic_level_3: Optional[str] = None
    overall_keywords: Optional[List[str]] = None
    domain_keywords: Optional[List[str]] = None
    
    # Error fields (optional)
    error: Optional[bool] = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Allow arbitrary fields if SentenceAnalyzer returns more
    # Alternatively, define all expected fields explicitly
    class Config:
        extra = 'allow' # or 'ignore' or remove if all fields are defined

class FileListResponse(BaseModel):
    """Schema for the response listing available analysis files."""
    filenames: List[str]

class AnalysisFileContent(BaseModel):
    """Schema for the response containing the content of an analysis file."""
    results: List[AnalysisResult]

# Add the new response model below
class FileContentResponse(BaseModel):
    """Response model for returning the content of a .jsonl file."""
    content: List[Dict[str, Any]]

# --- Schemas for Analysis Triggering ---

class AnalysisRequest(BaseModel):
    """Request model for triggering analysis on a specific input file."""
    input_filename: str

class AnalysisResponse(BaseModel):
    """Response model after successfully triggering an analysis."""
    message: str
    input_filename: str
    # Potentially add job_id here later if using background tasks 