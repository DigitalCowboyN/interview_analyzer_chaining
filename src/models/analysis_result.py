from pydantic import BaseModel
from typing import List  # Import List from typing

class AnalysisResult(BaseModel):
    # Ensure the fields match the expected output from the OpenAI API
    function_type: str
    structure_type: str
    purpose: str
    topic_level_1: str
    topic_level_3: str
    overall_keywords: List[str]  # Change to List of strings
    domain_keywords: List[str]    # Change to List of strings
