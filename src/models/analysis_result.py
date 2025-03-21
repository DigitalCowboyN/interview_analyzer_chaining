from pydantic import BaseModel

class AnalysisResult(BaseModel):
    # Ensure the fields match the expected output from the OpenAI API
    function_type: str
    structure_type: str
    purpose: str
    topic_level_1: str
    topic_level_3: str
    overall_keywords: str  # Ensure this is a string
    domain_keywords: str    # Ensure this is a string
