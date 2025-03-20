from pydantic import BaseModel

class AnalysisResult(BaseModel):
    overall_keywords: str  # Ensure this is a string
    domain_keywords: str    # Ensure this is a string
    function_type: str
    structure_type: str
    purpose: str
    topic_level_1: str
    topic_level_3: str
    overall_keywords: str
    domain_keywords: str
