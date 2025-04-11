"""
llm_responses.py

Defines Pydantic models for validating the structure of responses 
received from the language model for various classification tasks.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

# Base model for common fields like confidence, could be extended if needed
# class BaseResponse(BaseModel):
#     confidence: Optional[float] = Field(None, ge=0, le=1)

class SentenceFunctionResponse(BaseModel):
    function_type: str = Field(..., description="The function type of the sentence (e.g., declarative)")
    # confidence: Optional[float] = Field(None, ge=0, le=1) # Example if confidence was needed

class SentenceStructureResponse(BaseModel):
    structure_type: str = Field(..., description="The structure type of the sentence (e.g., simple)")
    # confidence: Optional[float] = Field(None, ge=0, le=1)

class SentencePurposeResponse(BaseModel):
    purpose: str = Field(..., description="The purpose of the sentence (e.g., Statement, Query)")
    # confidence: Optional[float] = Field(None, ge=0, le=1)

class TopicLevel1Response(BaseModel):
    topic_level_1: str = Field(..., description="The topic classification based on immediate context")
    # confidence: Optional[float] = Field(None, ge=0, le=1)

class TopicLevel3Response(BaseModel):
    topic_level_3: str = Field(..., description="The topic classification based on broader context")
    # confidence: Optional[float] = Field(None, ge=0, le=1)

class OverallKeywordsResponse(BaseModel):
    overall_keywords: List[str] = Field(default_factory=list, description="List of keywords representing main topics")

class DomainKeywordsResponse(BaseModel):
    domain_keywords: List[str] = Field(default_factory=list, description="List of domain-specific keywords found") 