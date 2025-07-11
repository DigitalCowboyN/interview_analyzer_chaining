"""
analysis_result.py

Defines the Pydantic model for representing the consolidated analysis results
for a single sentence after processing by the SentenceAnalyzer.
"""

from typing import List  # Import List from typing

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """
    Represents the complete set of analysis dimensions for a single sentence.

    This model aggregates the outputs obtained from various classification tasks
    performed by the SentenceAnalyzer.

    Attributes:
        function_type (str): The classified functional type of the sentence.
        structure_type (str): The classified structural type of the sentence.
        purpose (str): The classified purpose of the sentence.
        topic_level_1 (str): The Level 1 topic classification.
        topic_level_3 (str): The Level 3 topic classification.
        overall_keywords (List[str]): List of overall keywords extracted.
        domain_keywords (List[str]): List of domain-specific keywords extracted.
    """

    # Fields match the keys consolidated in SentenceAnalyzer.classify_sentence
    function_type: str = Field(
        ...,
        description="The classified functional type of the sentence (e.g., declarative)",
    )
    structure_type: str = Field(
        ..., description="The classified structural type of the sentence (e.g., simple)"
    )
    purpose: str = Field(
        ..., description="The classified purpose of the sentence (e.g., Statement)"
    )
    topic_level_1: str = Field(
        ..., description="The Level 1 topic classification based on immediate context"
    )
    topic_level_3: str = Field(
        ..., description="The Level 3 topic classification based on broader context"
    )
    overall_keywords: List[str] = Field(
        ..., description="List of keywords representing the main overall topics"
    )
    domain_keywords: List[str] = Field(
        ..., description="List of domain-specific keywords found in the sentence"
    )
