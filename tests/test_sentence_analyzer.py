# tests/test_sentence_analyzer.py
import pytest
from unittest.mock import patch, AsyncMock
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.models.analysis_result import AnalysisResult

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_contexts():
    """
    Fixture to provide mock contexts for testing.

    This fixture returns a dictionary containing various types of contexts
    that can be used in the tests for the SentenceAnalyzer class.

    Returns:
        dict: A dictionary with keys for different context types.
    """
    return {
        "structure": "",
        "immediate": "Immediate context around the sentence.",
        "observer": "Observer-level context.",
        "broader": "Broader context around the sentence.",
        "overall": "Overall context around the sentence."
    }

@pytest.fixture
def analyzer():
    """
    Fixture to create an instance of the SentenceAnalyzer for testing.

    This fixture initializes and returns an instance of the SentenceAnalyzer class,
    which can be used in the tests.

    Returns:
        SentenceAnalyzer: An instance of the SentenceAnalyzer class.
    """
    return SentenceAnalyzer()

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence(mock_call_model, analyzer, mock_contexts):
    """
    Test the classification of a single sentence.

    This test verifies that the classify_sentence method of the SentenceAnalyzer
    correctly classifies a given sentence and returns the expected results.

    Parameters:
        mock_call_model: The mocked OpenAIAgent's call_model method.
        analyzer: The instance of SentenceAnalyzer.
        mock_contexts: A fixture providing mock contexts for the sentence.

    Asserts:
        - The classification results match the expected values.
        - The call_model method is called the expected number of times.
    """
    mock_call_model.return_value = AnalysisResult(
        function_type="declarative",
        structure_type="simple sentence",
        purpose="informational",
        topic_level_1="testing",
        topic_level_3="evaluation",
        overall_keywords=["test"],
        domain_keywords=["assessment", "evaluation"]
    )

    sentence = "This is a test sentence."
    result = await analyzer.classify_sentence(sentence, mock_contexts)
    assert result["function_type"] == "declarative"
    assert result["structure_type"] == "simple sentence"
    assert result["purpose"] == "informational"
    assert result["topic_level_1"] == "testing"
    assert result["topic_level_3"] == "evaluation"
    assert result["overall_keywords"] == ["test"]
    assert result["domain_keywords"] == ["assessment", "evaluation"]
    assert mock_call_model.call_count == 7  # Seven classification calls expected

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_analyze_sentences(mock_call_model, analyzer):
    """
    Test the analysis of multiple sentences.

    This test verifies that the analyze_sentences method of the SentenceAnalyzer
    correctly analyzes a list of sentences and returns the expected results.

    Parameters:
        mock_call_model: The mocked OpenAIAgent's call_model method.
        analyzer: The instance of SentenceAnalyzer.

    Asserts:
        - The number of results matches the number of input sentences.
        - Each result contains the expected attributes.
    """
    mock_call_model.return_value = AnalysisResult(
        function_type="declarative",
        structure_type="simple sentence",
        purpose="informational",
        topic_level_1="testing",
        topic_level_3="evaluation",
        overall_keywords=["test"],
        domain_keywords=["assessment", "evaluation"]
    )
    sentences = ["First test sentence.", "Second test sentence."]
    results = await analyzer.analyze_sentences(sentences)
    assert len(results) == 2
    for idx, result in enumerate(results):
        assert result["sentence_id"] == idx
        assert result["sentence"] == sentences[idx]
        assert result["function_type"] == "declarative"
"""
test_sentence_analyzer.py

This module contains unit tests for the SentenceAnalyzer class, which is responsible for
analyzing sentences using the OpenAI API. The tests cover the classification of individual
sentences and the analysis of multiple sentences, ensuring that the results conform to the
expected structure and values.

Usage Example:

1. Run the tests using pytest:
   pytest tests/test_sentence_analyzer.py
"""
