"""
test_sentence_analyzer.py

This module contains unit tests for the SentenceAnalyzer class, which analyzes sentences
using the OpenAI API. It verifies that the classification of individual sentences and the
analysis of multiple sentences produce the expected results.

Usage:
    Run the tests using pytest:
        pytest tests/test_sentence_analyzer.py

Modifications:
    - If the classification prompts or expected keys change in SentenceAnalyzer, update the
      expected values in these tests.
    - New edge cases (such as domain_keywords returned as a string) are covered below.
"""

import pytest
from unittest.mock import patch, AsyncMock
from src.agents.sentence_analyzer import SentenceAnalyzer

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_contexts():
    """
    Fixture to provide mock contexts for testing.
    
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
    
    Returns:
        SentenceAnalyzer: An instance of the SentenceAnalyzer class.
    """
    return SentenceAnalyzer()

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence(mock_call_model, analyzer, mock_contexts):
    """
    Test the classification of a single sentence.
    
    This test verifies that the classify_sentence method correctly processes a given sentence,
    calls the OpenAI API the expected number of times, and returns the expected results.
    
    Asserts:
        - The classification results contain the expected keys and values.
        - The OpenAI API (call_model) is called exactly seven times.
    """
    # Return a plain dict instead of an AnalysisResult.
    fake_response = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"]
    }
    mock_call_model.return_value = fake_response

    sentence = "This is a test sentence."
    result = await analyzer.classify_sentence(sentence, mock_contexts)
    assert result["function_type"] == "declarative"
    assert result["structure_type"] == "simple sentence"
    assert result["purpose"] == "informational"
    assert result["topic_level_1"] == "testing"
    assert result["topic_level_3"] == "evaluation"
    assert result["overall_keywords"] == ["test"]
    assert result["domain_keywords"] == ["assessment", "evaluation"]
    # Expecting 7 calls for each classification dimension.
    assert mock_call_model.call_count == 7

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence_with_domain_keywords_string(mock_call_model, analyzer, mock_contexts):
    """
    Test classify_sentence when the API returns domain_keywords as a comma-separated string.
    
    This test ensures that if domain_keywords is returned as a string, it gets split into a list.
    
    Asserts:
        - The domain_keywords field in the result is a list of stripped keywords.
    """
    fake_result = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": "assessment, evaluation, extra"
    }
    mock_call_model.return_value = fake_result

    sentence = "Test sentence for domain keywords."
    result = await analyzer.classify_sentence(sentence, mock_contexts)
    assert isinstance(result["domain_keywords"], list)
    assert result["domain_keywords"] == ["assessment", "evaluation", "extra"]

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence_side_effect(mock_call_model, analyzer, mock_contexts):
    """
    Test classify_sentence with individual responses for each classification call.
    
    This test uses a side_effect list to simulate different responses for each call:
        1. function_type
        2. structure_type
        3. purpose
        4. topic_level_1
        5. topic_level_3
        6. overall_keywords
        7. domain_keywords (as a string)
    
    Asserts:
        - Each field in the final result matches the corresponding fake response.
        - The domain_keywords string is split into a list.
    """
    side_effect_responses = [
        {"function_type": "decl"},
        {"structure_type": "simple"},
        {"purpose": "info"},
        {"topic_level_1": "test"},
        {"topic_level_3": "eval"},
        {"overall_keywords": ["kw1", "kw2"]},
        {"domain_keywords": "dom1, dom2"}
    ]
    mock_call_model.side_effect = side_effect_responses

    sentence = "Side effect test sentence."
    result = await analyzer.classify_sentence(sentence, mock_contexts)
    # Check each field is extracted from its corresponding fake response.
    assert result["function_type"] == "decl"
    assert result["structure_type"] == "simple"
    assert result["purpose"] == "info"
    assert result["topic_level_1"] == "test"
    assert result["topic_level_3"] == "eval"
    assert result["overall_keywords"] == ["kw1", "kw2"]
    # Check that the domain_keywords string was split correctly.
    assert result["domain_keywords"] == ["dom1", "dom2"]
    # Total call count should equal 7.
    assert mock_call_model.call_count == 7

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_analyze_sentences(mock_call_model, analyzer):
    """
    Test the analysis of multiple sentences.
    
    This test verifies that analyze_sentences correctly processes a list of sentences,
    calling classify_sentence for each, and returns results that contain the expected attributes.
    
    Asserts:
        - The number of analysis results matches the number of input sentences.
        - Each result includes a sentence_id and the correct sentence text.
    """
    fake_response = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment", "evaluation"]
    }
    mock_call_model.return_value = fake_response
    sentences = ["First test sentence.", "Second test sentence."]
    results = await analyzer.analyze_sentences(sentences)
    assert len(results) == 2
    for idx, result in enumerate(results):
        assert result["sentence_id"] == idx
        assert result["sentence"] == sentences[idx]
        assert result["function_type"] == "declarative"

async def test_analyze_sentences_empty(analyzer):
    """
    Test analyze_sentences with an empty list of sentences.
    
    This test verifies that if an empty list is provided, the analyzer returns an empty list.
    
    Asserts:
        - The returned result is an empty list.
    """
    results = await analyzer.analyze_sentences([])
    assert results == []
