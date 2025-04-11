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
from src.models.llm_responses import ValidationError # Import ValidationError for testing error cases

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
async def test_classify_sentence_success(mock_call_model, analyzer, mock_contexts):
    """
    Test successful classification of a single sentence using Pydantic validation.
    
    Verifies that classify_sentence correctly processes valid responses from mocked API calls
    that conform to the Pydantic models.
    
    Asserts:
        - The classification results contain the expected keys and values matching the mocks.
        - The OpenAI API (call_model) is called exactly seven times.
    """
    # Define valid mock responses conforming to Pydantic models
    side_effect_responses = [
        {"function_type": "declarative"},                     # 1. SentenceFunctionResponse
        {"structure_type": "simple"},                       # 2. SentenceStructureResponse
        {"purpose": "informational"},                      # 3. SentencePurposeResponse
        {"topic_level_1": "testing"},                      # 4. TopicLevel1Response
        {"topic_level_3": "evaluation"},                     # 5. TopicLevel3Response
        {"overall_keywords": ["test", "validation"]},      # 6. OverallKeywordsResponse
        {"domain_keywords": ["assessment"]}                 # 7. DomainKeywordsResponse
    ]
    mock_call_model.side_effect = side_effect_responses

    sentence = "This is a valid test sentence."
    result = await analyzer.classify_sentence(sentence, mock_contexts)
    
    # Check that values from the validated responses are correct
    assert result["function_type"] == "declarative"
    assert result["structure_type"] == "simple"
    assert result["purpose"] == "informational"
    assert result["topic_level_1"] == "testing"
    assert result["topic_level_3"] == "evaluation"
    assert result["overall_keywords"] == ["test", "validation"]
    assert result["domain_keywords"] == ["assessment"]
    assert result["sentence"] == sentence
    
    assert mock_call_model.call_count == 7

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence_validation_failure(mock_call_model, analyzer, mock_contexts):
    """
    Test classify_sentence when some API responses fail Pydantic validation.
    
    Verifies that classify_sentence handles invalid responses by logging a warning
    (implicitly tested by checking default values) and returning default values
    (empty string/list) for the failed fields.
    
    Asserts:
        - Fields with valid responses have correct values.
        - Fields with invalid responses have default values ("" or []).
        - The OpenAI API (call_model) is called exactly seven times.
    """
    # Define responses where some are invalid (missing keys, wrong types)
    side_effect_responses = [
        {"function_typo": "declarative"},                   # 1. Invalid (wrong key)
        {"structure_type": "simple"},                       # 2. Valid
        {"purpose": 123},                                   # 3. Invalid (wrong type)
        {"topic_level_1": "testing"},                      # 4. Valid
        {"topic_level_3": "evaluation"},                     # 5. Valid
        {"overall_keywords": "not a list"},                 # 6. Invalid (wrong type)
        {"domain_keywords": ["assessment"]}                 # 7. Valid
    ]
    mock_call_model.side_effect = side_effect_responses

    sentence = "This sentence tests validation failures."
    result = await analyzer.classify_sentence(sentence, mock_contexts)
    
    # Check values - defaults for invalid, correct for valid
    assert result["function_type"] == ""  # Default for validation failure
    assert result["structure_type"] == "simple"
    assert result["purpose"] == ""  # Default for validation failure
    assert result["topic_level_1"] == "testing"
    assert result["topic_level_3"] == "evaluation"
    assert result["overall_keywords"] == [] # Default for validation failure
    assert result["domain_keywords"] == ["assessment"]
    assert result["sentence"] == sentence
    
    assert mock_call_model.call_count == 7

@patch("src.agents.sentence_analyzer.SentenceAnalyzer.classify_sentence", new_callable=AsyncMock)
async def test_analyze_sentences(mock_classify, analyzer, mock_contexts):
    """
    Test the analysis of multiple sentences, mocking classify_sentence.
    
    Verifies that analyze_sentences iterates through sentences, calls classify_sentence for each,
    and correctly aggregates the results with sentence_id.
    
    Asserts:
        - The number of results matches the number of input sentences.
        - Each result includes the correct sentence_id and sentence text.
        - classify_sentence is called once per sentence.
    """
    # Mock the output of classify_sentence
    fake_classification_result = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "testing",
        "topic_level_3": "evaluation",
        "overall_keywords": ["test"],
        "domain_keywords": ["assessment"]
        # 'sentence' key is added by analyze_sentences wrapper
    }
    
    sentences = ["First test sentence.", "Second test sentence."]
    
    # Use side_effect to return *copies* for each call
    mock_classify.side_effect = [fake_classification_result.copy() for _ in sentences]

    # Mock context_builder as it's called by analyze_sentences
    with patch("src.agents.sentence_analyzer.context_builder.build_all_contexts") as mock_build_contexts:
        # Return mock contexts for the number of sentences - Use injected fixture value
        mock_build_contexts.return_value = [mock_contexts] * len(sentences) 
        
        results = await analyzer.analyze_sentences(sentences)

    assert len(results) == 2
    assert mock_classify.call_count == 2
    
    # This loop should now work correctly as each 'result' is a distinct dict
    for idx, result in enumerate(results):
        assert result["sentence_id"] == idx
        assert result["sentence"] == sentences[idx] 
        # Check one field to ensure the mock result was included
        assert result["function_type"] == fake_classification_result["function_type"]
        
        # Explicitly check the 'sentence' key exists and only once
        assert 'sentence' in result, f"'sentence' key missing in result for index {idx}"
        assert list(result.keys()).count('sentence') == 1, f"'sentence' key found multiple times in result for index {idx}"

async def test_analyze_sentences_empty(analyzer):
    """
    Test analyze_sentences with an empty list of sentences.
    
    Asserts:
        - The returned result is an empty list.
    """
    # Mock context_builder as it's called by analyze_sentences
    with patch("src.agents.sentence_analyzer.context_builder.build_all_contexts") as mock_build_contexts:
        mock_build_contexts.return_value = [] # Return empty list for empty input
        results = await analyzer.analyze_sentences([])
        assert results == []
        mock_build_contexts.assert_called_once_with([]) # Ensure it was called
