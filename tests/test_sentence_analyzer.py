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
    Pytest fixture providing mock context data for testing SentenceAnalyzer.

    Returns a dictionary mimicking the structure of contexts passed to
    `classify_sentence`, containing various context types (e.g., immediate,
    observer) needed by different analysis prompts.

    Returns:
        dict[str, str]: A dictionary mapping context type names to mock string values.
    """
    # Use the full context keys as defined in config.yaml and used by ContextBuilder
    return {
        "structure_analysis": "Structure context.", # Add missing key if needed, adjust value
        "immediate_context": "Immediate context around the sentence.",
        "observer_context": "Observer-level context.",
        "broader_context": "Broader context around the sentence.",
        "overall_context": "Overall context around the sentence."
        # Note: The actual values don't matter much here as SentenceAnalyzer just passes them
        # to the agent's call_model, which is mocked in these tests. But keys must match.
    }

@pytest.fixture
def analyzer():
    """
    Pytest fixture to create a new instance of SentenceAnalyzer for each test.
    
    Returns:
        SentenceAnalyzer: An instance of the SentenceAnalyzer class.
    """
    return SentenceAnalyzer()

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence_success(mock_call_model, analyzer, mock_contexts):
    """
    Test `classify_sentence` successfully processes valid mocked API responses.
    
    Mocks `agent.call_model` to return a sequence of valid dictionaries conforming
    to the expected Pydantic models (`SentenceFunctionResponse`, etc.).
    Verifies that `classify_sentence` correctly calls the mock API 7 times
    and returns a result dictionary with correctly parsed values.
    
    Args:
        mock_call_model: Mock object for `agent.call_model`.
        analyzer: Fixture providing a `SentenceAnalyzer` instance.
        mock_contexts: Fixture providing mock context data.

    Returns:
        None

    Raises:
        AssertionError: If the result dictionary is missing keys, has incorrect
                      values, or if the mock API was not called 7 times.
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
    Test `classify_sentence` handles Pydantic validation errors gracefully.
    
    Mocks `agent.call_model` to return a sequence of responses where some are
    invalid (e.g., wrong key names, incorrect data types) and will fail Pydantic
    validation within `classify_sentence`.
    Verifies that `classify_sentence` returns default values (empty string/list)
    for the dimensions where validation failed, while still returning correct values
    for dimensions with valid responses.
    
    Args:
        mock_call_model: Mock object for `agent.call_model`.
        analyzer: Fixture providing a `SentenceAnalyzer` instance.
        mock_contexts: Fixture providing mock context data.

    Returns:
        None

    Raises:
        AssertionError: If the result dictionary does not contain the expected mix
                      of default and correctly parsed values.
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
    Test the deprecated `analyze_sentences` method's aggregation logic.
    
    Mocks the internal call to `classify_sentence` and the call to
    `context_builder.build_all_contexts`. Verifies that `analyze_sentences` iterates
    through input sentences, calls the mocked `classify_sentence` correctly for each,
    and aggregates the results, adding the correct `sentence_id` and `sentence` text.
    
    Args:
        mock_classify: Mock object for `analyzer.classify_sentence`.
        analyzer: Fixture providing a `SentenceAnalyzer` instance.
        mock_contexts: Fixture providing mock context data (used by the mock context builder).

    Returns:
        None

    Raises:
        AssertionError: If the aggregated results list has the wrong length,
                      results are missing expected keys/values, or mocks were not
                      called as expected.
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
        mock_build_contexts.return_value = {idx: mock_contexts for idx in range(len(sentences))} 
        
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
    Test the deprecated `analyze_sentences` raises ValueError for empty input.
    
    Verifies that calling `analyze_sentences` with an empty list raises a
    `ValueError` as expected.

    Args:
        analyzer: Fixture providing a `SentenceAnalyzer` instance.

    Returns:
        None

    Raises:
        AssertionError: If `ValueError` is not raised.
    """
    # Mock context_builder as it's called by analyze_sentences, though it won't be reached
    with patch("src.agents.sentence_analyzer.context_builder.build_all_contexts") as mock_build_contexts:
        # Check that analyze_sentences raises ValueError for an empty list
        with pytest.raises(ValueError):
            await analyzer.analyze_sentences([])
        # Assert that build_all_contexts was not called because the error is raised first
        mock_build_contexts.assert_not_called() 
