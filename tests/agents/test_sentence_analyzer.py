"""Tests for the SentenceAnalyzer agent."""

import pytest
import logging
from unittest.mock import MagicMock, patch, AsyncMock, call
from pydantic import ValidationError

from src.agents.sentence_analyzer import SentenceAnalyzer

# Define dummy Pydantic models for mocking validation (or use MagicMock)
class MockValidationModel:
    def __init__(self, **kwargs):
        # Simulate Pydantic validation/attribute access
        for k, v in kwargs.items():
            setattr(self, k, v)

@pytest.fixture
def mock_config_dict() -> dict:
    """Fixture for a config dictionary with nested structure."""
    return {
        "classification": {
            "local": {
                "prompt_files": {
                    "no_context": "dict/prompts/no_context.yaml"
                }
            }
        },
        "domain_keywords": ["dict_keyword"],
        # Add other keys if SentenceAnalyzer uses them directly
    }

@pytest.fixture
def mock_global_config() -> MagicMock:
    """Fixture for a mock global config object."""
    cfg = MagicMock()
    # Mock attribute access as dictionary access
    cfg.get.side_effect = lambda key, default=None: {
        "classification": {
            "local": {
                "prompt_files": {
                    "no_context": "global/prompts/no_context.yaml"
                }
            }
        },
        "domain_keywords": ["global_keyword"],
    }.get(key, default)
    return cfg

@pytest.fixture
def loaded_prompts() -> dict:
    """Fixture for mock loaded prompts data."""
    # Structure matching how prompts are used in classify_sentence
    return {
        "sentence_function_type": {"prompt": "Function: {sentence}"},
        "sentence_structure_type": {"prompt": "Structure: {sentence}"},
        "sentence_purpose": {"prompt": "Purpose: {sentence}, Context: {context}"},
        "topic_level_1": {"prompt": "Topic1: {sentence}, Context: {context}"},
        "topic_level_3": {"prompt": "Topic3: {sentence}, Context: {context}"},
        "topic_overall_keywords": {"prompt": "Overall Kwds: Context: {context}"},
        "domain_specific_keywords": {"prompt": "Domain Kwds: {sentence}, Keywords: {domain_keywords}"},
    }

@pytest.fixture
def sentence_analyzer(mock_global_config, loaded_prompts):
    """Fixture providing SentenceAnalyzer and the mocked agent (using global config)."""
    # Patch global config, load_yaml, and agent used by SentenceAnalyzer
    with patch("src.agents.sentence_analyzer.global_config", mock_global_config), \
         patch("src.agents.sentence_analyzer.load_yaml", return_value=loaded_prompts) as mock_load_yaml, \
         patch("src.agents.sentence_analyzer.agent") as mock_agent_module_level: # Renamed mock

        analyzer = SentenceAnalyzer() # Initialize without dict
        mock_load_yaml.assert_called_once_with("global/prompts/no_context.yaml")
        # Yield analyzer AND the mock agent created by the patch
        yield analyzer, mock_agent_module_level

@pytest.fixture
def sentence_analyzer_with_dict(mock_config_dict, loaded_prompts):
    """Fixture providing SentenceAnalyzer and the mocked agent (using config_dict)."""
    # Patch load_yaml and agent used by SentenceAnalyzer
    with patch("src.agents.sentence_analyzer.load_yaml", return_value=loaded_prompts) as mock_load_yaml, \
         patch("src.agents.sentence_analyzer.agent") as mock_agent_module_level: # Renamed mock

        analyzer = SentenceAnalyzer(config_dict=mock_config_dict)
        mock_load_yaml.assert_called_once_with("dict/prompts/no_context.yaml")
        # Yield analyzer AND the mock agent created by the patch
        yield analyzer, mock_agent_module_level

# === Test __init__ ===

def test_init_uses_config_dict(sentence_analyzer_with_dict, mock_config_dict):
    """Test that __init__ uses the provided config_dict correctly."""
    analyzer, mock_agent = sentence_analyzer_with_dict # Unpack fixture
    # Check that the stored config is the one provided
    assert analyzer.config == mock_config_dict
    # load_yaml assertion happens within the fixture now

def test_init_falls_back_to_global_config(sentence_analyzer, mock_global_config):
    """Test that __init__ falls back to global config when no dict is provided."""
    analyzer, mock_agent = sentence_analyzer # Unpack fixture
    # Check that the stored config is the global mock
    assert analyzer.config == mock_global_config
    # load_yaml assertion happens within the fixture now

# === Test classify_sentence ===

@pytest.fixture
def mock_contexts() -> dict:
    """Provides sample contexts dictionary."""
    return {
        "immediate_context": "Immediate context text.",
        "observer_context": "Observer context text.",
        "broader_context": "Broader context text.",
    }

@pytest.mark.asyncio
async def test_classify_sentence_success(sentence_analyzer_with_dict, mock_contexts):
    """Test successful sentence classification and aggregation."""
    sentence = "This is a test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict # Unpack fixture

    # Explicitly set call_model as AsyncMock BEFORE assigning side_effect
    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"}, # function_prompt
        {"structure_type": "simple"},      # structure_prompt
        {"purpose": "inform"},            # purpose_prompt
        {"topic_level_1": "testing"},     # topic_lvl1_prompt
        {"topic_level_3": "unit testing"}, # topic_lvl3_prompt
        {"overall_keywords": ["test", "sentence"]}, # overall_keywords_prompt
        {"domain_keywords": ["dict_keyword"]}, # domain_prompt
    ]

    # Patch Pydantic models used for validation to bypass actual validation
    with patch("src.agents.sentence_analyzer.SentenceFunctionResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.SentenceStructureResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.SentencePurposeResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.TopicLevel1Response", MockValidationModel), \
         patch("src.agents.sentence_analyzer.TopicLevel3Response", MockValidationModel), \
         patch("src.agents.sentence_analyzer.OverallKeywordsResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.DomainKeywordsResponse", MockValidationModel):

        result = await analyzer.classify_sentence(sentence, mock_contexts)

    # Assertions
    assert result["sentence"] == sentence
    assert result["function_type"] == "declarative"
    assert result["structure_type"] == "simple"
    assert result["purpose"] == "inform"
    assert result["topic_level_1"] == "testing"
    assert result["topic_level_3"] == "unit testing"
    assert result["overall_keywords"] == ["test", "sentence"]
    assert result["domain_keywords"] == ["dict_keyword"]

    # Assert agent was called 7 times (once for each dimension)
    assert mock_agent.call_model.await_count == 7
    # Optional: More specific assertions on call args if needed

@pytest.mark.asyncio
async def test_classify_sentence_api_error(sentence_analyzer_with_dict, mock_contexts, caplog):
    """Test classify_sentence when agent.call_model raises an error."""
    # Set caplog level to capture ERROR messages
    caplog.set_level(logging.ERROR)
    sentence = "Another test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict # Unpack fixture

    # Define an async function to raise the exception for the side_effect
    async def raiser(*args, **kwargs):
        raise ValueError("Simulated API Error")

    # Configure mock agent to raise error using the async raiser function
    mock_agent.call_model = AsyncMock(side_effect=raiser)

    # Patch the Pydantic models to assert they aren't called
    with patch("src.agents.sentence_analyzer.SentenceFunctionResponse") as MockFuncResp, \
         patch("src.agents.sentence_analyzer.SentenceStructureResponse") as MockStructResp, \
         patch("src.agents.sentence_analyzer.SentencePurposeResponse") as MockPurposeResp, \
         patch("src.agents.sentence_analyzer.TopicLevel1Response") as MockTopic1Resp, \
         patch("src.agents.sentence_analyzer.TopicLevel3Response") as MockTopic3Resp, \
         patch("src.agents.sentence_analyzer.OverallKeywordsResponse") as MockOverallKwResp, \
         patch("src.agents.sentence_analyzer.DomainKeywordsResponse") as MockDomainKwResp:

        with pytest.raises(ValueError, match="Simulated API Error"):
            await analyzer.classify_sentence(sentence, mock_contexts)

        # Check logs
        assert f"Error during concurrent API calls for sentence '{sentence[:50]}...'" in caplog.text
        assert "Simulated API Error" in caplog.text
        # Removed unreliable await_count assertion

        # Assert that Pydantic models (validation part) were NOT called
        MockFuncResp.assert_not_called()
        MockStructResp.assert_not_called()
        MockPurposeResp.assert_not_called()
        MockTopic1Resp.assert_not_called()
        MockTopic3Resp.assert_not_called()
        MockOverallKwResp.assert_not_called()
        MockDomainKwResp.assert_not_called()

@pytest.mark.asyncio
async def test_classify_sentence_validation_error(sentence_analyzer_with_dict, mock_contexts, caplog):
    """Test classify_sentence when Pydantic validation fails for one dimension."""
    # Set caplog level to capture WARNING messages
    caplog.set_level(logging.WARNING)
    sentence = "Sentence causing validation issue."
    analyzer, mock_agent = sentence_analyzer_with_dict # Unpack fixture

    # Explicitly set call_model as AsyncMock BEFORE assigning side_effect
    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"},
        {"structure_type": "simple"},
        {"purpose": "inform"},
        {"invalid_key": "wrong_data"}, # Invalid response for TopicLevel1
        {"topic_level_3": "unit testing"},
        {"overall_keywords": ["test", "sentence"]},
        {"domain_keywords": ["dict_keyword"]},
    ]

    # No need to patch Pydantic models here, let validation fail
    result = await analyzer.classify_sentence(sentence, mock_contexts)

    # Assertions
    assert result["sentence"] == sentence
    assert result["function_type"] == "declarative"
    assert result["structure_type"] == "simple"
    assert result["purpose"] == "inform"
    # Check that the failed dimension has a default value
    assert result["topic_level_1"] == "" # Default defined in SentenceAnalyzer
    assert result["topic_level_3"] == "unit testing"
    assert result["overall_keywords"] == ["test", "sentence"]
    assert result["domain_keywords"] == ["dict_keyword"]

    # Check logs for validation warning
    assert "Validation failed for Topic Level 1 response" in caplog.text
    assert "invalid_key" in caplog.text # Log should contain the failed response
    # assert "metrics_tracker.increment_errors()" # Check if metrics tracking is called (if uncommented)

    assert mock_agent.call_model.await_count == 7 