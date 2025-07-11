"""Tests for the SentenceAnalyzer agent."""

import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

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
            "local": {"prompt_files": {"no_context": "dict/prompts/no_context.yaml"}}
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
            "local": {"prompt_files": {"no_context": "global/prompts/no_context.yaml"}}
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
        "domain_specific_keywords": {
            "prompt": "Domain Kwds: {sentence}, Keywords: {domain_keywords}"
        },
    }


@pytest.fixture
def sentence_analyzer(mock_global_config, loaded_prompts):
    """Fixture providing SentenceAnalyzer and the mocked agent (using global config)."""
    # Patch global config, load_yaml, and agent used by SentenceAnalyzer
    with patch("src.agents.sentence_analyzer.global_config", mock_global_config), patch(
        "src.agents.sentence_analyzer.load_yaml", return_value=loaded_prompts
    ) as mock_load_yaml, patch(
        "src.agents.sentence_analyzer.agent"
    ) as mock_agent_module_level:  # Renamed mock

        analyzer = SentenceAnalyzer()  # Initialize without dict
        mock_load_yaml.assert_called_once_with("global/prompts/no_context.yaml")
        # Yield analyzer AND the mock agent created by the patch
        yield analyzer, mock_agent_module_level


@pytest.fixture
def sentence_analyzer_with_dict(mock_config_dict, loaded_prompts):
    """Fixture providing SentenceAnalyzer and the mocked agent (using config_dict)."""
    # Patch load_yaml and agent used by SentenceAnalyzer
    with patch(
        "src.agents.sentence_analyzer.load_yaml", return_value=loaded_prompts
    ) as mock_load_yaml, patch(
        "src.agents.sentence_analyzer.agent"
    ) as mock_agent_module_level:  # Renamed mock

        analyzer = SentenceAnalyzer(config_dict=mock_config_dict)
        mock_load_yaml.assert_called_once_with("dict/prompts/no_context.yaml")
        # Yield analyzer AND the mock agent created by the patch
        yield analyzer, mock_agent_module_level


# === Test __init__ ===


def test_init_uses_config_dict(sentence_analyzer_with_dict, mock_config_dict):
    """Test that __init__ uses the provided config_dict correctly."""
    analyzer, mock_agent = sentence_analyzer_with_dict  # Unpack fixture
    # Check that the stored config is the one provided
    assert analyzer.config == mock_config_dict
    # load_yaml assertion happens within the fixture now


def test_init_falls_back_to_global_config(sentence_analyzer, mock_global_config):
    """Test that __init__ falls back to global config when no dict is provided."""
    analyzer, mock_agent = sentence_analyzer  # Unpack fixture
    # Check that the stored config is the global mock
    assert analyzer.config == mock_global_config
    # load_yaml assertion happens within the fixture now


def test_init_missing_classification_key(caplog):
    """Test initialization when classification key is missing from config."""
    caplog.set_level(logging.ERROR)

    config_missing_classification = {"domain_keywords": ["test"]}

    with patch("src.agents.sentence_analyzer.load_yaml") as mock_load_yaml:
        analyzer = SentenceAnalyzer(config_dict=config_missing_classification)

        # Should not call load_yaml due to missing key
        mock_load_yaml.assert_not_called()

        # Should have empty prompts
        assert analyzer.prompts == {}

        # Should log error
        assert "Failed to get prompt path from config" in caplog.text
        assert "Using empty prompts" in caplog.text


def test_init_missing_nested_keys(caplog):
    """Test initialization when nested keys are missing from config."""
    caplog.set_level(logging.ERROR)

    config_missing_nested = {
        "classification": {
            "local": {}  # Missing prompt_files key
        }
    }

    with patch("src.agents.sentence_analyzer.load_yaml") as mock_load_yaml:
        analyzer = SentenceAnalyzer(config_dict=config_missing_nested)

        # Should not call load_yaml due to missing nested key
        mock_load_yaml.assert_not_called()

        # Should have empty prompts
        assert analyzer.prompts == {}

        # Should log error
        assert "Failed to get prompt path from config" in caplog.text


def test_init_empty_prompt_path(caplog):
    """Test initialization when prompt path is empty string."""
    caplog.set_level(logging.ERROR)

    config_empty_path = {
        "classification": {
            "local": {"prompt_files": {"no_context": ""}}  # Empty path
        }
    }

    with patch("src.agents.sentence_analyzer.load_yaml") as mock_load_yaml:
        analyzer = SentenceAnalyzer(config_dict=config_empty_path)

        # Should not call load_yaml due to empty path
        mock_load_yaml.assert_not_called()

        # Should have empty prompts
        assert analyzer.prompts == {}

        # Should log error
        assert "Failed to get prompt path from config" in caplog.text


def test_init_file_not_found_error(caplog):
    """Test initialization when prompt file is not found."""
    caplog.set_level(logging.ERROR)

    config_dict = {
        "classification": {
            "local": {"prompt_files": {"no_context": "nonexistent/prompts.yaml"}}
        }
    }

    with patch("src.agents.sentence_analyzer.load_yaml", side_effect=FileNotFoundError()):
        analyzer = SentenceAnalyzer(config_dict=config_dict)

        # Should have empty prompts
        assert analyzer.prompts == {}

        # Should log error
        assert "Prompt file not found at path" in caplog.text
        assert "Using empty prompts" in caplog.text


def test_init_general_exception(caplog):
    """Test initialization when load_yaml raises a general exception."""
    caplog.set_level(logging.ERROR)

    config_dict = {
        "classification": {
            "local": {"prompt_files": {"no_context": "prompts.yaml"}}
        }
    }

    with patch("src.agents.sentence_analyzer.load_yaml", side_effect=ValueError("YAML parsing error")):
        analyzer = SentenceAnalyzer(config_dict=config_dict)

        # Should have empty prompts
        assert analyzer.prompts == {}

        # Should log error with exception info
        assert "Failed to load prompts yaml from prompts.yaml" in caplog.text
        assert "YAML parsing error" in caplog.text


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
    analyzer, mock_agent = sentence_analyzer_with_dict  # Unpack fixture
    config_dict = analyzer.config  # Get config used by analyzer
    domain_keywords_str = ", ".join(config_dict.get("domain_keywords", []))

    # Explicitly set call_model as AsyncMock BEFORE assigning side_effect
    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"},  # function_prompt
        {"structure_type": "simple"},  # structure_prompt
        {"purpose": "inform"},  # purpose_prompt
        {"topic_level_1": "testing"},  # topic_lvl1_prompt
        {"topic_level_3": "unit testing"},  # topic_lvl3_prompt
        {"overall_keywords": ["test", "sentence"]},  # overall_keywords_prompt
        {"domain_keywords": ["dict_keyword"]},  # domain_prompt
    ]

    # Patch Pydantic models used for validation to bypass actual validation
    with patch(
        "src.agents.sentence_analyzer.SentenceFunctionResponse", MockValidationModel
    ), patch(
        "src.agents.sentence_analyzer.SentenceStructureResponse", MockValidationModel
    ), patch(
        "src.agents.sentence_analyzer.SentencePurposeResponse", MockValidationModel
    ), patch(
        "src.agents.sentence_analyzer.TopicLevel1Response", MockValidationModel
    ), patch(
        "src.agents.sentence_analyzer.TopicLevel3Response", MockValidationModel
    ), patch(
        "src.agents.sentence_analyzer.OverallKeywordsResponse", MockValidationModel
    ), patch(
        "src.agents.sentence_analyzer.DomainKeywordsResponse", MockValidationModel
    ):

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

    # Assert agent was called 7 times
    assert mock_agent.call_model.await_count == 7

    # --- New Assertions: Check call arguments more specifically --- #
    # Build expected prompts using the same logic as the SUT
    prompts = analyzer.prompts
    expected_calls = [
        call(prompts["sentence_function_type"]["prompt"].format(sentence=sentence)),
        call(prompts["sentence_structure_type"]["prompt"].format(sentence=sentence)),
        call(
            prompts["sentence_purpose"]["prompt"].format(
                sentence=sentence, context=mock_contexts["observer_context"]
            )
        ),
        call(
            prompts["topic_level_1"]["prompt"].format(
                sentence=sentence, context=mock_contexts["immediate_context"]
            )
        ),
        call(
            prompts["topic_level_3"]["prompt"].format(
                sentence=sentence, context=mock_contexts["broader_context"]
            )
        ),
        call(
            prompts["topic_overall_keywords"]["prompt"].format(
                context=mock_contexts["observer_context"]
            )
        ),
        call(
            prompts["domain_specific_keywords"]["prompt"].format(
                sentence=sentence, domain_keywords=domain_keywords_str
            )
        ),
    ]
    # Check that the mock was awaited with these specific calls (order might vary due to gather)
    mock_agent.call_model.assert_has_awaits(expected_calls, any_order=True)


@pytest.mark.asyncio
async def test_classify_sentence_api_error(
    sentence_analyzer_with_dict, mock_contexts, caplog
):
    """Test classify_sentence correctly handles and propagates API errors from agent.call_model."""
    caplog.set_level(logging.ERROR)  # Ensure ERROR logs are captured
    sentence = "Sentence triggering API error."
    analyzer, mock_agent = sentence_analyzer_with_dict  # Unpack fixture

    # Configure the mock agent's call_model to raise the desired exception
    # Raising it directly is enough, as asyncio.gather will propagate the first exception.
    mock_agent.call_model = AsyncMock(side_effect=ValueError("Simulated API Error"))

    # Expect the classify_sentence method to re-raise the same exception
    with pytest.raises(ValueError, match="Simulated API Error"):
        await analyzer.classify_sentence(sentence, mock_contexts)

    # Assert that the error was logged
    assert len(caplog.records) == 1  # Should be exactly one error log
    assert caplog.records[0].levelname == "ERROR"
    assert (
        f"Error during concurrent API calls for sentence '{sentence[:50]}...'"
        in caplog.text
    )
    assert "Simulated API Error" in caplog.text

    # Verify that call_model was attempted (likely just once before gather raises)
    mock_agent.call_model.assert_awaited()  # Check it was awaited at least once


@pytest.mark.asyncio
async def test_classify_sentence_validation_error(
    sentence_analyzer_with_dict, mock_contexts, caplog
):
    """Test classify_sentence when Pydantic validation fails for one dimension."""
    # Set caplog level to capture WARNING messages
    caplog.set_level(logging.WARNING)
    sentence = "Sentence causing validation issue."
    analyzer, mock_agent = sentence_analyzer_with_dict  # Unpack fixture

    # Explicitly set call_model as AsyncMock BEFORE assigning side_effect
    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"},
        {"structure_type": "simple"},
        {"purpose": "inform"},
        {"invalid_key": "wrong_data"},  # Invalid response for TopicLevel1
        {"topic_level_3": "unit testing"},
        {"overall_keywords": ["test", "sentence"]},
        {"domain_keywords": ["dict_keyword"]},
    ]

    # Need to patch metrics_tracker as it's now called
    with patch("src.agents.sentence_analyzer.metrics_tracker") as mock_metrics_tracker:
        # Don't patch TopicLevel1Response - let it fail with the invalid data
        result = await analyzer.classify_sentence(sentence, mock_contexts)

    # Assertions
    assert result["sentence"] == sentence
    assert result["function_type"] == "declarative"
    assert result["structure_type"] == "simple"
    assert result["purpose"] == "inform"
    # Check that the failed dimension has a default value
    assert result["topic_level_1"] == ""  # Default defined in SentenceAnalyzer
    assert result["topic_level_3"] == "unit testing"
    assert result["overall_keywords"] == ["test", "sentence"]
    assert result["domain_keywords"] == ["dict_keyword"]

    # Check logs for validation warning
    assert "Validation failed for Topic Level 1 response" in caplog.text

    # Assert metrics tracker was called for the specific error
    mock_metrics_tracker.increment_errors.assert_called_once_with(
        "validation_error_topic_level_1"
    )

    assert mock_agent.call_model.await_count == 7


# === Test validation error handling for each dimension ===


@pytest.mark.asyncio
async def test_classify_sentence_function_validation_error(
    sentence_analyzer_with_dict, mock_contexts, caplog
):
    """Test validation error handling for function type dimension."""
    caplog.set_level(logging.WARNING)
    sentence = "Test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict

    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"invalid": "data"},  # Invalid for function
        {"structure_type": "simple"},
        {"purpose": "inform"},
        {"topic_level_1": "testing"},
        {"topic_level_3": "unit testing"},
        {"overall_keywords": ["test"]},
        {"domain_keywords": ["keyword"]},
    ]

    with patch("src.agents.sentence_analyzer.metrics_tracker") as mock_metrics_tracker:
        result = await analyzer.classify_sentence(sentence, mock_contexts)

    assert result["function_type"] == ""  # Default value
    assert "Validation failed for Function Type response" in caplog.text
    mock_metrics_tracker.increment_errors.assert_called_with("validation_error_function_type")


@pytest.mark.asyncio
async def test_classify_sentence_structure_validation_error(
    sentence_analyzer_with_dict, mock_contexts, caplog
):
    """Test validation error handling for structure type dimension."""
    caplog.set_level(logging.WARNING)
    sentence = "Test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict

    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"},
        {"invalid": "data"},  # Invalid for structure
        {"purpose": "inform"},
        {"topic_level_1": "testing"},
        {"topic_level_3": "unit testing"},
        {"overall_keywords": ["test"]},
        {"domain_keywords": ["keyword"]},
    ]

    with patch("src.agents.sentence_analyzer.metrics_tracker") as mock_metrics_tracker:
        result = await analyzer.classify_sentence(sentence, mock_contexts)

    assert result["structure_type"] == ""  # Default value
    assert "Validation failed for Structure Type response" in caplog.text
    mock_metrics_tracker.increment_errors.assert_called_with("validation_error_structure_type")


@pytest.mark.asyncio
async def test_classify_sentence_purpose_validation_error(
    sentence_analyzer_with_dict, mock_contexts, caplog
):
    """Test validation error handling for purpose dimension."""
    caplog.set_level(logging.WARNING)
    sentence = "Test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict

    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"},
        {"structure_type": "simple"},
        {"invalid": "data"},  # Invalid for purpose
        {"topic_level_1": "testing"},
        {"topic_level_3": "unit testing"},
        {"overall_keywords": ["test"]},
        {"domain_keywords": ["keyword"]},
    ]

    with patch("src.agents.sentence_analyzer.metrics_tracker") as mock_metrics_tracker:
        result = await analyzer.classify_sentence(sentence, mock_contexts)

    assert result["purpose"] == ""  # Default value
    assert "Validation failed for Purpose response" in caplog.text
    mock_metrics_tracker.increment_errors.assert_called_with("validation_error_purpose")


@pytest.mark.asyncio
async def test_classify_sentence_topic_level_3_validation_error(
    sentence_analyzer_with_dict, mock_contexts, caplog
):
    """Test validation error handling for topic level 3 dimension."""
    caplog.set_level(logging.WARNING)
    sentence = "Test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict

    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"},
        {"structure_type": "simple"},
        {"purpose": "inform"},
        {"topic_level_1": "testing"},
        {"invalid": "data"},  # Invalid for topic level 3
        {"overall_keywords": ["test"]},
        {"domain_keywords": ["keyword"]},
    ]

    with patch("src.agents.sentence_analyzer.metrics_tracker") as mock_metrics_tracker:
        result = await analyzer.classify_sentence(sentence, mock_contexts)

    assert result["topic_level_3"] == ""  # Default value
    assert "Validation failed for Topic Level 3 response" in caplog.text
    mock_metrics_tracker.increment_errors.assert_called_with("validation_error_topic_level_3")


@pytest.mark.asyncio
async def test_classify_sentence_multiple_validation_errors(
    sentence_analyzer_with_dict, mock_contexts, caplog
):
    """Test handling of multiple validation errors in a single call."""
    caplog.set_level(logging.WARNING)
    sentence = "Test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict

    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"invalid": "data"},  # Invalid for function
        {"invalid": "data"},  # Invalid for structure
        {"purpose": "inform"},
        {"topic_level_1": "testing"},
        {"topic_level_3": "unit testing"},
        {"overall_keywords": ["test"]},  # Valid for overall keywords
        {"domain_keywords": ["keyword"]},
    ]

    # Patch only the successful validations, let the others fail
    with patch("src.agents.sentence_analyzer.SentencePurposeResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.TopicLevel1Response", MockValidationModel), \
         patch("src.agents.sentence_analyzer.TopicLevel3Response", MockValidationModel), \
         patch("src.agents.sentence_analyzer.OverallKeywordsResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.DomainKeywordsResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.metrics_tracker") as mock_metrics_tracker:

        result = await analyzer.classify_sentence(sentence, mock_contexts)

    # Check default values for failed validations
    assert result["function_type"] == ""
    assert result["structure_type"] == ""

    # Check successful validations
    assert result["purpose"] == "inform"
    assert result["topic_level_1"] == "testing"
    assert result["topic_level_3"] == "unit testing"
    assert result["overall_keywords"] == ["test"]
    assert result["domain_keywords"] == ["keyword"]

    # Check that function and structure errors were logged
    assert "Validation failed for Function Type response" in caplog.text
    assert "Validation failed for Structure Type response" in caplog.text

    # Check that metrics were tracked for the errors
    expected_calls = [
        call("validation_error_function_type"),
        call("validation_error_structure_type"),
    ]
    mock_metrics_tracker.increment_errors.assert_has_calls(expected_calls, any_order=True)


@pytest.mark.asyncio
async def test_classify_sentence_empty_domain_keywords(sentence_analyzer_with_dict, mock_contexts):
    """Test classify_sentence with empty domain_keywords in config."""
    sentence = "Test sentence."
    analyzer, mock_agent = sentence_analyzer_with_dict

    # Override config to have empty domain_keywords
    analyzer.config = {
        "classification": analyzer.config["classification"],
        "domain_keywords": []  # Empty list
    }

    mock_agent.call_model = AsyncMock()
    mock_agent.call_model.side_effect = [
        {"function_type": "declarative"},
        {"structure_type": "simple"},
        {"purpose": "inform"},
        {"topic_level_1": "testing"},
        {"topic_level_3": "unit testing"},
        {"overall_keywords": ["test"]},
        {"domain_keywords": []},
    ]

    with patch("src.agents.sentence_analyzer.SentenceFunctionResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.SentenceStructureResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.SentencePurposeResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.TopicLevel1Response", MockValidationModel), \
         patch("src.agents.sentence_analyzer.TopicLevel3Response", MockValidationModel), \
         patch("src.agents.sentence_analyzer.OverallKeywordsResponse", MockValidationModel), \
         patch("src.agents.sentence_analyzer.DomainKeywordsResponse", MockValidationModel):

        result = await analyzer.classify_sentence(sentence, mock_contexts)

    # Check that domain_keywords prompt was called with empty string
    expected_domain_prompt = analyzer.prompts["domain_specific_keywords"]["prompt"].format(
        sentence=sentence, domain_keywords=""
    )
    mock_agent.call_model.assert_any_await(expected_domain_prompt)

    assert result["domain_keywords"] == []
