"""
tests/agents/test_agent_factory.py

Contains unit tests for the `AgentFactory` class (`src.agents.agent_factory.py`).

Tests verify:
    - Correct provider instantiation (OpenAI, Anthropic)
    - Singleton pattern (same instance returned per provider)
    - Configuration-driven provider selection
    - Error handling for unknown providers
    - Provider registration for extensibility
    - Reset functionality for testing

Key Testing Techniques:
    - Fixtures for factory reset between tests
    - Mocking configuration
    - Testing provider registry
    - Validation of singleton behavior
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.agent_factory import AgentFactory
from src.agents.base_agent import BaseLLMAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.anthropic_agent import AnthropicAgent


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset factory state before and after each test."""
    AgentFactory.reset()
    AgentFactory._providers = {}  # Also reset providers for clean state
    yield
    AgentFactory.reset()
    AgentFactory._providers = {}


@pytest.fixture
def mock_openai_config():
    """Mock config for OpenAI."""
    return {
        "llm": {"provider": "openai"},
        "openai": {
            "api_key": "sk-test-openai-key",
            "model_name": "gpt-4o-mini",
            "max_tokens": 256,
            "temperature": 0.2,
        },
        "openai_api": {
            "retry": {"max_attempts": 5, "backoff_factor": 2}
        }
    }


@pytest.fixture
def mock_anthropic_config():
    """Mock config for Anthropic."""
    return {
        "llm": {"provider": "anthropic"},
        "anthropic": {
            "api_key": "sk-ant-test-key",
            "model_name": "claude-3-haiku-20240307",
            "max_tokens": 256,
            "temperature": 0.2,
        },
        "anthropic_api": {
            "retry": {"max_attempts": 5, "backoff_factor": 2}
        }
    }


# Test provider creation

def test_create_openai_agent(mock_openai_config):
    """Test creating OpenAI agent."""
    with patch("src.agents.openai_agent.config", mock_openai_config), \
         patch("src.agents.agent_factory.config", mock_openai_config):

        agent = AgentFactory.create_agent("openai")

        assert isinstance(agent, OpenAIAgent)
        assert isinstance(agent, BaseLLMAgent)
        assert agent.get_provider_name() == "openai"


def test_create_anthropic_agent(mock_anthropic_config):
    """Test creating Anthropic agent."""
    with patch("src.agents.anthropic_agent.config", mock_anthropic_config), \
         patch("src.agents.agent_factory.config", mock_anthropic_config):

        agent = AgentFactory.create_agent("anthropic")

        assert isinstance(agent, AnthropicAgent)
        assert isinstance(agent, BaseLLMAgent)
        assert agent.get_provider_name() == "anthropic"


# Test singleton pattern

def test_singleton_pattern_same_provider(mock_openai_config):
    """Test factory returns same instance for same provider."""
    with patch("src.agents.openai_agent.config", mock_openai_config), \
         patch("src.agents.agent_factory.config", mock_openai_config):

        agent1 = AgentFactory.create_agent("openai")
        agent2 = AgentFactory.create_agent("openai")

        assert agent1 is agent2  # Same instance


def test_different_providers_different_instances(mock_openai_config, mock_anthropic_config):
    """Test different providers get different instances."""
    with patch("src.agents.openai_agent.config", mock_openai_config), \
         patch("src.agents.anthropic_agent.config", mock_anthropic_config), \
         patch("src.agents.agent_factory.config", mock_openai_config):

        openai_agent = AgentFactory.create_agent("openai")

    with patch("src.agents.agent_factory.config", mock_anthropic_config):
        anthropic_agent = AgentFactory.create_agent("anthropic")

        assert openai_agent is not anthropic_agent
        assert type(openai_agent) != type(anthropic_agent)


# Test configuration-driven selection

def test_default_provider_from_config_openai(mock_openai_config):
    """Test reading default provider from config (OpenAI)."""
    with patch("src.agents.openai_agent.config", mock_openai_config), \
         patch("src.agents.agent_factory.config", mock_openai_config):

        agent = AgentFactory.create_agent()  # No provider specified

        assert isinstance(agent, OpenAIAgent)


def test_default_provider_from_config_anthropic(mock_anthropic_config):
    """Test reading default provider from config (Anthropic)."""
    with patch("src.agents.anthropic_agent.config", mock_anthropic_config), \
         patch("src.agents.agent_factory.config", mock_anthropic_config):

        agent = AgentFactory.create_agent()  # No provider specified

        assert isinstance(agent, AnthropicAgent)


def test_default_to_openai_if_no_config(mock_openai_config):
    """Test defaults to OpenAI if llm.provider not in config."""
    config_without_llm = {
        "openai": mock_openai_config["openai"],
        "openai_api": mock_openai_config["openai_api"]
    }

    with patch("src.agents.openai_agent.config", mock_openai_config), \
         patch("src.agents.agent_factory.config", config_without_llm):

        agent = AgentFactory.create_agent()

        assert isinstance(agent, OpenAIAgent)


# Test error handling

def test_unknown_provider_raises_error():
    """Test unknown provider raises ValueError."""
    mock_config = {"llm": {"provider": "openai"}}

    with patch("src.agents.agent_factory.config", mock_config):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            AgentFactory.create_agent("unknown_provider")


def test_case_insensitive_provider_name(mock_openai_config):
    """Test provider names are case-insensitive."""
    with patch("src.agents.openai_agent.config", mock_openai_config), \
         patch("src.agents.agent_factory.config", mock_openai_config):

        agent1 = AgentFactory.create_agent("openai")
        agent2 = AgentFactory.create_agent("OPENAI")
        agent3 = AgentFactory.create_agent("OpenAI")

        assert agent1 is agent2 is agent3


# Test provider registration

def test_register_custom_provider():
    """Test registering a custom provider."""

    class CustomAgent(BaseLLMAgent):
        async def call_model(self, prompt):
            return {"result": "custom"}

        def get_provider_name(self):
            return "custom"

        def get_model_name(self):
            return "custom-model"

    AgentFactory.register_provider("custom", CustomAgent)

    assert "custom" in AgentFactory.get_available_providers()

    mock_config = {
        "custom": {
            "api_key": "test",
            "model_name": "custom-model",
            "max_tokens": 100,
            "temperature": 0.5
        }
    }

    with patch("src.agents.agent_factory.config", mock_config):
        agent = AgentFactory.create_agent("custom")
        assert isinstance(agent, CustomAgent)
        assert agent.get_provider_name() == "custom"


def test_register_non_base_agent_raises_error():
    """Test registering non-BaseLLMAgent class raises TypeError."""

    class NotAnAgent:
        pass

    with pytest.raises(TypeError, match="must inherit from BaseLLMAgent"):
        AgentFactory.register_provider("invalid", NotAnAgent)


# Test available providers

def test_get_available_providers():
    """Test get_available_providers returns list of registered providers."""
    # After initialization, should have openai and anthropic
    providers = AgentFactory.get_available_providers()

    assert "openai" in providers
    assert "anthropic" in providers
    assert isinstance(providers, list)


# Test reset functionality

def test_reset_clears_instances(mock_openai_config):
    """Test reset clears all singleton instances."""
    with patch("src.agents.openai_agent.config", mock_openai_config), \
         patch("src.agents.agent_factory.config", mock_openai_config):

        agent1 = AgentFactory.create_agent("openai")

        AgentFactory.reset()

        agent2 = AgentFactory.create_agent("openai")

        # After reset, should get new instance
        assert agent1 is not agent2


# Test global singleton export

def test_global_agent_singleton():
    """Test that factory exports global agent singleton."""
    from src.agents.agent_factory import agent

    assert isinstance(agent, BaseLLMAgent)
    # Default should be OpenAI (from config)
    assert agent.get_provider_name() in ["openai", "anthropic"]


# Test error propagation

def test_initialization_error_propagated(mock_openai_config):
    """Test that agent initialization errors are propagated."""
    bad_config = {
        "llm": {"provider": "openai"},
        "openai": {"api_key": ""},  # Empty API key
        "openai_api": mock_openai_config["openai_api"]
    }

    with patch("src.agents.openai_agent.config", bad_config), \
         patch("src.agents.agent_factory.config", bad_config):

        with pytest.raises(ValueError, match="OpenAI API key is not set"):
            AgentFactory.create_agent("openai")
