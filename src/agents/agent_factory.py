"""
agent_factory.py

Factory for creating LLM agent instances based on configuration.
Manages singleton instances per provider and provides clean abstraction for provider selection.

Design Pattern: Factory + Singleton
- Factory: Select provider based on configuration
- Singleton: One instance per provider for connection reuse

Extensibility:
- Easy to add new providers via register_provider()
- Configuration-driven selection (no code changes needed)
- Supports testing with reset() method
"""

from typing import Dict, Optional

from src.config import config
from src.utils.logger import get_logger
from .base_agent import BaseLLMAgent

logger = get_logger()


class AgentFactory:
    """
    Factory for creating and managing LLM agent instances.

    Provides:
    - Configuration-driven provider selection
    - Singleton pattern for each provider (connection reuse)
    - Easy extensibility for new providers
    - Validation and error handling
    - Testing support (reset method)

    Usage:
        # Get agent from config
        agent = AgentFactory.create_agent()

        # Get specific provider
        openai_agent = AgentFactory.create_agent("openai")
        anthropic_agent = AgentFactory.create_agent("anthropic")

        # Register custom provider
        AgentFactory.register_provider("gemini", GeminiAgent)
    """

    # Registry of available providers (lazy imports to avoid circular dependencies)
    _providers: Dict[str, type] = {}

    # Singleton instances per provider
    _instances: Dict[str, BaseLLMAgent] = {}

    @classmethod
    def _initialize_providers(cls):
        """Initialize provider registry with lazy imports."""
        if not cls._providers:
            # Import here to avoid circular dependencies
            from .openai_agent import OpenAIAgent
            from .anthropic_agent import AnthropicAgent

            cls._providers = {
                "openai": OpenAIAgent,
                "anthropic": AnthropicAgent,
            }

    @classmethod
    def create_agent(cls, provider: Optional[str] = None) -> BaseLLMAgent:
        """
        Create or retrieve agent instance for specified provider.

        Implements singleton pattern: returns existing instance if already created.
        Configuration-driven: reads from config['llm']['provider'] if not specified.

        Args:
            provider: Provider name ('openai', 'anthropic').
                     If None, reads from config['llm']['provider'].
                     Defaults to 'openai' if config missing.

        Returns:
            BaseLLMAgent: Configured agent instance (singleton per provider)

        Raises:
            ValueError: If provider is unknown or config is invalid

        Examples:
            # Use configured provider
            agent = AgentFactory.create_agent()

            # Force specific provider
            agent = AgentFactory.create_agent("anthropic")
        """
        # Ensure providers are registered
        cls._initialize_providers()

        # Determine provider from config if not specified
        if provider is None:
            provider = config.get("llm", {}).get("provider", "openai")

        provider = provider.lower()

        # Validate provider
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available providers: {available}"
            )

        # Return existing instance if already created (singleton pattern)
        if provider in cls._instances:
            logger.debug(f"Returning existing {provider} agent instance")
            return cls._instances[provider]

        # Create new instance
        logger.info(f"Creating new {provider} agent instance")
        agent_class = cls._providers[provider]

        try:
            agent = agent_class()
            cls._instances[provider] = agent
            logger.info(
                f"Successfully created {provider} agent "
                f"(model: {agent.get_model_name()})"
            )
            return agent

        except Exception as e:
            logger.error(f"Failed to create {provider} agent: {e}", exc_info=True)
            raise

    @classmethod
    def register_provider(cls, name: str, agent_class: type):
        """
        Register a new provider class (for extensibility).

        Allows adding new providers without modifying factory code:
        - Gemini
        - Cohere
        - Custom providers

        Args:
            name: Provider name (e.g., 'gemini', 'cohere')
            agent_class: Agent class implementing BaseLLMAgent

        Raises:
            TypeError: If agent_class doesn't inherit from BaseLLMAgent

        Example:
            class GeminiAgent(BaseLLMAgent):
                # Implementation

            AgentFactory.register_provider("gemini", GeminiAgent)
            agent = AgentFactory.create_agent("gemini")
        """
        cls._initialize_providers()

        if not issubclass(agent_class, BaseLLMAgent):
            raise TypeError(
                f"Agent class must inherit from BaseLLMAgent, got {agent_class}"
            )

        cls._providers[name] = agent_class
        logger.info(f"Registered new LLM provider: {name}")

    @classmethod
    def get_available_providers(cls) -> list:
        """
        Return list of available provider names.

        Useful for:
        - Validation
        - UI dropdowns
        - Documentation
        - Testing

        Returns:
            list: Provider names (e.g., ['openai', 'anthropic'])
        """
        cls._initialize_providers()
        return list(cls._providers.keys())

    @classmethod
    def reset(cls):
        """
        Clear all singleton instances (useful for testing).

        Allows tests to create fresh instances with mocked configurations.
        Should be called in test fixtures to ensure test isolation.

        Example:
            @pytest.fixture(autouse=True)
            def reset_factory():
                AgentFactory.reset()
                yield
                AgentFactory.reset()
        """
        cls._instances = {}
        logger.debug("Reset all agent instances")


# Create global singleton for backward compatibility
# This maintains the existing pattern: `from src.agents.agent import agent`
# Now imports as: `from src.agents.agent_factory import agent`
agent = AgentFactory.create_agent()
