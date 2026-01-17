"""
Agent module exports.

Provides unified access to LLM agents through factory pattern.
Supports multiple providers (OpenAI, Anthropic) with configuration-driven selection.
"""

from .base_agent import BaseLLMAgent
from .openai_agent import OpenAIAgent
from .anthropic_agent import AnthropicAgent
from .agent_factory import AgentFactory, agent
from .sentence_analyzer import SentenceAnalyzer
from .context_builder import ContextBuilder

__all__ = [
    "BaseLLMAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "AgentFactory",
    "agent",  # Global singleton for backward compatibility
    "SentenceAnalyzer",
    "ContextBuilder",
]
