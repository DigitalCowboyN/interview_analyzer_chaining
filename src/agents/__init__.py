"""
Agent module exports.

Provides unified access to LLM agents through factory pattern.
Supports multiple providers (OpenAI, Anthropic, Claude Code) with
configuration-driven selection and a failover chain.
"""

from .base_agent import BaseLLMAgent
from .openai_agent import OpenAIAgent
from .anthropic_agent import AnthropicAgent
from .claude_code_agent import ClaudeCodeAgent
from .agent_factory import AgentFactory, agent
from .failover_agent import FailoverAgent, get_failover_agent

__all__ = [
    "BaseLLMAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "ClaudeCodeAgent",
    "AgentFactory",
    "agent",  # Global singleton for backward compatibility
    "FailoverAgent",
    "get_failover_agent",
]
