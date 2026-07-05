"""Provider chain with per-call failover on availability errors.

The baseline provider-strategy pattern: chat calls try the configured chain
in order; only quota/availability failures advance the chain (a malformed
request should fail loudly). Events record the provider/model that actually
served each call via CallResult.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.utils.logger import get_logger

from .base_agent import BaseLLMAgent

logger = get_logger()

_AVAILABILITY_ERROR_NAMES = (
    "RateLimitError",
    "APIStatusError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
    "OverloadedError",
)


def _is_availability_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if any(marker in name for marker in _AVAILABILITY_ERROR_NAMES):
        return True
    # ConnectionError/TimeoutError cover network failures; RuntimeError is the
    # ClaudeCodeAgent's CLI-failure signal.
    return isinstance(exc, (ConnectionError, TimeoutError, RuntimeError))


class CallResult(BaseModel):
    """A chat-call result with provenance."""

    data: Dict[str, Any]
    provider: str
    model: str


class FailoverAgent(BaseLLMAgent):
    """Tries providers in order; advances only on availability errors."""

    def __init__(self, providers: List[BaseLLMAgent]):
        super().__init__()
        if not providers:
            raise ValueError("FailoverAgent requires at least one provider")
        self.providers = providers

    async def call(
        self, function_prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> CallResult:
        """Call the chain; return data plus which provider/model served it."""
        last_error: Optional[Exception] = None
        for provider in self.providers:
            try:
                data = await provider.call_model(function_prompt, schema=schema)
                return CallResult(
                    data=data,
                    provider=provider.get_provider_name(),
                    model=provider.get_model_name(),
                )
            except Exception as exc:
                if not _is_availability_error(exc):
                    raise
                logger.warning(
                    f"Provider {provider.get_provider_name()} unavailable "
                    f"({type(exc).__name__}); failing over"
                )
                last_error = exc
        assert last_error is not None  # loop ran at least once (ctor guard)
        raise last_error

    async def call_model(
        self, function_prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return (await self.call(function_prompt, schema=schema)).data

    def get_provider_name(self) -> str:
        return "failover:" + ",".join(p.get_provider_name() for p in self.providers)

    def get_model_name(self) -> str:
        return self.providers[0].get_model_name()


def get_failover_agent(config_dict: Optional[Dict[str, Any]] = None) -> FailoverAgent:
    """Build the chain from config['llm']['chain'] (falls back to single provider)."""
    from src.agents.agent_factory import AgentFactory
    from src.config import config as global_config

    cfg = config_dict if config_dict is not None else global_config
    chain = cfg.get("llm", {}).get("chain")
    if not chain:
        chain = [cfg.get("llm", {}).get("provider", "anthropic")]
    return FailoverAgent([AgentFactory.create_agent(name) for name in chain])
