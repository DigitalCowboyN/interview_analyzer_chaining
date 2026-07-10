from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.failover_agent import CallResult, FailoverAgent


class RateLimitError(Exception):
    """Name-matched availability error (mirrors provider SDK class names)."""


def make_provider(name, model, result=None, error=None):
    p = MagicMock()
    p.get_provider_name.return_value = name
    p.get_model_name.return_value = model
    if error is not None:
        p.call_model = AsyncMock(side_effect=error)
    else:
        p.call_model = AsyncMock(return_value=result)
    return p


@pytest.mark.asyncio
async def test_first_healthy_provider_serves():
    a = make_provider("anthropic", "haiku", result={"x": 1})
    b = make_provider("openai", "gpt", result={"x": 2})
    agent = FailoverAgent([a, b])
    result = await agent.call("prompt")
    assert result == CallResult(data={"x": 1}, provider="anthropic", model="haiku")
    b.call_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_fails_over_on_rate_limit():
    a = make_provider("anthropic", "haiku", error=RateLimitError("429"))
    b = make_provider("openai", "gpt", result={"x": 2})
    agent = FailoverAgent([a, b])
    result = await agent.call("prompt")
    assert result.provider == "openai"
    assert result.data == {"x": 2}


@pytest.mark.asyncio
async def test_schema_forwarded_to_provider():
    a = make_provider("anthropic", "haiku", result={"x": 1})
    agent = FailoverAgent([a])
    schema = {"type": "object"}
    await agent.call("prompt", schema=schema)
    assert a.call_model.call_args.kwargs["schema"] == schema


@pytest.mark.asyncio
async def test_non_availability_error_propagates():
    a = make_provider("anthropic", "haiku", error=ValueError("bad prompt"))
    b = make_provider("openai", "gpt", result={"x": 2})
    agent = FailoverAgent([a, b])
    with pytest.raises(ValueError):
        await agent.call("prompt")
    b.call_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_cli_runtime_error_fails_over():
    a = make_provider("claude_code", "claude-code/haiku", error=RuntimeError("claude CLI failed"))
    b = make_provider("openai", "gpt", result={"x": 2})
    agent = FailoverAgent([a, b])
    result = await agent.call("prompt")
    assert result.provider == "openai"


@pytest.mark.asyncio
async def test_exhausted_chain_raises_last_error():
    a = make_provider("anthropic", "haiku", error=RateLimitError("429"))
    b = make_provider("openai", "gpt", error=RuntimeError("claude CLI failed"))
    agent = FailoverAgent([a, b])
    with pytest.raises(RuntimeError):
        await agent.call("prompt")


@pytest.mark.asyncio
async def test_call_model_returns_plain_dict():
    a = make_provider("anthropic", "haiku", result={"x": 1})
    agent = FailoverAgent([a])
    assert await agent.call_model("prompt") == {"x": 1}


def test_empty_chain_rejected():
    with pytest.raises(ValueError, match="at least one provider"):
        FailoverAgent([])


def test_factory_builds_chain_from_config():
    from unittest.mock import patch

    from src.agents.failover_agent import get_failover_agent

    fake = MagicMock()
    with patch("src.agents.agent_factory.AgentFactory.create_agent", return_value=fake) as create:
        agent = get_failover_agent({"llm": {"chain": ["anthropic", "claude_code"]}})
    assert [c.args[0] for c in create.call_args_list] == ["anthropic", "claude_code"]
    assert len(agent.providers) == 2


def test_factory_falls_back_to_single_provider():
    from unittest.mock import patch

    from src.agents.failover_agent import get_failover_agent

    fake = MagicMock()
    with patch("src.agents.agent_factory.AgentFactory.create_agent", return_value=fake) as create:
        agent = get_failover_agent({"llm": {"provider": "anthropic"}})
    assert [c.args[0] for c in create.call_args_list] == ["anthropic"]
    assert len(agent.providers) == 1


def test_factory_skips_unconstructible_providers():
    from unittest.mock import patch

    from src.agents.failover_agent import get_failover_agent

    good = MagicMock()

    def create(name):
        if name == "anthropic":
            raise ValueError("API key is not set.")
        return good

    with patch("src.agents.agent_factory.AgentFactory.create_agent", side_effect=create):
        agent = get_failover_agent({"llm": {"chain": ["anthropic", "openai"]}})
    assert agent.providers == [good]


def test_factory_raises_when_no_provider_constructible():
    from unittest.mock import patch

    import pytest as _pytest

    from src.agents.failover_agent import get_failover_agent

    with patch(
        "src.agents.agent_factory.AgentFactory.create_agent",
        side_effect=ValueError("no key"),
    ):
        with _pytest.raises(ValueError, match="No usable LLM provider"):
            get_failover_agent({"llm": {"chain": ["anthropic", "openai"]}})
