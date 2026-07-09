import json
from unittest.mock import AsyncMock, MagicMock

import pytest

SCHEMA = {
    "type": "object",
    "properties": {"purpose": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["purpose", "confidence"],
    "additionalProperties": False,
}


def make_openai_agent():
    from src.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent.__new__(OpenAIAgent)  # skip __init__ (no API key needed)
    agent.model = "gpt-4o-mini-2024-07-18"
    agent.max_tokens = 256
    agent.temperature = 0.2
    agent.retry_attempts = 1
    agent.backoff_factor = 1
    agent.client = MagicMock()
    return agent


def openai_response(payload: dict):
    content = MagicMock()
    content.text = json.dumps(payload)
    item = MagicMock()
    item.content = [content]
    response = MagicMock()
    response.output = [item]
    response.usage = MagicMock(total_tokens=10, input_tokens=5, output_tokens=5)
    return response


@pytest.mark.asyncio
async def test_openai_passes_json_schema_text_format():
    agent = make_openai_agent()
    agent.client.responses.create = AsyncMock(
        return_value=openai_response({"purpose": "Query", "confidence": 0.9})
    )

    result = await agent.call_model("classify this", schema=SCHEMA)

    assert result == {"purpose": "Query", "confidence": 0.9}
    kwargs = agent.client.responses.create.call_args.kwargs
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["schema"] == SCHEMA
    assert kwargs["text"]["format"]["strict"] is True


@pytest.mark.asyncio
async def test_openai_without_schema_keeps_json_object_format():
    agent = make_openai_agent()
    agent.client.responses.create = AsyncMock(return_value=openai_response({"ok": True}))

    await agent.call_model("classify this")

    kwargs = agent.client.responses.create.call_args.kwargs
    assert kwargs["text"] == {"format": {"type": "json_object"}}


def make_anthropic_agent():
    from src.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent.__new__(AnthropicAgent)
    agent.model = "claude-3-haiku-20240307"
    agent.max_tokens = 256
    agent.temperature = 0.2
    agent.retry_attempts = 1
    agent.backoff_factor = 1
    agent.system_prompt = "Respond only in JSON."
    agent.client = MagicMock()
    return agent


@pytest.mark.asyncio
async def test_anthropic_forces_tool_use_and_returns_input():
    agent = make_anthropic_agent()
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"purpose": "Query", "confidence": 0.9}
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=5, output_tokens=5)
    agent.client.messages.create = AsyncMock(return_value=response)

    result = await agent.call_model("classify this", schema=SCHEMA)

    assert result == {"purpose": "Query", "confidence": 0.9}
    kwargs = agent.client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "extraction"}
    assert kwargs["tools"][0]["input_schema"] == SCHEMA


@pytest.mark.asyncio
async def test_anthropic_without_schema_sends_no_tools():
    agent = make_anthropic_agent()
    text_block = MagicMock()
    text_block.text = json.dumps({"ok": True})
    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(input_tokens=5, output_tokens=5)
    agent.client.messages.create = AsyncMock(return_value=response)

    result = await agent.call_model("classify this")

    assert result == {"ok": True}
    kwargs = agent.client.messages.create.call_args.kwargs
    assert "tools" not in kwargs and "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_anthropic_missing_tool_use_block_returns_empty():
    agent = make_anthropic_agent()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "not a tool call"
    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(input_tokens=5, output_tokens=5)
    agent.client.messages.create = AsyncMock(return_value=response)

    result = await agent.call_model("classify this", schema=SCHEMA)

    assert result == {}
