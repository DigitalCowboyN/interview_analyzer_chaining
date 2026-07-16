from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.ask.engine import AskResult, SynthesisUnavailable
from src.main import app

PID = "33333333-3333-3333-3333-333333333333"


def make_result(**overrides):
    defaults = dict(
        project_id=PID,
        question="What did they say about pricing?",
        answer="They said pricing was too high.",
        citations=[{"fragment_id": "f1"}],
        retrieval={"channels": {"vector": 1}, "flags": {}, "fragments": ["f1"]},
        provider="openai",
        model="gpt-4",
    )
    defaults.update(overrides)
    return AskResult(**defaults)


def patch_engine(ask_mock):
    engine_instance = MagicMock()
    engine_instance.ask = ask_mock
    engine_class = MagicMock(return_value=engine_instance)
    return patch("src.api.routers.ask.AskEngine", new=engine_class), engine_class


@pytest.fixture
def client():
    return TestClient(app)


def test_ask_returns_result_json(client):
    result = make_result()
    ask_mock = AsyncMock(return_value=result)
    patcher, engine_class = patch_engine(ask_mock)
    with patcher:
        resp = client.post(f"/ask/{PID}", json={"question": "What did they say about pricing?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == result.answer
    assert body["citations"] == result.citations
    assert body["retrieval"] == result.retrieval
    ask_mock.assert_awaited_once_with(PID, "What did they say about pricing?", top_k=12)


def test_ask_passes_top_k(client):
    result = make_result()
    ask_mock = AsyncMock(return_value=result)
    patcher, _ = patch_engine(ask_mock)
    with patcher:
        resp = client.post(f"/ask/{PID}", json={"question": "q", "top_k": 3})
    assert resp.status_code == 200
    ask_mock.assert_awaited_once_with(PID, "q", top_k=3)


def test_unknown_project_404(client):
    ask_mock = AsyncMock(side_effect=ValueError(f"Project {PID} not found"))
    patcher, _ = patch_engine(ask_mock)
    with patcher:
        resp = client.post(f"/ask/{PID}", json={"question": "q"})
    assert resp.status_code == 404


def test_synthesis_unavailable_502_with_partial_body(client):
    partial = make_result(answer=None, citations=[], provider="", model="")
    ask_mock = AsyncMock(side_effect=SynthesisUnavailable("synthesis failed: RuntimeError", partial))
    patcher, _ = patch_engine(ask_mock)
    with patcher:
        resp = client.post(f"/ask/{PID}", json={"question": "q"})
    assert resp.status_code == 502
    assert resp.json() == partial.model_dump()


def test_empty_question_422(client):
    ask_mock = AsyncMock()
    patcher, _ = patch_engine(ask_mock)
    with patcher:
        resp = client.post(f"/ask/{PID}", json={"question": ""})
    assert resp.status_code == 422
    ask_mock.assert_not_awaited()
