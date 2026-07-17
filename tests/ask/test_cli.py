"""CLI: python -m src.ask <project_id> "<question>" [--top-k N] (M4.7 hardening)."""

import sys
from unittest.mock import AsyncMock, patch

import pytest

from src.ask.__main__ import main
from src.ask.engine import AskEngine, AskResult, SynthesisUnavailable


def make_result(**overrides):
    fields = dict(
        project_id="proj-1", question="why?", answer="Because.",
        citations=[], retrieval={"channels": {}, "flags": {}, "fragments": []},
    )
    fields.update(overrides)
    return AskResult(**fields)


def test_happy_path_prints_result_json_and_returns_normally(capsys):
    result = make_result()
    with patch.object(AskEngine, "ask", new=AsyncMock(return_value=result)):
        with patch.object(sys, "argv", ["ask", "proj-1", "why?"]):
            main()  # no SystemExit — returns normally

    out = capsys.readouterr().out
    assert '"project_id": "proj-1"' in out
    assert '"answer": "Because."' in out


def test_synthesis_unavailable_prints_partial_json_and_exits_1(capsys):
    partial = make_result(answer=None)
    with patch.object(
        AskEngine, "ask",
        new=AsyncMock(side_effect=SynthesisUnavailable("synthesis failed: RuntimeError", partial)),
    ):
        with patch.object(sys, "argv", ["ask", "proj-1", "why?"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert '"answer": null' in captured.out
    assert "synthesis unavailable" in captured.err
    assert "synthesis failed: RuntimeError" in captured.err


def test_top_k_zero_is_rejected_by_argparse():
    with patch.object(sys, "argv", ["ask", "proj-1", "why?", "--top-k", "0"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 2


def test_top_k_is_forwarded_to_engine_ask():
    result = make_result()
    ask_mock = AsyncMock(return_value=result)
    with patch.object(AskEngine, "ask", new=ask_mock):
        with patch.object(sys, "argv", ["ask", "proj-1", "why?", "--top-k", "7"]):
            main()

    ask_mock.assert_awaited_once_with("proj-1", "why?", top_k=7)
