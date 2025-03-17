# tests/test_sentence_analyzer.py
import pytest
from unittest.mock import patch
from src.agents.sentence_analyzer import SentenceAnalyzer


@pytest.fixture
def mock_contexts():
    return {
        "structure": "",
        "immediate": "Context around the sentence.",
        "observer": "Observer-level context.",
        "broader": "Broader context around the sentence.",
        "overall": "Overall context around the sentence."
    }


@pytest.fixture
def analyzer():
    return SentenceAnalyzer()


@patch("src.agents.agent.OpenAIAgent.call_model")
def test_classify_sentence(mock_call_model, analyzer, mock_contexts):
    mock_call_model.return_value = "<type> [0.9]"

    sentence = "This is a test sentence."
    result = analyzer.classify_sentence(sentence, mock_contexts)

    assert "function_type" in result
    assert result["function_type"] == "<type> [0.9]"

    assert mock_call_model.call_count == 7  # Seven classifications


@patch("src.agents.agent.OpenAIAgent.call_model")
def test_analyze_sentences(mock_call_model, analyzer):
    mock_call_model.return_value = "<type> [0.9]"

    sentences = ["First test sentence.", "Second test sentence."]
    results = list(analyzer.analyze_sentences(sentences))

    assert len(results) == 2

    for idx, result in enumerate(results):
        assert result["sentence_id"] == idx
        assert result["sentence"] == sentences[idx]
        assert "function_type" in result
