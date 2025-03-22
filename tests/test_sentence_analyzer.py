# tests/test_sentence_analyzer.py
import pytest
from unittest.mock import patch, AsyncMock
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.models.analysis_result import AnalysisResult

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_contexts():
    return {
        "structure": "",
        "immediate": "Immediate context around the sentence.",
        "observer": "Observer-level context.",
        "broader": "Broader context around the sentence.",
        "overall": "Overall context around the sentence."
    }

@pytest.fixture
def analyzer():
    return SentenceAnalyzer()

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence(mock_call_model, analyzer, mock_contexts):
    mock_call_model.return_value = AnalysisResult(
        function_type="declarative",
        structure_type="simple sentence",
        purpose="informational",
        topic_level_1="testing",
        topic_level_3="evaluation",
        overall_keywords=["test"],
        domain_keywords=["assessment", "evaluation"]
    )

    sentence = "This is a test sentence."
    result = await analyzer.classify_sentence(sentence, mock_contexts)
    assert result["function_type"] == "declarative"
    assert result["structure_type"] == "simple sentence"
    assert result["purpose"] == "informational"
    assert result["topic_level_1"] == "testing"
    assert result["topic_level_3"] == "evaluation"
    assert result["overall_keywords"] == ["test"]
    assert result["domain_keywords"] == ["assessment", "evaluation"]
    assert mock_call_model.call_count == 7  # Seven classification calls expected

@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_analyze_sentences(mock_call_model, analyzer):
    mock_call_model.return_value = AnalysisResult(
        function_type="declarative",
        structure_type="simple sentence",
        purpose="informational",
        topic_level_1="testing",
        topic_level_3="evaluation",
        overall_keywords=["test"],
        domain_keywords=["assessment", "evaluation"]
    )
    sentences = ["First test sentence.", "Second test sentence."]
    results = await analyzer.analyze_sentences(sentences)
    assert len(results) == 2
    for idx, result in enumerate(results):
        assert result["sentence_id"] == idx
        assert result["sentence"] == sentences[idx]
        assert result["function_type"] == "declarative"
