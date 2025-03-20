# tests/test_sentence_analyzer.py
import pytest

pytestmark = pytest.mark.asyncio
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
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


@pytest.mark.asyncio
@patch("src.agents.agent.OpenAIAgent.call_model", new_callable=AsyncMock)
async def test_classify_sentence(mock_call_model, analyzer, mock_contexts):
    mock_call_model.return_value = AnalysisResult(
        function_type="declarative",
        structure_type="simple sentence",
        purpose="to state a fact",
        topic_level_1="testing",
        topic_level_3="evaluation",
        overall_keywords="test",
        domain_keywords="assessment, evaluation"
    )

    sentence = "This is a test sentence."
    result = await analyzer.classify_sentence(sentence, mock_contexts)  # Ensure await is used

    assert "function_type" in result
    assert result["function_type"] == "<type> [0.9]"

    assert mock_call_model.call_count == 7  # Seven classifications


@pytest.mark.asyncio
@patch("src.agents.agent.OpenAIAgent.call_model")
async def test_analyze_sentences(mock_call_model, analyzer):
    mock_call_model.return_value = "<type> [0.9]"

    sentences = ["First test sentence.", "Second test sentence."]
    results = await analyzer.analyze_sentences(sentences)  # Await the function directly

    assert len(results) == 2

    for idx, result in enumerate(results):
        assert result["sentence_id"] == idx
        assert result["sentence"] == sentences[idx]
        assert "function_type" in result
