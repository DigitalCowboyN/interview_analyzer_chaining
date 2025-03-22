# tests/test_pipeline.py
import pytest
from pathlib import Path
from src.pipeline import segment_text, process_file
import json
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.asyncio

def mock_response(content_dict):
    import json
    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_output = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(content_dict)
    mock_output.content = [mock_content]
    mock_resp.output = [mock_output]
    return mock_resp

def test_segment_text():
    test_text = "Hello world. How are you today? This pipeline is running well!"
    sentences = segment_text(test_text)
    assert len(sentences) == 3

@pytest.fixture
def sample_text_file(tmp_path):
    file_content = "This is a test. Ensure proper segmentation."
    test_file = tmp_path / "test_file.txt"
    test_file.write_text(file_content)
    return test_file

async def test_process_file(sample_text_file, tmp_path):
    from src.pipeline import process_file
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    response_content = {
        "function_type": "declarative",
        "structure_type": "simple sentence",
        "purpose": "informational",
        "topic_level_1": "unit testing",
        "topic_level_3": "sentence segmentation",
        "overall_keywords": ["test", "segmentation"],
        "domain_keywords": ["unit test"]
    }
    with patch("openai.responses.create", return_value=mock_response(response_content)):
        await process_file(sample_text_file, output_dir)

    output_file = output_dir / f"{sample_text_file.stem}_analysis.json"
    assert output_file.exists()

    data = json.loads(output_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["sentence"] == "This is a test."
    assert data[0]["function_type"] == "declarative"
