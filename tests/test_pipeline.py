# tests/test_pipeline.py
import pytest
from pathlib import Path
from src.pipeline import segment_text


def test_segment_text():
    test_text = "Hello world. How are you today? This pipeline is running well!"

    sentences = segment_text(test_text)

    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you today?"
    assert sentences[2] == "This pipeline is running well!"


@pytest.fixture
def sample_text_file(tmp_path):
    file_content = "This is a test. Ensure proper segmentation."
    test_file = tmp_path / "test_file.txt"
    test_file.write_text(file_content)
    return test_file


@pytest.mark.asyncio
async def test_process_file(sample_text_file, tmp_path):
    from src.pipeline import process_file

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    await process_file(sample_text_file, output_dir)  # Ensure this is awaited

    output_file = output_dir / "test_file_analysis.json"
    assert output_file.exists()

    import json

    data = json.loads(output_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 2  # two sentences
    assert data[0]["sentence"] == "This is a test."
    assert data[1]["sentence"] == "Ensure proper segmentation."
