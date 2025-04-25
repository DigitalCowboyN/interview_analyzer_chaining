# tests/io/test_local_storage.py
"""
Unit tests for the local file storage implementations of IO protocols.
"""

import pytest
import aiofiles
import json
from pathlib import Path
from typing import List, Dict, Any, Set

# Import the classes to test (adjust path if necessary)
from src.io.local_storage import (
    LocalTextDataSource,
    LocalJsonlAnalysisWriter,
    LocalJsonlMapStorage
)

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# --- Tests for LocalTextDataSource ---

async def test_local_text_data_source_read_success(tmp_path: Path):
    """Tests reading text successfully from an existing file."""
    file_path = tmp_path / "source.txt"
    content = "Line 1\nLine 2\nLast line."
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(content)

    data_source = LocalTextDataSource(file_path)
    read_content = await data_source.read_text()

    assert read_content == content
    assert data_source.get_identifier() == str(file_path)

async def test_local_text_data_source_read_file_not_found(tmp_path: Path):
    """Tests FileNotFoundError is raised if the source file doesn't exist."""
    file_path = tmp_path / "non_existent.txt"
    data_source = LocalTextDataSource(file_path)

    with pytest.raises(FileNotFoundError):
        await data_source.read_text()


# --- Tests for LocalJsonlAnalysisWriter ---

async def test_local_analysis_writer_write_and_finalize(tmp_path: Path):
    """Tests initializing, writing results, and finalizing the writer."""
    file_path = tmp_path / "output" / "analysis.jsonl" # Test directory creation
    writer = LocalJsonlAnalysisWriter(file_path)
    results = [
        {"sentence_id": 0, "text": "one", "score": 0.9},
        {"sentence_id": 1, "text": "two", "score": 0.8}
    ]

    await writer.initialize()
    for res in results:
        await writer.write_result(res)
    await writer.finalize()

    # Verify content
    assert file_path.exists()
    lines = []
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        async for line in f:
            lines.append(json.loads(line.strip()))
    
    assert lines == results
    assert writer.get_identifier() == str(file_path)

async def test_local_analysis_writer_write_uninitialized(tmp_path: Path):
    """Tests writing before initialize raises RuntimeError."""
    file_path = tmp_path / "uninitialized.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)
    with pytest.raises(RuntimeError, match="Writer must be initialized"):
        await writer.write_result({"id": 0})

async def test_local_analysis_writer_read_ids(tmp_path: Path):
    """Tests reading sentence IDs back from the written analysis file."""
    file_path = tmp_path / "output" / "analysis_read_ids.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)
    results = [
        {"sentence_id": 5, "text": "five"},
        {"sentence_id": 2, "text": "two"},
        {"sentence_id": 5, "text": "five again"}, # Duplicate ID
        {"id_other": 9, "text": "no sentence id"}, # Missing ID
        {"sentence_id": 0, "text": "zero"}
    ]
    expected_ids = {0, 2, 5}

    await writer.initialize()
    for res in results:
        await writer.write_result(res)
    await writer.finalize()

    read_ids = await writer.read_analysis_ids() # Read back from the same instance
    assert read_ids == expected_ids

async def test_local_analysis_writer_read_ids_file_not_found(tmp_path: Path):
    """Tests read_analysis_ids returns empty set if file doesn't exist."""
    file_path = tmp_path / "non_existent_analysis.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)
    read_ids = await writer.read_analysis_ids()
    assert read_ids == set()


# --- Tests for LocalJsonlMapStorage ---

async def test_local_map_storage_write_read_finalize(tmp_path: Path):
    """Tests writing, reading (all entries and IDs), and finalizing map storage."""
    file_path = tmp_path / "maps" / "storage_test.jsonl"
    storage = LocalJsonlMapStorage(file_path)
    entries = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Sentence zero."},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "Sentence one."},
        {"sentence_id": 2, "sequence_order": 2, "sentence": "Sentence two."}
    ]
    expected_ids = {0, 1, 2}

    # Write
    await storage.initialize() # Initialize for writing
    for entry in entries:
        await storage.write_entry(entry)
    await storage.finalize() # Finalize writing

    assert file_path.exists()
    assert storage.get_identifier() == str(file_path)

    # Read All Entries
    read_entries = await storage.read_all_entries()
    assert read_entries == entries

    # Read IDs
    read_ids = await storage.read_sentence_ids()
    assert read_ids == expected_ids

async def test_local_map_storage_overwrite_on_initialize(tmp_path: Path):
    """Tests that initializing for write overwrites existing file."""
    file_path = tmp_path / "overwrite_map.jsonl"
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write('{"initial": true}\\n')

    storage = LocalJsonlMapStorage(file_path)
    new_entry = {"sentence_id": 0, "sentence": "New content"}
    
    await storage.initialize() # Should open in 'w' mode
    await storage.write_entry(new_entry)
    await storage.finalize()

    # Verify only new content exists
    lines = []
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        async for line in f:
             lines.append(json.loads(line))
    assert lines == [new_entry]

async def test_local_map_storage_read_file_not_found(tmp_path: Path):
    """Tests reading entries/IDs returns empty if file doesn't exist."""
    file_path = tmp_path / "non_existent_map.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    read_entries = await storage.read_all_entries()
    assert read_entries == []

    read_ids = await storage.read_sentence_ids()
    assert read_ids == set()

async def test_local_map_storage_write_uninitialized(tmp_path: Path):
    """Tests writing map entry before initialize raises RuntimeError."""
    file_path = tmp_path / "uninitialized_map.jsonl"
    storage = LocalJsonlMapStorage(file_path)
    with pytest.raises(RuntimeError, match="Map storage must be initialized"):
        await storage.write_entry({"id": 0}) 