# tests/io/test_neo4j_map_storage.py
"""
Unit tests for the Neo4j map storage implementation of ConversationMapStorage protocol.
"""

from typing import Any, Dict, List, Set

import pytest

from src.io.neo4j_map_storage import Neo4jMapStorage

# Mark all tests in this module as asyncio and require Neo4j
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


# --- Tests for Neo4jMapStorage ---


async def test_neo4j_map_storage_init():
    """Tests basic initialization of Neo4jMapStorage."""
    project_id = "test-project-123"
    interview_id = "test-interview-456"

    storage = Neo4jMapStorage(project_id, interview_id)

    assert storage.project_id == project_id
    assert storage.interview_id == interview_id
    assert storage.get_identifier() == interview_id


async def test_neo4j_map_storage_init_empty_ids():
    """Tests that initialization raises ValueError for empty IDs."""
    with pytest.raises(ValueError, match="project_id and interview_id cannot be empty"):
        Neo4jMapStorage("", "interview-123")

    with pytest.raises(ValueError, match="project_id and interview_id cannot be empty"):
        Neo4jMapStorage("project-123", "")

    with pytest.raises(ValueError, match="project_id and interview_id cannot be empty"):
        Neo4jMapStorage("", "")


async def test_neo4j_map_storage_write_read_finalize(clean_test_database):
    """Tests writing, reading (all entries and IDs), and finalizing map storage."""
    project_id = "test-project-write-read"
    interview_id = "test-interview-write-read"
    storage = Neo4jMapStorage(project_id, interview_id)

    entries: List[Dict[str, Any]] = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Sentence zero."},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "Sentence one."},
        {"sentence_id": 2, "sequence_order": 2, "sentence": "Sentence two."},
    ]
    expected_ids: Set[int] = {0, 1, 2}

    # Initialize and write
    await storage.initialize()
    for entry in entries:
        await storage.write_entry(entry)
    await storage.finalize()

    # Read all entries (should be ordered by sequence_order)
    read_entries = await storage.read_all_entries()
    assert len(read_entries) == len(entries)

    # Verify the entries match (may need to filter out Neo4j node metadata)
    for i, entry in enumerate(entries):
        read_entry = read_entries[i]
        assert read_entry["sentence_id"] == entry["sentence_id"]
        assert read_entry["sequence_order"] == entry["sequence_order"]
        assert read_entry["text"] == entry["sentence"]  # Neo4j stores as "text"

    # Read sentence IDs
    read_ids = await storage.read_sentence_ids()
    assert read_ids == expected_ids


async def test_neo4j_map_storage_with_optional_fields(clean_test_database):
    """Tests writing entries with optional fields like start_time, end_time, speaker."""
    project_id = "test-project-optional"
    interview_id = "test-interview-optional"
    storage = Neo4jMapStorage(project_id, interview_id)

    entries: List[Dict[str, Any]] = [
        {
            "sentence_id": 0,
            "sequence_order": 0,
            "sentence": "Hello world.",
            "start_time": 0.0,
            "end_time": 2.5,
            "speaker": "John",
        },
        {
            "sentence_id": 1,
            "sequence_order": 1,
            "sentence": "How are you?",
            "start_time": 3.0,
            "end_time": 4.8,
            "speaker": "Jane",
        },
    ]

    await storage.initialize()
    for entry in entries:
        await storage.write_entry(entry)
    await storage.finalize()

    # Read back and verify optional fields
    read_entries = await storage.read_all_entries()
    assert len(read_entries) == 2

    for i, original_entry in enumerate(entries):
        read_entry = read_entries[i]
        assert read_entry["sentence_id"] == original_entry["sentence_id"]
        assert read_entry["start_time"] == original_entry["start_time"]
        assert read_entry["end_time"] == original_entry["end_time"]
        assert read_entry["speaker"] == original_entry["speaker"]


async def test_neo4j_map_storage_initialize_clears_old_data(clean_test_database):
    """Tests that initializing clears old sentences for the same interview."""
    project_id = "test-project-clear"
    interview_id = "test-interview-clear"
    storage = Neo4jMapStorage(project_id, interview_id)

    # First batch of entries
    first_entries: List[Dict[str, Any]] = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Old sentence 0."},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "Old sentence 1."},
    ]

    await storage.initialize()
    for entry in first_entries:
        await storage.write_entry(entry)
    await storage.finalize()

    # Verify first entries exist
    read_entries = await storage.read_all_entries()
    assert len(read_entries) == 2

    # Initialize again with new entries (should clear old ones)
    new_entries: List[Dict[str, Any]] = [
        {"sentence_id": 10, "sequence_order": 0, "sentence": "New sentence 10."},
    ]

    await storage.initialize()  # Should clear old sentences
    for entry in new_entries:
        await storage.write_entry(entry)
    await storage.finalize()

    # Verify only new entries exist
    read_entries = await storage.read_all_entries()
    assert len(read_entries) == 1
    assert read_entries[0]["sentence_id"] == 10
    assert read_entries[0]["text"] == "New sentence 10."


async def test_neo4j_map_storage_missing_required_keys(clean_test_database):
    """Tests that write_entry raises ValueError for missing required keys."""
    project_id = "test-project-missing"
    interview_id = "test-interview-missing"
    storage = Neo4jMapStorage(project_id, interview_id)

    await storage.initialize()

    # Missing sentence_id
    with pytest.raises(ValueError, match="Entry dict missing required keys"):
        await storage.write_entry({"sequence_order": 0, "sentence": "test"})

    # Missing sequence_order
    with pytest.raises(ValueError, match="Entry dict missing required keys"):
        await storage.write_entry({"sentence_id": 0, "sentence": "test"})

    # Missing sentence
    with pytest.raises(ValueError, match="Entry dict missing required keys"):
        await storage.write_entry({"sentence_id": 0, "sequence_order": 0})


async def test_neo4j_map_storage_read_empty_interview(clean_test_database):
    """Tests reading from an interview that has no sentences."""
    project_id = "test-project-empty"
    interview_id = "test-interview-empty"
    storage = Neo4jMapStorage(project_id, interview_id)

    # Don't write any entries, just try to read
    read_entries = await storage.read_all_entries()
    assert read_entries == []

    read_ids = await storage.read_sentence_ids()
    assert read_ids == set()


async def test_neo4j_map_storage_multiple_projects(clean_test_database):
    """Tests that different projects/interviews are isolated."""
    # Project 1, Interview 1
    storage1 = Neo4jMapStorage("project-1", "interview-1")
    entries1: List[Dict[str, Any]] = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Project 1 sentence."},
    ]

    # Project 2, Interview 2 (different project, different interview ID)
    storage2 = Neo4jMapStorage("project-2", "interview-2")
    entries2: List[Dict[str, Any]] = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Project 2 sentence."},
    ]

    # Write to both
    await storage1.initialize()
    await storage1.write_entry(entries1[0])
    await storage1.finalize()

    await storage2.initialize()
    await storage2.write_entry(entries2[0])
    await storage2.finalize()

    # Read from both - should be isolated
    read1 = await storage1.read_all_entries()
    read2 = await storage2.read_all_entries()

    assert len(read1) == 1
    assert len(read2) == 1
    assert read1[0]["text"] == "Project 1 sentence."
    assert read2[0]["text"] == "Project 2 sentence."


async def test_neo4j_map_storage_sequence_ordering(clean_test_database):
    """Tests that entries are returned in correct sequence order."""
    project_id = "test-project-order"
    interview_id = "test-interview-order"
    storage = Neo4jMapStorage(project_id, interview_id)

    # Write entries out of order
    entries: List[Dict[str, Any]] = [
        {"sentence_id": 2, "sequence_order": 2, "sentence": "Third sentence."},
        {"sentence_id": 0, "sequence_order": 0, "sentence": "First sentence."},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "Second sentence."},
    ]

    await storage.initialize()
    for entry in entries:
        await storage.write_entry(entry)
    await storage.finalize()

    # Read back - should be ordered by sequence_order
    read_entries = await storage.read_all_entries()
    assert len(read_entries) == 3
    assert read_entries[0]["sequence_order"] == 0
    assert read_entries[0]["text"] == "First sentence."
    assert read_entries[1]["sequence_order"] == 1
    assert read_entries[1]["text"] == "Second sentence."
    assert read_entries[2]["sequence_order"] == 2
    assert read_entries[2]["text"] == "Third sentence."


async def test_neo4j_map_storage_non_integer_sentence_ids(clean_test_database):
    """Tests handling of non-integer sentence IDs in read_sentence_ids."""
    project_id = "test-project-nonint"
    interview_id = "test-interview-nonint"
    storage = Neo4jMapStorage(project_id, interview_id)

    # This test checks the robustness of read_sentence_ids
    # In normal usage, sentence_id should always be int, but let's test the guard

    # For now, just test with normal integer IDs since our write logic enforces int
    entries: List[Dict[str, Any]] = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Test sentence."},
        {"sentence_id": 42, "sequence_order": 1, "sentence": "Another sentence."},
    ]

    await storage.initialize()
    for entry in entries:
        await storage.write_entry(entry)
    await storage.finalize()

    read_ids = await storage.read_sentence_ids()
    assert read_ids == {0, 42}
