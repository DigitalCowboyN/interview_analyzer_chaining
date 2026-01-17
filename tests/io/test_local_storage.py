# tests/io/test_local_storage.py
"""
Unit tests for the local file storage implementations of IO protocols.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import aiofiles
import pytest

# Import the classes to test (adjust path if necessary)
from src.io.local_storage import (
    LocalJsonlAnalysisWriter,
    LocalJsonlMapStorage,
    LocalTextDataSource,
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


async def test_local_analysis_writer_write_and_finalize(tmp_path: Path) -> None:
    """Tests initializing, writing results, and finalizing the writer."""
    file_path = tmp_path / "output" / "analysis.jsonl"  # Test directory creation
    writer = LocalJsonlAnalysisWriter(file_path)
    results: List[Dict[str, Any]] = [
        {"sentence_id": 0, "text": "one", "score": 0.9},
        {"sentence_id": 1, "text": "two", "score": 0.8},
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


async def test_local_analysis_writer_write_uninitialized(tmp_path: Path) -> None:
    """Tests writing before initialize raises RuntimeError."""
    file_path = tmp_path / "uninitialized.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)
    with pytest.raises(RuntimeError, match="Writer must be initialized"):
        await writer.write_result({"id": 0})


async def test_local_analysis_writer_read_ids(tmp_path: Path) -> None:
    """Tests reading sentence IDs back from the written analysis file."""
    file_path = tmp_path / "output" / "analysis_read_ids.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)
    results: List[Dict[str, Any]] = [
        {"sentence_id": 5, "text": "five"},
        {"sentence_id": 2, "text": "two"},
        {"sentence_id": 5, "text": "five again"},  # Duplicate ID
        {"id_other": 9, "text": "no sentence id"},  # Missing ID
        {"sentence_id": 0, "text": "zero"},
    ]
    expected_ids: Set[int] = {0, 2, 5}

    await writer.initialize()
    for res in results:
        await writer.write_result(res)
    await writer.finalize()

    read_ids = await writer.read_analysis_ids()  # Read back from the same instance
    assert read_ids == expected_ids


async def test_local_analysis_writer_read_ids_file_not_found(tmp_path: Path) -> None:
    """Tests read_analysis_ids returns empty set if file doesn't exist."""
    file_path = tmp_path / "non_existent_analysis.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)
    read_ids: Set[int] = await writer.read_analysis_ids()
    assert read_ids == set()


# --- Tests for LocalJsonlMapStorage ---


async def test_local_map_storage_write_read_finalize(tmp_path: Path) -> None:
    """Tests writing, reading (all entries and IDs), and finalizing map storage."""
    file_path = tmp_path / "maps" / "storage_test.jsonl"
    storage = LocalJsonlMapStorage(file_path)
    entries: List[Dict[str, Any]] = [
        {"sentence_id": 0, "sequence_order": 0, "sentence": "Sentence zero."},
        {"sentence_id": 1, "sequence_order": 1, "sentence": "Sentence one."},
        {"sentence_id": 2, "sequence_order": 2, "sentence": "Sentence two."},
    ]
    expected_ids: Set[int] = {0, 1, 2}

    # Write
    await storage.initialize()  # Initialize for writing
    for entry in entries:
        await storage.write_entry(entry)
    await storage.finalize()  # Finalize writing

    assert file_path.exists()
    assert storage.get_identifier() == str(file_path)

    # Read All Entries
    read_entries = await storage.read_all_entries()
    assert read_entries == entries

    # Read IDs
    read_ids = await storage.read_sentence_ids()
    assert read_ids == expected_ids


async def test_local_map_storage_overwrite_on_initialize(tmp_path: Path) -> None:
    """Tests that initializing for write overwrites existing file."""
    file_path = tmp_path / "overwrite_map.jsonl"
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write('{"initial": true}\\n')

    storage = LocalJsonlMapStorage(file_path)
    new_entry: Dict[str, Any] = {"sentence_id": 0, "sentence": "New content"}

    await storage.initialize()  # Should open in 'w' mode
    await storage.write_entry(new_entry)
    await storage.finalize()

    # Verify only new content exists
    lines = []
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        async for line in f:
            lines.append(json.loads(line))
    assert lines == [new_entry]


async def test_local_map_storage_read_file_not_found(tmp_path: Path) -> None:
    """Tests reading entries/IDs returns empty if file doesn't exist."""
    file_path = tmp_path / "non_existent_map.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    read_entries: List[Dict[str, Any]] = await storage.read_all_entries()
    assert read_entries == []

    read_ids: Set[int] = await storage.read_sentence_ids()
    assert read_ids == set()


async def test_local_map_storage_write_uninitialized(tmp_path: Path) -> None:
    """Tests writing map entry before initialize raises RuntimeError."""
    file_path = tmp_path / "uninitialized_map.jsonl"
    storage = LocalJsonlMapStorage(file_path)
    with pytest.raises(RuntimeError, match="Map storage must be initialized"):
        await storage.write_entry({"id": 0})


# --- PHASE 1: FOUNDATION TESTS ---


# Constructor Validation Tests
async def test_local_text_data_source_constructor_validation():
    """Tests LocalTextDataSource constructor type validation."""
    # Valid Path object should work
    valid_path = Path("/test/path.txt")
    source = LocalTextDataSource(valid_path)
    assert source.get_identifier() == str(valid_path)

    # Invalid types should raise TypeError
    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalTextDataSource("/string/path")  # String instead of Path

    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalTextDataSource(123)  # Integer instead of Path

    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalTextDataSource(None)  # None instead of Path


async def test_local_analysis_writer_constructor_validation():
    """Tests LocalJsonlAnalysisWriter constructor type validation."""
    # Valid Path object should work
    valid_path = Path("/test/analysis.jsonl")
    writer = LocalJsonlAnalysisWriter(valid_path)
    assert writer.get_identifier() == str(valid_path)

    # Invalid types should raise TypeError
    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalJsonlAnalysisWriter("/string/path.jsonl")

    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalJsonlAnalysisWriter(42)

    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalJsonlAnalysisWriter(None)


async def test_local_map_storage_constructor_validation():
    """Tests LocalJsonlMapStorage constructor type validation."""
    # Valid Path object should work
    valid_path = Path("/test/map.jsonl")
    storage = LocalJsonlMapStorage(valid_path)
    assert storage.get_identifier() == str(valid_path)

    # Invalid types should raise TypeError
    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalJsonlMapStorage("/string/path.jsonl")

    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalJsonlMapStorage(3.14)

    with pytest.raises(TypeError, match="file_path must be a Path object"):
        LocalJsonlMapStorage(None)


# Advanced Error Handling Tests
async def test_local_text_data_source_os_error(tmp_path: Path):
    """Tests LocalTextDataSource handling of OS errors (permissions, etc.)."""
    import platform

    # Skip on Windows as permission model is different
    if platform.system() == "Windows":
        pytest.skip("Permission tests not reliable on Windows")

    file_path = tmp_path / "test_file.txt"

    # Create file and make it unreadable (if on Unix-like system)
    file_path.write_text("test content")

    try:
        file_path.chmod(0o000)  # Remove all permissions

        data_source = LocalTextDataSource(file_path)
        # Check if we can actually read the file (some systems ignore chmod)
        try:
            await data_source.read_text()
            # If no error was raised, skip the test (permissions not enforced)
            pytest.skip("File permissions not enforced in this environment")
        except (OSError, PermissionError):
            # This is expected - the test passes
            pass
    finally:
        # Restore permissions for cleanup
        try:
            file_path.chmod(0o644)
        except OSError:
            pass  # Ignore cleanup errors


async def test_local_text_data_source_empty_file(tmp_path: Path):
    """Tests reading from an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.touch()  # Create empty file

    data_source = LocalTextDataSource(file_path)
    content = await data_source.read_text()

    assert content == ""
    assert data_source.get_identifier() == str(file_path)


async def test_local_analysis_writer_os_error_during_initialize(tmp_path: Path):
    """Tests LocalJsonlAnalysisWriter handling OS errors during initialization."""
    import platform

    # Skip on Windows as permission model is different
    if platform.system() == "Windows":
        pytest.skip("Permission tests not reliable on Windows")

    # Create a path where parent directory cannot be created (if possible)
    file_path = tmp_path / "readonly" / "subdir" / "analysis.jsonl"

    # Make parent directory read-only (Unix-like systems)
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()

    try:
        readonly_dir.chmod(0o444)  # Read-only

        writer = LocalJsonlAnalysisWriter(file_path)
        try:
            await writer.initialize()
            # If no error was raised, skip the test (permissions not enforced)
            pytest.skip("Directory permissions not enforced in this environment")
        except (OSError, PermissionError):
            # This is expected - the test passes
            pass
    finally:
        # Restore permissions for cleanup
        try:
            readonly_dir.chmod(0o755)
        except OSError:
            pass


async def test_local_analysis_writer_write_error_simulation(tmp_path: Path):
    """Tests LocalJsonlAnalysisWriter error handling during write operations."""
    file_path = tmp_path / "write_error.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    await writer.initialize()

    # Test writing invalid JSON-serializable data
    invalid_data = {"key": set([1, 2, 3])}  # Sets are not JSON serializable

    with pytest.raises(Exception):  # Could be TypeError or other JSON error
        await writer.write_result(invalid_data)

    # Cleanup
    await writer.finalize()


async def test_local_analysis_writer_finalize_without_initialize(tmp_path: Path):
    """Tests finalizing writer without initialization."""
    file_path = tmp_path / "no_init.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    # Should handle finalize gracefully when not initialized
    await writer.finalize()  # Should not raise error

    # Verify file handle is None
    assert writer._file_handle is None


async def test_local_analysis_writer_multiple_initialize(tmp_path: Path):
    """Tests multiple initialization calls."""
    file_path = tmp_path / "multi_init.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    # First initialization
    await writer.initialize()
    first_handle = writer._file_handle
    assert first_handle is not None

    # Second initialization should work (might close previous handle)
    await writer.initialize()
    second_handle = writer._file_handle
    assert second_handle is not None

    # Should be able to write
    await writer.write_result({"test": "data"})

    # Cleanup
    await writer.finalize()


async def test_local_map_storage_os_error_during_initialize(tmp_path: Path):
    """Tests LocalJsonlMapStorage handling OS errors during initialization."""
    import platform

    # Skip on Windows as permission model is different
    if platform.system() == "Windows":
        pytest.skip("Permission tests not reliable on Windows")

    # Similar to analysis writer test
    file_path = tmp_path / "readonly_map" / "subdir" / "map.jsonl"

    readonly_dir = tmp_path / "readonly_map"
    readonly_dir.mkdir()

    try:
        readonly_dir.chmod(0o444)  # Read-only

        storage = LocalJsonlMapStorage(file_path)
        try:
            await storage.initialize()
            # If no error was raised, skip the test (permissions not enforced)
            pytest.skip("Directory permissions not enforced in this environment")
        except (OSError, PermissionError):
            # This is expected - the test passes
            pass
    finally:
        try:
            readonly_dir.chmod(0o755)
        except OSError:
            pass


async def test_local_map_storage_write_error_simulation(tmp_path: Path):
    """Tests LocalJsonlMapStorage error handling during write operations."""
    file_path = tmp_path / "map_write_error.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    await storage.initialize()

    # Test writing invalid JSON-serializable data
    invalid_entry = {"sentence_id": 1, "data": set([1, 2, 3])}

    with pytest.raises(Exception):
        await storage.write_entry(invalid_entry)

    # Cleanup
    await storage.finalize()


async def test_local_map_storage_finalize_without_initialize(tmp_path: Path):
    """Tests finalizing map storage without initialization."""
    file_path = tmp_path / "map_no_init.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    # Should handle finalize gracefully when not initialized
    await storage.finalize()  # Should not raise error

    # Verify write handle is None
    assert storage._write_handle is None


async def test_local_map_storage_multiple_initialize(tmp_path: Path):
    """Tests multiple initialization calls for map storage."""
    file_path = tmp_path / "map_multi_init.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    # First initialization
    await storage.initialize()
    first_handle = storage._write_handle
    assert first_handle is not None

    # Second initialization should work
    await storage.initialize()
    second_handle = storage._write_handle
    assert second_handle is not None

    # Should be able to write
    await storage.write_entry({"sentence_id": 1, "text": "test"})

    # Cleanup
    await storage.finalize()


# --- PHASE 2: ADVANCED SCENARIOS ---


# JSON Parsing Edge Cases
async def test_local_analysis_writer_read_ids_malformed_json(tmp_path: Path):
    """Tests read_analysis_ids handling of malformed JSON lines."""
    file_path = tmp_path / "malformed.jsonl"

    # Create file with mix of valid and malformed JSON
    content = """{"sentence_id": 1, "text": "valid"}
invalid json line without quotes
{"sentence_id": 2, "text": "another valid"}
{"incomplete": json without closing brace
{"sentence_id": 3, "text": "final valid"}
"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    writer = LocalJsonlAnalysisWriter(file_path)
    ids = await writer.read_analysis_ids()

    # Should only extract IDs from valid JSON lines
    assert ids == {1, 2, 3}


async def test_local_analysis_writer_read_ids_missing_sentence_id(tmp_path: Path):
    """Tests read_analysis_ids with entries missing sentence_id field."""
    file_path = tmp_path / "missing_ids.jsonl"

    # Create file with entries missing sentence_id
    content = """{"sentence_id": 1, "text": "has id"}
{"text": "missing id", "other_field": "value"}
{"sentence_id": 2, "text": "has id"}
{"id": 3, "text": "wrong field name"}
{"sentence_id": 4, "text": "has id"}
"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    writer = LocalJsonlAnalysisWriter(file_path)
    ids = await writer.read_analysis_ids()

    # Should only extract valid sentence_id fields
    assert ids == {1, 2, 4}


async def test_local_analysis_writer_read_ids_empty_lines(tmp_path: Path):
    """Tests read_analysis_ids handling of empty lines and whitespace."""
    file_path = tmp_path / "empty_lines.jsonl"

    # Create file with empty lines and whitespace
    content = """{"sentence_id": 1, "text": "first"}

{"sentence_id": 2, "text": "second"}


{"sentence_id": 3, "text": "third"}

"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    writer = LocalJsonlAnalysisWriter(file_path)
    ids = await writer.read_analysis_ids()

    # Should handle empty lines gracefully
    assert ids == {1, 2, 3}


async def test_local_map_storage_read_entries_malformed_json(tmp_path: Path):
    """Tests read_all_entries handling of malformed JSON lines."""
    file_path = tmp_path / "malformed_map.jsonl"

    # Create file with mix of valid and malformed JSON
    content = """{"sentence_id": 1, "text": "valid entry"}
{invalid json without quotes}
{"sentence_id": 2, "text": "another valid"}
{"incomplete": "json without closing brace"
{"sentence_id": 3, "text": "final valid"}
"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    storage = LocalJsonlMapStorage(file_path)
    entries = await storage.read_all_entries()

    # Should only return valid JSON entries
    expected = [
        {"sentence_id": 1, "text": "valid entry"},
        {"sentence_id": 2, "text": "another valid"},
        {"sentence_id": 3, "text": "final valid"},
    ]
    assert entries == expected


async def test_local_map_storage_read_sentence_ids_invalid_types(tmp_path: Path):
    """Tests read_sentence_ids with invalid sentence_id types."""
    file_path = tmp_path / "invalid_types.jsonl"

    # Create file with various sentence_id types
    content = """{"sentence_id": 1, "text": "valid int"}
{"sentence_id": "2", "text": "string id"}
{"sentence_id": 3.5, "text": "float id"}
{"sentence_id": null, "text": "null id"}
{"sentence_id": true, "text": "boolean id"}
{"sentence_id": 4, "text": "valid int"}
"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    storage = LocalJsonlMapStorage(file_path)
    ids = await storage.read_sentence_ids()

    # Should only extract valid integer sentence_ids
    assert ids == {1, 4}


async def test_local_map_storage_read_entries_empty_lines(tmp_path: Path):
    """Tests read_all_entries handling of empty lines."""
    file_path = tmp_path / "map_empty_lines.jsonl"

    content = """{"sentence_id": 1, "text": "first"}

{"sentence_id": 2, "text": "second"}

{"sentence_id": 3, "text": "third"}

"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    storage = LocalJsonlMapStorage(file_path)
    entries = await storage.read_all_entries()

    expected = [
        {"sentence_id": 1, "text": "first"},
        {"sentence_id": 2, "text": "second"},
        {"sentence_id": 3, "text": "third"},
    ]
    assert entries == expected


# Advanced State Management Tests
async def test_local_analysis_writer_error_during_finalize(tmp_path: Path):
    """Tests error handling during finalize operation."""
    file_path = tmp_path / "finalize_error.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    await writer.initialize()
    await writer.write_result({"sentence_id": 1, "text": "test"})

    # Manually close the handle to simulate an error condition
    if writer._file_handle:
        await writer._file_handle.close()

    # Finalize should handle the error gracefully or raise appropriately
    try:
        await writer.finalize()
    except Exception as e:
        # If an exception is raised, it should be a reasonable one
        assert isinstance(e, (OSError, ValueError, AttributeError))


async def test_local_map_storage_error_during_finalize(tmp_path: Path):
    """Tests error handling during map storage finalize operation."""
    file_path = tmp_path / "map_finalize_error.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    await storage.initialize()
    await storage.write_entry({"sentence_id": 1, "text": "test"})

    # Manually close the handle to simulate an error condition
    if storage._write_handle:
        await storage._write_handle.close()

    # Finalize should handle the error gracefully or raise appropriately
    try:
        await storage.finalize()
    except Exception as e:
        # If an exception is raised, it should be a reasonable one
        assert isinstance(e, (OSError, ValueError, AttributeError))


async def test_local_analysis_writer_write_after_finalize(tmp_path: Path):
    """Tests writing after finalization raises appropriate error."""
    file_path = tmp_path / "write_after_finalize.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    await writer.initialize()
    await writer.write_result({"sentence_id": 1, "text": "test"})
    await writer.finalize()

    # Writing after finalize should raise RuntimeError
    with pytest.raises(RuntimeError, match="Writer must be initialized"):
        await writer.write_result({"sentence_id": 2, "text": "after finalize"})


async def test_local_map_storage_write_after_finalize(tmp_path: Path):
    """Tests writing to map storage after finalization raises appropriate error."""
    file_path = tmp_path / "map_write_after_finalize.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    await storage.initialize()
    await storage.write_entry({"sentence_id": 1, "text": "test"})
    await storage.finalize()

    # Writing after finalize should raise RuntimeError
    with pytest.raises(RuntimeError, match="Map storage must be initialized"):
        await storage.write_entry({"sentence_id": 2, "text": "after finalize"})


async def test_local_analysis_writer_read_from_empty_file(tmp_path: Path):
    """Tests reading analysis IDs from completely empty file."""
    file_path = tmp_path / "completely_empty.jsonl"
    file_path.touch()  # Create empty file

    writer = LocalJsonlAnalysisWriter(file_path)
    ids = await writer.read_analysis_ids()

    assert ids == set()


async def test_local_map_storage_read_from_empty_file(tmp_path: Path):
    """Tests reading from completely empty map file."""
    file_path = tmp_path / "empty_map.jsonl"
    file_path.touch()  # Create empty file

    storage = LocalJsonlMapStorage(file_path)

    entries = await storage.read_all_entries()
    assert entries == []

    ids = await storage.read_sentence_ids()
    assert ids == set()


# Unicode and Special Character Tests
async def test_local_text_data_source_unicode_content(tmp_path: Path):
    """Tests reading files with Unicode content."""
    file_path = tmp_path / "unicode.txt"
    unicode_content = "Hello 世界! 🌍 Café naïve résumé"

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(unicode_content)

    data_source = LocalTextDataSource(file_path)
    content = await data_source.read_text()

    assert content == unicode_content


async def test_local_analysis_writer_unicode_json(tmp_path: Path):
    """Tests writing and reading JSON with Unicode content."""
    file_path = tmp_path / "unicode_analysis.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    unicode_data = {"sentence_id": 1, "text": "Unicode test: 世界 🌍 café", "keywords": ["世界", "café", "naïve"]}

    await writer.initialize()
    await writer.write_result(unicode_data)
    await writer.finalize()

    # Read back and verify
    entries = []
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        async for line in f:
            entries.append(json.loads(line.strip()))

    assert entries == [unicode_data]

    # Test ID reading with Unicode content
    ids = await writer.read_analysis_ids()
    assert ids == {1}


# --- PHASE 3: INTEGRATION & POLISH ---


# Exception Path Coverage
async def test_local_text_data_source_generic_exception(tmp_path: Path, monkeypatch):
    """Tests LocalTextDataSource handling of generic exceptions."""
    file_path = tmp_path / "exception_test.txt"
    file_path.write_text("test content")

    data_source = LocalTextDataSource(file_path)

    # Mock aiofiles.open to raise a generic exception
    class MockAsyncContextManager:
        async def __aenter__(self):
            raise ValueError("Simulated generic error")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_open_exception(*args, **kwargs):
        return MockAsyncContextManager()

    monkeypatch.setattr("aiofiles.open", mock_open_exception)

    with pytest.raises(ValueError, match="Simulated generic error"):
        await data_source.read_text()


async def test_local_analysis_writer_initialization_os_error(tmp_path: Path, monkeypatch):
    """Tests LocalJsonlAnalysisWriter OS error during file operations."""
    file_path = tmp_path / "os_error_test.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    # Mock aiofiles.open to raise OSError
    async def mock_open_os_error(*args, **kwargs):
        raise OSError("Simulated OS error")

    monkeypatch.setattr("aiofiles.open", mock_open_os_error)

    with pytest.raises(OSError, match="Simulated OS error"):
        await writer.initialize()


async def test_local_analysis_writer_finalize_error_propagation(tmp_path: Path, monkeypatch):
    """Tests that finalize errors are properly propagated."""
    file_path = tmp_path / "finalize_error_prop.jsonl"
    writer = LocalJsonlAnalysisWriter(file_path)

    await writer.initialize()

    # Mock the file handle close to raise an exception
    async def mock_close_error():
        raise OSError("Simulated close error")

    writer._file_handle.close = mock_close_error

    with pytest.raises(OSError, match="Simulated close error"):
        await writer.finalize()


async def test_local_map_storage_initialization_os_error(tmp_path: Path, monkeypatch):
    """Tests LocalJsonlMapStorage OS error during initialization."""
    file_path = tmp_path / "map_os_error.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    # Mock aiofiles.open to raise OSError
    async def mock_open_os_error(*args, **kwargs):
        raise OSError("Simulated map OS error")

    monkeypatch.setattr("aiofiles.open", mock_open_os_error)

    with pytest.raises(OSError, match="Simulated map OS error"):
        await storage.initialize()


async def test_local_map_storage_finalize_error_propagation(tmp_path: Path, monkeypatch):
    """Tests that map storage finalize errors are properly propagated."""
    file_path = tmp_path / "map_finalize_error_prop.jsonl"
    storage = LocalJsonlMapStorage(file_path)

    await storage.initialize()

    # Mock the file handle close to raise an exception
    async def mock_close_error():
        raise OSError("Simulated map close error")

    storage._write_handle.close = mock_close_error

    with pytest.raises(OSError, match="Simulated map close error"):
        await storage.finalize()


# Advanced Read Operation Error Handling
async def test_local_analysis_writer_read_ids_unexpected_error(tmp_path: Path, monkeypatch):
    """Tests read_analysis_ids handling of unexpected errors."""
    file_path = tmp_path / "read_error.jsonl"

    # Create a valid file first
    async with aiofiles.open(file_path, "w") as f:
        await f.write('{"sentence_id": 1}\n')

    writer = LocalJsonlAnalysisWriter(file_path)

    # Mock aiofiles.open to raise a generic exception
    class MockAsyncContextManager:
        async def __aenter__(self):
            raise RuntimeError("Simulated read error")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_open_error(*args, **kwargs):
        return MockAsyncContextManager()

    monkeypatch.setattr("aiofiles.open", mock_open_error)

    # Should handle the exception gracefully and return empty set (resilient behavior)
    ids = await writer.read_analysis_ids()
    assert ids == set()  # Should return empty set on error, not raise


async def test_local_map_storage_read_all_entries_unexpected_error(tmp_path: Path, monkeypatch):
    """Tests read_all_entries handling of unexpected errors."""
    file_path = tmp_path / "map_read_error.jsonl"

    # Create a valid file first
    async with aiofiles.open(file_path, "w") as f:
        await f.write('{"sentence_id": 1}\n')

    storage = LocalJsonlMapStorage(file_path)

    # Mock aiofiles.open to raise a generic exception
    class MockAsyncContextManager:
        async def __aenter__(self):
            raise RuntimeError("Simulated map read error")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_open_error(*args, **kwargs):
        return MockAsyncContextManager()

    monkeypatch.setattr("aiofiles.open", mock_open_error)

    # Should raise the exception (line 247)
    with pytest.raises(RuntimeError, match="Simulated map read error"):
        await storage.read_all_entries()


async def test_local_map_storage_read_sentence_ids_unexpected_error(tmp_path: Path, monkeypatch):
    """Tests read_sentence_ids handling of unexpected errors."""
    file_path = tmp_path / "map_read_ids_error.jsonl"

    # Create a valid file first
    async with aiofiles.open(file_path, "w") as f:
        await f.write('{"sentence_id": 1}\n')

    storage = LocalJsonlMapStorage(file_path)

    # Mock aiofiles.open to raise a generic exception
    class MockAsyncContextManager:
        async def __aenter__(self):
            raise RuntimeError("Simulated map read IDs error")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_open_error(*args, **kwargs):
        return MockAsyncContextManager()

    monkeypatch.setattr("aiofiles.open", mock_open_error)

    # Should raise the exception (line 285)
    with pytest.raises(RuntimeError, match="Simulated map read IDs error"):
        await storage.read_sentence_ids()


# Edge Cases for JSON Parsing
async def test_local_analysis_writer_read_ids_json_decode_error_coverage(tmp_path: Path):
    """Tests specific JSON decode error handling paths."""
    file_path = tmp_path / "json_decode_coverage.jsonl"

    # Create file with JSON that will trigger specific decode error handling
    content = """{"sentence_id": 1, "text": "valid"}
{"sentence_id": 2, "text": "valid"}
{this is completely invalid json that will trigger the JSONDecodeError path}
{"sentence_id": 3, "text": "valid"}
"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    writer = LocalJsonlAnalysisWriter(file_path)
    ids = await writer.read_analysis_ids()

    # Should handle the JSON decode error and continue (lines 139-140)
    assert ids == {1, 2, 3}


async def test_local_map_storage_json_decode_error_coverage(tmp_path: Path):
    """Tests specific JSON decode error handling in map storage."""
    file_path = tmp_path / "map_json_decode_coverage.jsonl"

    # Create file with JSON that will trigger specific decode error handling
    content = """{"sentence_id": 1, "text": "valid"}
{"sentence_id": 2, "text": "valid"}
{completely invalid json to trigger JSONDecodeError}
{"sentence_id": 3, "text": "valid"}
"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    storage = LocalJsonlMapStorage(file_path)

    # Test read_all_entries with JSON decode error (lines 230-231)
    entries = await storage.read_all_entries()
    expected = [
        {"sentence_id": 1, "text": "valid"},
        {"sentence_id": 2, "text": "valid"},
        {"sentence_id": 3, "text": "valid"},
    ]
    assert entries == expected

    # Test read_sentence_ids with JSON decode error (lines 270-271)
    ids = await storage.read_sentence_ids()
    assert ids == {1, 2, 3}


# Missing ID Warning Coverage
async def test_local_map_storage_missing_sentence_id_warning(tmp_path: Path):
    """Tests warning when sentence_id is missing in map entries."""
    file_path = tmp_path / "missing_id_warning.jsonl"

    content = """{"sentence_id": 1, "text": "has id"}
{"text": "missing sentence_id", "other": "data"}
{"sentence_id": 2, "text": "has id"}
{"no_id": "completely missing"}
{"sentence_id": 3, "text": "has id"}
"""

    async with aiofiles.open(file_path, "w") as f:
        await f.write(content)

    storage = LocalJsonlMapStorage(file_path)
    ids = await storage.read_sentence_ids()

    # Should handle missing sentence_id gracefully (lines 266-271)
    assert ids == {1, 2, 3}
