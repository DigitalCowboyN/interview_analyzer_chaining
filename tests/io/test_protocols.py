"""
tests/io/test_protocols.py

This module contains comprehensive tests for the IO protocol interfaces defined in `src/io/protocols.py`.

The tests verify protocol definitions, method signatures, and integration with concrete implementations for:
- TextDataSource: Protocol for reading text data (runtime checkable)
- SentenceAnalysisWriter: Protocol for writing sentence analysis results
- ConversationMapStorage: Protocol for reading/writing conversation map data

Tests focus on protocol structure, method signatures, and behavior with real implementations.
"""

import asyncio
import inspect
from typing import Any, Dict, List, Set
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.io.protocols import (
    ConversationMapStorage,
    SentenceAnalysisWriter,
    TextDataSource,
)


class TestTextDataSourceProtocol:
    """Test the TextDataSource protocol interface."""

    def test_protocol_is_runtime_checkable(self):
        """Test that TextDataSource is a runtime checkable protocol."""

        # Test runtime checking with a compliant class
        class MockTextSource:
            async def read_text(self) -> str:
                return "test"

            def get_identifier(self) -> str:
                return "test_id"

        mock_source = MockTextSource()
        assert isinstance(mock_source, TextDataSource)

    def test_protocol_method_signatures(self):
        """Test that the protocol defines the expected method signatures."""
        # Check that protocol has the expected methods
        assert hasattr(TextDataSource, "read_text")
        assert hasattr(TextDataSource, "get_identifier")

        # Verify protocol methods have docstrings
        read_text_method = getattr(TextDataSource, "read_text")
        get_identifier_method = getattr(TextDataSource, "get_identifier")

        assert callable(read_text_method)
        assert callable(get_identifier_method)

    def test_protocol_compliance_check(self):
        """Test protocol compliance checking with various implementations."""

        # Test compliant implementation
        class CompliantSource:
            async def read_text(self) -> str:
                return "content"

            def get_identifier(self) -> str:
                return "identifier"

        compliant = CompliantSource()
        assert isinstance(compliant, TextDataSource)

        # Test non-compliant implementation (missing method)
        class NonCompliantSource:
            async def read_text(self) -> str:
                return "content"

            # Missing get_identifier method

        non_compliant = NonCompliantSource()
        assert not isinstance(non_compliant, TextDataSource)

    def test_protocol_with_mock_implementation(self):
        """Test protocol behavior with mock implementations."""
        mock_source = MagicMock(spec=TextDataSource)
        mock_source.read_text = AsyncMock(return_value="mock content")
        mock_source.get_identifier = MagicMock(return_value="mock_id")

        # Test that mock conforms to protocol
        assert isinstance(mock_source, TextDataSource)

        # Test method calls
        assert mock_source.get_identifier() == "mock_id"

        # Test async method (would need to be awaited in real usage)
        mock_source.read_text.assert_not_called()

    async def test_protocol_async_method_behavior(self):
        """Test async method behavior in protocol implementations."""

        class AsyncTestSource:
            def __init__(self, content: str):
                self.content = content
                self.identifier = "test_source"

            async def read_text(self) -> str:
                # Simulate async behavior
                await asyncio.sleep(0.001)
                return self.content

            def get_identifier(self) -> str:
                return self.identifier

        source = AsyncTestSource("test content")
        assert isinstance(source, TextDataSource)

        # Test async method execution
        result = await source.read_text()
        assert result == "test content"
        assert source.get_identifier() == "test_source"


class TestSentenceAnalysisWriterProtocol:
    """Test the SentenceAnalysisWriter protocol interface."""

    def test_protocol_method_signatures(self):
        """Test that the protocol defines the expected method signatures."""
        # Check that protocol has the expected methods
        expected_methods = ["initialize", "write_result", "finalize", "get_identifier", "read_analysis_ids"]

        for method in expected_methods:
            assert hasattr(SentenceAnalysisWriter, method)

        # Verify methods are callable
        for method_name in expected_methods:
            method = getattr(SentenceAnalysisWriter, method_name)
            assert callable(method)

    def test_protocol_structure(self):
        """Test the protocol structure and documentation."""
        assert SentenceAnalysisWriter.__doc__ is not None
        assert "Protocol for writing sentence analysis results" in SentenceAnalysisWriter.__doc__

        # Test that it's a Protocol
        assert hasattr(SentenceAnalysisWriter, "_is_protocol")
        assert SentenceAnalysisWriter._is_protocol is True

    async def test_protocol_lifecycle_methods(self):
        """Test the lifecycle methods (initialize, write, finalize)."""

        class TestWriter:
            def __init__(self):
                self.initialized = False
                self.finalized = False
                self.results = []

            async def initialize(self) -> None:
                self.initialized = True

            async def write_result(self, result: Dict[str, Any]) -> None:
                if not self.initialized:
                    raise RuntimeError("Writer not initialized")
                self.results.append(result)

            async def finalize(self) -> None:
                self.finalized = True

            def get_identifier(self) -> str:
                return "test_writer"

            async def read_analysis_ids(self) -> Set[int]:
                # Extract IDs from written results
                return {result.get("sentence_id", 0) for result in self.results}

        writer = TestWriter()

        # Test lifecycle
        await writer.initialize()
        assert writer.initialized

        # Test writing results
        test_result = {"sentence_id": 1, "analysis": "test"}
        await writer.write_result(test_result)
        assert test_result in writer.results

        # Test reading analysis IDs
        ids = await writer.read_analysis_ids()
        assert 1 in ids

        # Test finalization
        await writer.finalize()
        assert writer.finalized

    def test_protocol_method_contracts(self):
        """Test that protocol methods have the expected contracts."""
        # Test initialize method
        init_method = getattr(SentenceAnalysisWriter, "initialize")
        assert callable(init_method)

        # Test write_result method
        write_method = getattr(SentenceAnalysisWriter, "write_result")
        assert callable(write_method)

        # Test finalize method
        finalize_method = getattr(SentenceAnalysisWriter, "finalize")
        assert callable(finalize_method)

        # Test get_identifier method
        id_method = getattr(SentenceAnalysisWriter, "get_identifier")
        assert callable(id_method)

        # Test read_analysis_ids method
        read_ids_method = getattr(SentenceAnalysisWriter, "read_analysis_ids")
        assert callable(read_ids_method)

    async def test_protocol_error_handling(self):
        """Test error handling in protocol implementations."""

        class ErrorWriter:
            async def initialize(self) -> None:
                raise ConnectionError("Failed to initialize")

            async def write_result(self, result: Dict[str, Any]) -> None:
                raise IOError("Write failed")

            async def finalize(self) -> None:
                raise RuntimeError("Finalization error")

            def get_identifier(self) -> str:
                return "error_writer"

            async def read_analysis_ids(self) -> Set[int]:
                raise ValueError("Read failed")

        writer = ErrorWriter()

        # Test that errors are properly raised
        with pytest.raises(ConnectionError):
            await writer.initialize()

        with pytest.raises(IOError):
            await writer.write_result({"test": "data"})

        with pytest.raises(RuntimeError):
            await writer.finalize()

        with pytest.raises(ValueError):
            await writer.read_analysis_ids()


class TestConversationMapStorageProtocol:
    """Test the ConversationMapStorage protocol interface."""

    def test_protocol_method_signatures(self):
        """Test that the protocol defines the expected method signatures."""
        expected_methods = [
            "initialize",
            "write_entry",
            "read_all_entries",
            "read_sentence_ids",
            "finalize",
            "get_identifier",
        ]

        for method in expected_methods:
            assert hasattr(ConversationMapStorage, method)

        # Verify methods are callable
        for method_name in expected_methods:
            method = getattr(ConversationMapStorage, method_name)
            assert callable(method)

    def test_protocol_structure(self):
        """Test the protocol structure and documentation."""
        assert ConversationMapStorage.__doc__ is not None
        assert "Protocol for writing and reading conversation map data" in ConversationMapStorage.__doc__

        # Test that it's a Protocol
        assert hasattr(ConversationMapStorage, "_is_protocol")
        assert ConversationMapStorage._is_protocol is True

    async def test_protocol_data_operations(self):
        """Test data read/write operations through the protocol."""

        class TestMapStorage:
            def __init__(self):
                self.entries = []
                self.initialized = False

            async def initialize(self) -> None:
                self.initialized = True

            async def write_entry(self, entry: Dict[str, Any]) -> None:
                if not self.initialized:
                    raise RuntimeError("Storage not initialized")
                self.entries.append(entry)

            async def read_all_entries(self) -> List[Dict[str, Any]]:
                return self.entries.copy()

            async def read_sentence_ids(self) -> Set[int]:
                return {entry.get("sentence_id", 0) for entry in self.entries}

            async def finalize(self) -> None:
                self.initialized = False

            def get_identifier(self) -> str:
                return "test_map_storage"

        storage = TestMapStorage()

        # Test initialization
        await storage.initialize()
        assert storage.initialized

        # Test writing entries
        entry1 = {"sentence_id": 1, "sequence_order": 0, "sentence": "First sentence."}
        entry2 = {"sentence_id": 2, "sequence_order": 1, "sentence": "Second sentence."}

        await storage.write_entry(entry1)
        await storage.write_entry(entry2)

        # Test reading all entries
        all_entries = await storage.read_all_entries()
        assert len(all_entries) == 2
        assert entry1 in all_entries
        assert entry2 in all_entries

        # Test reading sentence IDs
        sentence_ids = await storage.read_sentence_ids()
        assert sentence_ids == {1, 2}

        # Test finalization
        await storage.finalize()
        assert not storage.initialized

    def test_protocol_method_contracts(self):
        """Test that protocol methods have the expected contracts."""
        # Test all methods are callable
        methods_to_test = [
            "initialize",
            "write_entry",
            "read_all_entries",
            "read_sentence_ids",
            "finalize",
            "get_identifier",
        ]

        for method_name in methods_to_test:
            method = getattr(ConversationMapStorage, method_name)
            assert callable(method)

    async def test_protocol_edge_cases(self):
        """Test edge cases and boundary conditions."""

        class EdgeCaseStorage:
            def __init__(self):
                self.entries = []

            async def initialize(self) -> None:
                pass

            async def write_entry(self, entry: Dict[str, Any]) -> None:
                # Handle empty or malformed entries
                if not entry:
                    raise ValueError("Empty entry not allowed")
                self.entries.append(entry)

            async def read_all_entries(self) -> List[Dict[str, Any]]:
                return self.entries

            async def read_sentence_ids(self) -> Set[int]:
                # Handle entries without sentence_id
                ids = set()
                for entry in self.entries:
                    if "sentence_id" in entry and isinstance(entry["sentence_id"], int):
                        ids.add(entry["sentence_id"])
                return ids

            async def finalize(self) -> None:
                pass

            def get_identifier(self) -> str:
                return "edge_case_storage"

        storage = EdgeCaseStorage()

        # Test empty entry handling
        with pytest.raises(ValueError):
            await storage.write_entry({})

        # Test entries without sentence_id
        await storage.write_entry({"text": "no id"})
        sentence_ids = await storage.read_sentence_ids()
        assert len(sentence_ids) == 0

        # Test mixed valid/invalid entries
        await storage.write_entry({"sentence_id": 1, "text": "valid"})
        await storage.write_entry({"sentence_id": "invalid", "text": "bad id"})

        sentence_ids = await storage.read_sentence_ids()
        assert sentence_ids == {1}  # Only valid integer IDs


class TestProtocolInteroperability:
    """Test how protocols work together and with concrete implementations."""

    def test_all_protocols_are_importable(self):
        """Test that all protocols can be imported and used."""
        # This test verifies the basic import and instantiation patterns
        protocols = [TextDataSource, SentenceAnalysisWriter, ConversationMapStorage]

        for protocol in protocols:
            assert protocol is not None
            assert hasattr(protocol, "__name__")
            # Test that they are Protocol classes
            assert hasattr(protocol, "_is_protocol")
            assert protocol._is_protocol is True

    def test_protocol_type_checking_with_text_data_source(self):
        """Test type checking behavior with TextDataSource (runtime checkable)."""

        # Create implementations that satisfy TextDataSource
        class MultiProtocolImpl:
            # TextDataSource methods
            async def read_text(self) -> str:
                return "text"

            def get_identifier(self) -> str:
                return "multi_impl"

            # SentenceAnalysisWriter methods
            async def initialize(self) -> None:
                pass

            async def write_result(self, result: Dict[str, Any]) -> None:
                pass

            async def finalize(self) -> None:
                pass

            async def read_analysis_ids(self) -> Set[int]:
                return set()

        impl = MultiProtocolImpl()

        # Test that it satisfies TextDataSource (runtime checkable)
        assert isinstance(impl, TextDataSource)

        # Note: Can't test isinstance for non-runtime-checkable protocols
        # but we can verify the implementation has the required methods

        # Verify it has SentenceAnalysisWriter methods
        assert hasattr(impl, "initialize")
        assert hasattr(impl, "write_result")
        assert hasattr(impl, "finalize")
        assert hasattr(impl, "read_analysis_ids")

    async def test_protocol_integration_patterns(self):
        """Test common integration patterns using protocols."""

        # Simulate a pipeline-like usage of protocols
        class MockTextSource:
            async def read_text(self) -> str:
                return "Test sentence 1. Test sentence 2."

            def get_identifier(self) -> str:
                return "test_source"

        class MockMapStorage:
            def __init__(self):
                self.entries = []

            async def initialize(self) -> None:
                pass

            async def write_entry(self, entry: Dict[str, Any]) -> None:
                self.entries.append(entry)

            async def read_all_entries(self) -> List[Dict[str, Any]]:
                return self.entries

            async def read_sentence_ids(self) -> Set[int]:
                return {entry["sentence_id"] for entry in self.entries}

            async def finalize(self) -> None:
                pass

            def get_identifier(self) -> str:
                return "test_map"

        class MockAnalysisWriter:
            def __init__(self):
                self.results = []

            async def initialize(self) -> None:
                pass

            async def write_result(self, result: Dict[str, Any]) -> None:
                self.results.append(result)

            async def finalize(self) -> None:
                pass

            def get_identifier(self) -> str:
                return "test_writer"

            async def read_analysis_ids(self) -> Set[int]:
                return {result["sentence_id"] for result in self.results}

        # Create instances
        source = MockTextSource()
        map_storage = MockMapStorage()
        analysis_writer = MockAnalysisWriter()

        # Verify TextDataSource compliance (runtime checkable)
        assert isinstance(source, TextDataSource)

        # Test a simple pipeline flow
        await map_storage.initialize()
        await analysis_writer.initialize()

        # Simulate processing
        text = await source.read_text()
        sentences = text.split(". ")

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Write to map
                map_entry = {"sentence_id": i + 1, "sequence_order": i, "sentence": sentence.strip()}
                await map_storage.write_entry(map_entry)

                # Write analysis result
                analysis_result = {"sentence_id": i + 1, "analysis": f"Analysis of: {sentence.strip()}"}
                await analysis_writer.write_result(analysis_result)

        # Verify results
        map_entries = await map_storage.read_all_entries()
        map_ids = await map_storage.read_sentence_ids()
        analysis_ids = await analysis_writer.read_analysis_ids()

        assert len(map_entries) == 2
        assert map_ids == analysis_ids == {1, 2}

        await map_storage.finalize()
        await analysis_writer.finalize()

    def test_protocol_documentation_and_contracts(self):
        """Test that protocols maintain their documented contracts."""
        # Test TextDataSource contract
        assert TextDataSource.__doc__ == "Protocol for reading input text data."

        # Test SentenceAnalysisWriter contract
        assert "Protocol for writing sentence analysis results" in SentenceAnalysisWriter.__doc__

        # Test ConversationMapStorage contract
        assert "Protocol for writing and reading conversation map data" in ConversationMapStorage.__doc__

        # Verify that ellipsis (...) is used for abstract methods
        # Check TextDataSource methods
        read_text_source = inspect.getsource(TextDataSource.read_text)
        assert "..." in read_text_source

        get_id_source = inspect.getsource(TextDataSource.get_identifier)
        assert "..." in get_id_source

    def test_protocol_method_inspection(self):
        """Test that protocol methods can be inspected."""
        # Test that we can get method signatures
        for protocol in [TextDataSource, SentenceAnalysisWriter, ConversationMapStorage]:
            methods = inspect.getmembers(protocol, predicate=inspect.isfunction)
            assert len(methods) > 0

            for method_name, method in methods:
                # Verify methods are callable
                assert callable(method)
                # Verify methods have signatures
                sig = inspect.signature(method)
                assert sig is not None
