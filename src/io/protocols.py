# src/io/protocols.py
"""
Defines Protocol interfaces for data source and sink abstractions in the pipeline.
"""

from typing import Protocol, List, Dict, Any, AsyncIterable, Iterable, Set
# Note: Path might not be strictly necessary in protocols, 
# identifiers could be strings (URIs, etc.)
from pathlib import Path 


class TextDataSource(Protocol):
    """Protocol for reading input text data."""

    async def read_text(self) -> str:
        """Reads and returns the entire text content as a string."""
        ...

    def get_identifier(self) -> str:
        """Returns a unique string identifier for the data source (e.g., filename, URI)."""
        ...


class SentenceAnalysisWriter(Protocol):
    """Protocol for writing sentence analysis results (typically JSONL)."""

    async def initialize(self) -> None:
        """Initializes the writer (e.g., opens file, establishes connection)."""
        # This method is optional to implement if initialization is not needed
        # or handled elsewhere (e.g., __init__ or async context manager).
        pass 

    async def write_result(self, result: Dict[str, Any]) -> None:
        """Writes a single analysis result dictionary."""
        ...

    async def finalize(self) -> None:
        """Finalizes writing (e.g., closes file, commits transaction)."""
        # Optional if finalization is handled by async context manager.
        pass

    def get_identifier(self) -> str:
        """Returns a unique string identifier for the destination."""
        ...

    # Added based on verification needs
    async def read_analysis_ids(self) -> Set[int]:
        """Reads unique sentence IDs present in the written analysis results."""
        ...


class ConversationMapStorage(Protocol):
    """Protocol for writing and reading conversation map data (typically JSONL)."""

    async def initialize(self) -> None:
        """Initializes the storage for writing/reading."""
        pass

    async def write_entry(self, entry: Dict[str, Any]) -> None:
        """Writes a single map entry dictionary ({sentence_id, sequence_order, sentence})."""
        ...

    async def read_all_entries(self) -> List[Dict[str, Any]]:
        """Reads all entries from the map storage."""
        ...

    async def read_sentence_ids(self) -> Set[int]:
        """Reads only the unique sentence_id values from the map storage."""
        ...

    async def finalize(self) -> None:
        """Releases resources associated with the map storage."""
        pass

    def get_identifier(self) -> str:
        """Returns a unique string identifier for the map storage."""
        ...
