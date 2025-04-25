# src/io/local_storage.py
"""
Concrete implementations of IO protocols using the local filesystem and aiofiles.
"""

import aiofiles
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

# Assuming protocols are defined in src.io.protocols
# Adjust import if the directory structure is different
from .protocols import TextDataSource, SentenceAnalysisWriter, ConversationMapStorage 

logger = logging.getLogger(__name__)

class LocalTextDataSource(TextDataSource):
    """Reads text data from a local file."""
    def __init__(self, file_path: Path):
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be a Path object")
        self._file_path = file_path

    async def read_text(self) -> str:
        """Reads text content asynchronously."""
        try:
            async with aiofiles.open(self._file_path, mode='r', encoding='utf-8') as f:
                logger.debug(f"Reading text from: {self._file_path}")
                return await f.read()
        except FileNotFoundError:
            logger.error(f"Local text data source file not found: {self._file_path}")
            raise
        except OSError as e:
            logger.error(f"OS error reading local text file {self._file_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading local text file {self._file_path}: {e}", exc_info=True)
            raise
            
    def get_identifier(self) -> str:
        """Returns the string representation of the file path."""
        return str(self._file_path)

class LocalJsonlAnalysisWriter(SentenceAnalysisWriter):
    """Writes analysis results to a local JSON Lines file."""
    def __init__(self, file_path: Path):
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be a Path object")
        self._file_path = file_path
        self._file_handle: Optional[aiofiles.threadpool.binary.AsyncBufferedIOBase] = None # Type hint for clarity

    async def initialize(self) -> None:
        """Opens the file for appending, creating directories if needed."""
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            # Open in append mode 'a' for analysis results
            self._file_handle = await aiofiles.open(self._file_path, mode='a', encoding='utf-8')
            logger.debug(f"Initialized local JSONL writer for: {self._file_path}")
        except OSError as e:
            logger.error(f"Failed to initialize/open local analysis writer {self._file_path}: {e}", exc_info=True)
            raise

    async def write_result(self, result: Dict[str, Any]) -> None:
        """Appends a JSON line representation of the result dictionary."""
        if self._file_handle is None:
            # Optionally raise an error or try to initialize implicitly?
            # Raising error is safer to enforce initialization.
            logger.error(f"Attempted to write to uninitialized writer: {self.get_identifier()}")
            raise RuntimeError("Writer must be initialized before writing.")
            
        try:
            line = json.dumps(result, ensure_ascii=False) + '\n'
            await self._file_handle.write(line)
        except Exception as e:
            # Catch potential errors during write (e.g., disk full)
            logger.error(f"Error writing analysis result to {self.get_identifier()}: {e}", exc_info=True)
            # Decide if we should raise here or just log
            raise # Propagate write errors

    async def finalize(self) -> None:
        """Closes the file handle."""
        if self._file_handle:
            try:
                await self._file_handle.close()
                logger.debug(f"Finalized local JSONL writer for: {self._file_path}")
                self._file_handle = None
            except Exception as e:
                logger.error(f"Error closing analysis writer file {self.get_identifier()}: {e}", exc_info=True)
                # Potentially raise here as well, data might be lost/corrupted
                raise

    def get_identifier(self) -> str:
        """Returns the string representation of the file path."""
        return str(self._file_path)
        
    async def read_analysis_ids(self) -> Set[int]:
        """Reads unique sentence IDs from the analysis JSONL file."""
        actual_ids: Set[int] = set()
        try:
            async with aiofiles.open(self._file_path, mode="r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "sentence_id" in entry:
                            actual_ids.add(entry["sentence_id"])
                        else:
                            logger.warning(f"Missing 'sentence_id' in analysis file {self.get_identifier()}, line content: {line[:50]}...")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse line in analysis file {self.get_identifier()}: '{line[:100]}...'", exc_info=True)
        except FileNotFoundError:
            logger.warning(f"Analysis file not found when reading IDs: {self.get_identifier()}")
            # Return empty set if file doesn't exist yet (or was deleted)
        except Exception as e:
            logger.error(f"Unexpected error reading analysis IDs from {self.get_identifier()}: {e}", exc_info=True)
            # Depending on desired behavior, could raise or return empty set
        return actual_ids


class LocalJsonlMapStorage(ConversationMapStorage):
    """Writes and reads conversation map data to/from a local JSON Lines file."""
    def __init__(self, file_path: Path):
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be a Path object")
        self._file_path = file_path
        self._write_handle: Optional[aiofiles.threadpool.binary.AsyncBufferedIOBase] = None

    async def initialize(self) -> None:
        """Initializes the storage for writing, creating directories if needed."""
        # Map creation typically overwrites, so use 'w' mode.
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_handle = await aiofiles.open(self._file_path, mode='w', encoding='utf-8')
            logger.debug(f"Initialized local JSONL map storage for writing: {self._file_path}")
        except OSError as e:
             logger.error(f"Failed to initialize/open local map storage {self._file_path}: {e}", exc_info=True)
             raise

    async def write_entry(self, entry: Dict[str, Any]) -> None:
        """Writes a single map entry as a JSON line."""
        if self._write_handle is None:
             logger.error(f"Attempted to write map entry to uninitialized storage: {self.get_identifier()}")
             raise RuntimeError("Map storage must be initialized before writing.")
             
        try:
            line = json.dumps(entry, ensure_ascii=False) + '\n'
            await self._write_handle.write(line)
        except Exception as e:
            logger.error(f"Error writing map entry to {self.get_identifier()}: {e}", exc_info=True)
            raise

    async def finalize(self) -> None:
        """Closes the write file handle."""
        if self._write_handle:
            try:
                await self._write_handle.close()
                logger.debug(f"Finalized local JSONL map storage writing: {self._file_path}")
                self._write_handle = None
            except Exception as e:
                logger.error(f"Error closing map storage file {self.get_identifier()}: {e}", exc_info=True)
                raise

    async def read_all_entries(self) -> List[Dict[str, Any]]:
        """Reads all entries from the map JSONL file."""
        entries: List[Dict[str, Any]] = []
        try:
            async with aiofiles.open(self._file_path, mode="r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                         logger.error(f"Failed to parse line in map file {self.get_identifier()}: '{line[:100]}...'", exc_info=True)
        except FileNotFoundError:
             logger.warning(f"Map file not found when reading entries: {self.get_identifier()}")
             # Return empty list as per original behavior assumption
        except Exception as e:
             logger.error(f"Unexpected error reading map entries from {self.get_identifier()}: {e}", exc_info=True)
             # Consider raising here? Or return empty list?
             # Raising seems safer if caller expects data or an error.
             raise
        return entries

    async def read_sentence_ids(self) -> Set[int]:
        """Reads unique sentence IDs from the map JSONL file."""
        ids: Set[int] = set()
        try:
            async with aiofiles.open(self._file_path, mode="r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "sentence_id" in entry and isinstance(entry["sentence_id"], int):
                            ids.add(entry["sentence_id"])
                        else:
                             logger.warning(f"Missing or invalid 'sentence_id' in map file {self.get_identifier()}, line content: {line[:50]}...")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse line in map file {self.get_identifier()}: '{line[:100]}...'", exc_info=True)
        except FileNotFoundError:
            logger.warning(f"Map file not found when reading IDs: {self.get_identifier()}")
        except Exception as e:
             logger.error(f"Unexpected error reading map IDs from {self.get_identifier()}: {e}", exc_info=True)
             # Raising seems safer than returning potentially incorrect empty set
             raise
        return ids

    def get_identifier(self) -> str:
        """Returns the string representation of the file path."""
        return str(self._file_path) 