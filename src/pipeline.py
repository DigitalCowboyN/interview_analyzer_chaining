"""
pipeline.py

Defines the core pipeline functions for processing text files.

Main workflow orchestrated by `run_pipeline`:
    - Iterates through input text files.
    - For each file, `process_file` is called:
        - `create_conversation_map`: Segments text, creates a map file (.jsonl)
          listing sentences with IDs.
        - Injected `AnalysisService`:
            Builds context for sentences and analyzes them.
        - `_result_writer`:
            Asynchronously writes analysis results to an output file (.jsonl).
    - `verify_output_completeness`: Checks if analysis outputs match map files.

Also includes `analyze_specific_sentences` for targeted re-analysis.

Dependencies:
    - `src.services.analysis_service.AnalysisService`:
        Performs context building and analysis.
    - `src.agents.*`:
        Underlying components used by AnalysisService (indirect dependency).
    - `src.utils.helpers.append_json_line`: Writes JSON Lines data.
    - `src.utils.text_processing.segment_text`: Splits text into sentences.
    - `src.utils.logger.get_logger`: Centralized logging.
    - `src.utils.metrics.MetricsTracker`: Tracks operational metrics.
"""

import asyncio
import time
from pathlib import Path

# Remove commented-out import: import json
# Import necessary types
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import agent classes needed for instantiation
from src.agents.context_builder import ContextBuilder
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.config import config  # Import the global config object

# Import the local storage implementations
from src.io.local_storage import (
    LocalJsonlAnalysisWriter,
    LocalJsonlMapStorage,
    LocalTextDataSource,
)

# Import dataclass if not already (it's part of typing)
# from dataclasses import dataclass
# Import the new protocols
from src.io.protocols import (
    ConversationMapStorage,
    SentenceAnalysisWriter,
    TextDataSource,
)

# Import graph persistence components
from src.persistence.graph_persistence import save_analysis_to_graph
from src.services.analysis_service import AnalysisService

# SentenceAnalyzer is used indirectly via AnalysisService
# from src.agents.sentence_analyzer import SentenceAnalyzer
from src.utils.logger import get_logger

# ContextBuilder is used indirectly via AnalysisService
# from src.agents.context_builder import ContextBuilder
# Import the singleton instance, not the class
from src.utils.metrics import metrics_tracker
from src.utils.neo4j_driver import connection_manager  # Import the singleton

# Import the new path helper
from src.utils.path_helpers import PipelinePaths, generate_pipeline_paths
from src.utils.text_processing import segment_text

logger = get_logger()


# Define a helper for log prefixing
def _log_prefix(task_id: Optional[str] = None) -> str:
    return f"[Task {task_id}] " if task_id else ""


# --- Refactored Orchestrator Class ---
class PipelineOrchestrator:
    """Orchestrates the text analysis pipeline execution."""
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        map_dir: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ):
        """Initializes the orchestrator and sets up the environment."""
        self.task_id = task_id
        self.prefix = _log_prefix(self.task_id)
        logger.info(f"{self.prefix}Initializing Pipeline Orchestrator...")

        # Use provided config or load default
        self.config = config_dict if config_dict is not None else config
        logger.debug(f"{self.prefix}Using configuration: {self.config}")

        # --- Path Setup (moved from _setup_pipeline_environment) ---
        try:
            self.input_dir_path = Path(input_dir).resolve(strict=True)
            logger.info(f"{self.prefix}Input directory set to: {self.input_dir_path}")

            # Use output_dir from config if not provided, default to input_dir/output
            output_dir_str = output_dir or self.config.get("paths", {}).get("output_dir")
            if not output_dir_str:
                self.output_dir_path = self.input_dir_path / "output"
                logger.warning(f"{self.prefix}Output directory not specified, defaulting to: {self.output_dir_path}")
            else:
                self.output_dir_path = Path(output_dir_str).resolve()
                logger.info(f"{self.prefix}Output directory set to: {self.output_dir_path}")
            self.output_dir_path.mkdir(parents=True, exist_ok=True)

            # Use map_dir from config if not provided, default to input_dir/maps
            map_dir_str = map_dir or self.config.get("paths", {}).get("map_dir")
            if not map_dir_str:
                self.map_dir_path = self.input_dir_path / "maps"
                logger.warning(f"{self.prefix}Map directory not specified, defaulting to: {self.map_dir_path}")
            else:
                self.map_dir_path = Path(map_dir_str).resolve()
                logger.info(f"{self.prefix}Map directory set to: {self.map_dir_path}")
            self.map_dir_path.mkdir(parents=True, exist_ok=True)

            # Get suffixes from config with defaults
            self.map_suffix = self.config.get("paths", {}).get("map_suffix", "_map.jsonl")
            self.analysis_suffix = self.config.get("paths", {}).get("analysis_suffix", "_analysis.jsonl")

        except FileNotFoundError as e:
            logger.critical(f"{self.prefix}Input directory not found: {input_dir}. {e}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"{self.prefix}Error setting up pipeline directories: {e}", exc_info=True)
            raise

        # --- Service Initialization (moved from _setup_pipeline_environment) ---
        try:
            # Metrics must be initialized *before* AnalysisService
            self.metrics_tracker = metrics_tracker
            self.metrics_tracker.reset()
            logger.info(f"{self.prefix}MetricsTracker initialized and reset.")

            context_builder = ContextBuilder(self.config)
            sentence_analyzer = SentenceAnalyzer(self.config)  # Instantiate analyser
            # Pass the metrics_tracker instance AND config
            self.analysis_service = AnalysisService(
                self.config, context_builder, sentence_analyzer, self.metrics_tracker
            )
            logger.info(f"{self.prefix}AnalysisService initialized.")
        except Exception as e:
            logger.critical(f"{self.prefix}Failed to initialize AnalysisService: {e}", exc_info=True)
            raise

        # --- Metrics Initialization (moved from _setup_pipeline_environment) ---
        # REMOVED - Moved up before AnalysisService init
        # # Use the imported singleton instance
        # self.metrics_tracker = metrics_tracker
        # # Reset explicitly at the start of an orchestration run
        # self.metrics_tracker.reset()
        # logger.info(f"{self.prefix}MetricsTracker initialized and reset.\")

        # --- Concurrency Setup (moved from _setup_pipeline_environment) ---
        try:
            self.num_concurrent_files = int(self.config.get("pipeline", {}).get("num_concurrent_files", 1))
            if self.num_concurrent_files < 1:
                logger.warning(f"{self.prefix}num_concurrent_files < 1, setting to 1.")
                self.num_concurrent_files = 1
            logger.info(f"{self.prefix}Concurrent file processing limit: {self.num_concurrent_files}")
        except (ValueError, TypeError) as e:
            logger.warning(f"{self.prefix}Invalid num_concurrent_files in config, defaulting to 1. Error: {e}")
            self.num_concurrent_files = 1

    # --- Result Handling Methods (MOVED INSIDE CLASS) ---
    async def _handle_jsonl_write(
        self,
        result: Dict[str, Any],
        analysis_writer: SentenceAnalysisWriter,
        prefix: str
    ) -> bool:
        """Handles writing a single result to the JSONL analysis writer."""
        writer_id = analysis_writer.get_identifier()
        sentence_id = result.get('sentence_id', 'N/A')
        try:
            await analysis_writer.write_result(result)
            return True  # Indicate success
        except Exception as write_e:
            sentence_id_info = f"result {sentence_id}" if sentence_id != 'N/A' else "result (unknown ID)"
            logger.error(f"{prefix}Writer failed writing {sentence_id_info} to {writer_id}: {write_e}", exc_info=True)
            self.metrics_tracker.increment_errors(f"{writer_id}_jsonl_write_error")  # Use instance tracker
            return False  # Indicate failure

    async def _handle_graph_save(
        self,
        result: Dict[str, Any],
        prefix: str
    ) -> bool:
        """Handles saving a single result to the graph database."""
        sentence_id = result.get('sentence_id', 'N/A')
        filename = result.get('filename')

        if not filename:
            logger.warning(
                f"{prefix}Filename missing in result data for sentence_id {sentence_id}, skipping graph save."
            )
            return False  # Indicate failure (or skip)

        try:
            await save_analysis_to_graph(
                result, filename, connection_manager
            )  # Use the imported singleton connection manager directly
            return True  # Indicate success
        except Exception as graph_e:
            logger.error(
                f"{prefix}Writer failed saving result ID: {sentence_id} from {filename} to graph: {graph_e}",
                exc_info=True
            )
            self.metrics_tracker.increment_errors(f"{filename}_graph_save_error")  # Use instance tracker
            return False  # Indicate failure
    # --- End Result Handling Methods ---

    def _discover_files_to_process(self, specific_file: Optional[str] = None) -> List[Path]:
        """Discovers .txt files in the input directory. (Logic moved from _discover_files_to_process)"""
        prefix = self.prefix  # Use instance prefix
        logger.info(f"{prefix}Discovering files... Specific: {specific_file}")

        files_to_process: List[Path] = []
        # If a specific file is requested, prioritize it
        if specific_file:
            potential_path = self.input_dir_path / specific_file
            if potential_path.is_file() and potential_path.suffix == ".txt":
                logger.info(f"{prefix}Processing specific file: {potential_path}")
                files_to_process.append(potential_path)
            else:
                logger.warning(
                    f"{prefix}Specific file '{specific_file}' not found or not a .txt file in "
                    f"{self.input_dir_path}. Processing all files."
                )
                # Fall through to process all if specific not found/valid

        # If no specific file was processed or if the specific file was invalid, process all
        if not files_to_process:
            all_files = sorted([p for p in self.input_dir_path.glob("*.txt") if p.is_file()])
            logger.info(f"{prefix}Found {len(all_files)} total .txt files. Processing all.")
            files_to_process = all_files

        # Log final list
        if files_to_process:
            logger.debug(f"{prefix}Final list of files to process: {[f.name for f in files_to_process]}")
        else:
            logger.info(f"{prefix}No .txt files found to process in {self.input_dir_path}")

        # Log metric for files discovered
        self.metrics_tracker.set_metric("pipeline", "files_discovered", len(files_to_process))
        return files_to_process

    # --- Helper methods for _process_single_file ---
    def _setup_file_io(
        self, file_path: Path
    ) -> Tuple[TextDataSource, ConversationMapStorage, SentenceAnalysisWriter, PipelinePaths]:
        """Sets up the IO handlers (DataSource, MapStorage, AnalysisWriter) for a given file."""
        prefix = self.prefix
        task_id = self.task_id
        logger.debug(f"{prefix}Setting up IO handlers for file: {file_path.name}")
        paths = generate_pipeline_paths(
            input_file=file_path,
            map_dir=self.map_dir_path,
            output_dir=self.output_dir_path,
            map_suffix=self.map_suffix,
            analysis_suffix=self.analysis_suffix,
            task_id=task_id
        )
        data_source = LocalTextDataSource(file_path)
        map_storage = LocalJsonlMapStorage(paths.map_file)
        analysis_writer = LocalJsonlAnalysisWriter(paths.analysis_file)
        return data_source, map_storage, analysis_writer, paths

    # Renamed from _read_segment_and_map_sentences
    async def _read_and_segment_sentences(self, data_source: TextDataSource) -> Tuple[int, List[str]]:
        """
        Reads text from data source, segments it, and returns sentences.
        (Map writing logic moved to _write_map_file)
        """
        prefix = f"{self.prefix}[File: {data_source.get_identifier()}] "  # Use file-specific prefix
        source_id = data_source.get_identifier()
        logger.info(f"{prefix}Reading and segmenting source '{source_id}'...")

        # --- Read Source Text ---
        try:
            text = await data_source.read_text()
        except Exception as e:
            logger.error(f"{prefix}Failed to read text from {source_id}: {e}", exc_info=True)
            self.metrics_tracker.increment_errors(f"{source_id}_read_error")
            raise

        # --- Segment Text ---
        sentences = segment_text(text)
        num_sentences = len(sentences)
        self.metrics_tracker.set_metric(source_id, "sentences_segmented", num_sentences)
        logger.debug(f"{prefix}Segmented into {num_sentences} sentences.")

        return num_sentences, sentences

    async def _write_map_file(self, sentences: List[str], map_storage: ConversationMapStorage, source_id: str):
        """Writes the conversation map file from segmented sentences."""
        prefix = f"{self.prefix}[File: {source_id}] "  # Use file-specific prefix
        storage_id = map_storage.get_identifier()
        num_sentences = len(sentences)
        logger.info(f"{prefix}Writing conversation map with {num_sentences} sentences to '{storage_id}'...")

        # --- Write Map Data ---
        initialized = False
        try:
            await map_storage.initialize()
            initialized = True
            for idx, sentence_text in enumerate(sentences):
                entry = {
                    "sentence_id": idx,
                    "sequence_order": idx,
                    "sentence": sentence_text
                }
                await map_storage.write_entry(entry)
            await map_storage.finalize()
            initialized = False  # Prevent finalizing again in finally block on success
            self.metrics_tracker.set_metric(source_id, "sentences_in_map", num_sentences)
            logger.info(f"{prefix}Conversation map successfully written to '{storage_id}'.")
        except Exception as e:
            logger.error(f"{prefix}Failed during map storage write operation for '{storage_id}': {e}", exc_info=True)
            self.metrics_tracker.increment_errors(f"{source_id}_map_write_error")
            raise  # Re-raise the write/init exception
        finally:
            # Attempt to finalize only if initialized but not successfully finalized
            if initialized:
                try:
                    await map_storage.finalize()
                except Exception as final_e:
                    logger.error(
                        f"{prefix}Failed to finalize map storage '{storage_id}' after write error: {final_e}",
                        exc_info=True
                    )
                    self.metrics_tracker.increment_errors(f"{source_id}_map_finalize_error")
                    # Original exception (if any) is already being raised

    def _build_contexts(self, sentences: List[str], file_name: str) -> List[Dict[str, str]]:
        """Builds analysis contexts using the AnalysisService."""
        # Call the analysis service method
        return self.analysis_service.build_contexts(sentences)

    async def _save_analysis_results(
        self,
        analysis_results: List[Dict[str, Any]],
        analysis_writer: SentenceAnalysisWriter,
        file_name: str,
        prefix: str  # Pass prefix for logging consistency
    ):
        """
        Saves analysis results via JSONL and Graph DB.
        Handles writer initialization/finalization and metric updates.
        (Logic moved from _analyze_and_save_results)
        """
        writer_id = analysis_writer.get_identifier()
        logger.info(f"{prefix}Saving {len(analysis_results)} results via {writer_id} and graph DB...")
        initialized = False
        num_results_written = 0
        num_results_failed = 0

        try:
            # 1. Initialize the writer
            await analysis_writer.initialize()
            initialized = True
            logger.debug(f"{prefix}Initialized analysis writer: {writer_id}")

            # 2. Process and save each result
            num_results_total = len(analysis_results)
            for i, result in enumerate(analysis_results):
                # Add filename if not already present (though it should be)
                if isinstance(result, dict):
                    result.setdefault('filename', file_name)  # Ensure filename exists
                    sentence_id = result.get('sentence_id', 'N/A')
                    logger.debug(
                        f"{prefix}Processing result {i+1}/{num_results_total} (ID: {sentence_id}) for saving..."
                    )

                    # --- Check for error flag ---
                    if result.get("error") is True:
                        logger.warning(
                            f"{prefix}Skipping save for result ID {sentence_id} due to analysis error: "
                            f"{result.get('error_message', 'Unknown')}"
                        )
                        num_results_failed += 1  # Count as failed save
                        # Increment a specific metric for analysis errors encountered during saving phase
                        self.metrics_tracker.increment_errors(f"{file_name}_analysis_error_skipped_save")
                        continue  # Skip to the next result
                    # --- End error check ---

                    # Call internal handlers for saving (only if not an error result)
                    jsonl_success = await self._handle_jsonl_write(result, analysis_writer, prefix)
                    graph_success = await self._handle_graph_save(result, prefix)  # DEBUG: Re-enabled graph save
                    # graph_success = True # DEBUG: Assume graph save succeeded

                    if jsonl_success and graph_success:
                        num_results_written += 1
                        # Increment results_processed metric only on successful save to both targets
                        self.metrics_tracker.increment_results_processed(file_name)
                    else:
                        num_results_failed += 1
                        logger.warning(
                            f"{prefix}Result ID {sentence_id} encountered an error during saving "
                            f"(JSONL: {jsonl_success}, Graph: {graph_success})"
                        )
                        self.metrics_tracker.increment_errors(f"{file_name}_result_save_failure")
                else:
                    logger.warning(
                        f"{prefix}Invalid non-dictionary item received for saving at index {i}: "
                        f"{type(result)}. Skipping save."
                    )
                    num_results_failed += 1  # Count as failed save
                    self.metrics_tracker.increment_errors(f"{file_name}_invalid_analysis_result_type")

            logger.info(
                f"{prefix}Finished saving results. Total: {num_results_total}, "
                f"Written: {num_results_written}, Failed Saves: {num_results_failed}"
            )
            self.metrics_tracker.set_metric(
                file_name, "results_saved_successfully", num_results_written
            )  # Renamed metric key
            self.metrics_tracker.set_metric(file_name, "results_save_failures", num_results_failed)

        except Exception as e:
            logger.error(f"{prefix}Error during saving results for {file_name}: {e}", exc_info=True)
            self.metrics_tracker.increment_errors(f"{file_name}_save_results_error")
            raise  # Re-raise the error to be caught by _analyze_and_save_results or _process_single_file
        finally:
            # Ensure writer is finalized even if saving failed
            if initialized:
                try:
                    logger.debug(f"{prefix}>>> ATTEMPTING TO FINALIZE analysis writer {writer_id}...")
                    await analysis_writer.finalize()
                    logger.debug(f"{prefix}>>> SUCCESSFULLY FINALIZED analysis writer {writer_id}.")
                    logger.debug(f"{prefix}Analysis writer {writer_id} finalized.")
                except Exception as final_e:
                    logger.error(
                        f"{prefix}>>> FAILED TO FINALIZE analysis writer {writer_id} after saving: {final_e}",
                        exc_info=True
                    )

    # Renamed and moved from standalone _orchestrate_analysis_and_writing
    async def _analyze_and_save_results(
        self,
        sentences: List[str],
        contexts: List[Dict[str, str]],
        analysis_writer: SentenceAnalysisWriter,
        file_name: str  # Renamed from input_file_name
    ):
        """
        Analyzes sentences and orchestrates the saving of results.
        (Separated from saving logic)
        """
        prefix = f"{self.prefix}[File: {file_name}] "  # Add file context to prefix
        logger.info(f"{prefix}Analyzing {len(sentences)} sentences...")
        analysis_results = []
        try:
            # 1. Run analysis
            logger.debug(f"{prefix}Calling analysis_service.analyze_sentences...")
            analysis_results = await self.analysis_service.analyze_sentences(
                sentences, contexts, task_id=self.task_id  # Pass orchestrator task_id
            )
            logger.debug(f"{prefix}Received {len(analysis_results)} analysis results.")

            # Add filename to results immediately after analysis
            for result in analysis_results:
                if isinstance(result, dict):
                    result['filename'] = file_name

            # 2. Orchestrate Saving
            await self._save_analysis_results(analysis_results, analysis_writer, file_name, prefix)

        except Exception as e:
            # Original simple error handling
            logger.error(
                f"{prefix}Error during sentence analysis or save orchestration for {file_name}: {e}",
                exc_info=True
            )
            self.metrics_tracker.increment_errors(
                f"{file_name}_analysis_or_save_orchestration_error"
            )  # Original metric key
            raise  # Re-raise the error to be caught by _process_single_file

        # Note: Finalization of analysis_writer happens within _save_analysis_results

    # --- End Helper methods ---

    async def _process_single_file(self, file_path: Path):
        """Processes a single input file using helper methods."""
        prefix = f"{self.prefix}[File: {file_path.name}] "  # Add file context early
        file_name = file_path.name
        logger.info(f"{prefix}Starting processing...")
        self.metrics_tracker.start_file_timer(file_name)

        try:
            # 1. Setup IO
            data_source, map_storage, analysis_writer, _ = self._setup_file_io(file_path)

            # 2. Read and Segment Sentences
            num_sentences, sentences = await self._read_and_segment_sentences(data_source)

            # 3. Write Map File (if sentences exist)
            if num_sentences > 0:
                await self._write_map_file(sentences, map_storage, data_source.get_identifier())
            else:
                # Handle case of 0 sentences (ensure map file is appropriately handled)
                logger.warning(f"{prefix}Skipping map file write for source '{file_name}' as it contains no sentences.")
                try:
                    # Explicitly initialize/finalize to truncate/create empty map file if needed by storage impl.
                    await map_storage.initialize()
                    await map_storage.finalize()
                except Exception as map_init_e:
                    logger.error(
                        f"{prefix}Error initializing/finalizing empty map storage "
                        f"'{map_storage.get_identifier()}': {map_init_e}"
                    )
                # --- Early Exit for 0 sentences ---
                # Count as processed (even if empty)
                self.metrics_tracker.increment_files_processed(1)
                self.metrics_tracker.set_metric(file_name, "sentences_processed", 0)
                logger.info(f"{prefix}Finished processing empty file.")
                return  # Stop processing this file

            # 4. Build Contexts (Only if sentences existed)
            contexts = self._build_contexts(sentences, file_name)

            # 5. Run Analysis and Saving
            await self._analyze_and_save_results(sentences, contexts, analysis_writer, file_name)

            # --- Success Metrics ---
            # Metrics are now updated within _analyze_and_save_results
            self.metrics_tracker.increment_files_processed(1)  # Increment file processed count
            self.metrics_tracker.set_metric(
                file_name, "sentences_processed", num_sentences
            )  # Keep this sentence count metric
            logger.info(f"{prefix}Successfully finished processing.")

        except Exception as e:
            logger.error(f"{prefix}>>> CAUGHT EXCEPTION IN _process_single_file: {type(e).__name__}")
            logger.error(f"{prefix}Error processing file {file_name}: {e}", exc_info=True)
            self.metrics_tracker.increment_files_failed(1)
            raise
        finally:
            self.metrics_tracker.stop_file_timer(file_name)  # Keep timer stop here

    async def _process_files_concurrently(self, files_to_process: List[Path]):
        """Runs _process_single_file concurrently for all files using a semaphore.
        (Logic moved from _run_processing_tasks)"""
        prefix = self.prefix
        logger.info(
            f"{prefix}Processing {len(files_to_process)} files concurrently with limit "
            f"{self.num_concurrent_files}..."
        )
        semaphore = asyncio.Semaphore(self.num_concurrent_files)
        tasks = []
        for file_path in files_to_process:
            # Define the coroutine to be run with semaphore protection
            async def process_with_semaphore(fp: Path):
                # Add file-specific prefix for clearer logs within concurrent tasks
                file_prefix = f"{prefix}[File: {fp.name}] "
                logger.debug(f"{file_prefix}Acquiring semaphore...")
                async with semaphore:
                    logger.debug(f"{file_prefix}Semaphore acquired. Starting processing.")
                    try:
                        # Call the main processing method for the single file
                        await self._process_single_file(fp)
                        logger.debug(f"{file_prefix}Processing finished successfully.")
                        return fp  # Indicate success by returning path
                    except Exception as e:
                        # Error should have been logged within _process_single_file
                        logger.error(f"{file_prefix}Processing failed (exception returned to gather). Error: {e}")
                        return e  # Indicate failure by returning exception
                    finally:
                        logger.debug(f"{file_prefix}Releasing semaphore.")

            # Create and schedule the task
            tasks.append(asyncio.create_task(process_with_semaphore(file_path), name=f"process_{file_path.name}"))

        # Wait for all tasks to complete and collect results
        results = await asyncio.gather(*tasks)
        logger.info(f"{prefix}Finished concurrent processing of all scheduled files.")
        return results  # List containing file paths (success) or Exceptions (failure)

    def _log_summary(self, results: List[Union[Exception, Any]], files_to_process: List[Path]):
        """Logs the final processing summary. (Logic moved from _log_processing_summary)"""
        prefix = self.prefix
        # Use the initial list count for total, as results might be shorter if cancelled
        total_files_attempted = len(files_to_process)
        # Success is indicated by returning the Path object, failure by Exception
        success_count = sum(1 for r in results if isinstance(r, Path))
        # Failure count derived from results list length
        failure_count = sum(1 for r in results if isinstance(r, Exception))
        # Files that might not have a result if gather was interrupted (though unlikely here)
        # For simplicity, we assume results list matches files_to_process length or errors indicate non-completion.

        # Update global metrics based on final counts
        self.metrics_tracker.set_metric("pipeline", "files_processed_successfully", success_count)
        self.metrics_tracker.set_metric("pipeline", "files_failed", failure_count)

        logger.info(f"{prefix}--- Processing Summary ---")
        logger.info(f"{prefix}Total files initially discovered: {total_files_attempted}")
        logger.info(f"{prefix}Successfully processed files: {success_count}")
        logger.info(f"{prefix}Failed files: {failure_count}")

        if failure_count > 0:
            # Map results back to filenames for logging failed files
            failed_map = {res: fp.name for fp, res in zip(files_to_process, results) if isinstance(res, Exception)}
            failed_files_str = ", ".join(failed_map.values())
            logger.warning(f"{prefix}Failed files list: [{failed_files_str}]")
            # Optionally log the specific errors
            # for err, fname in failed_map.items():
            #     logger.debug(f"{prefix}  - {fname}: {type(err).__name__}: {err}")

        # Log overall metrics from the tracker
        try:
            summary_metrics = self.metrics_tracker.get_summary()
            logger.info(f"{prefix}Overall Metrics:")
            for key, value in summary_metrics.items():
                logger.info(f"{prefix}  - {key}: {value}")
        except Exception as e:
            logger.error(f"{prefix}Failed to retrieve or log metrics summary: {e}")

    async def _run_verification(self, files_to_process: List[Path]):
        """
        Runs output completeness verification for all processed files.
        (Logic moved from standalone verify_output_completeness)
        """
        prefix = self.prefix
        logger.info(f"{prefix}Starting output verification...")
        verification_results = []
        total_missing_overall = 0
        total_expected_overall = 0
        verification_errors = 0

        if not files_to_process:
            logger.info(f"{prefix}No files were processed, skipping verification.")
            # Set metrics to 0 if no files were processed
            self.metrics_tracker.set_metric("pipeline", "verification_total_missing", 0)
            self.metrics_tracker.set_metric("pipeline", "verification_errors", 0)
            return

        for file_path in files_to_process:
            file_prefix = f"{prefix}[File: {file_path.name}] "  # File specific prefix
            map_storage = None
            analysis_reader = None
            verification_result = None
            # Instantiate IO objects for verification
            try:
                # Use orchestrator paths and suffixes
                map_file = self.map_dir_path / f"{file_path.stem}{self.map_suffix}"
                map_storage = LocalJsonlMapStorage(map_file)

                analysis_file = self.output_dir_path / f"{file_path.stem}{self.analysis_suffix}"
                # We only need the reading capability of the writer protocol here
                analysis_reader = LocalJsonlAnalysisWriter(analysis_file)

                # Call the core verification logic (which is now part of this method)
                logger.debug(
                    f"{file_prefix}Verifying completeness between map "
                    f"'{map_storage.get_identifier()}' and analysis "
                    f"'{analysis_reader.get_identifier()}'..."
                )

                expected_ids: Set[int] = set()
                actual_ids: Set[int] = set()
                error_msg: Optional[str] = None

                # Process Map Storage
                try:
                    expected_ids = await map_storage.read_sentence_ids()
                except Exception as map_read_e:
                    logger.error(
                        f"{file_prefix}Failed to read IDs from map storage "
                        f"{map_storage.get_identifier()}: {map_read_e}",
                        exc_info=True
                    )
                    error_msg = f"Error reading map storage: {type(map_read_e).__name__}"
                    # Cannot continue verification for this file if map is unreadable
                    verification_result = {
                        "total_expected": 0, "total_actual": 0, "total_missing": 0,
                        "missing_ids": [], "error": error_msg
                    }
                    verification_errors += 1  # Count this as a verification error
                    # Skip to next file using continue
                    verification_results.append(verification_result)
                    logger.warning(f"{file_prefix}Verification check failed: {error_msg}")
                    continue

                # Process Analysis Storage
                try:
                    actual_ids = await analysis_reader.read_analysis_ids()
                except Exception as analysis_read_e:
                    logger.error(
                        f"{file_prefix}Failed to read IDs from analysis storage "
                        f"{analysis_reader.get_identifier()}: {analysis_read_e}",
                        exc_info=True
                    )
                    error_msg = f"Error reading analysis storage: {type(analysis_read_e).__name__}"
                    # Proceed with calculation, actual_ids will be empty

                # Calculate differences
                missing_ids_set = expected_ids - actual_ids
                total_expected = len(expected_ids)
                total_actual = len(actual_ids)
                total_missing = len(missing_ids_set)

                logger.debug(
                    f"{file_prefix}Completeness check: Expected={total_expected}, "
                    f"Actual={total_actual}, Missing={total_missing}"
                )

                verification_result = {
                    "total_expected": total_expected,
                    "total_actual": total_actual,
                    "total_missing": total_missing,
                    "missing_ids": sorted(list(missing_ids_set)),
                    "error": error_msg
                }

            except Exception as e:  # Catch errors during IO instantiation
                logger.error(
                    f"{file_prefix}Unexpected error during verification setup for "
                    f"{file_path.name}: {e}",
                    exc_info=True
                )
                # Create a default error result if setup fails
                verification_result = {
                    "total_expected": 0, "total_actual": 0, "total_missing": 0,
                    "missing_ids": [], "error": f"Verification setup error: {type(e).__name__}"
                }
                verification_errors += 1

            verification_results.append(verification_result)

            # Log individual file verification details and update aggregate counts
            if verification_result.get("error"):
                # Error already logged during exception handling for this file
                # No need to increment verification_errors again here if already done
                # But we need to make sure total_expected isn't counted if there was an error
                logger.warning(f"{file_prefix}Verification check: ERROR - {verification_result['error']}")
            else:
                # Only add to totals if there was no error reading map/analysis for this file
                total_expected_overall += verification_result["total_expected"]
                total_missing_overall += verification_result["total_missing"]
                if verification_result["total_missing"] > 0:
                    logger.warning(
                        f"{file_prefix}Verification check: MISSING "
                        f"{verification_result['total_missing']}/{verification_result['total_expected']} "
                        f"sentences. IDs: {str(verification_result['missing_ids'][:10])[:100]}..."
                    )
                else:
                    logger.info(
                        f"{file_prefix}Verification check: OK "
                        f"({verification_result['total_actual']}/{verification_result['total_expected']} "
                        f"sentences found)."
                    )

        # Log overall verification summary using accumulated totals
        logger.info(f"{prefix}--- Verification Summary ---")
        logger.info(f"{prefix}Total Files Verified: {len(files_to_process)}")
        logger.info(f"{prefix}Verification Errors (file setup/read): {verification_errors}")
        # Ensure division by zero is avoided if all files had verification errors
        expected_log = total_expected_overall if total_expected_overall > 0 else 'N/A'
        logger.info(f"{prefix}Total Missing Sentences (across valid files): {total_missing_overall}/{expected_log}")

        # Update overall pipeline metrics using orchestrator tracker
        self.metrics_tracker.set_metric("pipeline", "verification_total_missing", total_missing_overall)
        self.metrics_tracker.set_metric("pipeline", "verification_errors", verification_errors)

    async def execute(self, specific_file: Optional[str] = None):
        """Runs the entire pipeline orchestration."""
        start_time = time.time()
        self.metrics_tracker.reset()  # Ensure reset at start of execution
        self.metrics_tracker.start_pipeline_timer()
        logger.info(f"{self.prefix}Pipeline execution started.")

        files_to_process = []
        results = []
        try:
            files_to_process = self._discover_files_to_process(specific_file)
            if not files_to_process:
                logger.warning(f"{self.prefix}No files found to process. Exiting.")
                return

            results = await self._process_files_concurrently(files_to_process)

            # --- Run Verification Step ---
            # Only run verification if concurrent processing seems to have finished
            # (We might refine error handling later, but for now, run if no exception bubbled up)
            logger.info(f"{self.prefix}Concurrent processing finished. Proceeding to verification...")
            await self._run_verification(files_to_process)
            logger.info(f"{self.prefix}Verification finished.")
            # --- End Verification Step ---

        except Exception as e:
            logger.critical(f"{self.prefix}Critical error during pipeline execution: {e}", exc_info=True)
            # Ensure metrics reflect potential failures even if loop didn't finish
        finally:
            self.metrics_tracker.stop_pipeline_timer()
            # Log summary now includes verification metrics potentially set in _run_verification
            self._log_summary(results, files_to_process)
            end_time = time.time()
            logger.info(f"{self.prefix}Pipeline execution finished in {end_time - start_time:.2f} seconds.")

# --- End Orchestrator Class ---

# Remove standalone create_conversation_map function
# async def create_conversation_map(...): ...

# --- Result Handling Helpers (Refactored from _result_writer) ---
# THESE ARE NOW METHODS OF PipelineOrchestrator - REMOVE STANDALONE DEFINITIONS
# async def _handle_jsonl_write(...): ...
# async def _handle_graph_save(...): ...
# --- End Result Handling Helpers ---

# Remove standalone verify_output_completeness function
# async def verify_output_completeness(...): ...

# --- Helper Functions for process_file Refactoring --- #
# ... (Redundant helpers remain removed) ...

# --- End Helper Functions for process_file --- #


async def run_pipeline(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    map_dir: Optional[Union[str, Path]] = None,
    specific_file: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,  # Changed name for clarity
    task_id: Optional[str] = None  # Add task_id parameter
):
    """
    Sets up and executes the analysis pipeline using PipelineOrchestrator.

    Initializes the orchestrator with paths and config, then runs its execute method.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}--- Starting Pipeline Run --- Task ID: {task_id if task_id else 'N/A'}")
    try:
        orchestrator = PipelineOrchestrator(
            input_dir=input_dir,
            output_dir=output_dir,
            map_dir=map_dir,
            config_dict=config_dict,
            task_id=task_id
        )
        await orchestrator.execute(specific_file=specific_file)
    except Exception as e:
        # Catch critical setup errors from orchestrator init
        logger.critical(f"{prefix}Pipeline setup failed: {e}", exc_info=True)
        # Potentially re-raise or handle differently depending on desired top-level behavior
        raise
    finally:
        logger.info(f"{prefix}--- Pipeline Run Finished --- Task ID: {task_id if task_id else 'N/A'}")


# --- Refactoring Helpers for analyze_specific_sentences ---

async def _prepare_data_for_specific_analysis(
    map_storage: ConversationMapStorage,
    sentence_ids: List[int],
    prefix: str  # Pass prefix for logging
) -> Tuple[List[str], List[int], List[str]]:
    """
    Reads map, validates IDs, and prepares target sentences/indices and full context list.
    """
    map_id = map_storage.get_identifier()
    logger.info(f"{prefix}Reading map entries from: {map_id}")
    try:
        all_entries = await map_storage.read_all_entries()
        if not all_entries:
            logger.error(f"{prefix}Map storage '{map_id}' is empty or could not be read.")
            raise ValueError(f"Map storage '{map_id}' is empty or unreadable.")
    except Exception as read_e:
        logger.error(f"{prefix}Error reading map entries from {map_id}: {read_e}", exc_info=True)
        raise  # Re-raise the storage access error

    # --- Prepare Sentences and Check IDs ---
    all_sentences_map: Dict[int, str] = {}
    all_entries.sort(key=lambda x: x.get('sequence_order', -1))

    max_id = -1
    for entry in all_entries:
        s_id = entry.get('sentence_id')
        text = entry.get('sentence')
        if isinstance(s_id, int) and text is not None:
            all_sentences_map[s_id] = text
            max_id = max(max_id, s_id)
        else:
            logger.warning(f"{prefix}Skipping invalid/incomplete entry in map {map_id}: {str(entry)[:100]}...")

    if max_id == -1:
        logger.warning(f"{prefix}No valid sentence entries found in map {map_id}. Cannot perform specific analysis.")
        raise ValueError(f"No valid sentence entries found in map '{map_id}'.")

    full_sentence_list_for_context = ["" for _ in range(max_id + 1)]
    for s_id, text in all_sentences_map.items():
        if 0 <= s_id <= max_id:
            full_sentence_list_for_context[s_id] = text

    target_sentences: List[str] = []
    target_indices: List[int] = []  # Store original indices for context building
    missing_ids = []

    target_id_set = set(sentence_ids)
    for target_id in target_id_set:
        if target_id in all_sentences_map:
            target_sentences.append(all_sentences_map[target_id])
            target_indices.append(target_id)  # Store the original index
        else:
            missing_ids.append(target_id)

    if missing_ids:
        logger.error(f"{prefix}Requested sentence IDs not found in map {map_id}: {sorted(missing_ids)}")
        raise ValueError(f"Sentence IDs not found in map '{map_id}': {sorted(missing_ids)}")

    if not target_sentences:
        # This case should be unlikely if missing_ids didn't raise, but check anyway
        logger.warning(f"{prefix}No target sentences identified from IDs {sentence_ids} in map {map_id}.")
        raise ValueError(f"No target sentences identified for IDs in map '{map_id}'.")

    # Sort target_sentences and target_indices together based on original index (target_indices)
    # This ensures the order matches the analysis service input if it relies on sequence
    paired_sorted = sorted(zip(target_indices, target_sentences))
    target_indices = [pair[0] for pair in paired_sorted]
    target_sentences = [pair[1] for pair in paired_sorted]

    logger.debug(f"{prefix}Prepared {len(target_sentences)} target sentences with indices: {target_indices}")
    return target_sentences, target_indices, full_sentence_list_for_context


def _build_contexts_for_specific_analysis(
    full_sentence_list_for_context: List[str],
    target_indices: List[int],
    analysis_service: AnalysisService,
    prefix: str
) -> List[Dict[str, Any]]:  # Return type is List[ContextDict]
    """
    Builds contexts for all sentences and extracts contexts for target indices.
    """
    logger.info(f"{prefix}Building contexts for {len(target_indices)} specific sentences...")
    try:
        # Build contexts for the *entire* list first
        all_contexts_dict = analysis_service.context_builder.build_all_contexts(full_sentence_list_for_context)
        # Extract contexts only for the target sentences using their original indices
        target_contexts = [all_contexts_dict.get(idx, {}) for idx in target_indices]

        # Basic validation
        if len(target_contexts) != len(target_indices):
            logger.error(
                f"{prefix}Context count ({len(target_contexts)}) mismatch with "
                f"target index count ({len(target_indices)})."
            )
            # This indicates a logic error in context building or extraction
            raise RuntimeError("Internal error: Context and target index count mismatch.")

        logger.debug(f"{prefix}Successfully built {len(target_contexts)} target contexts.")
        return target_contexts

    except Exception as e:
        logger.error(f"{prefix}Failed to build contexts for specific sentences: {e}", exc_info=True)
        raise  # Re-raise context building errors


def _post_process_specific_results(
    analysis_results: List[Dict[str, Any]],
    target_indices: List[int],
    prefix: str
) -> List[Dict[str, Any]]:
    """Remaps sentence_id and sequence_order in results based on original indices."""
    logger.debug(f"{prefix}Post-processing {len(analysis_results)} analysis results...")
    final_results = []
    if len(analysis_results) != len(target_indices):
        logger.warning(
            f"{prefix}Result count ({len(analysis_results)}) mismatch with "
            f"target index count ({len(target_indices)}). "
            f"Results might be incomplete or misaligned."
        )
        # Decide how to handle this - return partial, raise error? For now, process what we have.

    for i, result in enumerate(analysis_results):
        # Protect against index out of bounds if counts mismatch
        if i < len(target_indices):
            original_sentence_id = target_indices[i]
            if isinstance(result, dict):  # Ensure result is a dict before modifying
                result['sentence_id'] = original_sentence_id
                result['sequence_order'] = original_sentence_id
                final_results.append(result)
            else:
                logger.warning(f"{prefix}Skipping post-processing for non-dict result at index {i}: {type(result)}")
        else:
            logger.warning(f"{prefix}Extra result found at index {i} beyond the number of target indices. Skipping.")

    logger.info(f"{prefix}Finished post-processing. Returning {len(final_results)} results.")
    return final_results

# --- End Refactoring Helpers for analyze_specific_sentences ---


# --- analyze_specific_sentences function ---
# Refactored to use helper functions
async def analyze_specific_sentences(
    map_storage: ConversationMapStorage,  # New: Inject map storage
    sentence_ids: List[int],
    analysis_service: AnalysisService,  # Inject AnalysisService
    task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Analyzes specific sentences using ConversationMapStorage by orchestrating helper functions.
    """
    prefix = _log_prefix(task_id)
    map_id = map_storage.get_identifier()
    logger.info(f"{prefix}Starting analysis for specific sentences using map: {map_id}, IDs: {sentence_ids}")

    try:
        # 1. Prepare Data (Read map, find sentences/indices, build full context list)
        target_sentences, target_indices, full_context_list = await _prepare_data_for_specific_analysis(
            map_storage, sentence_ids, prefix
        )

        # 2. Build Contexts (For the specific target sentences)
        target_contexts = _build_contexts_for_specific_analysis(
            full_context_list, target_indices, analysis_service, prefix
        )

        # 3. Analyze Specific Sentences
        logger.info(f"{prefix}Analyzing {len(target_sentences)} specific sentences...")
        analysis_results = await analysis_service.analyze_sentences(
            target_sentences, target_contexts, task_id=task_id
        )

        # 4. Post-Process Results (Remap IDs)
        final_results = _post_process_specific_results(
            analysis_results, target_indices, prefix
        )

        logger.info(
            f"{prefix}Finished specific sentence analysis for map {map_id}. "
            f"Returning {len(final_results)} results."
        )
        return final_results

    except ValueError as ve:
        # Catch specific validation errors from helpers (e.g., missing IDs, empty map)
        logger.error(f"{prefix}Specific analysis failed due to validation error: {ve}")
        raise  # Re-raise ValueError
    except Exception as e:
        # Catch other unexpected errors during orchestration
        logger.error(f"{prefix}Error during specific sentence analysis orchestration: {e}", exc_info=True)
        raise  # Re-raise other exceptions
