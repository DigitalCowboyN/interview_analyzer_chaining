"""
pipeline.py

Defines the core pipeline functions for processing text files.

Main workflow orchestrated by `run_pipeline`:
    - Iterates through input text files.
    - For each file, `process_file` is called:
        - `create_conversation_map`: Segments text, creates a map file (.jsonl)
          listing sentences with IDs.
        - Injected `AnalysisService`: Builds context for sentences and analyzes them.
        - `_result_writer`: Asynchronously writes analysis results to an output file (.jsonl).
    - `verify_output_completeness`: Checks if analysis outputs match map files.

Also includes `analyze_specific_sentences` for targeted re-analysis.

Dependencies:
    - `src.services.analysis_service.AnalysisService`: Performs context building and analysis.
    - `src.agents.*`: Underlying components used by AnalysisService (indirect dependency).
    - `src.utils.helpers.append_json_line`: Writes JSON Lines data.
    - `src.utils.text_processing.segment_text`: Splits text into sentences.
    - `src.utils.logger.get_logger`: Centralized logging.
    - `src.utils.metrics.MetricsTracker`: Tracks operational metrics.
"""

from pathlib import Path
import asyncio
# SentenceAnalyzer is used indirectly via AnalysisService
# from src.agents.sentence_analyzer import SentenceAnalyzer
from src.utils.logger import get_logger
# ContextBuilder is used indirectly via AnalysisService
# from src.agents.context_builder import ContextBuilder
from src.utils.metrics import MetricsTracker
from src.utils.text_processing import segment_text
# Remove commented-out import: import json
# Import necessary types
from typing import Dict, Any, List, Tuple, Set, Optional, Union
from src.services.analysis_service import AnalysisService
import time
# Import agent classes needed for instantiation
from src.agents.context_builder import ContextBuilder
from src.agents.sentence_analyzer import SentenceAnalyzer
# Import the new path helper
from src.utils.path_helpers import generate_pipeline_paths, PipelinePaths
# Import dataclass if not already (it's part of typing)
from dataclasses import dataclass
# Import the new protocols
from src.io.protocols import TextDataSource, ConversationMapStorage, SentenceAnalysisWriter
# Import the local storage implementations
from src.io.local_storage import (
    LocalTextDataSource,
    LocalJsonlMapStorage,
    LocalJsonlAnalysisWriter
)

logger = get_logger()

# Define a helper for log prefixing
def _log_prefix(task_id: Optional[str] = None) -> str:
    return f"[Task {task_id}] " if task_id else ""

async def create_conversation_map(
    data_source: TextDataSource, 
    map_storage: ConversationMapStorage, 
    task_id: Optional[str] = None
) -> Tuple[int, List[str]]:
    """
    Creates conversation map data using IO protocols and returns sentence count and sentences.

    Reads text from the data_source, segments it into sentences, and writes map entries 
    ({sentence_id, sequence_order, sentence}) to the map_storage.

    Args:
        data_source (TextDataSource): Protocol object for reading input text.
        map_storage (ConversationMapStorage): Protocol object for writing map data.
        task_id (Optional[str]): Optional identifier for logging task context.

    Returns:
        Tuple[int, List[str]]: The number of sentences found and the list of sentence strings.

    Raises:
        # Note: Specific exceptions depend on the protocol implementations.
        # FileNotFoundError might be raised by LocalTextDataSource.read_text()
        # OSError or other IOErrors might be raised by map_storage methods.
        Exception: Propagates exceptions from data_source or map_storage operations.
    """
    prefix = _log_prefix(task_id)
    source_id = data_source.get_identifier()
    storage_id = map_storage.get_identifier()
    logger.info(f"{prefix}Creating conversation map from '{source_id}' to '{storage_id}'")

    # --- Read Source Text ---
    # Error handling (like FileNotFoundError) is now within data_source.read_text()
    try:
        text = await data_source.read_text() 
    except Exception as e:
        logger.error(f"{prefix}Failed to read text from {source_id}: {e}", exc_info=True)
        raise # Re-raise read errors

    # --- Segment Text ---
    sentences = segment_text(text)
    num_sentences = len(sentences)
    
    # --- Handle Empty Input ---
    if num_sentences == 0:
        logger.warning(f"{prefix}Source '{source_id}' contains no sentences after segmentation. Map storage '{storage_id}' will be empty or truncated")
        # Initialize and finalize to potentially create/truncate the map storage
        try:
            await map_storage.initialize()
            await map_storage.finalize()
        except Exception as e:
            logger.error(f"{prefix}Failed to initialize/finalize empty map storage '{storage_id}': {e}", exc_info=True)
            raise # Re-raise storage init/finalize errors
        return 0, [] # Return 0 sentences and empty list

    # --- Write Map Data ---
    try:
        await map_storage.initialize() # Initialize the storage for writing
        for idx, sentence_text in enumerate(sentences):
            entry = {
                "sentence_id": idx,
                "sequence_order": idx,
                "sentence": sentence_text
            }
            await map_storage.write_entry(entry)
        await map_storage.finalize() # Finalize writing
    except Exception as e:
        # Catch errors during initialize, write_entry, or finalize
        logger.error(f"{prefix}Failed during map storage operation for '{storage_id}': {e}", exc_info=True)
        # Attempt to finalize even if write failed, though it might also fail
        try:
            await map_storage.finalize() 
        except Exception as final_e:
             logger.error(f"{prefix}Failed to finalize map storage '{storage_id}' after write error: {final_e}", exc_info=True)
        raise # Re-raise the original exception 

    logger.info(f"{prefix}Conversation map created via storage '{storage_id}' with {num_sentences} sentences.")
    return num_sentences, sentences

async def _result_writer(
    analysis_writer: SentenceAnalysisWriter, 
    results_queue: asyncio.Queue, 
    metrics_tracker: MetricsTracker,
    task_id: Optional[str] = None
):
    """
    Consumes analysis results from a queue and writes them using a SentenceAnalysisWriter.

    Runs asynchronously, fetching results from the queue until a `None` sentinel
    is received. Initializes the writer, writes results using `write_result`,
    handles potential errors, logs errors, updates metrics, and ensures the writer
    is finalized.

    Args:
        analysis_writer (SentenceAnalysisWriter): Protocol object for writing analysis results.
        results_queue (asyncio.Queue): Queue to get analysis result dictionaries from.
                                       Expects `None` as a termination signal.
        metrics_tracker (MetricsTracker): Instance used to track errors during writing.
        task_id (Optional[str]): Optional identifier for logging task context.
    """
    prefix = _log_prefix(task_id)
    writer_id = analysis_writer.get_identifier()
    logger.debug(f"{prefix}Result writer starting for: {writer_id}")
    results_written = 0
    initialized = False
    
    try:
        # Initialize the writer before starting the loop
        await analysis_writer.initialize()
        initialized = True
        logger.debug(f"{prefix}Initialized writer: {writer_id}")

        while True:
            try:
                logger.debug(f"{prefix}Writer waiting for item...")
                result = await results_queue.get()
                logger.debug(f"{prefix}Writer received item: {type(result)}")
                if result is None: # Sentinel value indicates completion
                    logger.info(f"{prefix}Writer received sentinel. Exiting loop for {writer_id}.")
                    results_queue.task_done() # Mark sentinel as processed
                    break

                try:
                    sentence_id = result.get('sentence_id', 'N/A')
                    logger.debug(f"{prefix}Writer attempting to write result ID: {sentence_id} to {writer_id}")
                    # Use the analysis_writer protocol method
                    await analysis_writer.write_result(result)
                    results_written += 1
                    logger.debug(f"{prefix}Writer successfully wrote result ID: {sentence_id} to {writer_id}")
                    # Optionally track write success metric
                    # metrics_tracker.increment_write_success()
                except Exception as write_e:
                    # Log error details including sentence ID if available
                    sentence_id_info = f"result {result.get('sentence_id', 'N/A')}" if isinstance(result, dict) else "result (unknown ID)"
                    logger.error(f"{prefix}Writer failed writing {sentence_id_info} to {writer_id}: {write_e}", exc_info=True)
                    metrics_tracker.increment_errors() # Use passed-in tracker
                    # Decide if we should break the loop on write error? 
                    # For now, continue processing other items in the queue.
                finally:
                    results_queue.task_done() # Mark result as processed

            except asyncio.CancelledError:
                logger.info(f"{prefix}Writer for {writer_id} cancelled during queue processing.")
                raise # Re-raise CancelledError to be handled by outer block/caller
            except Exception as queue_e:
                logger.critical(f"{prefix}Critical error getting item from queue for {writer_id}: {queue_e}", exc_info=True)
                break # Exit loop on critical queue error
                
    except asyncio.CancelledError:
        # Handle cancellation requested before or during initialization, or re-raised from inner loop
        logger.info(f"{prefix}Result writer task for {writer_id} cancelled.")
        # The finally block will handle finalization.
        raise # Re-raise so the caller knows cancellation happened
    except Exception as init_e:
        # Handle errors during initialization
        logger.critical(f"{prefix}Critical error initializing writer {writer_id}: {init_e}", exc_info=True)
        # The finally block will attempt finalization if needed.
        raise # Re-raise critical initialization error
    finally:
        # Ensure finalization happens regardless of how the try block exited
        if initialized:
            try:
                logger.debug(f"{prefix}Finalizing writer {writer_id}...")
                await analysis_writer.finalize()
                logger.info(f"{prefix}Result writer finalized for: {writer_id}. Total written: {results_written}")
            except Exception as final_e:
                logger.error(f"{prefix}Error finalizing writer {writer_id}: {final_e}", exc_info=True)
                # Potentially raise this? Depends on whether finalize failure is critical
        else:
             # Log if finalization is skipped because initialization failed/was skipped
             logger.debug(f"{prefix}Writer {writer_id} finalization skipped (was not initialized).")

async def verify_output_completeness(
    map_storage: ConversationMapStorage, 
    analysis_writer: SentenceAnalysisWriter,
    task_id: Optional[str] = None # Added task_id for logging consistency
) -> Dict[str, Any]:
    """
    Compares map data and analysis data using IO protocols to verify completeness.

    Reads sentence IDs asynchronously from the map_storage (expected) and 
    analysis_writer (actual) and calculates metrics on how many are missing from 
    the analysis output. Handles errors during reading gracefully.

    Args:
        map_storage (ConversationMapStorage): Protocol object for reading map data.
        analysis_writer (SentenceAnalysisWriter): Protocol object for reading analysis data.
        task_id (Optional[str]): Optional identifier for logging task context.

    Returns:
        Dict[str, Any]: A dictionary containing completeness metrics:
            - total_expected (int): Number of unique sentence IDs read from map storage.
            - total_actual (int): Number of unique sentence IDs read from analysis writer.
            - total_missing (int): Count of expected IDs not found in the actual IDs.
            - missing_ids (List[int]): Sorted list of sentence IDs missing from analysis.
            - error (Optional[str]): Description of critical error encountered during reading,
                                     otherwise None.
    """
    prefix = _log_prefix(task_id)
    map_id = map_storage.get_identifier()
    analysis_id = analysis_writer.get_identifier()
    logger.debug(f"{prefix}Verifying completeness between map '{map_id}' and analysis '{analysis_id}'...")
    
    expected_ids: Set[int] = set()
    actual_ids: Set[int] = set()
    error_msg: Optional[str] = None

    # Process Map Storage
    try:
        logger.debug(f"{prefix}Reading expected IDs from map storage: {map_id}")
        expected_ids = await map_storage.read_sentence_ids()
        logger.debug(f"{prefix}Found {len(expected_ids)} expected IDs from {map_id}")
    except Exception as e:
        # Catch any exception from the protocol implementation
        logger.error(f"{prefix}Failed to read IDs from map storage {map_id}: {e}", exc_info=True)
        error_msg = f"Error reading map storage '{map_id}': {type(e).__name__}"
        # Cannot determine expected, return zero/empty results
        return {
            "total_expected": 0, "total_actual": 0, "total_missing": 0,
            "missing_ids": [], "error": error_msg
        }

    # Process Analysis Storage
    try:
        logger.debug(f"{prefix}Reading actual IDs from analysis writer storage: {analysis_id}")
        actual_ids = await analysis_writer.read_analysis_ids()
        logger.debug(f"{prefix}Found {len(actual_ids)} actual IDs from {analysis_id}")
    except Exception as e:
        # Catch any exception from the protocol implementation
        logger.error(f"{prefix}Failed to read IDs from analysis storage {analysis_id}: {e}", exc_info=True)
        # We have expected IDs, but actual reading failed. 
        # Record the error and proceed with calculation (actual will be empty set).
        error_msg = f"Error reading analysis storage '{analysis_id}': {type(e).__name__}"
        # Proceed with actual_ids as an empty set

    # Calculate differences
    missing_ids_set = expected_ids - actual_ids
    total_expected = len(expected_ids)
    total_actual = len(actual_ids) # Use length of actual_ids read (or 0 if read failed)
    total_missing = len(missing_ids_set)

    logger.debug(f"{prefix}Completeness check: Expected={total_expected}, Actual={total_actual}, Missing={total_missing}")

    return {
        "total_expected": total_expected,
        "total_actual": total_actual,
        "total_missing": total_missing,
        "missing_ids": sorted(list(missing_ids_set)),
        "error": error_msg # Include error message if reading failed
    }


# --- Helper Functions for process_file Refactoring --- #

# Helper function refactored to use protocols
async def _handle_map_creation(
    data_source: TextDataSource,         # New: Pass data source
    map_storage: ConversationMapStorage, # New: Pass map storage
    metrics_tracker: MetricsTracker,
    task_id: Optional[str] = None
) -> Tuple[int, List[str]]:
    """
    Handles the creation of the conversation map using IO Protocols.

    Calls create_conversation_map (which now uses protocols) and manages specific 
    exceptions, logging, and metrics updates related to map creation.

    Args:
        data_source: Protocol object for reading input text.
        map_storage: Protocol object for writing map data.
        metrics_tracker: Metrics tracker instance.
        task_id: Optional task identifier.

    Returns:
        Tuple[int, List[str]]: Number of sentences and the list of sentences.
        Returns (0, []) if the input source resulted in no sentences.

    Raises:
        # Exceptions are now determined by the specific protocol implementations used.
        Exception: For unexpected errors during map creation.
    """
    prefix = _log_prefix(task_id)
    source_id = data_source.get_identifier()
    storage_id = map_storage.get_identifier()
    # Use a representative name for metrics, like the source identifier
    metric_key = source_id 
    
    try:
        # Call the refactored create_conversation_map
        num_sentences, sentences = await create_conversation_map(data_source, map_storage, task_id)
        
        # Update metrics using the source identifier as the key
        metrics_tracker.set_metric(metric_key, "sentences_found_in_map", num_sentences)
        
        if num_sentences == 0:
             logger.warning(f"{prefix}Skipping analysis for source '{source_id}' as it contains no sentences.")
             # Return 0, [] - the caller (process_file) will handle the early exit
        return num_sentences, sentences
    except Exception as e:
        # Catch broader errors from either data_source or map_storage
        logger.error(f"{prefix}Unexpected error during map creation (source '{source_id}', storage '{storage_id}'): {e}", exc_info=True)
        metrics_tracker.increment_errors(metric_key)
        raise # Re-raise

def _handle_context_building(
    sentences: List[str],
    analysis_service: AnalysisService,
    metrics_tracker: MetricsTracker,
    input_file_name: str, # Pass name for logging/metrics
    task_id: Optional[str] = None
) -> Dict[int, Dict[str, Any]]: # Assuming context is Dict[int, Dict]
    """
    Handles the context building step, generating surrounding text for analysis.

    Calls the analysis_service's context builder (`build_all_contexts`) and 
    manages exceptions, logging, and metrics related to this step.

    Args:
        sentences: List of sentence strings.
        analysis_service: The analysis service instance.
        metrics_tracker: Metrics tracker instance.
        input_file_name: Name of the input file for context.
        task_id: Optional task identifier.

    Returns:
        Dict[int, Dict[str, Any]]: The generated context dictionary.

    Raises:
        Exception: If context building fails.
    """
    prefix = _log_prefix(task_id)
    logger.debug(f"{prefix}Building contexts for {input_file_name}...")
    try:
        contexts = analysis_service.build_contexts(sentences)
        logger.debug(f"{prefix}Context building successful for {input_file_name}.")
        return contexts
    except Exception as e:
        logger.error(f"{prefix}Failed to build contexts for {input_file_name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file_name)
        # Stop timer early? No, timer is stopped in process_file's finally block.
        raise # Re-raise to process_file


async def _orchestrate_analysis_and_writing(
    sentences: List[str],
    contexts: Dict[int, Dict[str, Any]],
    analysis_writer: SentenceAnalysisWriter,
    analysis_service: AnalysisService,
    metrics_tracker: MetricsTracker,
    input_file_name: str, # Keep for logging/metrics
    task_id: Optional[str] = None
) -> None:
    """
    Orchestrates sentence analysis and asynchronous result writing via protocols.

    Creates an `asyncio.Queue` and an `asyncio.Task` for the `_result_writer` 
    (using the passed `analysis_writer`), calls the analysis service to get results, 
    puts results onto the queue, and manages task/queue completion and cancellation.

    Args:
        sentences: List of sentence strings.
        contexts: Generated context dictionary.
        analysis_writer: Protocol object for writing analysis results.
        analysis_service: The analysis service instance.
        metrics_tracker: Metrics tracker instance.
        input_file_name: Name of the input file for context.
        task_id: Optional task identifier.

    Raises:
        Exception: Propagates exceptions from analysis or unexpected writer issues.
    """
    prefix = _log_prefix(task_id)
    results_queue = asyncio.Queue()
    writer_task = None # Initialize to None
    writer_id = analysis_writer.get_identifier() # Get writer ID for logging
    
    try:
        # Create writer task first - ensure it's always cancellable in finally
        logger.debug(f"{prefix}Creating result writer task for {writer_id}...")
        # Pass the analysis_writer object instead of the path
        writer_task = asyncio.create_task(_result_writer(analysis_writer, results_queue, metrics_tracker, task_id))

        # Run analysis
        logger.info(f"{prefix}Starting sentence analysis for {input_file_name} using AnalysisService...")
        analysis_results = await analysis_service.analyze_sentences(sentences, contexts, task_id=task_id)
        logger.info(f"{prefix}AnalysisService completed analysis for {input_file_name}. Found {len(analysis_results)} results.")
        
        # Queue results for writing
        logger.info(f"{prefix}Queueing {len(analysis_results)} analysis results for writing to {writer_id}...")
        num_results = len(analysis_results)
        for result in analysis_results:
            await results_queue.put(result)
            # Increment processed metric immediately after queuing
            metrics_tracker.increment_results_processed(input_file_name)

        # Signal writer completion and wait for processing
        logger.debug(f"{prefix}Signalling writer task completion for {writer_id}...")
        await results_queue.put(None)
        await results_queue.join() # Wait for queue to be emptied
        logger.debug(f"{prefix}Result queue joined. Waiting for writer task ({writer_id}) to finish...")
        await writer_task # Wait for writer task coroutine to fully finish
        await asyncio.sleep(0.01) # Keep small sleep for now
        logger.info(f"{prefix}Result writing complete via {writer_id} for {input_file_name}.")
        metrics_tracker.set_metric(input_file_name, "results_written", num_results) # Track written count

    except Exception as e:
        logger.error(f"{prefix}Error during sentence analysis or queuing for {input_file_name} (writer {writer_id}): {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file_name)
        raise # Re-raise the analysis/writing error
    finally:
        # Ensure writer task is cancelled if it exists and analysis failed 
        # or if an unexpected error occurred during queueing/waiting
        if writer_task and not writer_task.done():
            logger.warning(f"{prefix}Analysis/writing orchestration failed; cancelling writer task ({writer_id}) for {input_file_name}...")
            writer_task.cancel()
            try:
                await writer_task # Wait for cancellation to complete gracefully
            except asyncio.CancelledError:
                 logger.info(f"{prefix}Result writer task ({writer_id}) successfully cancelled for {input_file_name}.")
            except Exception as cancel_e:
                 # Log error during cancellation itself, but don't overshadow original error
                 logger.error(f"{prefix}Error awaiting writer task cancellation ({writer_id}) for {input_file_name}: {cancel_e}")
        elif writer_task and writer_task.done() and writer_task.exception():
            # If writer task finished but with an internal exception
            writer_exc = writer_task.exception()
            logger.error(f"{prefix}Result writer task ({writer_id}) for {input_file_name} finished with an exception: {writer_exc}", exc_info=writer_exc)
            # This indicates an error within _result_writer itself. Should we re-raise?
            # The original exception from the analysis block will likely be raised anyway.
            
# --- End Helper Functions for process_file --- #


# Refactor process_file to use IO protocols
async def process_file(
    data_source: TextDataSource,
    map_storage: ConversationMapStorage,
    analysis_writer: SentenceAnalysisWriter,
    config: Dict[str, Any],
    analysis_service: AnalysisService,
    metrics_tracker: MetricsTracker,
    task_id: Optional[str] = None
):
    """
    Processes data from a source using injected IO protocols and AnalysisService.

    Orchestrates the following steps:
    1. Creates conversation map data via `_handle_map_creation` (using protocols).
    2. Calls the injected `analysis_service` to build contexts for all sentences.
    3. Calls `_orchestrate_analysis_and_writing` (using protocols) to analyze sentences 
       and write results.

    Handles errors during map creation, context building, and analysis orchestration.

    Args:
        data_source (TextDataSource): Protocol object for reading input text.
        map_storage (ConversationMapStorage): Protocol object for writing/reading map data.
        analysis_writer (SentenceAnalysisWriter): Protocol object for writing analysis results.
        config (Dict[str, Any]): Application configuration dictionary.
        analysis_service (AnalysisService): An initialized and injected instance.
        metrics_tracker (MetricsTracker): Metrics tracker instance.
        task_id (Optional[str]): Optional identifier for logging task context.

    Raises:
        # Exceptions depend on protocol implementations and analysis service
        Exception: For critical, unexpected errors during processing.
    """
    prefix = _log_prefix(task_id)
    source_id = data_source.get_identifier()
    logger.info(f"{prefix}Processing source: {source_id}")
    
    file_timer_start = time.monotonic()
    metrics_tracker.start_file_timer(source_id)

    # Wrap all steps in one try/finally to ensure metrics stop always executes
    try:
        # --- Step 1: Create Conversation Map --- 
        try:
            num_sentences, sentences = await _handle_map_creation(
                data_source=data_source, 
                map_storage=map_storage, 
                metrics_tracker=metrics_tracker, 
                task_id=task_id
            )
            if num_sentences == 0:
                logger.warning(f"{prefix}Skipping analysis for source '{source_id}' as it contains no sentences.")
                # Return must happen *inside* the main try block but *before* the finally
                return 
        except Exception as map_e:
            logger.error(f"{prefix}Failed during map creation phase for source '{source_id}': {map_e}", exc_info=True)
            raise

        # --- Step 2: Build Contexts --- 
        try:
            contexts = _handle_context_building(sentences, analysis_service, metrics_tracker, source_id, task_id)
        except Exception as ctx_e:
            logger.error(f"{prefix}Failed during context building phase for source '{source_id}': {ctx_e}", exc_info=True)
            raise

        # --- Step 3: Orchestrate Analysis & Writing --- 
        try:
            await _orchestrate_analysis_and_writing(
                sentences=sentences, 
                contexts=contexts, 
                analysis_writer=analysis_writer, 
                analysis_service=analysis_service, 
                metrics_tracker=metrics_tracker, 
                input_file_name=source_id, 
                task_id=task_id
            )
        except Exception as orch_e:
            logger.error(f"{prefix}Failed during analysis/writing orchestration for source '{source_id}': {orch_e}", exc_info=True)
            raise
    # --- Outer Finally Block --- 
    # This block now executes regardless of where the try block exits (return or raise)
    finally:
        metrics_tracker.stop_file_timer(source_id)
        elapsed_time = time.monotonic() - file_timer_start
        logger.info(f"{prefix}Finished processing source '{source_id}'. Time taken: {elapsed_time:.2f} seconds.")


# --- Helper Functions for run_pipeline Refactoring --- #

# Define a dataclass to hold environment setup results
@dataclass
class PipelineEnvironment:
    config_dict: Dict[str, Any]
    analysis_service: AnalysisService
    metrics_tracker: MetricsTracker
    input_dir_path: Path
    output_dir_path: Path
    map_dir_path: Path
    num_concurrent_files: int
    map_suffix: str
    analysis_suffix: str

def _setup_pipeline_environment(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]],
    map_dir: Optional[Union[str, Path]],
    config: Optional[Dict[str, Any]],
    task_id: Optional[str] = None
) -> PipelineEnvironment:
    """
    Loads configuration, sets up paths, creates directories, and instantiates services.
    
    Initializes the core components needed for a pipeline run, including paths,
    the singleton `MetricsTracker`, and the `AnalysisService` (which in turn
    instantiates `ContextBuilder` and `SentenceAnalyzer`).

    Args:
        input_dir: Path to the input directory.
        output_dir: Optional path to the output directory.
        map_dir: Optional path to the map directory.
        config: Optional configuration dictionary.
        task_id: Optional task identifier for logging.

    Returns:
        A PipelineEnvironment object containing the setup results.

    Raises:
        ValueError: If configuration is invalid or paths cannot be processed.
        RuntimeError: If core services cannot be instantiated or directories cannot be created.
    """
    prefix = _log_prefix(task_id)
    logger.debug(f"{prefix}Setting up pipeline environment...")

    # --- Use provided config or load global --- 
    if config is None:
        from src.config import config as global_config # Import locally
        config_dict = global_config.config
    else:
        config_dict = config
    
    # Determine directories using provided args or config defaults
    try:
        input_dir_path = Path(input_dir)
        output_dir_str = output_dir or config_dict.get("paths", {}).get("output_dir", "./data/output")
        output_dir_path = Path(output_dir_str)
        map_dir_str = map_dir or config_dict.get("paths", {}).get("map_dir", "./data/maps")
        map_dir_path = Path(map_dir_str)
        
        # Get other config values
        num_concurrent_files = config_dict.get("pipeline", {}).get("num_concurrent_files", 1)
        analysis_suffix = config_dict.get("paths", {}).get("analysis_suffix", "_analysis.jsonl")
        map_suffix = config_dict.get("paths", {}).get("map_suffix", "_map.jsonl")

    except KeyError as e:
        logger.critical(f"{prefix}Configuration missing required path key: {e}", exc_info=True)
        raise ValueError(f"Configuration missing required path key: {e}") from e
    except Exception as e:
        logger.critical(f"{prefix}Error processing configuration paths: {e}", exc_info=True)
        raise ValueError("Error processing configuration paths") from e
        
    logger.info(f"{prefix}Using Input Dir: {input_dir_path}")
    logger.info(f"{prefix}Using Output Dir: {output_dir_path}")
    logger.info(f"{prefix}Using Map Dir: {map_dir_path}")
    logger.info(f"{prefix}Max concurrent file processing: {num_concurrent_files}")

    # Ensure directories exist
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        map_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         logger.critical(f"{prefix}Failed to create output/map directories: {e}", exc_info=True)
         raise RuntimeError(f"Failed to create required directories ({output_dir_path}, {map_dir_path})" ) from e

    # Initialize MetricsTracker 
    # Assuming singleton usage for now as per original code
    from src.utils.metrics import metrics_tracker 

    # --- Instantiate Services --- 
    try:
        logger.debug(f"{prefix}Instantiating AnalysisService and its dependencies...")
        context_builder_instance = ContextBuilder(config_dict=config_dict)
        sentence_analyzer_instance = SentenceAnalyzer(config_dict=config_dict)
        analysis_service_instance = AnalysisService(
            config=config_dict,
            context_builder=context_builder_instance,
            sentence_analyzer=sentence_analyzer_instance,
            metrics_tracker=metrics_tracker
        )
        logger.debug(f"{prefix}AnalysisService instantiated.")
    except Exception as e:
        logger.critical(f"{prefix}Failed to instantiate core services: {e}", exc_info=True)
        raise RuntimeError("Failed to instantiate core services") from e

    return PipelineEnvironment(
        config_dict=config_dict,
        analysis_service=analysis_service_instance,
        metrics_tracker=metrics_tracker,
        input_dir_path=input_dir_path,
        output_dir_path=output_dir_path,
        map_dir_path=map_dir_path,
        num_concurrent_files=num_concurrent_files,
        map_suffix=map_suffix,
        analysis_suffix=analysis_suffix
    )

def _discover_files_to_process(
    input_dir_path: Path,
    specific_file: Optional[str] = None,
    metrics_tracker: Optional[MetricsTracker] = None, # Optional for error tracking
    task_id: Optional[str] = None
) -> List[Path]:
    """
    Discovers input files (.txt) to be processed.

    Args:
        input_dir_path: Path to the input directory.
        specific_file: Optional specific filename to process within input_dir.
        metrics_tracker: Optional metrics tracker for incrementing errors.
        task_id: Optional task identifier for logging.

    Returns:
        A list of Path objects for the files to process. Returns empty list if none found.

    Raises:
        FileNotFoundError: If specific_file is provided but not found.
    """
    prefix = _log_prefix(task_id)
    files_to_process: List[Path] = []

    if specific_file:
        specific_file_path = input_dir_path / specific_file
        if not specific_file_path.is_file():
            logger.error(f"{prefix}Specified input file not found: {specific_file_path}")
            if metrics_tracker:
                 metrics_tracker.increment_errors() # Track pipeline setup error
            raise FileNotFoundError(f"Specified input file not found: {specific_file_path}")
        files_to_process = [specific_file_path]
        logger.info(f"{prefix}Processing specific file: {specific_file}")
    else:
        logger.debug(f"{prefix}Discovering .txt files in: {input_dir_path}")
        files_to_process = list(input_dir_path.glob("*.txt"))
        if not files_to_process:
            logger.warning(f"{prefix}No .txt files found in input directory: {input_dir_path}")
            # Return empty list, caller (run_pipeline) handles exit
        else:
             logger.info(f"{prefix}Found {len(files_to_process)} .txt files to process.")

    return files_to_process


async def _run_processing_tasks(
    files_to_process: List[Path],
    env: PipelineEnvironment, # Pass the setup environment object
    task_id: Optional[str] = None
) -> List[Union[Exception, Any]]:
    """
    Creates and executes concurrent tasks for processing each file using IO protocols.

    Iterates through `files_to_process`, instantiating `LocalTextDataSource`,
    `LocalJsonlMapStorage`, and `LocalJsonlAnalysisWriter` for each file path.
    It then schedules `process_file` calls using an `asyncio.Semaphore` to 
    limit concurrency.

    Args:
        files_to_process: List of file paths to process.
        env: The PipelineEnvironment object containing config, services, paths etc.
        task_id: Optional task identifier for logging.

    Returns:
        A list containing the results of asyncio.gather (either None on success
        per task, or the Exception raised).
    """
    prefix = _log_prefix(task_id)
    semaphore = asyncio.Semaphore(env.num_concurrent_files)
    tasks = []
    total_files = len(files_to_process)
    env.metrics_tracker.set_metric("pipeline", "total_files_to_process", total_files)

    # Define nested helper for semaphore usage
    async def process_with_semaphore(file_path: Path):
        async with semaphore:
            # Instantiate IO objects for this specific file
            try:
                data_source = LocalTextDataSource(file_path)
                
                map_file = env.map_dir_path / f"{file_path.stem}{env.map_suffix}"
                map_storage = LocalJsonlMapStorage(map_file)
                
                analysis_file = env.output_dir_path / f"{file_path.stem}{env.analysis_suffix}"
                analysis_writer = LocalJsonlAnalysisWriter(analysis_file)
            
            except Exception as io_setup_e:
                # Handle potential errors during IO object instantiation (e.g., invalid path parts)
                logger.error(f"{prefix}Failed to set up IO objects for {file_path.name}: {io_setup_e}", exc_info=True)
                env.metrics_tracker.increment_errors(str(file_path)) # Track error against file path
                raise # Re-raise the setup error to be caught by gather

            # Call the refactored process_file with instantiated IO objects
            await process_file(
                data_source=data_source,
                map_storage=map_storage,
                analysis_writer=analysis_writer,
                config=env.config_dict, 
                analysis_service=env.analysis_service, 
                metrics_tracker=env.metrics_tracker, 
                task_id=task_id # Pass task_id down
            )

    logger.info(f"{prefix}Scheduling {total_files} file processing tasks...")
    for i, file_path in enumerate(files_to_process):
        tasks.append(process_with_semaphore(file_path))

    logger.info(f"{prefix}Starting concurrent file processing with {env.num_concurrent_files} workers...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"{prefix}Concurrent file processing finished.")
    return results


def _log_processing_summary(
    results: List[Union[Exception, Any]],
    files_to_process: List[Path],
    metrics_tracker: MetricsTracker,
    task_id: Optional[str] = None
) -> Tuple[int, int]:
    """
    Logs the outcome of each processing task and updates overall metrics.

    Args:
        results: The list of results/exceptions from asyncio.gather.
        files_to_process: The corresponding list of file paths.
        metrics_tracker: The metrics tracker instance.
        task_id: Optional task identifier for logging.
        
    Returns:
        Tuple[int, int]: Count of files processed successfully and files failed.
    """
    prefix = _log_prefix(task_id)
    files_processed_successfully = 0
    files_failed = 0
    logger.info(f"{prefix}Summarizing processing results...")
    
    for i, result in enumerate(results):
        # Ensure index is valid before accessing files_to_process
        if i < len(files_to_process):
            file_path = files_to_process[i]
            if isinstance(result, Exception):
                files_failed += 1
                # Log error type and message
                logger.error(f"{prefix}Processing failed for {file_path.name}: {type(result).__name__}: {result}")
                # Detailed error should have been logged within process_file or its callees
            else:
                # Check if process_file returned something unexpected, though it shouldn't
                if result is not None:
                     logger.warning(f"{prefix}Processing for {file_path.name} completed but returned unexpected value: {result}")
                files_processed_successfully += 1
                logger.info(f"{prefix}Successfully processed {file_path.name}.")
        else:
            # This case should ideally not happen if results and files_to_process align
            logger.error(f"{prefix}Result found at index {i} with no corresponding file path.")
            
    # Update overall pipeline metrics
    metrics_tracker.set_metric("pipeline", "files_processed_successfully", files_processed_successfully)
    metrics_tracker.set_metric("pipeline", "files_failed", files_failed)
    logger.info(f"{prefix}Processing Summary: Successful={files_processed_successfully}, Failed={files_failed}")
    return files_processed_successfully, files_failed


# Make async and use protocols
async def _run_verification(
    files_to_process: List[Path],
    env: PipelineEnvironment, # Pass the setup environment object
    task_id: Optional[str] = None
) -> None:
    """
    Runs the output completeness verification step for all processed files using IO Protocols.
    
    Iterates through `files_to_process`, instantiating `LocalJsonlMapStorage`
    and `LocalJsonlAnalysisWriter` for each corresponding map/analysis file pair.
    Calls `verify_output_completeness` for each pair.

    Args:
        files_to_process: List of file paths that were attempted.
        env: The PipelineEnvironment object containing paths and suffixes.
        task_id: Optional task identifier for logging.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Starting output verification...")
    verification_results = []
    
    if not files_to_process:
        logger.info(f"{prefix}No files were processed, skipping verification.")
        return
        
    for file_path in files_to_process:
        # Instantiate IO objects for verification
        try:
            map_file = env.map_dir_path / f"{file_path.stem}{env.map_suffix}"
            map_storage = LocalJsonlMapStorage(map_file)
            
            analysis_file = env.output_dir_path / f"{file_path.stem}{env.analysis_suffix}"
            # We only need the reading capability of the writer protocol here
            analysis_reader = LocalJsonlAnalysisWriter(analysis_file) 
            
            # Call the async verify function with the protocol objects
            verification_result = await verify_output_completeness(
                map_storage=map_storage,
                analysis_writer=analysis_reader,
                task_id=task_id # Pass task_id down
            )
            
        except Exception as e: # Catch errors during IO instantiation or verification call
             logger.error(f"{prefix}Unexpected error during verification setup/run for {file_path.name}: {e}", exc_info=True)
             # Create a default error result if setup/verification fails
             verification_result = {
                 "total_expected": 0, "total_actual": 0, "total_missing": 0,
                 "missing_ids": [], "error": f"Verification error: {type(e).__name__}"
             }
             
        verification_results.append(verification_result)
        
        # Log individual file verification details (remains the same)
        if verification_result.get("error"):
            logger.warning(f"{prefix}Verification check for {file_path.name}: ERROR - {verification_result['error']}")
        elif verification_result["total_missing"] > 0:
            logger.warning(f"{prefix}Verification check for {file_path.name}: MISSING {verification_result['total_missing']}/{verification_result['total_expected']} sentences. Missing IDs: {verification_result['missing_ids'][:10]}..." if verification_result['missing_ids'] else "")
        else:
            expected_count = verification_result['total_expected']
            actual_count = verification_result['total_actual']
            logger.info(f"{prefix}Verification check for {file_path.name}: OK ({actual_count}/{expected_count} sentences found).")
            
    # Log overall verification summary (remains the same)
    total_missing_overall = sum(vr["total_missing"] for vr in verification_results if vr.get("error") is None)
    total_expected_overall = sum(vr["total_expected"] for vr in verification_results if vr.get("error") is None)
    verification_errors = sum(1 for vr in verification_results if vr.get("error") is not None)
    logger.info(f"{prefix}Verification Summary: Total Missing Sentences={total_missing_overall}/{total_expected_overall}, Verification Errors={verification_errors}")
    env.metrics_tracker.set_metric("pipeline", "verification_total_missing", total_missing_overall)
    env.metrics_tracker.set_metric("pipeline", "verification_errors", verification_errors)

# --- End Helper Functions --- #


async def run_pipeline(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    map_dir: Optional[Union[str, Path]] = None,
    specific_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None # Add task_id parameter
):
    """
    Runs the analysis pipeline on all .txt files in the input directory.

    Orchestrates the main pipeline flow:
    1. Setup: Calls `_setup_pipeline_environment` to initialize config, paths, services.
    2. Discover: Calls `_discover_files_to_process` to find input files.
    3. Process: Calls `_run_processing_tasks` to analyze files concurrently.
    4. Summarize: Calls `_log_processing_summary` to log outcomes.
    5. Verify: Calls `_run_verification` to check output completeness.

    Args:
        input_dir: Directory containing input .txt files or path to specific file's dir.
        output_dir: Optional directory to save analysis output files.
        map_dir: Optional directory to save conversation map files.
        specific_file: Optional filename to process within input_dir.
        config: Optional application configuration dictionary.
        task_id: Optional unique ID for tracking/logging.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Starting pipeline run...")

    try:
        # --- Step 1: Setup Environment --- 
        env = _setup_pipeline_environment(input_dir, output_dir, map_dir, config, task_id)

        # --- Step 2: Discover Files --- 
        files_to_process = _discover_files_to_process(
            env.input_dir_path, specific_file, env.metrics_tracker, task_id
        )

        # Exit early if no files found
        if not files_to_process:
            logger.info(f"{prefix}Pipeline run finished: No files to process.")
            return

        # --- Step 3: Process Files Concurrently --- 
        results = await _run_processing_tasks(files_to_process, env, task_id)

        # --- Step 4: Log Processing Summary --- 
        # This step also updates metrics_tracker for successful/failed files
        _log_processing_summary(results, files_to_process, env.metrics_tracker, task_id)

        # --- Step 5: Verification Step --- 
        await _run_verification(files_to_process, env, task_id)

    except (ValueError, RuntimeError, FileNotFoundError) as setup_error:
        # Catch critical errors during setup or file discovery
        logger.critical(f"{prefix}Pipeline run failed during setup/discovery: {setup_error}", exc_info=True)
        # Potentially increment a general pipeline error metric if tracker is available
        # (Metrics tracker might not be initialized if config loading failed)
        # Consider how to handle metrics in very early failures.
        raise # Re-raise the critical error to the caller
    except Exception as e:
        # Catch unexpected errors during the main processing/verification phases
        logger.critical(f"{prefix}Unexpected critical error during pipeline execution: {e}", exc_info=True)
        # Increment error metric if possible
        # if 'env' in locals() and hasattr(env, 'metrics_tracker'):
        #     env.metrics_tracker.increment_errors("pipeline_critical") # Add specific error key?
        raise # Re-raise
    finally:
        # --- Final Logging --- 
        # Pipeline timer stopped by the caller (main() or API endpoint wrapper if needed)
        logger.info(f"{prefix}Pipeline run finished.")


# --- analyze_specific_sentences function --- 
# ... (This function likely also needs task_id added if called from API)

async def analyze_specific_sentences(
    # input_file_path: Path, # Replaced by map_storage
    map_storage: ConversationMapStorage, # New: Inject map storage
    sentence_ids: List[int],
    config: Dict[str, Any], # Keep for analysis service
    analysis_service: AnalysisService, # Inject AnalysisService
    task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Analyzes specific sentences using ConversationMapStorage.

    Reads map data from the injected map_storage, extracts target sentences and 
    their contexts, calls the analysis service, and returns remapped results.

    Args:
        map_storage (ConversationMapStorage): Protocol object for reading map data.
        sentence_ids (List[int]): List of sentence IDs to analyze.
        config (Dict[str, Any]): Application configuration (potentially used by analysis service).
        analysis_service (AnalysisService): Injected analysis service instance.
        task_id (Optional[str]): Task identifier for logging.

    Returns:
        List[Dict[str, Any]]: List of analysis result dictionaries for the requested sentences.

    Raises:
        ValueError: If requested sentence IDs are not found in the map storage.
        Exception: Propagates exceptions from map storage reading or analysis service.
    """
    prefix = _log_prefix(task_id)
    map_id = map_storage.get_identifier()
    logger.info(f"{prefix}Starting analysis for specific sentences using map: {map_id}, IDs: {sentence_ids}")

    # Remove path setup logic
    # try: ... except ...

    # --- Read Map Data using Protocol --- 
    logger.info(f"{prefix}Reading map entries from: {map_id}")
    try:
        all_entries = await map_storage.read_all_entries()
        if not all_entries:
            logger.error(f"{prefix}Map storage '{map_id}' is empty or could not be read.")
            # Decide if this is a ValueError or other exception type
            raise ValueError(f"Map storage '{map_id}' is empty or unreadable.")
    except Exception as read_e:
        logger.error(f"{prefix}Error reading map entries from {map_id}: {read_e}", exc_info=True)
        raise # Re-raise the storage access error
        
    # --- Prepare Sentences and Check IDs --- 
    all_sentences_map: Dict[int, str] = {}
    # Ensure sequence_order is present for sorting, default to -1 if missing
    all_entries.sort(key=lambda x: x.get('sequence_order', -1))
    
    # Build map and full list for context from potentially unsorted/sparse entries
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
        logger.warning(f"{prefix}No valid sentence entries found in map {map_id}. Returning empty list.")
        return []
        
    full_sentence_list_for_context = ["" for _ in range(max_id + 1)]
    for s_id, text in all_sentences_map.items():
        if 0 <= s_id <= max_id:
            full_sentence_list_for_context[s_id] = text
            
    target_sentences: List[str] = []
    target_indices: List[int] = [] # Store original indices for context building
    missing_ids = []

    target_id_set = set(sentence_ids)
    for target_id in target_id_set:
        if target_id in all_sentences_map:
            target_sentences.append(all_sentences_map[target_id])
            target_indices.append(target_id) # Store the original index
        else:
            missing_ids.append(target_id)

    if missing_ids:
        logger.error(f"{prefix}Requested sentence IDs not found in map {map_id}: {sorted(missing_ids)}")
        raise ValueError(f"Sentence IDs not found in map '{map_id}': {sorted(missing_ids)}")

    if not target_sentences:
        logger.warning(f"{prefix}No valid target sentences found after checking map {map_id}. Returning empty list.")
        return []

    # --- Build Contexts for Target Sentences ONLY --- 
    logger.info(f"{prefix}Building contexts for {len(target_indices)} specific sentences from map {map_id}...")
    try:
        all_contexts_dict = analysis_service.context_builder.build_all_contexts(full_sentence_list_for_context)
        target_contexts = [all_contexts_dict.get(idx, {}) for idx in target_indices]
        
        if len(target_contexts) != len(target_sentences):
             logger.error(f"{prefix}Context count ({len(target_contexts)}) mismatch with target sentence count ({len(target_sentences)}) for map {map_id}.")
             raise RuntimeError(f"Context and sentence count mismatch for map '{map_id}'.")
             
    except Exception as e:
        logger.error(f"{prefix}Failed to build contexts for specific sentences from map {map_id}: {e}", exc_info=True)
        raise

    # --- Analyze Specific Sentences --- 
    logger.info(f"{prefix}Analyzing {len(target_sentences)} specific sentences from map {map_id}...")
    try:
        analysis_results = await analysis_service.analyze_sentences(
            target_sentences, target_contexts, task_id=task_id
        )
    except Exception as e:
         logger.error(f"{prefix}Error during specific sentence analysis from map {map_id}: {e}", exc_info=True)
         raise
         
    # --- Post-Process Results (Remains the same logic) --- 
    final_results = []
    if len(analysis_results) != len(target_indices):
        logger.warning(f"{prefix}Result count ({len(analysis_results)}) mismatch with target index count ({len(target_indices)}) for map {map_id}. Results might be incomplete.")
        
    for i, result in enumerate(analysis_results):
         if i < len(target_indices):
              original_sentence_id = target_indices[i]
              result['sentence_id'] = original_sentence_id
              result['sequence_order'] = original_sentence_id 
              final_results.append(result)
         else:
              logger.warning(f"{prefix}Extra result found at index {i} beyond the number of target indices for map {map_id}.")

    logger.info(f"{prefix}Finished specific sentence analysis for map {map_id}. Returning {len(final_results)} results.")
    return final_results
