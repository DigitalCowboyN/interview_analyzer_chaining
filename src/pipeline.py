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
from src.utils.helpers import append_json_line
import asyncio
# SentenceAnalyzer is used indirectly via AnalysisService
# from src.agents.sentence_analyzer import SentenceAnalyzer
from src.utils.logger import get_logger
# ContextBuilder is used indirectly via AnalysisService
# from src.agents.context_builder import ContextBuilder
from src.utils.metrics import MetricsTracker
from src.utils.text_processing import segment_text
import json
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

logger = get_logger()

# Define a helper for log prefixing
def _log_prefix(task_id: Optional[str] = None) -> str:
    return f"[Task {task_id}] " if task_id else ""

async def create_conversation_map(input_file: Path, map_dir: Path, map_suffix: str, task_id: Optional[str] = None) -> Tuple[int, List[str]]:
    """
    Creates a conversation map file (.jsonl) and returns sentence count and sentences.

    Segments the input text file into sentences. Each line in the output map file
    corresponds to a sentence and contains its ID (index), sequence order (same as ID),
    and the sentence text.

    Args:
        input_file (Path): Path to the input text file.
        map_dir (Path): Directory where the map file will be saved.
        map_suffix (str): Suffix to append to the input file stem for the map filename
                          (e.g., "_map.jsonl").

    Returns:
        Tuple[int, List[str]]: The number of sentences found and the list of sentence strings.

    Raises:
        FileNotFoundError: If the input file does not exist.
        OSError: If the map directory cannot be created or the map file cannot be written.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Creating conversation map for: {input_file}")
    if not input_file.exists():
        # No need to raise FileNotFoundError here if run_pipeline checks first
        # Let run_pipeline handle the main file check? Or keep redundant check?
        # For now, keep it, but maybe simplify later.
        logger.error(f"{prefix}Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    map_file = map_dir / f"{input_file.stem}{map_suffix}"
    map_dir.mkdir(parents=True, exist_ok=True)

    # Check if map file already exists
    # if map_file.exists():
    #    logger.warning(f"Map file {map_file} already exists. Overwriting.") # Or skip?

    text = input_file.read_text(encoding="utf-8")
    # Use the imported segment_text function
    sentences = segment_text(text)
    num_sentences = len(sentences)
    
    # Check for empty sentences list early
    if num_sentences == 0:
        logger.warning(f"{prefix}Input file {input_file} contains no sentences after segmentation. Map file will be empty.")
        # Create an empty map file
        map_file.touch()
        return 0, [] # Return 0 sentences and empty list

    # --- Write Map File ---
    try:
        with map_file.open("w", encoding="utf-8") as f:
            for idx, sentence_text in enumerate(sentences):
                entry = {
                    "sentence_id": idx,
                    "sequence_order": idx,
                    "sentence": sentence_text
                }
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + '\n')
    except OSError as e:
        logger.error(f"{prefix}Failed to write map file {map_file}: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by process_file

    logger.info(f"{prefix}Conversation map created: {map_file} with {num_sentences} sentences.")
    return num_sentences, sentences

async def _result_writer(
    output_file: Path, 
    results_queue: asyncio.Queue, 
    metrics_tracker: MetricsTracker,
    task_id: Optional[str] = None
):
    """
    Consumes analysis results from a queue and writes them to a JSON Lines file.

    Runs asynchronously, fetching results from the queue until a `None` sentinel
    is received. Handles potential errors during file writing using `append_json_line`,
    logs errors, and updates the provided `metrics_tracker`.

    Args:
        output_file (Path): Path to the target output .jsonl file.
        results_queue (asyncio.Queue): Queue to get analysis result dictionaries from.
                                       Expects `None` as a termination signal.
        metrics_tracker (MetricsTracker): Instance used to track errors during writing.
    """
    prefix = _log_prefix(task_id)
    logger.debug(f"{prefix}Result writer starting for: {output_file}")
    results_written = 0
    while True:
        try:
            logger.debug(f"{prefix}Writer waiting for item...")
            result = await results_queue.get()
            logger.debug(f"{prefix}Writer received item: {type(result)}")
            if result is None: # Sentinel value indicates completion
                logger.info(f"{prefix}Writer received sentinel. Exiting.")
                results_queue.task_done() # Mark sentinel as processed
                break

            try:
                logger.debug(f"{prefix}Writer attempting to append result ID: {result.get('sentence_id', 'N/A')}")
                append_json_line(result, output_file)
                results_written += 1
                logger.debug(f"{prefix}Writer successfully appended result ID: {result.get('sentence_id', 'N/A')}")
                # Optionally track write success metric
                # metrics_tracker.increment_write_success()
            except Exception as e:
                # Log error details including sentence ID if available
                sentence_id_info = f"result {result.get('sentence_id', 'N/A')}" if isinstance(result, dict) else "result (unknown ID)"
                logger.error(f"{prefix}Writer failed writing {sentence_id_info} to {output_file}: {e}", exc_info=True)
                metrics_tracker.increment_errors() # Use passed-in tracker
            finally:
                results_queue.task_done() # Mark result as processed

        except asyncio.CancelledError:
            logger.info(f"{prefix}Writer for {output_file} cancelled.")
            break
        except Exception as e:
            logger.critical(f"{prefix}Critical error in writer for {output_file}: {e}", exc_info=True)
            break # Exit loop on critical error
    logger.debug(f"{prefix}Result writer loop finished for: {output_file}. Total written: {results_written}")

def verify_output_completeness(map_path: Path, analysis_path: Path) -> Dict[str, Any]:
    """
    Compares a map file and an analysis file to verify processing completeness.

    Reads sentence IDs from both the map file (expected) and the analysis file (actual)
    and calculates metrics on how many are missing from the analysis output.
    Handles file not found errors and JSON parsing errors gracefully by logging and
    continuing where possible.

    Args:
        map_path (Path): Path to the conversation map file (.jsonl).
        analysis_path (Path): Path to the analysis results file (.jsonl).

    Returns:
        Dict[str, Any]: A dictionary containing completeness metrics:
            - total_expected (int): Number of unique sentence IDs found in the map file.
            - total_actual (int): Number of unique sentence IDs found in the analysis file.
            - total_missing (int): Count of expected IDs not found in the actual IDs.
            - missing_ids (List[int]): Sorted list of sentence IDs missing from analysis.
            - error (Optional[str]): Description of critical error encountered (e.g., file not found),
                                     otherwise None.
    """
    expected_ids: Set[int] = set()
    actual_ids: Set[int] = set()
    error_msg: Optional[str] = None

    # Process Map File
    try:
        with map_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "sentence_id" in entry:
                        expected_ids.add(entry["sentence_id"])
                    else:
                        logger.warning(f"Missing 'sentence_id' in map file {map_path}, line {i+1}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line in map file {map_path}, line {i+1}: {e}. Line: '{line[:100]}...'")
    except FileNotFoundError:
        logger.warning(f"Map file not found: {map_path}")
        error_msg = f"Map file not found: {map_path.name}"
        # Cannot determine expected, so return zero/empty
        return {
            "total_expected": 0, "total_actual": 0, "total_missing": 0,
            "missing_ids": [], "error": error_msg
        }
    except Exception as e:
        logger.error(f"Unexpected error reading map file {map_path}: {e}", exc_info=True)
        error_msg = f"Error reading map file: {type(e).__name__}"
        # Treat as if expected is unknown
        return {
            "total_expected": 0, "total_actual": 0, "total_missing": 0,
            "missing_ids": [], "error": error_msg
        }

    # Process Analysis File
    try:
        with analysis_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Also check for error flag to potentially exclude failed analysis if needed
                    # For completeness check, we count any entry with a sentence_id
                    if "sentence_id" in entry:
                        actual_ids.add(entry["sentence_id"])
                    else:
                         logger.warning(f"Missing 'sentence_id' in analysis file {analysis_path}, line {i+1}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line in analysis file {analysis_path}, line {i+1}: {e}. Line: '{line[:100]}...'")
    except FileNotFoundError:
        logger.warning(f"Analysis file not found: {analysis_path}")
        # We know the expected count, but actual is zero
        error_msg = f"Analysis file not found: {analysis_path.name}"
        # Proceed with calculation based on expected_ids found from map
    except Exception as e:
         logger.error(f"Unexpected error reading analysis file {analysis_path}: {e}", exc_info=True)
         error_msg = f"Error reading analysis file: {type(e).__name__}"
         # Treat as if actual is unknown or potentially incomplete

    # Calculate differences
    missing_ids_set = expected_ids - actual_ids
    total_expected = len(expected_ids)
    total_actual = len(actual_ids)
    total_missing = len(missing_ids_set)

    return {
        "total_expected": total_expected,
        "total_actual": total_actual,
        "total_missing": total_missing,
        "missing_ids": sorted(list(missing_ids_set)),
        "error": error_msg # Include error message if a file was not found/read
    }


# --- Helper Functions for process_file Refactoring --- #

async def _handle_map_creation(
    input_file: Path,
    map_dir: Path,
    map_suffix: str,
    metrics_tracker: MetricsTracker,
    task_id: Optional[str] = None
) -> Tuple[int, List[str]]:
    """
    Handles the creation of the conversation map file.

    Calls create_conversation_map and manages specific exceptions,
    logging, and metrics updates related to map creation.

    Args:
        input_file: Path to the input text file.
        map_dir: Directory where the map file will be saved.
        map_suffix: Suffix for the map filename.
        metrics_tracker: Metrics tracker instance.
        task_id: Optional task identifier.

    Returns:
        Tuple[int, List[str]]: Number of sentences and the list of sentences.
        Returns (0, []) if the input file resulted in no sentences.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        OSError: If the map file cannot be written.
        Exception: For other unexpected errors during map creation.
    """
    prefix = _log_prefix(task_id)
    try:
        num_sentences, sentences = await create_conversation_map(input_file, map_dir, map_suffix, task_id)
        metrics_tracker.set_metric(input_file.name, "sentences_found_in_map", num_sentences)
        if num_sentences == 0:
             logger.warning(f"{prefix}Skipping analysis for {input_file.name} as it contains no sentences.")
             # Return 0, [] - the caller (process_file) will handle the early exit
        return num_sentences, sentences
    except FileNotFoundError as e:
        logger.error(f"{prefix}Input file not found during map creation: {e}")
        metrics_tracker.increment_errors(input_file.name)
        # Stop timer early? No, timer is stopped in process_file's finally block.
        raise # Re-raise to be caught by process_file
    except OSError as e:
        logger.error(f"{prefix}OS error during map creation for {input_file.name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file.name)
        raise # Re-raise
    except Exception as e:
        logger.error(f"{prefix}Unexpected error during map creation for {input_file.name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file.name)
        raise # Re-raise

def _handle_context_building(
    sentences: List[str],
    analysis_service: AnalysisService,
    metrics_tracker: MetricsTracker,
    input_file_name: str, # Pass name for logging/metrics
    task_id: Optional[str] = None
) -> Dict[int, Dict[str, Any]]: # Assuming context is Dict[int, Dict]
    """
    Handles the context building step.

    Calls the analysis_service's context builder and manages exceptions,
    logging, and metrics related to this step.

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
    analysis_file_path: Path,
    analysis_service: AnalysisService,
    metrics_tracker: MetricsTracker,
    input_file_name: str, # Pass name for logging/metrics
    task_id: Optional[str] = None
) -> None:
    """
    Orchestrates sentence analysis and asynchronous result writing.

    Creates a queue and a writer task, calls the analysis service,
    puts results on the queue, and manages task/queue completion and cancellation.

    Args:
        sentences: List of sentence strings.
        contexts: Generated context dictionary.
        analysis_file_path: Path for the output analysis file.
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
    
    try:
        # Create writer task first - ensure it's always cancellable in finally
        logger.debug(f"{prefix}Creating result writer task for {analysis_file_path.name}...")
        writer_task = asyncio.create_task(_result_writer(analysis_file_path, results_queue, metrics_tracker, task_id))

        # Run analysis
        logger.info(f"{prefix}Starting sentence analysis for {input_file_name} using AnalysisService...")
        analysis_results = await analysis_service.analyze_sentences(sentences, contexts, task_id=task_id)
        logger.info(f"{prefix}AnalysisService completed analysis for {input_file_name}. Found {len(analysis_results)} results.")
        
        # Queue results for writing
        logger.info(f"{prefix}Queueing {len(analysis_results)} analysis results for writing...")
        num_results = len(analysis_results)
        for result in analysis_results:
            await results_queue.put(result)
            # Increment processed metric immediately after queuing
            metrics_tracker.increment_results_processed(input_file_name)

        # Signal writer completion and wait for processing
        logger.debug(f"{prefix}Signalling writer task completion...")
        await results_queue.put(None)
        await results_queue.join() # Wait for queue to be emptied
        logger.debug(f"{prefix}Result queue joined. Waiting for writer task to finish...")
        await writer_task # Wait for writer task coroutine to fully finish
        await asyncio.sleep(0.01) # Keep small sleep for now
        logger.info(f"{prefix}Result writing complete for {input_file_name}.")
        metrics_tracker.set_metric(input_file_name, "results_written", num_results) # Track written count

    except Exception as e:
        logger.error(f"{prefix}Error during sentence analysis or queuing for {input_file_name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file_name)
        raise # Re-raise the analysis/writing error
    finally:
        # Ensure writer task is cancelled if it exists and analysis failed 
        # or if an unexpected error occurred during queueing/waiting
        if writer_task and not writer_task.done():
            logger.warning(f"{prefix}Analysis/writing orchestration failed; cancelling writer task for {input_file_name}...")
            writer_task.cancel()
            try:
                await writer_task # Wait for cancellation to complete gracefully
            except asyncio.CancelledError:
                 logger.info(f"{prefix}Result writer task successfully cancelled for {input_file_name}.")
            except Exception as cancel_e:
                 # Log error during cancellation itself, but don't overshadow original error
                 logger.error(f"{prefix}Error awaiting writer task cancellation for {input_file_name}: {cancel_e}")
        elif writer_task and writer_task.done() and writer_task.exception():
            # If writer task finished but with an internal exception
            logger.error(f"{prefix}Result writer task for {input_file_name} finished with an exception: {writer_task.exception()}", exc_info=writer_task.exception())
            # This indicates an error within _result_writer itself. Should we re-raise?
            # The original exception from the analysis block will likely be raised anyway.
            
# --- End Helper Functions for process_file --- #


async def process_file(
    input_file: Path,
    output_dir: Path,
    map_dir: Path,
    config: Dict[str, Any],
    analysis_service: AnalysisService, # Inject AnalysisService
    metrics_tracker: MetricsTracker, # Accept MetricsTracker
    task_id: Optional[str] = None
):
    """
    Processes a single text file using an injected AnalysisService instance.

    Orchestrates the following steps:
    1. Creates a conversation map file (`create_conversation_map`).
    2. Calls the injected `analysis_service` to build contexts for all sentences.
    3. Calls the `analysis_service` to analyze sentences with their contexts.
    4. Creates and awaits an asynchronous task (`_result_writer`) to write analysis
       results to a JSON Lines file.

    Handles errors during map creation, context building, and analysis. Errors
    during result writing are handled within `_result_writer`.

    Args:
        input_file (Path): Path to the input text file.
        output_dir (Path): Directory to save the analysis output file.
        map_dir (Path): Directory containing (or to contain) the map file.
        config (Dict[str, Any]): Application configuration dictionary (used for paths).
        analysis_service (AnalysisService): An initialized and injected instance responsible
                                           for analysis and providing a `MetricsTracker`.

    Raises:
        FileNotFoundError: If `create_conversation_map` fails due to missing input file.
        ValueError: If context building or analysis returns unexpected results or raises
                    an error within the `analysis_service`.
        OSError: If map/output directory creation fails, or if propagated from `_result_writer`.
        Exception: For other critical, unexpected errors during processing.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Processing file: {input_file.name}")
    # Start timer for this file
    file_timer_start = time.monotonic()
    metrics_tracker.start_file_timer(input_file.name)

    # Derive output and map filenames using the utility
    try:
        map_suffix = config.get("paths", {}).get("map_suffix", "_map.jsonl")
        analysis_suffix = config.get("paths", {}).get("analysis_suffix", "_analysis.jsonl")
        # Generate paths using the helper
        pipeline_paths = generate_pipeline_paths(
            input_file=input_file, 
            map_dir=map_dir, 
            output_dir=output_dir, 
            map_suffix=map_suffix, 
            analysis_suffix=analysis_suffix, 
            task_id=task_id
        )
    except ValueError as e: # Handle potential error from path generation
        logger.error(f"{prefix}Failed to generate pipeline paths for {input_file.name}: {e}")
        metrics_tracker.increment_errors(input_file.name)
        metrics_tracker.stop_file_timer(input_file.name)
        raise

    # Ensure output/map directories exist (create_map also does map_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Create Conversation Map --- 
    try:
        num_sentences, sentences = await _handle_map_creation(input_file, map_dir, map_suffix, metrics_tracker, task_id)
        if num_sentences == 0:
             logger.warning(f"{prefix}Skipping analysis for {input_file.name} as it contains no sentences.")
             metrics_tracker.stop_file_timer(input_file.name)
             return # Exit early if no sentences
    except FileNotFoundError as e:
        logger.error(f"{prefix}Input file not found during map creation: {e}")
        metrics_tracker.increment_errors(input_file.name)
        metrics_tracker.stop_file_timer(input_file.name)
        raise # Re-raise to be caught by run_pipeline
    except OSError as e:
        logger.error(f"{prefix}OS error during map creation for {input_file.name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file.name)
        metrics_tracker.stop_file_timer(input_file.name)
        raise # Re-raise
    except Exception as e:
        logger.error(f"{prefix}Unexpected error during map creation for {input_file.name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file.name)
        metrics_tracker.stop_file_timer(input_file.name)
        raise # Re-raise

    # --- Step 2: Build Contexts --- 
    try:
        contexts = _handle_context_building(sentences, analysis_service, metrics_tracker, input_file.name, task_id)
    except Exception:
        # Error logged within helper, stop timer is handled in finally
        raise # Re-raise to ensure timer stops and error propagates

    # --- Step 3: Orchestrate Analysis & Writing --- 
    try:
        # This function now contains the core analysis call and writer task management
        await _orchestrate_analysis_and_writing(
            sentences, 
            contexts, 
            pipeline_paths.analysis_file, 
            analysis_service, 
            metrics_tracker, 
            input_file.name, 
            task_id
        )
    except Exception:
        # Error logged within helper, stop timer is handled in finally
        raise # Re-raise
        
    # --- Outer Finally Block --- 
    # This ensures the timer stops even if map creation or context building failed
    # The analysis/writing helper has its own internal finally for cancellation
    finally:
        metrics_tracker.stop_file_timer(input_file.name)
        elapsed_time = time.monotonic() - file_timer_start
        logger.info(f"{prefix}Finished processing {input_file.name}. Time taken: {elapsed_time:.2f} seconds.")


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
    Creates and executes concurrent tasks for processing each file.

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
    async def process_with_semaphore(file_path):
        async with semaphore:
            # Pass task_id down to process_file (assuming process_file signature matches)
            # Need to ensure process_file exists and has the correct signature
            # Let's assume it does for now.
            await process_file(
                file_path, 
                env.output_dir_path, 
                env.map_dir_path, 
                env.config_dict, 
                env.analysis_service, 
                env.metrics_tracker, 
                task_id # Pass task_id down
            )

    logger.info(f"{prefix}Scheduling {total_files} file processing tasks...")
    for i, file_path in enumerate(files_to_process):
        # logger.info(f"{prefix}Scheduling processing for file {i+1}/{total_files}: {file_path.name}") # Reduced verbosity
        tasks.append(process_with_semaphore(file_path))

    logger.info(f"{prefix}Starting concurrent file processing with {env.num_concurrent_files} workers...")
    # Use gather with return_exceptions=True 
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


def _run_verification(
    files_to_process: List[Path],
    env: PipelineEnvironment, # Pass the setup environment object
    task_id: Optional[str] = None
) -> None:
    """
    Runs the output completeness verification step for all processed files.

    Args:
        files_to_process: List of file paths that were attempted.
        env: The PipelineEnvironment object containing paths and suffixes.
        task_id: Optional task identifier for logging.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Starting output verification...")
    verification_results = []
    
    # Ensure we only verify files that potentially ran (might include failed ones)
    if not files_to_process:
        logger.info(f"{prefix}No files were processed, skipping verification.")
        return
        
    for file_path in files_to_process:
        try:
            # Use helper function to get paths for verification
            paths = generate_pipeline_paths(
                input_file=file_path, 
                map_dir=env.map_dir_path, 
                output_dir=env.output_dir_path, 
                map_suffix=env.map_suffix, 
                analysis_suffix=env.analysis_suffix, 
                task_id=task_id
            )
            verification_result = verify_output_completeness(paths.map_file, paths.analysis_file)
            
        except ValueError as e: # Error generating paths
            logger.error(f"{prefix}Skipping verification for {file_path.name}, cannot generate paths: {e}")
            verification_result = {
                 "total_expected": 0, "total_actual": 0, "total_missing": 0,
                 "missing_ids": [], "error": f"Path generation error: {e}"
            }
        except Exception as e: # Catch other potential errors during verification setup
             logger.error(f"{prefix}Unexpected error setting up verification for {file_path.name}: {e}", exc_info=True)
             verification_result = {
                 "total_expected": 0, "total_actual": 0, "total_missing": 0,
                 "missing_ids": [], "error": f"Verification setup error: {type(e).__name__}"
             }
             
        verification_results.append(verification_result)
        
        # Log individual file verification details
        if verification_result.get("error"):
            logger.warning(f"{prefix}Verification check for {file_path.name}: ERROR - {verification_result['error']}")
        elif verification_result["total_missing"] > 0:
            logger.warning(f"{prefix}Verification check for {file_path.name}: MISSING {verification_result['total_missing']}/{verification_result['total_expected']} sentences. Missing IDs: {verification_result['missing_ids'][:10]}..." if verification_result['missing_ids'] else "")
        else:
            # Avoid division by zero if total_expected is 0 (e.g., empty input file)
            expected_count = verification_result['total_expected']
            actual_count = verification_result['total_actual']
            logger.info(f"{prefix}Verification check for {file_path.name}: OK ({actual_count}/{expected_count} sentences found).")
            
    # Log overall verification summary (optional)
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

    Orchestrates the pipeline by calling helper functions for setup, discovery,
    processing, logging, and verification.

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
        _run_verification(files_to_process, env, task_id)

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
    input_file_path: Path,
    sentence_ids: List[int],
    config: Dict[str, Any],
    analysis_service: AnalysisService, # Inject AnalysisService
    task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Analyzes specific sentences identified by their IDs within a given input file.

    Reads the corresponding map file to find the text of all sentences, then extracts
    the target sentences and their contexts. Finally, calls the analysis service
    for the target sentences and returns the remapped results.

    Args:
        input_file_path (Path): Path to the original input text file (used to find the map).
        sentence_ids (List[int]): List of sentence IDs to analyze.
        config (Dict[str, Any]): Application configuration.
        analysis_service (AnalysisService): Injected analysis service instance.
        task_id (Optional[str]): Task identifier for logging.

    Returns:
        List[Dict[str, Any]]: List of analysis result dictionaries for the requested sentences.

    Raises:
        FileNotFoundError: If the map file cannot be found.
        ValueError: If requested sentence IDs are not found in the map file.
        Exception: Propagates exceptions from analysis service or file reading.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Starting analysis for specific sentences in: {input_file_path.name}, IDs: {sentence_ids}")

    # --- Configuration and Path Setup --- 
    try:
        map_dir_str = config.get("paths", {}).get("map_dir", "./data/maps")
        map_suffix = config.get("paths", {}).get("map_suffix", "_map.jsonl")
        # Analysis path details not strictly needed here, but need valid args for helper
        output_dir_str = config.get("paths", {}).get("output_dir", "./data/output") # Dummy
        analysis_suffix = config.get("paths", {}).get("analysis_suffix", "_analysis.jsonl") # Dummy

        map_dir = Path(map_dir_str)
        output_dir = Path(output_dir_str) # Dummy path object
        
        # Use path helper to get the map file path
        pipeline_paths = generate_pipeline_paths(
            input_file=input_file_path,
            map_dir=map_dir,
            output_dir=output_dir, # Pass dummy output dir
            map_suffix=map_suffix,
            analysis_suffix=analysis_suffix, # Pass dummy analysis suffix
            task_id=task_id
        )
        map_file = pipeline_paths.map_file
        # map_file = map_dir / f"{input_file_path.stem}{map_suffix}" # Old way
    except KeyError as e:
        logger.error(f"{prefix}Configuration missing required path key: {e}")
        raise ValueError(f"Configuration missing required path key: {e}") from e
    except ValueError as e: # Catch error from generate_pipeline_paths
        logger.error(f"{prefix}Failed to generate map path for {input_file_path.name}: {e}")
        raise
    except Exception as e:
        logger.error(f"{prefix}Unexpected error during path setup: {e}", exc_info=True)
        raise

    # --- Read Map File --- 
    logger.info(f"{prefix}Reading map file: {map_file}")
    if not map_file.is_file():
        logger.error(f"{prefix}Map file not found: {map_file}")
        raise FileNotFoundError(f"Map file not found: {map_file}")

    # Read all sentences from map file first
    all_sentences_map: Dict[int, str] = {}
    try:
        with map_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "sentence_id" in entry and "sentence" in entry:
                        all_sentences_map[entry["sentence_id"]] = entry["sentence"]
                except json.JSONDecodeError:
                    logger.warning(f"{prefix}Skipping malformed line in map file {map_file}")
                    continue
    except OSError as e:
        logger.error(f"{prefix}Error reading map file {map_file}: {e}", exc_info=True)
        raise
        
    # --- Prepare Sentences and Check IDs --- 
    target_sentences: List[str] = []
    target_indices: List[int] = [] # Store original indices for context building
    missing_ids = []
    
    # Convert map dict to ordered list based on sentence_id keys for context building
    # Find max ID to determine list size
    max_id = -1
    if all_sentences_map:
        max_id = max(all_sentences_map.keys())
        
    full_sentence_list_for_context = ["" for _ in range(max_id + 1)]
    for s_id, text in all_sentences_map.items():
        if 0 <= s_id <= max_id:
             full_sentence_list_for_context[s_id] = text
        else:
             logger.warning(f"{prefix}Found sentence ID {s_id} outside expected range [0, {max_id}] in map file {map_file}")
             
    # Check if requested IDs exist and get corresponding sentences
    target_id_set = set(sentence_ids)
    for target_id in target_id_set:
        if target_id in all_sentences_map:
            target_sentences.append(all_sentences_map[target_id])
            target_indices.append(target_id) # Store the original index
        else:
            missing_ids.append(target_id)

    if missing_ids:
        logger.error(f"{prefix}Requested sentence IDs not found in map file {map_file}: {sorted(missing_ids)}")
        raise ValueError(f"Sentence IDs not found in map: {sorted(missing_ids)}")

    if not target_sentences:
        logger.warning(f"{prefix}No valid target sentences found after checking map. Returning empty list.")
        return []

    # --- Build Contexts for Target Sentences ONLY --- 
    # We need the full sentence list context is built relative to the original positions.
    # Build contexts only for the indices corresponding to target_ids.
    logger.info(f"{prefix}Building contexts for {len(target_indices)} specific sentences...")
    try:
        # Use the full list for context building, but only build for specific indices
        all_contexts_dict = analysis_service.context_builder.build_all_contexts(full_sentence_list_for_context)
        # Extract contexts only for the target indices
        target_contexts = [all_contexts_dict.get(idx, {}) for idx in target_indices]
        
        # --- Validate context length --- 
        if len(target_contexts) != len(target_sentences):
             logger.error(f"{prefix}Context count ({len(target_contexts)}) mismatch with target sentence count ({len(target_sentences)}) after context building.")
             # Handle this error - perhaps raise or return partial?
             # Raising for now, as it indicates a logic error.
             raise RuntimeError("Context and sentence count mismatch during specific analysis.")
             
    except Exception as e:
        logger.error(f"{prefix}Failed to build contexts for specific sentences: {e}", exc_info=True)
        raise

    # --- Analyze Specific Sentences --- 
    logger.info(f"{prefix}Analyzing {len(target_sentences)} specific sentences...")
    try:
        # Pass only the target sentences and their corresponding contexts
        # Pass task_id as well
        analysis_results = await analysis_service.analyze_sentences(
            target_sentences, target_contexts, task_id=task_id
        )
    except Exception as e:
         logger.error(f"{prefix}Error during specific sentence analysis: {e}", exc_info=True)
         raise
         
    # --- Post-Process Results (if needed) --- 
    # The results from analyze_sentences might need adjustment.
    # Currently, analyze_sentences uses the *index within the passed list* for sequence_order.
    # We need to map this back to the original sentence_id.
    # Let's modify the return to ensure original sentence_id is preserved.
    
    # Assuming analyze_sentences adds `sequence_order` based on the input list index.
    # We need to replace that with the original `sentence_id`.
    final_results = []
    if len(analysis_results) != len(target_indices):
        logger.warning(f"{prefix}Result count ({len(analysis_results)}) mismatch with target index count ({len(target_indices)}). Results might be incomplete.")
        # Decide how to handle - return partial, raise error? Returning partial for now.
        
    for i, result in enumerate(analysis_results):
         if i < len(target_indices):
              original_sentence_id = target_indices[i]
              result['sentence_id'] = original_sentence_id
              # Overwrite sequence_order if it was based on the sublist index
              result['sequence_order'] = original_sentence_id 
              final_results.append(result)
         else:
              logger.warning(f"{prefix}Extra result found at index {i} beyond the number of target indices.")

    logger.info(f"{prefix}Finished specific sentence analysis. Returning {len(final_results)} results.")
    return final_results
