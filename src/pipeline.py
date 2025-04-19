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
    logger.info(f"{prefix}Result writer started for: {output_file}")
    results_written = 0
    while True:
        try:
            result = await results_queue.get()
            if result is None: # Sentinel value indicates completion
                logger.info(f"{prefix}Writer received sentinel. Exiting.")
                results_queue.task_done() # Mark sentinel as processed
                break

            try:
                append_json_line(result, output_file)
                results_written += 1
                logger.debug(f"{prefix}Writer appended result for sentence_id: {result.get('sentence_id', 'N/A')}")
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
    logger.info(f"{prefix}Result writer finished for: {output_file}. Total results written: {results_written}")

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

    # Initialize AnalysisService (moved to run_pipeline)
    # analyzer = SentenceAnalyzer(config=config)
    # context_builder = ContextBuilder(config_dict=config)
    # analysis_service = AnalysisService(
    #     config=config, 
    #     context_builder=context_builder, 
    #     sentence_analyzer=analyzer, 
    #     metrics_tracker=metrics_tracker 
    # )

    # Derive output and map filenames
    analysis_suffix = config.get("paths", {}).get("analysis_suffix", "_analysis.jsonl")
    map_suffix = config.get("paths", {}).get("map_suffix", "_map.jsonl")
    output_file = output_dir / f"{input_file.stem}{analysis_suffix}"
    # map_file_path = map_dir / f"{input_file.stem}{map_suffix}" # Map path handled by create_map

    # Ensure output/map directories exist (create_map also does map_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Create Conversation Map --- 
    try:
        num_sentences, sentences = await create_conversation_map(input_file, map_dir, map_suffix, task_id)
        metrics_tracker.set_metric(input_file.name, "sentences_found_in_map", num_sentences)
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

    # --- Step 2: Build Contexts (Part of AnalysisService) --- 
    # Moved context building logic inside AnalysisService.build_contexts
    try:
        contexts = analysis_service.build_contexts(sentences)
    except Exception as e:
        logger.error(f"{prefix}Failed to build contexts for {input_file.name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file.name)
        metrics_tracker.stop_file_timer(input_file.name)
        # Depending on policy, we might want to raise here or try to continue without context?
        # Raising seems safer as subsequent analysis depends on context.
        raise 

    # --- Step 3: Analyze Sentences & Write Results --- 
    results_queue = asyncio.Queue()
    # Pass metrics_tracker and task_id to writer
    writer_task = asyncio.create_task(_result_writer(output_file, results_queue, metrics_tracker, task_id))

    try:
        # Use AnalysisService to perform analysis, putting results on queue
        # Analyze sentences - this now happens internally in AnalysisService
        # Pass the actual results queue to the service method (if designed that way)
        # OR have the service method return the results list directly.
        
        # Assuming analyze_sentences returns a list of result dicts
        logger.info(f"{prefix}Starting sentence analysis for {input_file.name} using AnalysisService...")
        analysis_results = await analysis_service.analyze_sentences(sentences, contexts)
        logger.info(f"{prefix}AnalysisService completed analysis for {input_file.name}. Found {len(analysis_results)} results.")
        
        # --- Step 4: Write results using the writer task --- 
        logger.info(f"{prefix}Queueing {len(analysis_results)} analysis results for writing...")
        for result in analysis_results:
            await results_queue.put(result)
            metrics_tracker.increment_results_processed(input_file.name)

        # Signal writer completion
        await results_queue.put(None) 
        await writer_task # Wait for writer to finish processing queue
        logger.info(f"{prefix}Result writing complete for {input_file.name}.")
        metrics_tracker.set_metric(input_file.name, "results_written", len(analysis_results)) # Track written count

    except Exception as e:
        logger.error(f"{prefix}Error during sentence analysis or writing for {input_file.name}: {e}", exc_info=True)
        metrics_tracker.increment_errors(input_file.name)
        # Ensure writer task is cancelled if analysis fails mid-way
        if not writer_task.done():
            writer_task.cancel()
            try:
                await writer_task # Wait for cancellation to complete
            except asyncio.CancelledError:
                 logger.info(f"{prefix}Result writer task successfully cancelled for {input_file.name}.")
            except Exception as cancel_e:
                 logger.error(f"{prefix}Error awaiting writer task cancellation for {input_file.name}: {cancel_e}")
        raise # Re-raise the analysis/writing error
    finally:
        # Stop timer for this file regardless of success/failure in analysis/writing stage
        metrics_tracker.stop_file_timer(input_file.name)
        elapsed_time = time.monotonic() - file_timer_start
        logger.info(f"{prefix}Finished processing {input_file.name}. Time taken: {elapsed_time:.2f} seconds.")


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

    Instantiates necessary components (`ContextBuilder`, `SentenceAnalyzer`,
    `MetricsTracker`, `AnalysisService`) once for the run. Iterates through
    text files, processing each using `process_file` with the *same* injected
    `AnalysisService` instance. Finally, verifies output completeness against
    map files and logs a summary of the run.

    Args:
        input_dir (Path): Directory containing input .txt files.
        output_dir (Path): Directory to save analysis output files.
        map_dir (Path): Directory to save conversation map files.
        config (Dict[str, Any]): Application configuration dictionary.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Starting pipeline run...")
    
    # --- Use provided config or load global --- 
    if config is None:
        from src.config import config as global_config # Import locally
        config_obj = global_config
        config_dict = config_obj.config # Assign the dict from the loaded Config object
    else:
        # If a dict is passed, wrap it temporarily or assume it has needed methods/keys
        # For simplicity, assume passed config is the dict from the Config instance
        # config_obj = config # This assumes it IS the dict
        # Safer: If DI requires the Config object, pass that instead of the dict.
        # For now, let's assume the functions below just need the dictionary part.
        # If process_file or services expect the Config class instance, this needs adjustment.
        # Let's assume the structure config['paths']['key'] works.
        config_dict = config # Rename for clarity
    
    # Determine directories using provided args or config defaults
    # Ensure Paths are created correctly
    try:
        input_dir_path = Path(input_dir) # Input is required
        
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
    output_dir_path.mkdir(parents=True, exist_ok=True)
    map_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize MetricsTracker (should probably be singleton or passed in)
    # For now, re-importing the singleton instance
    from src.utils.metrics import metrics_tracker
    # Don't reset here if called from API; API call should manage overall timer
    # metrics_tracker.reset()
    # metrics_tracker.start_pipeline_timer()

    # --- Instantiate Services (outside the loop for reuse) --- 
    try:
        logger.debug(f"{prefix}Instantiating AnalysisService and its dependencies...")
        # Pass the config dictionary to components that need it
        context_builder_instance = ContextBuilder(config_dict=config_dict)
        sentence_analyzer_instance = SentenceAnalyzer(config_dict=config_dict)
        analysis_service_instance = AnalysisService(
            config=config_dict,
            context_builder=context_builder_instance,
            sentence_analyzer=sentence_analyzer_instance,
            metrics_tracker=metrics_tracker # Pass the singleton tracker
        )
        logger.debug(f"{prefix}AnalysisService instantiated.")
    except Exception as e:
        logger.critical(f"{prefix}Failed to instantiate core services: {e}", exc_info=True)
        raise RuntimeError("Failed to instantiate core services") from e

    # --- Determine files to process --- 
    if specific_file:
        files_to_process = [input_dir_path / specific_file]
        if not files_to_process[0].is_file():
            logger.error(f"{prefix}Specified input file not found: {files_to_process[0]}")
            metrics_tracker.increment_errors() # Track pipeline setup error
            raise FileNotFoundError(f"Specified input file not found: {files_to_process[0]}")
        logger.info(f"{prefix}Processing specific file: {specific_file}")
    else:
        # Find all .txt files in the input directory
        files_to_process = list(input_dir_path.glob("*.txt"))
        if not files_to_process:
            logger.warning(f"{prefix}No .txt files found in input directory: {input_dir_path}")
            # Stop timer if started, log summary, and exit cleanly
            # metrics_tracker.stop_pipeline_timer()
            # summary = metrics_tracker.get_summary()
            # logger.info(f"Pipeline Execution Summary (No files processed): {json.dumps(summary, indent=2)}")
            return
        logger.info(f"{prefix}Found {len(files_to_process)} .txt files to process.")

    # --- Process Files Concurrently --- 
    semaphore = asyncio.Semaphore(num_concurrent_files)
    tasks = []
    total_files = len(files_to_process)
    metrics_tracker.set_metric("pipeline", "total_files_to_process", total_files)

    async def process_with_semaphore(file_path):
        async with semaphore:
            # Pass task_id down to process_file
            await process_file(
                file_path, output_dir_path, map_dir_path, config_dict, 
                analysis_service_instance, metrics_tracker, task_id
            )

    for i, file_path in enumerate(files_to_process):
        logger.info(f"{prefix}Scheduling processing for file {i+1}/{total_files}: {file_path.name}")
        tasks.append(process_with_semaphore(file_path))

    # Execute tasks concurrently
    logger.info(f"{prefix}Starting concurrent file processing...")
    # Use gather with return_exceptions=True to ensure all tasks run
    # even if some fail, allowing us to capture metrics for all attempts.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"{prefix}Concurrent file processing finished.")

    # --- Log results and handle exceptions from gather --- 
    files_processed_successfully = 0
    files_failed = 0
    for i, result in enumerate(results):
        file_path = files_to_process[i] # Get corresponding file path
        if isinstance(result, Exception):
            files_failed += 1
            logger.error(f"{prefix}Processing failed for {file_path.name}: {type(result).__name__}: {result}")
            # Detailed error logged within process_file or its callees
        else:
            # Successful completion (process_file doesn't return anything on success)
            files_processed_successfully += 1
            logger.info(f"{prefix}Successfully processed {file_path.name}.")
            
    # Update overall pipeline metrics
    metrics_tracker.set_metric("pipeline", "files_processed_successfully", files_processed_successfully)
    metrics_tracker.set_metric("pipeline", "files_failed", files_failed)

    # --- Verification Step --- 
    logger.info(f"{prefix}Starting output verification...")
    verification_results = []
    for file_path in files_to_process:
        map_path = map_dir_path / f"{file_path.stem}{map_suffix}"
        analysis_path = output_dir_path / f"{file_path.stem}{analysis_suffix}"
        verification_result = verify_output_completeness(map_path, analysis_path)
        verification_results.append(verification_result)
        # Log individual file verification details
        if verification_result.get("error"):
            logger.warning(f"{prefix}Verification check for {file_path.name} encountered an error: {verification_result['error']}")
        elif verification_result["total_missing"] > 0:
            logger.warning(f"{prefix}Verification check for {file_path.name}: MISSING {verification_result['total_missing']}/{verification_result['total_expected']} sentences. Missing IDs: {verification_result['missing_ids'][:10]}..." if verification_result['missing_ids'] else "")
        else:
            logger.info(f"{prefix}Verification check for {file_path.name}: OK ({verification_result['total_actual']}/{verification_result['total_expected']} sentences found)." )
            
    # Log overall verification summary (optional)
    total_missing_overall = sum(vr["total_missing"] for vr in verification_results if vr.get("error") is None)
    total_expected_overall = sum(vr["total_expected"] for vr in verification_results if vr.get("error") is None)
    verification_errors = sum(1 for vr in verification_results if vr.get("error") is not None)
    logger.info(f"{prefix}Verification Summary: Total Missing Sentences={total_missing_overall}/{total_expected_overall}, Verification Errors={verification_errors}")
    metrics_tracker.set_metric("pipeline", "verification_total_missing", total_missing_overall)
    metrics_tracker.set_metric("pipeline", "verification_errors", verification_errors)

    # --- Final Logging --- 
    # Pipeline timer stopped by the caller (main() or API endpoint wrapper if needed)
    # summary = metrics_tracker.get_summary()
    # logger.info(f"Pipeline Execution Summary:\n{json.dumps(summary, indent=2)}") # Commented out problematic line
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
    Analyzes only specific sentences from an input file.

    This function is intended for re-analysis or targeted analysis.
    It reads the corresponding map file to get sentence text and then uses
    the AnalysisService to build contexts and analyze only the specified IDs.

    Args:
        input_file_path (Path): Path to the original input text file.
        sentence_ids (List[int]): A list of sentence IDs to analyze.
        config (Dict[str, Any]): The application configuration dictionary.
        analysis_service (AnalysisService): Injected instance for analysis.
        task_id (Optional[str]): Unique ID for tracking/logging.

    Returns:
        List[Dict[str, Any]]: A list of analysis result dictionaries for the
                              specified sentence IDs.

    Raises:
        FileNotFoundError: If the input file or its corresponding map file is not found.
        ValueError: If a requested sentence_id is not found in the map file.
        Exception: Propagates errors from context building or sentence analysis.
    """
    prefix = _log_prefix(task_id)
    logger.info(f"{prefix}Starting specific sentence analysis for {input_file_path.name}, IDs: {sentence_ids}")
    if not sentence_ids:
        logger.warning(f"{prefix}No sentence IDs provided for specific analysis. Returning empty list.")
        return []
        
    if not input_file_path.is_file():
        logger.error(f"{prefix}Input file not found for specific analysis: {input_file_path}")
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    # --- Find and Read Map File --- 
    map_suffix = config.get("paths", {}).get("map_suffix", "_map.jsonl")
    map_dir_str = config.get("paths", {}).get("map_dir", "./data/maps")
    map_dir = Path(map_dir_str)
    map_file = map_dir / f"{input_file_path.stem}{map_suffix}"

    if not map_file.is_file():
        logger.error(f"{prefix}Map file not found for specific analysis: {map_file}")
        raise FileNotFoundError(f"Map file not found for {input_file_path.name}: {map_file}")

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
        analysis_results = await analysis_service.analyze_sentences(target_sentences, target_contexts)
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
