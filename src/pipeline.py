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
from typing import Dict, Any, List, Tuple, Set, Optional
from src.services.analysis_service import AnalysisService
import time

logger = get_logger()

async def create_conversation_map(input_file: Path, map_dir: Path, map_suffix: str) -> Tuple[int, List[str]]:
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
    logger.info(f"Creating conversation map for: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    map_file = map_dir / f"{input_file.stem}{map_suffix}"
    map_dir.mkdir(parents=True, exist_ok=True)

    text = input_file.read_text(encoding="utf-8")
    # Use the imported segment_text function
    sentences = segment_text(text)
    num_sentences = len(sentences)
    
    # Check for empty sentences list early
    if num_sentences == 0:
        logger.warning(f"Input file {input_file} contains no sentences after segmentation. Map file will be empty.")
        # Create an empty map file
        map_file.touch()
        return 0, [] # Return 0 sentences and empty list

    with map_file.open("w", encoding="utf-8") as f:
        for idx, sentence_text in enumerate(sentences):
            entry = {
                "sentence_id": idx,
                "sequence_order": idx,
                "sentence": sentence_text
            }
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

    logger.info(f"Conversation map created: {map_file} with {num_sentences} sentences.")
    return num_sentences, sentences

async def _result_writer(output_file: Path, results_queue: asyncio.Queue, metrics_tracker: MetricsTracker):
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
    logger.info(f"Result writer started for: {output_file}")
    results_written = 0
    while True:
        try:
            result = await results_queue.get()
            if result is None: # Sentinel value indicates completion
                logger.info("Writer received sentinel. Exiting.")
                results_queue.task_done() # Mark sentinel as processed
                break

            try:
                append_json_line(result, output_file)
                results_written += 1
                logger.debug(f"Writer appended result for sentence_id: {result.get('sentence_id', 'N/A')}")
                # Optionally track write success metric
                # metrics_tracker.increment_write_success()
            except Exception as e:
                # Log error details including sentence ID if available
                sentence_id_info = f"result {result.get('sentence_id', 'N/A')}" if isinstance(result, dict) else "result (unknown ID)"
                logger.error(f"Writer failed writing {sentence_id_info} to {output_file}: {e}", exc_info=True)
                metrics_tracker.increment_errors() # Use passed-in tracker
            finally:
                results_queue.task_done() # Mark result as processed

        except asyncio.CancelledError:
            logger.info(f"Writer for {output_file} cancelled.")
            break
        except Exception as e:
            logger.critical(f"Critical error in writer for {output_file}: {e}", exc_info=True)
            break # Exit loop on critical error
    logger.info(f"Result writer finished for: {output_file}. Total results written: {results_written}")

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
    analysis_service: AnalysisService # Inject AnalysisService
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
    logger.info(f"Processing file: {input_file}")
    analysis_file_path = output_dir / f"{input_file.stem}{config.get('paths', {}).get('analysis_suffix', '_analysis.jsonl')}"
    map_suffix = config.get('paths', {}).get('map_suffix', '_map.jsonl')
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tracker instance from the service early on
    metrics_tracker = analysis_service.metrics_tracker

    try:
        # 1. Create map file and get sentences
        num_sentences, sentences = await create_conversation_map(input_file, map_dir, map_suffix)
        if num_sentences == 0:
            logger.warning(f"Skipping analysis for {input_file} as it has no sentences.")
            return
            
        # 2. Use the injected AnalysisService

        # 3. Build Contexts using the service
        logger.info(f"Building context for {num_sentences} sentences...")
        contexts = analysis_service.build_contexts(sentences) # Call on injected instance
        if not contexts or len(contexts) != num_sentences:
             logger.error(f"Context building failed or returned unexpected count for {input_file}. Expected {num_sentences}, got {len(contexts)}. Aborting analysis.")
             # Track all sentences as errors if context fails
             metrics_tracker.increment_errors(num_sentences) 
             return # Stop processing this file
        logger.info("Context building complete.")

        # 4. Analyze Sentences using the service
        logger.info(f"[{input_file.name}] Analyzing {num_sentences} sentences...")
        # This call might raise errors (e.g., ValueError from LLM) which will be caught below
        analysis_results = await analysis_service.analyze_sentences(sentences, contexts) 
        logger.info(f"[{input_file.name}] Analysis complete. Received {len(analysis_results)} results.")
        
        # Check if analysis returned anything before writing
        if not analysis_results:
            logger.warning(f"[{input_file.name}] Analysis service returned no results. Output file will not be created/modified.")
            # If analysis ran but gave no results, consider if this is an error state
            # Currently, we just stop processing the file.
            return

        # 5. Write results asynchronously
        logger.debug(f"[{input_file.name}] Preparing to write {len(analysis_results)} results.")
        temp_results_queue = asyncio.Queue()
        for result in analysis_results:
            await temp_results_queue.put(result)
        await temp_results_queue.put(None) # Sentinel for the writer
        logger.debug(f"[{input_file.name}] Results loaded onto temp queue.")

        logger.info(f"[{input_file.name}] Starting result writer task for {analysis_file_path}...")
        # Pass the metrics_tracker instance to the writer
        writer_task = asyncio.create_task(_result_writer(analysis_file_path, temp_results_queue, metrics_tracker))
        
        logger.debug(f"[{input_file.name}] Awaiting result writer task...")
        await writer_task # Await completion. Errors inside _result_writer are logged there.
        logger.debug(f"[{input_file.name}] Result writer task finished.")
        
        # If writer_task completed without propagating an exception, assume success
        logger.info(f"[{input_file.name}] Successfully processed and wrote results to {analysis_file_path}.")
        metrics_tracker.increment_files_processed()

    except FileNotFoundError as e:
        logger.error(f"File not found during processing {input_file}: {e}")
        metrics_tracker.increment_errors()
    except (ValueError, OSError) as e: # Catch specific errors from analysis/IO
        logger.error(f"Error processing file {input_file}: {type(e).__name__}: {e}")
        metrics_tracker.increment_errors(num_sentences if 'num_sentences' in locals() else 1) # Count all sentences as error
        # Re-raise the exception to allow the caller (run_pipeline) or tests to see it
        raise 
    except Exception as e:
        logger.critical(f"Critical unexpected error processing file {input_file}: {e}", exc_info=True)
        metrics_tracker.increment_errors(num_sentences if 'num_sentences' in locals() else 1)
        raise # Re-raise critical errors too


async def run_pipeline(input_dir: Path, output_dir: Path, map_dir: Path, config: Dict[str, Any]):
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
    logger.info(f"Starting pipeline run. Input: {input_dir}, Output: {output_dir}, Maps: {map_dir}")
    start_time = time.monotonic()

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob("*.txt"))
    if not input_files:
        logger.warning(f"No input files found in {input_dir}")
        return

    # Instantiate dependencies needed for AnalysisService using the provided config
    try:
        logger.debug("Instantiating pipeline dependencies...")
        # Import necessary classes locally if not already imported at module level
        from src.agents.context_builder import ContextBuilder
        from src.agents.sentence_analyzer import SentenceAnalyzer
        
        # Instantiate components with config
        context_builder_instance = ContextBuilder(config_dict=config)
        sentence_analyzer_instance = SentenceAnalyzer(config_dict=config)
        metrics_tracker_instance = MetricsTracker() # Assuming no config needed for init

        # Instantiate AnalysisService ONCE with its dependencies
        analysis_service = AnalysisService(
            config=config,
            context_builder=context_builder_instance,
            sentence_analyzer=sentence_analyzer_instance,
            metrics_tracker=metrics_tracker_instance # Pass the new instance
        )
        logger.info("AnalysisService instantiated for pipeline run.")
    except Exception as e:
        logger.critical(f"Failed to instantiate pipeline components or AnalysisService: {e}", exc_info=True)
        return # Cannot proceed without the service

    processed_files_for_verification = []
    map_suffix = config.get('paths', {}).get('map_suffix', '_map.jsonl')
    analysis_suffix = config.get('paths', {}).get('analysis_suffix', '_analysis.jsonl')

    # Process files sequentially for now, could be parallelized
    process_tasks = []
    for input_file in input_files:
        logger.info(f"--- Scheduling processing for {input_file.name} ---")
        # Pass the SAME analysis_service instance to each process_file call
        task = asyncio.create_task(
            process_file(input_file, output_dir, map_dir, config, analysis_service),
            name=f"process_{input_file.name}"
        )
        process_tasks.append(task)
        # Store paths for verification regardless of processing success initially
        map_path = map_dir / f"{input_file.stem}{map_suffix}"
        analysis_path = output_dir / f"{input_file.stem}{analysis_suffix}"
        processed_files_for_verification.append((map_path, analysis_path))

    # Await all processing tasks concurrently
    if process_tasks:
        logger.info(f"Awaiting completion of {len(process_tasks)} file processing tasks...")
        await asyncio.gather(*process_tasks, return_exceptions=True) # Use return_exceptions to handle errors
        logger.info("All file processing tasks completed (or encountered errors).")

    # Verification step (checks files even if processing task failed)
    logger.info("--- Starting Output Verification --- ")
    total_expected = 0
    total_actual = 0
    total_missing = 0
    files_with_missing = 0
    files_with_errors = 0

    for map_path, analysis_path in processed_files_for_verification:
        logger.debug(f"Verifying: {analysis_path} against {map_path}")
        verification_result = verify_output_completeness(map_path, analysis_path)
        total_expected += verification_result["total_expected"]
        total_actual += verification_result["total_actual"]
        total_missing += verification_result["total_missing"]
        if verification_result["total_missing"] > 0:
            files_with_missing += 1
            logger.warning(f"Missing {verification_result['total_missing']} sentences for {map_path.name}. Missing IDs: {verification_result['missing_ids']}")
        if verification_result.get("error"): 
             files_with_errors += 1
             logger.warning(f"Verification error for {map_path.name}/{analysis_path.name}: {verification_result['error']}")

    # Log Verification Summary
    summary_level = logger.warning if total_missing > 0 or files_with_errors > 0 else logger.info
    summary_level(
        f"Verification Summary: Checked {len(processed_files_for_verification)} files. "
        f"Total Expected Sentences (from maps): {total_expected}. Total Actual Entries (in analysis files): {total_actual}. "
        f"Total Missing Sentences: {total_missing}{f' across {files_with_missing} files' if files_with_missing > 0 else ''}. "
        f"Verification Errors Encountered: {files_with_errors}."
    )

    # Log overall pipeline summary using metrics_tracker.get_summary()
    end_time = time.monotonic()
    pipeline_runtime = end_time - start_time
    metrics_summary = metrics_tracker_instance.get_summary()
    total_errors_reported = metrics_summary.get("total_errors", "N/A")
    total_files_processed_metric = metrics_summary.get("total_files_processed", "N/A")
    
    logger.info(
        f"Pipeline Execution Summary: "
        f"Files Attempted: {len(input_files)}. "
        f"Files Successfully Processed (by tracker): {total_files_processed_metric}. "
        f"Total Errors Reported by Tracker: {total_errors_reported}. " 
        f"Total Pipeline Runtime: {pipeline_runtime:.2f}s."
    )


async def analyze_specific_sentences(
    input_file_path: Path,
    sentence_ids: List[int],
    config: Dict[str, Any],
    analysis_service: AnalysisService # Inject AnalysisService
) -> List[Dict[str, Any]]:
    """
    Analyzes a specified list of sentences from an input file using an injected service.

    Ensures the conversation map file exists (creates it if necessary using
    `create_conversation_map`). Reads the map to get all sentences, then uses
    the injected `analysis_service` to build contexts for *all* sentences and
    analyze only the *requested* subset based on `sentence_ids`. Finally, it
    remaps the results back to their original sentence IDs.

    Args:
        input_file_path (Path): Path to the original input text file.
        sentence_ids (List[int]): A list of integer sentence IDs to analyze.
        config (Dict[str, Any]): Application configuration dictionary (used for paths).
        analysis_service (AnalysisService): An initialized and injected instance responsible
                                           for analysis.

    Returns:
        List[Dict[str, Any]]: A list of analysis result dictionaries for the requested
                              and successfully analyzed sentences.

    Raises:
        FileNotFoundError: If the input file `input_file_path` does not exist, or if
                           `create_conversation_map` fails to find it.
        ValueError: If context building or analysis raises an error within the service.
        Exception: For other critical, unexpected errors.
    """
    logger.info(f"Starting specific analysis for {len(sentence_ids)} sentences from {input_file_path}")
    if not sentence_ids:
        logger.warning("Received empty list of sentence IDs. No analysis to perform.")
        return []

    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    try:
        # Use injected service
        # analysis_service = analysis_service # Already passed as arg
        
        # Get paths from config (ensure defaults are reasonable)
        paths_config = config.get("paths", {})
        # input_dir = Path(paths_config.get("input_dir", ".")) # Not needed?
        # output_dir = Path(paths_config.get("output_dir", "./data/output")) # Not needed?
        map_dir = Path(paths_config.get("map_dir", "./data/maps"))
        map_suffix = paths_config.get("map_suffix", "_map.jsonl")
        # analysis_suffix = paths_config.get("analysis_suffix", "_analysis.jsonl") # Not needed?

        # Ensure map file exists or create it
        map_file_path = map_dir / f"{input_file_path.stem}{map_suffix}"

        if not map_file_path.exists():
            logger.info(f"Map file {map_file_path} not found. Creating it first.")
            try:
                # create_conversation_map raises FileNotFoundError if input_file_path is bad
                num_created, _ = await create_conversation_map(input_file_path, map_dir, map_suffix)
                if num_created == 0:
                    logger.warning(f"Map file created but input {input_file_path} had no sentences. Cannot proceed.")
                    return []
            except Exception as e:
                logger.error(f"Failed to create map file {map_file_path} for specific analysis: {e}")
                raise # Re-raise error if map creation fails
        
        # Read map file to get all sentences
        all_sentences_data = []
        try:
            with map_file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # Ensure required keys exist
                        if "sentence_id" in entry and "sentence" in entry:
                            all_sentences_data.append((entry["sentence_id"], entry["sentence"]))
                        else:
                            logger.warning(f"Skipping line in map file {map_file_path} due to missing keys: {line.strip()}")
                    except (json.JSONDecodeError) as e:
                        logger.error(f"Skipping invalid JSON line in map file {map_file_path}: {line.strip()} - Error: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error reading map file {map_file_path}: {e}")
            raise # Re-raise error if map reading fails

        if not all_sentences_data:
            logger.warning(f"Map file {map_file_path} is empty or invalid. Cannot analyze specific sentences.")
            return []

        # Sort by sentence ID just in case map isn't ordered
        all_sentences_data.sort(key=lambda x: x[0])
        all_sentences_text = [text for _, text in all_sentences_data]
        
        # Create mapping from original ID to its index AFTER sorting
        original_id_to_index = {s_id: index for index, (s_id, _) in enumerate(all_sentences_data)}

        # Filter sentence IDs requested against valid IDs found in the map
        valid_ids_in_map = set(original_id_to_index.keys())
        target_ids = sorted([s_id for s_id in set(sentence_ids) if s_id in valid_ids_in_map]) # Use set for unique IDs
        
        # Log skipped IDs
        skipped_ids = set(sentence_ids) - set(target_ids)
        if skipped_ids:
            logger.warning(f"Skipping requested sentence IDs not found in map file or invalid: {sorted(list(skipped_ids))}")

        if not target_ids:
            logger.warning(f"No valid sentence IDs found in map file for requested IDs: {sentence_ids}. No analysis performed.")
            return []

        # Build context for *all* sentences
        logger.info(f"Building context for all {len(all_sentences_text)} sentences...")
        all_contexts = analysis_service.build_contexts(all_sentences_text)
        if len(all_contexts) != len(all_sentences_text):
            # This indicates a critical internal error in build_contexts
            logger.error("Context count mismatch with total sentence count. Aborting specific analysis.")
            raise ValueError("ContextBuilder failed to return context for all sentences.")
        logger.info("Full context building complete.")
        
        # Select the target sentences and contexts based on target_ids
        sentences_to_analyze = []
        contexts_to_analyze = []
        original_indices_for_targets = [] # Store original ID for remapping
        
        for target_id in target_ids:
             original_index = original_id_to_index[target_id]
             # Ensure index is valid for both text and context lists (should always be if logic above is correct)
             if original_index < len(all_sentences_text) and original_index < len(all_contexts):
                 sentences_to_analyze.append(all_sentences_text[original_index])
                 contexts_to_analyze.append(all_contexts[original_index])
                 original_indices_for_targets.append(target_id) # Keep original ID
             else: 
                  # This case should ideally not be reachable if map reading/indexing is correct
                  logger.error(f"Internal error: Index {original_index} for target ID {target_id} out of bounds. Skipping.")

        if not sentences_to_analyze:
            logger.warning("Could not prepare any target sentences/contexts for analysis after filtering. Check logs for reasons.")
            return []

        # Call the analysis service for the subset
        logger.info(f"Calling analysis service for {len(sentences_to_analyze)} specific sentences (Original IDs: {original_indices_for_targets})...")
        # This call might raise errors (e.g., ValueError from LLM)
        specific_results = await analysis_service.analyze_sentences(sentences_to_analyze, contexts_to_analyze)

        # Remap results back to original sentence IDs and sequence order
        final_results = []
        if len(specific_results) == len(original_indices_for_targets):
            for i, result_dict in enumerate(specific_results):
                original_id = original_indices_for_targets[i]
                # Ensure basic structure before assigning
                if isinstance(result_dict, dict):
                    result_dict["sentence_id"] = original_id
                    result_dict["sequence_order"] = original_id # Use original ID for consistency
                    final_results.append(result_dict)
                else:
                    logger.warning(f"Received non-dict result from analyze_sentences for original ID {original_id}. Skipping result: {result_dict}")
            logger.info(f"Remapped {len(final_results)} results to original sentence IDs.")
        else:
            # This indicates an error within analyze_sentences if counts don't match
            logger.error(f"Result count ({len(specific_results)}) mismatch with targeted sentence count ({len(original_indices_for_targets)}). Cannot reliably map back IDs.")
            # Decide how to handle - return partial, empty, or raise? Returning partial for now.
            final_results = specific_results # Return potentially incorrectly mapped results
             
        return final_results

    except Exception as e:
        # Catch any unexpected error during the process
        logger.critical(f"Unexpected error during analyze_specific_sentences for {input_file_path}: {e}", exc_info=True)
        raise # Re-raise critical errors
