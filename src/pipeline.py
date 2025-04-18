"""
pipeline.py

This module defines the core functions for processing text files in the interview
analyzer pipeline. The pipeline consists of the following steps:

    1. Segmenting the input text into individual sentences using spaCy.
    2. Analyzing each sentence via the SentenceAnalyzer (which interacts with OpenAI's Responses API).
    3. Saving the analysis results to a JSON file for further processing.

Key functions:
    - segment_text: Splits input text into a list of sentences.
    - process_file: Processes a single text file by segmenting its content, analyzing the sentences,
      and saving the results.
    - run_pipeline: Iterates over all text files in an input directory and processes each file.

Usage:
    To run the pipeline for a set of text files, call run_pipeline with the input and output directories:
    
        await run_pipeline(Path("input_directory"), Path("output_directory"))
    
Modifications:
    - If the segmentation logic changes (e.g., different criteria for sentence boundaries), update segment_text.
    - If the analysis output or structure changes (e.g., additional fields), update process_file accordingly.
    - When altering file I/O behavior, ensure that both process_file and run_pipeline continue to handle errors
      (e.g., missing files or directories) appropriately.
      
Dependencies:
    - spaCy: For natural language processing and sentence segmentation.
    - SentenceAnalyzer: A class that performs sentence-level analysis using the OpenAI API.
    - append_json_line: A helper function to write JSON data to a file.
    - Logging: Uses a centralized logger for traceability.
"""

from pathlib import Path
from src.utils.helpers import append_json_line  # Helper function for saving JSON files.
import asyncio
from src.agents.sentence_analyzer import SentenceAnalyzer  # Class that analyzes sentences using OpenAI API.
from src.utils.logger import get_logger  # Centralized logger.
from src.agents.context_builder import context_builder
from src.utils.metrics import metrics_tracker
from src.utils.text_processing import segment_text # Import the new segment_text function
import json
from typing import Dict, Any, List, Tuple, Set, Optional
from src.services.analysis_service import AnalysisService # Import AnalysisService
import time

# Initialize the logger and load the spaCy model.
logger = get_logger()
# nlp = spacy.load("en_core_web_sm") # Removed global nlp instance

async def create_conversation_map(input_file: Path, map_dir: Path, map_suffix: str) -> Tuple[int, List[str]]:
    """
    Creates a conversation map file (.jsonl) and returns sentence count and sentences.

    Each line in the map file corresponds to a sentence and contains its ID,
    sequence order (same as ID here), and the sentence text.

    Parameters:
        input_file (Path): Path to the input text file.
        map_dir (Path): Directory to save the map file.
        map_suffix (str): Suffix for the map file (e.g., "_map.jsonl").

    Returns:
        Tuple[int, List[str]]: The number of sentences found and the list of sentences.

    Raises:
        FileNotFoundError: If the input file does not exist.
        OSError: If the map directory cannot be created.
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

async def _load_tasks(map_file: Path, contexts: List[Dict[str, str]], task_queue: asyncio.Queue):
    """
    Coroutine to read a map file and load analysis tasks onto the task queue.

    Parses each JSON line from the map file, retrieves the corresponding pre-built
    context, and enqueues a tuple representing the task.

    Args:
        map_file (Path): Path to the conversation map file (.jsonl).
        contexts (List[Dict[str, str]]): A list where the index corresponds to sentence_id
                                         and the value is the dictionary of contexts for that sentence.
        task_queue (asyncio.Queue): The queue to put task tuples onto.
                                    Task tuple format: (sentence_id, sequence_order, sentence, context)

    Raises:
        FileNotFoundError: If the map_file cannot be found.
        Exception: For JSON parsing errors or other unexpected issues during file reading.
    """
    logger.debug(f"Starting task loader for map file: {map_file}")
    try:
        with map_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    sentence_id = entry["sentence_id"]
                    sequence_order = entry["sequence_order"]
                    sentence = entry["sentence"]
                    
                    # Ensure context exists for the sentence ID
                    if sentence_id < len(contexts):
                        context = contexts[sentence_id]
                        task_item = (sentence_id, sequence_order, sentence, context)
                        await task_queue.put(task_item)
                        logger.debug(f"Loaded task for sentence_id: {sentence_id}")
                    else:
                        logger.warning(f"Context missing for sentence_id {sentence_id} from map file {map_file}. Skipping.")
                        metrics_tracker.increment_errors() # Track context mismatch

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Failed to parse line in map file {map_file}: {line.strip()}. Error: {e}")
                    metrics_tracker.increment_errors()
                    # Decide whether to continue or stop? Continue for robustness.
        logger.info(f"Finished loading tasks from map file: {map_file}")
    except FileNotFoundError:
        logger.error(f"Map file not found during task loading: {map_file}")
        metrics_tracker.increment_errors()
        raise # Propagate error to stop processing this file
    except Exception as e:
        logger.error(f"Unexpected error loading tasks from {map_file}: {e}", exc_info=True)
        metrics_tracker.increment_errors()
        raise # Propagate error

async def _analysis_worker(
    worker_id: int,
    analyzer: SentenceAnalyzer,
    task_queue: asyncio.Queue,
    results_queue: asyncio.Queue
):
    """
    Async worker that consumes sentence analysis tasks from a queue, performs analysis,
    and puts results onto another queue.

    Runs in a loop, fetching tasks. Exits when it receives a `None` sentinel value.
    If analysis via `analyzer.classify_sentence` succeeds, the result dictionary
    (augmented with IDs) is put onto `results_queue`.
    If analysis fails, an error dictionary containing identifying information and
    error details is put onto `results_queue` instead.

    Args:
        worker_id (int): Identifier for the worker (for logging).
        analyzer (SentenceAnalyzer): An instance of the sentence analyzer.
        task_queue (asyncio.Queue): Queue to get task tuples from.
                                    Expected tuple: (sentence_id, sequence_order, sentence, context).
                                    Receives `None` to signal termination.
        results_queue (asyncio.Queue): Queue to put successful analysis results (dict) or
                                       error dictionaries onto.
                                       Error dict structure: {"sentence_id": int, "sequence_order": int,
                                       "sentence": str, "error": True, "error_type": str, "error_message": str}
    """
    logger.info(f"Analysis worker {worker_id} started.")
    while True:
        try:
            task_item = await task_queue.get()
            # Check for the termination sentinel
            if task_item is None: 
                logger.info(f"Worker {worker_id} received sentinel. Exiting.")
                task_queue.task_done() # Mark sentinel as processed before exiting
                break 
            
            sentence_id, sequence_order, sentence, context = task_item
            logger.debug(f"Worker {worker_id} processing sentence_id: {sentence_id}")
            
            try:
                # Perform the core analysis
                analysis_result = await analyzer.classify_sentence(sentence, context)
                
                # Augment result with IDs and original sentence
                analysis_result["sentence_id"] = sentence_id
                analysis_result["sequence_order"] = sequence_order
                analysis_result["sentence"] = sentence # Ensure original sentence is included
                
                await results_queue.put(analysis_result)
                logger.debug(f"Worker {worker_id} completed sentence_id: {sentence_id}")
                
            except Exception as e:
                # Include exception type in the log message
                logger.error(f"Worker {worker_id} failed analyzing sentence_id {sentence_id}: {type(e).__name__}: {e}", exc_info=False) # Keep log concise but add type
                metrics_tracker.increment_errors()
                # Create and queue an error result instead of dropping
                error_result = {
                    "sentence_id": sentence_id,
                    "sequence_order": sequence_order,
                    "sentence": sentence,
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                await results_queue.put(error_result)
                # Do not put anything on results_queue for failed analysis
            finally:
                 # Signal task completion regardless of success or failure
                 task_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} cancelled.")
            break
        except Exception as e:
            logger.critical(f"Critical error in worker {worker_id}: {e}", exc_info=True)
            # How to handle critical worker errors? Maybe break the loop.
            break
    logger.info(f"Analysis worker {worker_id} finished.")

async def _result_writer(output_file: Path, results_queue: asyncio.Queue):
    """
    Async worker that consumes analysis results from a queue and writes them to a JSON Lines file.

    Runs in a loop, fetching results. Exits when it receives a `None` sentinel value.

    Args:
        output_file (Path): Path to the output .jsonl file.
        results_queue (asyncio.Queue): Queue to get analysis result dictionaries from.
                                       Receives `None` to signal termination.
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
                logger.error(f"Writer failed writing result {result.get('sentence_id', 'N/A')} to {output_file}: {e}", exc_info=True)
                metrics_tracker.increment_errors() # Treat write errors as pipeline errors
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
    Processes a single text file using the provided AnalysisService.

    Orchestrates map creation, calls the injected AnalysisService for context building
    and sentence analysis, and manages the result writing process.

    Args:
        input_file (Path): Path to the input text file.
        output_dir (Path): Directory to save the analysis output file.
        map_dir (Path): Directory containing (or to contain) the map file.
        config (Dict[str, Any]): Application configuration dictionary.
        analysis_service (AnalysisService): An initialized AnalysisService instance.
    """
    logger.info(f"Processing file: {input_file}")
    analysis_file_path = output_dir / f"{input_file.stem}{config.get('paths', {}).get('analysis_suffix', '_analysis.jsonl')}"
    map_suffix = config.get('paths', {}).get('map_suffix', '_map.jsonl')
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Create map file and get sentences
        num_sentences, sentences = await create_conversation_map(input_file, map_dir, map_suffix)
        if num_sentences == 0:
            logger.warning(f"Skipping analysis for {input_file} as it has no sentences.")
            return
            
        # 2. Use the injected AnalysisService
        # analysis_service = AnalysisService(config) # REMOVE internal instantiation

        # 3. Build Contexts using the service
        logger.info(f"Building context for {num_sentences} sentences...")
        contexts = analysis_service.build_contexts(sentences) # Call on injected instance
        if not contexts or len(contexts) != num_sentences:
             logger.error(f"Context building failed or returned unexpected count for {input_file}. Expected {num_sentences}, got {len(contexts)}. Aborting analysis.")
             metrics_tracker.increment_errors(num_sentences) # Count all as errors if context fails
             return
        logger.info("Context building complete.")

        # 4. Analyze Sentences using the service
        logger.info(f"[{input_file.name}] Analyzing {num_sentences} sentences...")
        analysis_results = await analysis_service.analyze_sentences(sentences, contexts) 
        logger.info(f"[{input_file.name}] Analysis complete. Received {len(analysis_results)} results.")
        
        # Check if analysis returned anything before writing
        if not analysis_results:
            logger.warning(f"[{input_file.name}] Analysis service returned no results. Output file will not be created/modified.")
            return

        # 5. Write results asynchronously
        logger.debug(f"[{input_file.name}] Preparing to write {len(analysis_results)} results.") # DEBUG
        temp_results_queue = asyncio.Queue()
        for result in analysis_results:
            await temp_results_queue.put(result)
        await temp_results_queue.put(None) # Sentinel for the writer
        logger.debug(f"[{input_file.name}] Results loaded onto temp queue.") # DEBUG

        logger.info(f"[{input_file.name}] Starting result writer task for {analysis_file_path}...")
        writer_task = asyncio.create_task(_result_writer(analysis_file_path, temp_results_queue))
        
        # Add log before awaiting writer
        logger.debug(f"[{input_file.name}] Awaiting result writer task...") # DEBUG
        await writer_task # This might raise OSError etc. from the writer task
        # Add log after awaiting writer
        logger.debug(f"[{input_file.name}] Result writer task finished.") # DEBUG
        
        logger.info(f"[{input_file.name}] Result writer finished successfully for {analysis_file_path}.") # Changed log level
        metrics_tracker.increment_files_processed()

    except FileNotFoundError as e:
        logger.error(f"File not found during processing {input_file}: {e}")
        metrics_tracker.increment_errors()
    except (ValueError, OSError) as e:
        logger.error(f"Propagating error during processing file {input_file}: {type(e).__name__}: {e}")
        raise
    except Exception as e:
        logger.critical(f"Critical error processing file {input_file}: {e}", exc_info=True)
        metrics_tracker.increment_errors()
        # Potentially track expected sentences as errors if process fails mid-way?
        # if 'num_sentences' in locals():
        #     metrics_tracker.increment_errors(num_sentences - len(analysis_results or []))


async def run_pipeline(input_dir: Path, output_dir: Path, map_dir: Path, config: Dict[str, Any]):
    """
    Runs the analysis pipeline on all .txt files in the input directory.

    Iterates through text files, processes each using `process_file`, 
    verifies output completeness against map files, and logs a summary.

    Args:
        input_dir (Path): Directory containing input .txt files.
        output_dir (Path): Directory to save analysis output files.
        map_dir (Path): Directory to save conversation map files.
        config (Dict[str, Any]): Application configuration.
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

    # Instantiate dependencies needed for AnalysisService
    # Assuming ContextBuilder is a singleton instance `context_builder`
    # Assuming MetricsTracker is a singleton instance `metrics_tracker`
    sentence_analyzer = SentenceAnalyzer() # Assuming SentenceAnalyzer doesn't need config for now
    
    # Instantiate AnalysisService ONCE with its dependencies
    try:
        analysis_service = AnalysisService(
            config=config,
            context_builder=context_builder, # Pass the singleton instance
            sentence_analyzer=sentence_analyzer,
            metrics_tracker=metrics_tracker # Pass the singleton instance
        )
        logger.info("AnalysisService instantiated for pipeline run.")
    except Exception as e:
        logger.critical(f"Failed to instantiate AnalysisService: {e}", exc_info=True)
        return # Cannot proceed without the service

    processed_files_for_verification = []
    map_suffix = config.get('paths', {}).get('map_suffix', '_map.jsonl')
    analysis_suffix = config.get('paths', {}).get('analysis_suffix', '_analysis.jsonl')

    # Process files sequentially for now, could be parallelized
    for input_file in input_files:
        logger.info(f"--- Starting processing for {input_file.name} ---")
        try:
            # Pass the SAME analysis_service instance to each process_file call
            await process_file(input_file, output_dir, map_dir, config, analysis_service)
            # If process_file completes without critical error, store paths for verification
            map_path = map_dir / f"{input_file.stem}{map_suffix}"
            analysis_path = output_dir / f"{input_file.stem}{analysis_suffix}"
            processed_files_for_verification.append((map_path, analysis_path))
            logger.info(f"--- Finished processing for {input_file.name} ---")
        except Exception as e:
            # process_file should handle its internal errors, but catch critical ones here
            logger.critical(f"Critical error during processing of {input_file.name}, stopping its processing: {e}", exc_info=True)
            # Decide if pipeline should continue with next file or stop entirely
            # continue 

    # Verification step
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
    # Get the actual summary data from the tracker
    # Note: metrics_tracker might still be the singleton if not injected everywhere
    # Or it could be the one injected into AnalysisService if run_pipeline uses that.
    # Assuming access to the correct tracker instance here.
    metrics_summary = metrics_tracker.get_summary() 
    total_errors_reported = metrics_summary.get("total_errors", "N/A")
    # pipeline_duration = metrics_summary.get("pipeline_duration_seconds", "N/A") # Often less accurate than monotonic
    
    logger.info(
        f"Pipeline Execution Summary: "
        # --- Remove calls to non-existent getters --- 
        # f"Processed {metrics_tracker.get_files_processed()} files. " 
        # f"Total Sentences Attempted (by workers): {metrics_tracker.get_sentences_processed()}. "
        # f"Total Sentences Succeeded: {metrics_tracker.get_sentences_success()}. " 
        # f"Total Processing Time: {metrics_tracker.get_total_processing_time():.2f}s. " 
        # --- Use values from verification and get_summary() ---
        f"Files Checked: {len(processed_files_for_verification)}. "
        f"Total Sentences Expected: {total_expected}. " 
        f"Total Sentences Found in Output: {total_actual}. "
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
    Analyzes only specific sentences using the provided AnalysisService.

    Args:
        input_file_path (Path): Path to the original input text file.
        sentence_ids (List[int]): A list of integer sentence IDs to analyze.
        config (Dict[str, Any]): Application configuration dictionary.
        analysis_service (AnalysisService): An initialized AnalysisService instance.

    Returns:
        List[Dict[str, Any]]: Analysis results for the requested sentences.
    """
    logger.info(f"Starting specific analysis for {len(sentence_ids)} sentences from {input_file_path}")
    if not sentence_ids:
        logger.warning("Received empty list of sentence IDs. No analysis to perform.")
        return []

    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    try:
        # Initialize services and paths from config
        analysis_service = analysis_service # Use injected service
        input_dir = Path(config.get("paths", {}).get("input_dir", "./data/input"))
        output_dir = Path(config.get("paths", {}).get("output_dir", "./data/output"))
        map_dir = Path(config.get("paths", {}).get("map_dir", "./data/maps")) # Fix default here
        map_suffix = config.get("paths", {}).get("map_suffix", "_map.jsonl")
        analysis_suffix = config.get("paths", {}).get("analysis_suffix", "_analysis.jsonl")

        # Read map file (assuming it must exist or be created first)
        map_file_path = map_dir / f"{input_file_path.stem}{map_suffix}"

        if not map_file_path.exists():
            logger.info(f"Map file {map_file_path} not found. Creating it first.")
            try:
                _, _ = await create_conversation_map(input_file_path, map_dir, map_suffix)
            except Exception as e:
                logger.error(f"Failed to create map file {map_file_path} for specific analysis: {e}")
                return []
        
        # Read map file to get all sentences
        all_sentences_data = []
        try:
            with map_file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        all_sentences_data.append((entry["sentence_id"], entry["sentence"]))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Skipping invalid line in map file {map_file_path}: {line.strip()} - Error: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error reading map file {map_file_path}: {e}")
            return []

        if not all_sentences_data:
            logger.warning(f"Map file {map_file_path} is empty or invalid. Cannot analyze specific sentences.")
            return []

        all_sentences_data.sort(key=lambda x: x[0])
        all_sentences_text = [text for _, text in all_sentences_data]
        sentence_map = {s_id: text for s_id, text in all_sentences_data}

        # Filter sentence IDs requested
        valid_ids_in_map = {s_id for s_id, _ in all_sentences_data}
        target_ids = sorted([s_id for s_id in sentence_ids if s_id in valid_ids_in_map])
        
        # Log skipped IDs
        skipped_ids = set(sentence_ids) - set(target_ids)
        if skipped_ids:
            logger.warning(f"Skipping requested sentence IDs not found in map file or invalid: {sorted(list(skipped_ids))}")

        if not target_ids:
            logger.warning(f"No valid sentence IDs found in map file for requested IDs: {sentence_ids}. No analysis performed.")
            return []

        # Use the injected AnalysisService
        # analysis_service = AnalysisService(config) # REMOVE internal instantiation

        # Build context for *all* sentences
        logger.info(f"Building context for all {len(all_sentences_text)} sentences...")
        all_contexts = analysis_service.build_contexts(all_sentences_text)
        if len(all_contexts) != len(all_sentences_text):
            logger.error("Context count mismatch with total sentence count. Aborting specific analysis.")
            return []
        logger.info("Full context building complete.")
        
        # Create mapping from original ID to its index in the full list
        original_id_to_index = {s_id: index for index, (s_id, _) in enumerate(all_sentences_data)}

        # Select the target sentences and contexts based on target_ids
        sentences_to_analyze = []
        contexts_to_analyze = []
        original_indices_for_targets = [] # Store original index for remapping
        for target_id in target_ids:
             if target_id in original_id_to_index:
                 original_index = original_id_to_index[target_id]
                 if original_index < len(all_contexts):
                     sentences_to_analyze.append(all_sentences_text[original_index])
                     contexts_to_analyze.append(all_contexts[original_index])
                     original_indices_for_targets.append(target_id) # Keep original ID
                 else: 
                      logger.warning(f"Context missing for original index {original_index} (ID {target_id}). Skipping.")
             else:
                  logger.warning(f"Original ID {target_id} not found in index map. Skipping.") # Should not happen if target_ids logic is correct

        if not sentences_to_analyze:
            logger.warning("Could not prepare any target sentences/contexts for analysis after filtering.")
            return []

        # Call the analysis service for the subset
        logger.info(f"Calling analysis service for {len(sentences_to_analyze)} specific sentences (Original IDs: {original_indices_for_targets})...")
        specific_results = await analysis_service.analyze_sentences(sentences_to_analyze, contexts_to_analyze)

        # Remap results back to original sentence IDs
        final_results = []
        if len(specific_results) == len(original_indices_for_targets):
            for i, result_dict in enumerate(specific_results):
                original_id = original_indices_for_targets[i]
                result_dict["sentence_id"] = original_id
                result_dict["sequence_order"] = original_id # Use original ID for consistency
                final_results.append(result_dict)
            logger.info(f"Remapped {len(final_results)} results to original sentence IDs.")
        else:
            logger.error(f"Result count ({len(specific_results)}) mismatch with targeted sentence count ({len(original_indices_for_targets)}). Cannot reliably map back IDs.")
            final_results = specific_results # Return potentially incorrect results
             
        return final_results

    except Exception as e:
        logger.critical(f"Unexpected error during analyze_specific_sentences for {input_file_path}: {e}", exc_info=True)
        raise
