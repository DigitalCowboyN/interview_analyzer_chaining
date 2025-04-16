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
import spacy
import json
from typing import Dict, Any, List, Tuple, Set, Optional

# Initialize the logger and load the spaCy model.
logger = get_logger()
nlp = spacy.load("en_core_web_sm")

def segment_text(text: str) -> List[str]:
    """
    Segment input text into sentences using spaCy.
    
    Uses the spaCy NLP library ('en_core_web_sm') to split the provided text into sentences,
    filtering out any empty strings after stripping whitespace.
    
    Args:
        text (str): The input text to be segmented.
        
    Returns:
        List[str]: A list of sentences extracted from the text.
                   Returns an empty list if the input text is empty or contains no sentences.
    """
    # Process the text using the spaCy model.
    doc = nlp(text)
    # Extract sentences and remove any that are empty after stripping whitespace.
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    logger.debug(f"Segmented text into {len(sentences)} sentences.")
    return sentences

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


async def process_file(input_file: Path, output_dir: Path, map_dir: Path, config: Dict[str, Any]):
    """
    Orchestrates the concurrent processing pipeline for a single input text file.

    Steps:
    1. Creates a conversation map file containing sentence IDs and text.
    2. Builds all necessary textual contexts for each sentence using ContextBuilder.
    3. Initializes asyncio Queues for tasks and results.
    4. Creates and starts the task loader coroutine.
    5. Creates and starts multiple analysis worker coroutines.
    6. Creates and starts the result writer coroutine.
    7. Waits for the task queue to be fully processed.
    8. Sends termination sentinels (None) to workers and the writer.
    9. Waits for all coroutines to complete.
    10. Logs success or failure for the file processing.

    Args:
        input_file (Path): Path to the input text file.
        output_dir (Path): Directory to save the final analysis results JSONL file.
        map_dir (Path): Directory to save the intermediate conversation map JSONL file.
        config (Dict[str, Any]): The application configuration dictionary, expected to contain
                                 keys like 'pipeline', 'context_map', etc.

    Raises:
        FileNotFoundError: If the input file cannot be found by `create_conversation_map`.
        Exception: Propagates exceptions from `create_conversation_map`, `_load_tasks`,
                   or potentially critical errors during queue/coroutine management.
    """
    logger.info(f"Processing file: {input_file}")
    # Revert config access to original method
    map_suffix = config["paths"].get("map_suffix", "_map.jsonl")
    analysis_suffix = config["paths"].get("analysis_suffix", "_analysis.jsonl")
    num_workers = config["pipeline"].get("num_analysis_workers", 10)

    map_file = map_dir / f"{input_file.stem}{map_suffix}"
    analysis_output_file = output_dir / f"{input_file.stem}{analysis_suffix}"

    try:
        # 1. Create Conversation Map & Get Sentences
        num_sentences, sentences = await create_conversation_map(input_file, map_dir, map_suffix)
        if num_sentences == 0:
            logger.warning(f"Input file {input_file} contains 0 processable sentences. Skipping analysis.")
            # Ensure empty map file is created IF map creation succeeded but returned 0
            if not map_file.exists(): # Double-check if map creation failed earlier
                 map_dir.mkdir(parents=True, exist_ok=True)
                 map_file.touch()
                 logger.info(f"Created empty map file for empty input: {map_file}")
            # We need to ensure the analysis file is also created here
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            analysis_output_file.touch() # Create the empty file
            logger.info(f"Created empty analysis file for empty input: {analysis_output_file}")
            return # Skip the rest of the processing

        # 2. Build Contexts using the exact sentences from map creation
        logger.debug(f"Building contexts for {len(sentences)} sentences from map creation.")
        contexts = context_builder.build_all_contexts(sentences)
        if len(contexts) != num_sentences:
             # This should ideally not happen now, but good to keep a check
             logger.error(f"Context count ({len(contexts)}) mismatch with sentence count ({num_sentences}) after map creation for {input_file}. Aborting.")
             metrics_tracker.increment_errors()
             return

        # 3. Setup Queues & Instances
        task_queue = asyncio.Queue()
        results_queue = asyncio.Queue()
        analyzer = SentenceAnalyzer() # Instantiate once
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        # Clear analysis file before starting
        with analysis_output_file.open("w", encoding="utf-8") as f:
             f.write("") 
        logger.debug(f"Cleared/prepared analysis output file: {analysis_output_file}")

        # Handle zero workers case
        if num_workers <= 0:
            logger.error(f"Configuration error: num_analysis_workers set to {num_workers}. Must be > 0. Skipping file {input_file}.")
            metrics_tracker.increment_errors()
            return

        # 4. Start Coroutines
        loader_task = asyncio.create_task(
            _load_tasks(map_file, contexts, task_queue),
            name=f"Loader-{input_file.stem}"
        )
        
        worker_tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(
                _analysis_worker(i, analyzer, task_queue, results_queue),
                name=f"Worker-{i}-{input_file.stem}"
            )
            worker_tasks.append(task)
            
        writer_task = asyncio.create_task(
            _result_writer(analysis_output_file, results_queue),
            name=f"Writer-{input_file.stem}"
        )

        # 5. Wait for Completion & Cleanup
        try:
            logger.info(f"Waiting for loader task for {input_file.stem}...")
            await loader_task
            logger.info(f"Loader finished for {input_file.stem}. Waiting for workers ({num_sentences} tasks)... ")
            
            await task_queue.join()
            logger.info(f"All tasks processed by workers for {input_file.stem}. Signaling workers to stop.")
            
            # Signal worker completion FIRST
            for _ in range(num_workers):
                await task_queue.put(None)
            
            # Wait for all results placed on the queue to be processed by the writer
            logger.info(f"Waiting for result writer to process {results_queue.qsize()} items...")
            await results_queue.join()
            logger.info(f"Result writer finished processing queue items for {input_file.stem}.")

            # NOW Signal writer completion
            await results_queue.put(None) 
            
            # Wait for the writer task to finish (it will consume the None sentinel)
            logger.debug(f"Waiting for writer task to complete for {input_file.stem}...")
            await writer_task 
            logger.info(f"Writer task completed for {input_file.stem}.")

            # Wait for workers to finish processing sentinels
            logger.debug(f"Gathering worker tasks for {input_file.stem}...")
            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True) # Wait for worker tasks to finish
            for i, res in enumerate(worker_results):
                 if isinstance(res, Exception):
                      logger.warning(f"Worker task {i} for {input_file.stem} finished with exception: {res}")
            logger.debug(f"Worker tasks gathered for {input_file.stem}.")
        
        except Exception as inner_e:
             logger.error(f"Exception during task coordination for {input_file}: {inner_e}", exc_info=True)
             metrics_tracker.increment_errors()
             # Re-raise the exception after attempting cleanup
             raise
        finally:
            # Ensure all tasks are truly finished or cancelled
            logger.debug(f"Entering final cleanup for {input_file.stem} tasks.")
            # Cancel any potentially lingering tasks (idempotent if already done)
            if writer_task and not writer_task.done():
                 writer_task.cancel()
            for task in worker_tasks:
                 if task and not task.done():
                     task.cancel()
            
            # Await cancellation completion (gather cancelled tasks)
            # Need to gather writer_task and worker_tasks again here to await cancellation
            all_tasks = [t for t in ([writer_task] + worker_tasks) if t] # Filter out potential None
            if all_tasks: # Only gather if there are tasks
                await asyncio.gather(*all_tasks, return_exceptions=True) 
            logger.debug(f"Final cleanup gather complete for {input_file.stem}.")

    except FileNotFoundError as e:
        logger.error(f"Processing failed for {input_file}: Map file could not be read or input file missing. Error: {e}")
        metrics_tracker.increment_errors()
    # Outer Exception handler removed as the inner one handles and re-raises
    # Consider specific exception handling if needed beyond FileNotFoundError
    finally:
        # This outer finally just logs the overall completion
        logger.info(f"Finished processing file: {input_file}")


async def run_pipeline(input_dir: Path, output_dir: Path, map_dir: Path, config: Dict[str, Any]):
    """
    Runs the analysis pipeline for all .txt files in the specified input directory.

    Iterates through input files, calls `process_file` for each, and tracks overall metrics.

    Args:
        input_dir (Path): Directory containing input .txt files.
        output_dir (Path): Root directory for output analysis results (.jsonl files).
        map_dir (Path): Root directory for intermediate conversation map files (.jsonl files).
        config (Dict[str, Any]): The application configuration dictionary.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input_dir does not exist.
    """
    # Start global pipeline timer
    metrics_tracker.start_pipeline_timer()
    processed_file_paths = [] # Keep track of (map_path, analysis_path) tuples

    logger.info(f"Starting pipeline run. Input: {input_dir}, Output: {output_dir}, Map: {map_dir}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        logger.warning(f"No input files found in {input_dir}")
        # Still stop timer and log basic metrics even if no files processed
        metrics_tracker.stop_pipeline_timer()
        summary = metrics_tracker.get_summary()
        logger.info(f"Pipeline Execution Summary: {json.dumps(summary, indent=2)}")
        return

    # Ensure base output/map directories exist (process_file handles specifics within)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    # Get suffixes from config once
    map_suffix = config["paths"].get("map_suffix", "_map.jsonl")
    analysis_suffix = config["paths"].get("analysis_suffix", "_analysis.jsonl")

    # Process each input file using the refactored process_file
    # Consider running process_file calls concurrently too if needed
    # for now, process files sequentially but sentences within concurrently
    for input_file in input_files:
        try:
            await process_file(input_file, output_dir, map_dir, config)
            # If process_file completes without critical error, store paths for verification
            map_path = map_dir / f"{input_file.stem}{map_suffix}"
            analysis_path = output_dir / f"{input_file.stem}{analysis_suffix}"
            processed_file_paths.append((map_path, analysis_path))
        except Exception as e:
            # Log critical errors from process_file itself if they propagate
            logger.critical(f"Unhandled exception during processing of {input_file} in run_pipeline: {e}", exc_info=True)
            metrics_tracker.increment_errors() # Count file-level processing errors

    # --- Verification Step ---
    total_files_processed = len(processed_file_paths)
    total_expected_overall = 0
    total_actual_overall = 0
    total_missing_overall = 0
    files_with_missing = 0
    files_with_errors = 0 # Files where verification check itself failed

    logger.info("--- Starting Output Verification --- ")
    for map_path, analysis_path in processed_file_paths:
        logger.info(f"Verifying output for: {analysis_path.name}")
        completeness_result = verify_output_completeness(map_path, analysis_path)
        
        total_expected_overall += completeness_result["total_expected"]
        total_actual_overall += completeness_result["total_actual"]
        total_missing_overall += completeness_result["total_missing"]
        
        if completeness_result["total_missing"] > 0:
            files_with_missing += 1
            logger.warning(f"Verification for {analysis_path.name}: MISSING {completeness_result['total_missing']} sentences. IDs: {completeness_result['missing_ids']}")
        elif completeness_result["error"]:
             files_with_errors += 1
             logger.error(f"Verification for {analysis_path.name}: FAILED ({completeness_result['error']})")
        else:
             logger.info(f"Verification for {analysis_path.name}: OK ({completeness_result['total_actual']}/{completeness_result['total_expected']})")
             
    logger.info("--- Output Verification Complete --- ")

    # Stop timer and log metrics summary including verification
    metrics_tracker.stop_pipeline_timer()
    summary = metrics_tracker.get_summary()
    # Add verification summary to the log output
    verification_summary = (
        f"Verification Summary: Checked {total_files_processed} files. "
        f"Total Expected Sentences: {total_expected_overall}, "
        f"Total Actual Entries: {total_actual_overall}, "
        f"Total Missing Sentences: {total_missing_overall} across {files_with_missing} files. "
        f"Verification Errors: {files_with_errors} files."
    )
    logger.info(f"Pipeline Execution Summary: {json.dumps(summary, indent=2)}")
    log_level = logger.warning if total_missing_overall > 0 or files_with_errors > 0 else logger.info
    log_level(verification_summary)

    logger.info("Pipeline run complete.")


async def analyze_specific_sentences(input_file_path: Path, sentence_ids: List[int], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyzes a specific subset of sentences from an input file.

    Reads the input file, segments it, builds context, and then concurrently
    analyzes only the sentences corresponding to the provided `sentence_ids`
    using the `SentenceAnalyzer`.

    This function is useful for reprocessing specific sentences, such as retrying
    failures or applying updated analysis logic to a subset.

    Args:
        input_file_path (Path): Path to the original input text file.
        sentence_ids (List[int]): A list of integer sentence IDs (0-based index)
                                  to be analyzed.
        config (Dict[str, Any]): The application configuration dictionary (currently unused
                                 in this function but passed for potential future use).

    Returns:
        List[Dict[str, Any]]: A list containing the analysis result dictionaries for
                              the requested sentences. Each dictionary will either be
                              a successful analysis result or an error dictionary
                              (containing 'error': True, 'error_type', 'error_message').
                              Returns an empty list if `sentence_ids` is empty or
                              if the input file is empty/invalid.

    Raises:
        FileNotFoundError: If `input_file_path` does not exist.
        Exception: Can propagate exceptions from file reading, context building,
                   or sentence analysis if not caught by the internal wrapper.
    """
    logger.info(f"Starting specific analysis for {len(sentence_ids)} sentences from {input_file_path}")
    if not sentence_ids:
        logger.warning("Received empty list of sentence IDs. No analysis to perform.")
        return []

    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    try:
        text = input_file_path.read_text(encoding="utf-8")
        all_sentences = segment_text(text)
        if not all_sentences:
             logger.warning(f"Input file {input_file_path} yielded no sentences. Cannot analyze specific IDs.")
             return []
        
        num_total_sentences = len(all_sentences)
        
        # Validate requested sentence IDs
        valid_ids_to_process: List[int] = []
        for idx in sentence_ids:
            if 0 <= idx < num_total_sentences:
                 valid_ids_to_process.append(idx)
            else:
                 logger.warning(f"Invalid sentence_id requested: {idx}. Total sentences: {num_total_sentences}. Skipping.")
        
        if not valid_ids_to_process:
            logger.warning("No valid sentence IDs remain after filtering. No analysis to perform.")
            return []
            
        logger.debug(f"Building contexts for all {num_total_sentences} sentences to analyze {len(valid_ids_to_process)} specific ones.")
        all_contexts = context_builder.build_all_contexts(all_sentences)
        analyzer = SentenceAnalyzer()

        # --- Define internal wrapper for concurrent execution --- 
        async def _wrapper_analyze(idx: int):
            sentence = all_sentences[idx]
            context = all_contexts.get(idx) # Get context for this specific ID
            
            if context is None:
                # Should not happen if build_all_contexts is correct, but handle defensively
                logger.error(f"Internal error: Context missing for sentence_id {idx} during specific analysis.")
                return {
                    "sentence_id": idx,
                    "sequence_order": idx, # Assume sequence_order == sentence_id here
                    "sentence": sentence,
                    "error": True,
                    "error_type": "ContextMissingError",
                    "error_message": "Failed to retrieve context during analysis."
                }

            try:
                analysis_result = await analyzer.classify_sentence(sentence, context)
                # Augment result with IDs (sequence_order might be ambiguous here, use id)
                analysis_result["sentence_id"] = idx
                analysis_result["sequence_order"] = idx # Assuming sequence order matches ID
                analysis_result["sentence"] = sentence # Ensure original sentence is included
                return analysis_result
            except Exception as e:
                 logger.error(f"Analysis failed for specific sentence_id {idx}: {type(e).__name__}: {e}", exc_info=False)
                 # Return the standard error dictionary
                 return {
                    "sentence_id": idx,
                    "sequence_order": idx,
                    "sentence": sentence,
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                 }
        # --- End internal wrapper --- 

        # Create tasks only for the valid requested IDs
        tasks = [asyncio.create_task(_wrapper_analyze(idx)) for idx in valid_ids_to_process]
        
        logger.info(f"Running analysis for {len(tasks)} specific sentences concurrently...")
        results = await asyncio.gather(*tasks)
        logger.info(f"Completed analysis for {len(results)} specific sentences.")
        
        return results # Return the list of results (success or error dicts)

    except Exception as e:
        logger.critical(f"Unexpected error during analyze_specific_sentences for {input_file_path}: {e}", exc_info=True)
        # Re-raise or return empty list? Returning empty list might hide critical errors.
        # Let's re-raise for now to make upstream aware.
        raise
