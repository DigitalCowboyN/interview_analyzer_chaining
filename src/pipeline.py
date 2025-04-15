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
from typing import Dict, Any, List, Tuple

# Initialize the logger and load the spaCy model.
logger = get_logger()
nlp = spacy.load("en_core_web_sm")

def segment_text(text: str) -> List[str]:
    """
    Segment input text into sentences using spaCy.
    
    This function uses the spaCy NLP library to split the provided text into sentences.
    It filters out any empty sentences after stripping whitespace.
    
    Parameters:
        text (str): The input text to be segmented.
        
    Returns:
        list: A list of sentences as strings. For example, if the text contains three
              sentences, the function returns a list with three elements.
              
    Raises:
        ValueError: If desired behavior changes (e.g., you want to raise an error for empty input).
                   Currently, if the input is empty, an empty list is returned.
                   
    Usage:
        sentences = segment_text("Hello world. How are you?")
        # sentences => ["Hello world.", "How are you?"]
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
    Coroutine to read the map file and load analysis tasks onto the task queue.
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
    Coroutine that consumes tasks from task_queue, analyzes sentences, and puts results on results_queue.
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
                metrics_tracker.increment_success() # Assuming classify_sentence doesn't throw
                
            except Exception as e:
                logger.error(f"Worker {worker_id} failed analyzing sentence_id {sentence_id}: {e}", exc_info=False) # Keep log concise
                metrics_tracker.increment_errors()
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
    Coroutine that consumes results from results_queue and writes them to the output file.
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


async def process_file(input_file: Path, output_dir: Path, map_dir: Path, config: Dict[str, Any]):
    """
    Process a single text file using the concurrent queue-based pipeline.
    
    1. Creates a conversation map file.
    2. Builds sentence contexts.
    3. Sets up task and result queues.
    4. Starts loader, worker, and writer coroutines.
    5. Waits for completion and handles errors.
    """
    logger.info(f"Processing file: {input_file} with concurrent pipeline.")
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
            logger.info(f"All tasks processed by workers for {input_file.stem}. Signaling workers and writer to stop.")
            
            # Signal worker completion
            for _ in range(num_workers):
                await task_queue.put(None)
                
            # Signal writer completion
            await results_queue.put(None) 
            
            # Wait for writer first (as it depends on results queue)
            await writer_task # Wait for writer to finish processing results and the sentinel
            logger.info(f"Writer finished for {input_file.stem}.")

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
    Run the concurrent pipeline across all text files in the input directory.
    """
    logger.info(f"Starting pipeline run. Input: {input_dir}, Output: {output_dir}, Map: {map_dir}")
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        logger.warning(f"No input files found in {input_dir}")
        return

    # Ensure base output/map directories exist (process_file handles specifics)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    # Process each input file using the refactored process_file
    # Consider running process_file calls concurrently too if needed
    # for now, process files sequentially but sentences within concurrently
    for input_file in input_files:
        try:
            await process_file(input_file, output_dir, map_dir, config)
        except Exception as e:
            logger.critical(f"Unhandled exception during processing of {input_file} in run_pipeline: {e}", exc_info=True)
            metrics_tracker.increment_errors()

    logger.info("Pipeline run complete.")
