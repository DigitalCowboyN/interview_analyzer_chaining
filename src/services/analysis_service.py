# src/services/analysis_service.py
import asyncio
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path # Added for potential future use if map reading moves here

from src.utils.logger import get_logger
# from src.config import config # Decided to pass config in __init__
# from src.agents.context_builder import context_builder as context_builder_agent # REMOVE GLOBAL IMPORT
from src.agents.sentence_analyzer import SentenceAnalyzer # Import SentenceAnalyzer CLASS
# from src.utils.metrics import metrics_tracker # REMOVE GLOBAL IMPORT
# --- Import types for dependency injection ---
from src.agents.context_builder import ContextBuilder # Assuming ContextBuilder class exists or using duck typing
from src.utils.metrics import MetricsTracker # Assuming MetricsTracker class exists or using duck typing

logger = get_logger()

class AnalysisService:
    def __init__(
        self, 
        config: Dict[str, Any],
        context_builder: ContextBuilder, # Inject ContextBuilder instance
        sentence_analyzer: SentenceAnalyzer, # Inject SentenceAnalyzer instance
        metrics_tracker: MetricsTracker # Inject MetricsTracker instance
    ):
         """
         Initializes the AnalysisService with configuration and injected dependencies.
         
         Args:
             config (Dict[str, Any]): Application configuration.
             context_builder (ContextBuilder): Instance for building sentence contexts.
             sentence_analyzer (SentenceAnalyzer): Instance for analyzing sentences.
             metrics_tracker (MetricsTracker): Instance for tracking metrics.
         """
         self.config = config
         self.context_builder = context_builder
         self.analyzer = sentence_analyzer # Use the injected analyzer
         self.metrics_tracker = metrics_tracker # Use the injected tracker
         logger.info("AnalysisService initialized with injected dependencies.")

    def build_contexts(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Builds context dictionaries for a list of sentences using the context_builder agent."""
        logger.debug(f"Building contexts for {len(sentences)} sentences.")
        try:
            # Use the injected context_builder instance
            contexts_dict = self.context_builder.build_all_contexts(sentences)
            
            # Convert dict to list, ensuring order and handling potential missing indices
            contexts_list = [contexts_dict.get(i, {}) for i in range(len(sentences))] 
            
            logger.info(f"Successfully built contexts for {len(sentences)} sentences.")
            return contexts_list
        except Exception as e:
            logger.error(f"Failed to build contexts: {e}", exc_info=True)
            raise

    async def analyze_sentences(
        self, 
        sentences: List[str], 
        contexts: List[Dict[str, str]], 
        task_id: Optional[str] = None # Keep task_id parameter
    ) -> List[Dict[str, Any]]:
        """Analyzes a list of sentences with their corresponding contexts using concurrent workers."""
        # Remove prefix usage
        logger.debug(f"Analyzing {len(sentences)} sentences.") # Revert to original log
        if not sentences:
            logger.warning("analyze_sentences called with no sentences. Returning empty list.")
            return []
        if len(sentences) != len(contexts):
             logger.error(f"Sentence count ({len(sentences)}) and context count ({len(contexts)}) mismatch in analyze_sentences. Aborting.")
             return []

        num_analysis_workers = self.config.get("pipeline", {}).get("num_analysis_workers", 1)
        task_queue = asyncio.Queue(maxsize=num_analysis_workers * 2) 
        results_queue = asyncio.Queue()

        # Use the injected analyzer instance passed in __init__
        # analyzer = SentenceAnalyzer() # REMOVE internal instantiation

        # Create worker tasks - pass the injected analyzer
        worker_tasks = []
        for i in range(num_analysis_workers):
            # Pass self.analyzer to the worker
            task = asyncio.create_task(self._analysis_worker(i, self.analyzer, task_queue, results_queue))
            worker_tasks.append(task)
        logger.info(f"Started {num_analysis_workers} analysis workers.")

        # Start task loading 
        loader_task = asyncio.create_task(self._load_tasks_from_memory(sentences, contexts, task_queue))
        
        # Wait for loader to finish and then signal workers to stop
        await loader_task
        logger.info("Task loading complete. Signaling workers to stop.")
        for _ in range(num_analysis_workers):
            await task_queue.put(None) # Sentinel value

        # Wait for workers to finish processing
        await asyncio.gather(*worker_tasks)
        logger.info("All analysis workers finished.")

        # Collect results
        analysis_results = []
        while not results_queue.empty():
            result = await results_queue.get()
            analysis_results.append(result)
            results_queue.task_done()
            
        # Sort results by sequence order to maintain original order
        # Note: The original pipeline relied on _result_writer processing in order.
        # Here, we need to ensure the returned list is ordered if order matters.
        # Let's add sequence_order to the result dict in the worker if not already present
        # and sort here.
        # Assuming worker puts result dicts that include 'sequence_order'
        try:
            analysis_results.sort(key=lambda x: x.get('sequence_order', float('inf'))) # Sort, handle potential missing key
        except TypeError as e:
            logger.error(f"Could not sort analysis results by sequence_order: {e}. Results might be out of order.")
        
        logger.info(f"Collected {len(analysis_results)} analysis results.")
        return analysis_results

    # --- Helper methods moved from pipeline.py --- 
    
    async def _load_tasks_from_memory(self, sentences: List[str], contexts: List[Dict[str, str]], task_queue: asyncio.Queue):
        """
        Coroutine to load analysis tasks onto the task queue directly from memory.
        """
        logger.debug("Starting task loader from memory.")
        try:
            for idx, sentence_text in enumerate(sentences):
                if idx < len(contexts):
                    context = contexts[idx]
                    sentence_id = idx
                    sequence_order = idx
                    task_item = (sentence_id, sequence_order, sentence_text, context)
                    await task_queue.put(task_item)
                    logger.debug(f"Loaded task for sentence_id: {sentence_id}")
                else:
                    logger.warning(f"Context missing for sentence index {idx}. Skipping loading task.")
                    # self.metrics_tracker.increment_errors() # TODO (Metrics): Reinstate with proper Celery solution
            logger.info(f"Finished loading {len(sentences)} tasks from memory.")
        except Exception as e:
            logger.error(f"Unexpected error loading tasks from memory: {e}", exc_info=True)
            # self.metrics_tracker.increment_errors() # TODO (Metrics): Reinstate with proper Celery solution
            raise

    async def _analysis_worker(
        self, worker_id: int, analyzer: SentenceAnalyzer, task_queue: asyncio.Queue, results_queue: asyncio.Queue
    ):
        """
        Async worker moved from pipeline.py.
        Consumes tasks, performs analysis, puts results/errors onto results_queue.
        Uses injected metrics_tracker.
        """
        logger.info(f"Analysis worker {worker_id} started.")
        while True:
            task_item = await task_queue.get()
            if task_item is None:  
                logger.info(f"Analysis worker {worker_id} received stop signal.")
                task_queue.task_done()
                break

            sentence_id, sequence_order, sentence, context = task_item
            logger.debug(f"Worker {worker_id} processing sentence_id: {sentence_id}")
            
            try:
                start_time = asyncio.get_event_loop().time()
                analysis_result: Dict[str, Any] = await analyzer.classify_sentence(sentence, context)
                end_time = asyncio.get_event_loop().time()
                
                full_result = {
                    "sentence_id": sentence_id,
                    "sequence_order": sequence_order,
                    "sentence": sentence,
                    **analysis_result
                }
                
                await results_queue.put(full_result)
                logger.debug(f"Worker {worker_id} successfully analyzed sentence_id: {sentence_id}")

            except Exception as e:
                logger.error(f"Worker {worker_id} failed analyzing sentence_id {sentence_id}: {e}", exc_info=True)
                # self.metrics_tracker.increment_errors() # TODO (Metrics): Reinstate with proper Celery solution
                error_result = {
                    "sentence_id": sentence_id,
                    "sequence_order": sequence_order,
                    "sentence": sentence,
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                await results_queue.put(error_result)
            finally:
                task_queue.task_done()

        logger.info(f"Analysis worker {worker_id} finished.") 