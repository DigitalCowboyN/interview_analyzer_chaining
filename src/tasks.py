"""
src/tasks.py

Celery task definitions for background processing.
"""

import asyncio
from pathlib import Path
from src.celery_app import celery_app
from src.utils.logger import get_logger

# Import necessary pipeline components and dependency CLASSES
from src.pipeline import process_file 
from src.services.analysis_service import AnalysisService 
from src.agents.context_builder import ContextBuilder # Import class
from src.agents.sentence_analyzer import SentenceAnalyzer # Import class
from src.utils.metrics import MetricsTracker # Import class

logger = get_logger()

@celery_app.task(bind=True)
def run_pipeline_for_file(self, input_file_path_str: str, output_dir_str: str, map_dir_str: str, config_dict: dict):
    """
    Celery task to run the analysis pipeline for a single input file.

    Instantiates necessary services and calls the core process_file function.

    Args:
        self: The task instance (available via bind=True).
        input_file_path_str (str): Path to the input file as a string.
        output_dir_str (str): Path to the output directory as a string.
        map_dir_str (str): Path to the map directory as a string.
        config_dict (dict): Configuration dictionary (passed from API).
    """
    task_id = self.request.id
    logger.info(f"[Task {task_id}] Received task for file: {input_file_path_str}")

    try:
        input_file = Path(input_file_path_str)
        output_dir = Path(output_dir_str)
        map_dir = Path(map_dir_str)

        # --- Instantiate Dependencies for this Task Run --- 
        # Worker loads global config via src.config when imported.
        # We instantiate dependencies here using that global config or passed config_dict.
        try:
            # SentenceAnalyzer init uses global config implicitly
            sentence_analyzer_instance = SentenceAnalyzer()
            # ContextBuilder init uses global config implicitly
            context_builder_instance = ContextBuilder()
            # Create a *local* metrics tracker for this specific task run
            # This avoids issues with the global singleton state across workers/tasks.
            # Note: Metrics will only reflect this single task, not aggregated.
            metrics_tracker_instance = MetricsTracker() 
            
            logger.info(f"[Task {task_id}] Dependencies (SentenceAnalyzer, ContextBuilder, MetricsTracker) instantiated.")
        except Exception as dep_e:
            logger.error(f"[Task {task_id}] Failed to instantiate dependencies: {dep_e}", exc_info=True)
            raise
            
        # --- Instantiate AnalysisService --- 
        try:
            # Pass the instances created above and the config dict from the task args
            analysis_service = AnalysisService(
                config=config_dict, 
                context_builder=context_builder_instance,
                sentence_analyzer=sentence_analyzer_instance,
                metrics_tracker=metrics_tracker_instance # Pass the local instance
            )
            logger.info(f"[Task {task_id}] AnalysisService instantiated.")
        except Exception as service_e:
            logger.error(f"[Task {task_id}] Failed to instantiate AnalysisService: {service_e}", exc_info=True)
            raise 

        # --- Run Core Pipeline Logic --- 
        logger.info(f"[Task {task_id}] Starting process_file for {input_file}")
        
        asyncio.run(process_file(
            input_file=input_file, 
            output_dir=output_dir, 
            map_dir=map_dir, 
            config=config_dict, 
            analysis_service=analysis_service
        ))
        
        logger.info(f"[Task {task_id}] Successfully processed: {input_file_path_str}")
        return {"status": "Success", "file": input_file_path_str}

    except FileNotFoundError:
        logger.error(f"[Task {task_id}] Input file not found during processing: {input_file_path_str}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"[Task {task_id}] Error processing file {input_file_path_str}: {e}", exc_info=True)
        raise # Re-raise the exception so Celery marks the task as FAILED 