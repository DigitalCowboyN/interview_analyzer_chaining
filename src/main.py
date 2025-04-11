"""                                                                                                                                                                                                                                    
main.py                                                                                                                                                                                                                                
                                                                                                                                                                                                                                       
This module serves as the entry point for the Enriched Sentence Analysis Pipeline.                                                                                                                                                     
It handles command-line arguments for input and output directories and initiates                                                                                                                                                       
the pipeline processing.                                                                                                                                                                                                               
""" 
import argparse
import asyncio
import json # Import json for logging summary
from pathlib import Path
from src.pipeline import run_pipeline
from src.config import config
from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker # Import metrics tracker

logger = get_logger()

def main():
    """
    Main function to execute the Enriched Sentence Analysis Pipeline.

    Sets up argument parsing, resets and starts metrics tracking, runs the pipeline,
    stops metrics tracking, and logs a summary of the execution metrics.
    """
    # Create an argument parser to handle command-line arguments 
    parser = argparse.ArgumentParser(description="Enriched Sentence Analysis Pipeline")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(config["paths"]["input_dir"]),
        help="Path to the input directory containing transcript files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(config["paths"]["output_dir"]),
        help="Path to the directory for saving output analysis JSON files",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Reset and start metrics tracking
    metrics_tracker.reset()
    metrics_tracker.start_pipeline_timer()
    
    # Log the start of the pipeline execution 
    logger.info("Starting the Enriched Sentence Analysis Pipeline")
    try:
        asyncio.run(run_pipeline(args.input_dir, args.output_dir))
        logger.info("Pipeline execution completed.")
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
        metrics_tracker.increment_errors() # Track pipeline-level errors
    finally:
        # Stop timer and log metrics summary
        metrics_tracker.stop_pipeline_timer()
        summary = metrics_tracker.get_summary()
        logger.info(f"Pipeline Execution Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
