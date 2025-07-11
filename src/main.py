"""
main.py

This module serves as the entry point for the
Enriched Sentence Analysis Pipeline.
It handles command-line arguments for input
and output directories and initiates
the pipeline processing.
"""
import argparse
import asyncio
import json  # Import json for logging summary
from pathlib import Path

from fastapi import FastAPI

from src.api.routers import analysis as analysis_router

# --- Add router imports ---
from src.api.routers import files as files_router
from src.config import config
from src.pipeline import run_pipeline
from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker  # Import metrics tracker

# -------------------------

logger = get_logger()

app = FastAPI(
    title="Interview Analyzer API",
    description="API for analyzing interview transcripts.",
    version="0.1.0"
)

# --- Include the routers ---
app.include_router(files_router.router)
app.include_router(analysis_router.router)  # Add the analysis router
# ------------------------------


@app.get("/", tags=["Health Check"])
async def read_root():
    """Basic health check endpoint."""
    return {"status": "ok"}


def main():
    """
    Main function to execute the Enriched Sentence Analysis Pipeline.

    Parses command-line arguments for input/output directories (using defaults
    from configuration if not provided). Resets and manages metrics tracking
    (start/stop timer). Initiates the asynchronous `run_pipeline` function.
    Logs a final summary of execution metrics upon completion or failure.

    Command-line arguments override configuration defaults.

    Args:
        None (arguments are parsed from `sys.argv` via `argparse`).

    Returns:
        None

    Raises:
        SystemExit: If argument parsing fails.
        Exception: If `run_pipeline` encounters a critical, unhandled error.
    """
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Enriched Sentence Analysis Pipeline"
    )
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
    # Add an argument for map_dir
    parser.add_argument(
        "--map_dir",
        type=Path,
        default=Path(config["paths"].get("map_dir", "data/maps")),
        # Use get with default
        help="Path to directory for saving intermediate map files (.jsonl)",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # --- Use map_dir from args ---
    # Ensure map_dir key exists in config["paths"]
    # map_dir_path = Path(config["paths"].get("map_dir", "data/maps"))
    # Add an argument for map_dir to be command-line configurable
    # parser.add_argument(
    #     "--map_dir",
    #     type=Path,
    #     default=map_dir_path,
    #     help="Path to the directory for saving map files",
    # )
    # args = parser.parse_args() # Re-parse if adding new arg
    # map_dir_to_use = args.map_dir
    # map_dir_to_use = map_dir_path # Using config value for now
    map_dir_to_use = args.map_dir  # Use the parsed argument

    # Reset and start metrics tracking
    metrics_tracker.reset()
    metrics_tracker.start_pipeline_timer()

    # Log the start of the pipeline execution
    logger.info("Starting the Enriched Sentence Analysis Pipeline")
    try:
        # Pass the necessary arguments to run_pipeline
        asyncio.run(run_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            map_dir=map_dir_to_use,  # Pass map directory
            config_dict=config  # Pass the loaded config object
        ))
        logger.info("Pipeline execution completed.")
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
        metrics_tracker.increment_errors()  # Track pipeline-level errors
    finally:
        # Stop timer and log metrics summary
        metrics_tracker.stop_pipeline_timer()
        summary = metrics_tracker.get_summary()
        logger.info(
            f"Pipeline Execution Summary: {json.dumps(summary, indent=2)}"
        )


if __name__ == "__main__":
    main()
