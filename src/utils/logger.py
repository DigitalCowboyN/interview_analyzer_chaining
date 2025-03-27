# src/utils/logger.py
from loguru import logger
import os
from pathlib import Path
from src.config import config

# Ensure log directory exists
log_dir = Path(config["paths"]["logs_dir"])
log_dir.mkdir(parents=True, exist_ok=True)

log_file_path = log_dir / "pipeline.log"

logger.remove()  # Remove default handlers to configure custom ones

# Add structured JSON logging to file
logger.add(
    log_file_path,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="INFO",
    serialize=True  # This outputs the log as a JSON string
)

# Optional: Add logging to stdout with different verbosity in human-readable format.
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    level="DEBUG",
)

def get_logger():
    """
    Returns the configured logger instance.

    The logger outputs logs to a file in structured JSON format for easier integration
    with log management systems, and also prints human-readable logs to stdout.
    """
    return logger
