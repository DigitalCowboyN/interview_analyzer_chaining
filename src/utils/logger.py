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

# Add formatted logging to file
logger.add(
    log_file_path,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="INFO",
)

# Optional: Add logging to stdout with different verbosity
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    level="DEBUG",
)


def get_logger():
    return logger
from loguru import logger
import os
from pathlib import Path
from src.config import config

# Ensure log directory exists
log_dir = Path(config["paths"]["logs_dir"])
log_dir.mkdir(parents=True, exist_ok=True)

log_file_path = log_dir / "pipeline.log"

logger.remove()  # Remove default handlers to configure custom ones

# Add formatted logging to file
logger.add(
    log_file_path,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="INFO",
)

# Optional: Add logging to stdout with different verbosity
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    level="DEBUG",
)


def get_logger():
    return logger
