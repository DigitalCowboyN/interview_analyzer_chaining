"""
logger.py

Configures the application-wide logger using Python's standard `logging` module.

This module sets up a single logger instance named 'pipeline' (accessible via
`get_logger`) configured with two handlers upon first access:

1.  File Handler:
    - Writes logs to a file specified in the configuration
      (`config["paths"]["logs_dir"]/pipeline.log`, defaults to `./logs/pipeline.log`).
    - Level: `INFO` and above.
    - Format: Detailed (Timestamp, Level, LoggerName:FuncName:LineNo, Message).
    - Handles potential config errors or OS errors during setup.

2.  Stream Handler (stdout):
    - Prints logs to standard output (`sys.stdout`).
    - Level: `DEBUG` and above.
    - Format: Simpler (Time, Level, Message).
    - Uses basicConfig as a fallback if stream handler setup fails.

The configuration is performed only once by the internal `_setup_logger` function.
Subsequent calls to `get_logger` return the same configured logger instance.
"""

import logging
import sys
from pathlib import Path

from src.config import config

# --- Configuration Constants ---
_logger_instance = None
_DEFAULT_LOG_DIR = "./logs"
_DEFAULT_LOG_FILENAME = "pipeline.log"
_APP_LOGGER_NAME = "pipeline"


def _setup_logger() -> logging.Logger:
    """
    Internal function to configure and return the singleton logger instance.

    This function is called by `get_logger` and should not be called directly.
    It performs the setup only once.

    - Retrieves/creates the logger instance for `_APP_LOGGER_NAME`.
    - Sets the logger's base level to DEBUG.
    - Clears existing handlers to prevent duplication if called again.
    - Creates, configures (level, formatter), and adds a StreamHandler for console output.
    - Creates, configures (level, formatter), and adds a FileHandler using paths
      from the global `config` object, with error handling for path/OS issues.

    Returns:
        logging.Logger: The configured singleton logger instance.
    """
    global _logger_instance
    if _logger_instance:
        return _logger_instance

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # Set root level to lowest handler level

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Formatter Definitions ---
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # --- Stream Handler (Console) ---
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to console
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    except Exception as e:
        # Fallback basic logging if stream handler fails
        logging.basicConfig(level=logging.WARNING)  # Use basicConfig on the root logger
        logging.error(f"Failed to configure console logging: {e}", exc_info=True)
        # Continue to try setting up file handler

    # --- File Handler ---
    try:
        log_dir_path_str = config.get("paths", {}).get("logs_dir", _DEFAULT_LOG_DIR)
        log_dir = Path(log_dir_path_str)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / _DEFAULT_LOG_FILENAME

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)  # Log INFO and above to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    except KeyError as e:
        # Log error using the already configured console handler (or basicConfig)
        logger.error(
            f"Log directory path key missing in config: {e}. File logging disabled."
        )
    except OSError as e:
        logger.error(
            f"Failed to create log directory or file at '{log_dir_path_str}': {e}. File logging disabled."
        )
    except Exception as e:
        logger.error(f"Unexpected error configuring file logging: {e}", exc_info=True)

    _logger_instance = logger
    return _logger_instance


def get_logger() -> logging.Logger:
    """
    Public function to retrieve the application's configured logger instance.

    Calls the internal `_setup_logger` function to perform the one-time
    configuration on the first call. Subsequent calls return the existing
    singleton logger instance.

    Returns:
        logging.Logger: The application's configured logger instance.
    """
    return _setup_logger()


# Initialize logger on import (optional, ensures setup happens early)
# get_logger()
