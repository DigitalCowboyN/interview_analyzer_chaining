"""
config.py

This module provides a robust configuration management system for the application.
It securely loads settings from a YAML file, explicitly expands environment variables,
and provides convenient access to these settings through a singleton instance of the Config class.

Usage Example:

1. Import the config instance:
   from src.config import config

2. Access a configuration value:
   api_key = config["openai"]["api_key"]
"""

import os
import yaml
import re
from pathlib import Path


class Config:
    """
    Manages application configuration, loading settings from a YAML file.

    Loads configuration settings from a specified YAML file, expands environment 
    variables explicitly (format: ${ENV_VAR_NAME}), and provides dictionary-like 
    access. This class is typically used as a singleton via the `config` instance 
    defined at the module level.

    Attributes:
        config_path (Path): Path to the configuration YAML file.
        config (dict): The loaded and processed configuration settings.

    Args:
        config_path (str): Relative path to the configuration file from the project root.
                           Defaults to "config.yaml".
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the Config instance by setting the path and loading the config.

        Args:
            config_path (str): Relative path to the configuration file. 
                               Defaults to "config.yaml".
        """
        self.config_path = Path(__file__).parent.parent / config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads configuration from the YAML file and expands environment variables.

        Environment variables in the YAML file should be formatted as ${VAR_NAME}.
        Variables not found in the environment retain their ${VAR_NAME} format.

        Returns:
            dict: The parsed and processed configuration settings.

        Raises:
            FileNotFoundError: If the configuration file specified by `config_path` does not exist.
            yaml.YAMLError: If the file content is not valid YAML.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        content = self.config_path.read_text(encoding="utf-8")

        # Explicitly expand environment variables formatted as ${VAR_NAME}
        content = re.sub(
            r"\$\{([^}]+)\}",
            lambda match: os.environ.get(match.group(1), match.group(0)),
            content
        )

        return yaml.safe_load(content)

    def get(self, key, default=None):
        """
        Retrieves a configuration value by key, returning a default if not found.

        Args:
            key (str): The dot-separated key or simple key of the configuration setting 
                     (e.g., 'openai.api_key' or 'some_top_level_key').
            default (optional): The value to return if the key is not found. Defaults to None.

        Returns:
            Any: The configuration value or the provided default.
        """
        # Simple implementation for top-level keys; can be extended for nested keys if needed
        return self.config.get(key, default)

    def __getitem__(self, key):
        """
        Enables dictionary-style access (e.g., `config['openai']`).

        Args:
            key (str): The top-level key of the configuration setting.

        Returns:
            Any: The configuration value associated with the given key.

        Raises:
            KeyError: If the key does not exist in the configuration.
        """
        return self.config[key]

    def __repr__(self):
        """
        Provides a developer-friendly string representation of the Config instance.

        Returns:
            str: Representation including the config file path (e.g., "Config(path/to/config.yaml)").
        """
        return f"Config(path={self.config_path})" # Added path= for clarity


# Singleton instance for global configuration access
config = Config()

# --- Add validation or specific accessors for new config keys if needed ---
# Example (optional validation):
# if config.get("pipeline", {}).get("num_analysis_workers", 1) < 0:
#     raise ValueError("num_analysis_workers cannot be negative")
