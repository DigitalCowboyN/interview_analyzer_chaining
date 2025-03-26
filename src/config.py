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

    This class loads configuration settings from a specified YAML file,
    expands environment variables explicitly, and provides dictionary-like access.

    Attributes:
        config_path (Path): Path to the configuration YAML file.
        config (dict): Loaded configuration settings.

    Parameters:
        config_path (str): Relative path to the configuration file (default: "config.yaml").
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(__file__).parent.parent / config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads and returns configuration from the YAML file after explicitly expanding environment variables.

        Environment variables in the YAML file should be formatted as ${ENV_VAR_NAME}.

        Returns:
            dict: Parsed configuration settings.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
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
        Retrieves a configuration value by key.

        Parameters:
            key (str): Key of the configuration setting.
            default (optional): Default value if key is not found.

        Returns:
            Any: The configuration value or the provided default.
        """
        return self.config.get(key, default)

    def __getitem__(self, key):
        """
        Enables dictionary-style access to configuration.

        Parameters:
            key (str): Key of the configuration setting.

        Returns:
            Any: The configuration value associated with the given key.

        Raises:
            KeyError: If the key does not exist.
        """
        return self.config[key]

    def __repr__(self):
        """
        Provides a string representation of the Config instance.

        Returns:
            str: Representation including the config file path.
        """
        return f"Config({self.config_path})"


# Singleton instance for global configuration access
config = Config()
