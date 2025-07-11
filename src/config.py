"""
config.py

This module provides a robust configurations
management system for the application.
It securely loads settings from a YAML file,
explicitly expands environment variables,
and provides convenient access to these settings
through a singleton instance of the Config class.

Usage Example:

1. Import the config instance:
   from src.config import config

2. Access a configuration value:
   api_key = config["openai"]["api_key"]
"""

import json
import os

# import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

# Define a Pydantic model for nested configuration sections if
# desired for validation
# (Example - can be expanded)
# class PathsConfig(BaseModel):
#     input_dir: str
#     output_dir: str
#     map_dir: str
#     # ... other path related configs ...


class Neo4jConfig(BaseModel):
    uri: str
    username: str
    password: str
    # database: Optional[str] = "neo4j"


class PipelineConfig(BaseModel):
    num_analysis_workers: int = 10  # Keep existing worker count
    num_concurrent_files: int = 4
    # Add the new cardinality limits dictionary
    default_cardinality_limits: Dict[str, Optional[int]] = {
        "HAS_FUNCTION": 1,
        "HAS_STRUCTURE": 1,
        "HAS_PURPOSE": 1,
        "MENTIONS_KEYWORD": 6,
        "MENTIONS_TOPIC": None,  # Use None for unlimited
        "MENTIONS_DOMAIN_KEYWORD": None,
    }
    # Optional: Add retry settings for pipeline steps if needed


class ConfigModel(BaseModel):
    openai: Dict[str, Any] = Field(default_factory=dict)
    openai_api: Dict[str, Any] = Field(default_factory=dict)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    classification: Dict[str, Any] = Field(default_factory=dict)
    paths: Dict[str, Any] = Field(default_factory=dict)
    domain_keywords: List[str] = Field(default_factory=list)
    pipeline: PipelineConfig = PipelineConfig()
    logging: Dict[str, Any] = Field(default_factory=dict)
    neo4j: Optional[Neo4jConfig] = None
    api: Dict[str, Any] = Field(default_factory=dict)
    # Add other top-level sections as needed


class Config:
    """
    Manages application configuration, loading settings from a YAML file.

    Loads configuration settings from a specified YAML file,
    expands environment variables explicitly (format: ${ENV_VAR_NAME}),
    and provides dictionary-like access. This class is typically
    used as a singleton via the `config` instance defined at the module level.

    Attributes:
        config_path (Path): Path to the configuration YAML file.
        config (dict): The loaded and processed configuration settings.

    Args:
        config_path (str): Relative path to
        the configuration file from the project root.
                           Defaults to "config.yaml".
    """

    _instance = None
    _config: Optional[Dict[str, Any]] = None  # Type hint for the config attribute

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> Dict[str, Any]:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
        # env_path = project_root / ".env"
        # Path reference might still be useful for debugging/context

        # Remove the call to load_dotenv,
        # as Docker Compose/Dev Container handles it
        # load_dotenv(dotenv_path=env_path)

        config_dict = {}
        if config_path.exists():
            with open(config_path, "r") as stream:
                try:
                    config_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(f"Error loading config.yaml: {exc}")
                    # Decide how to handle: raise error, use defaults, exit?
                    raise

        # Substitute environment variables
        config_str = json.dumps(config_dict)
        config_str = os.path.expandvars(config_str.replace("${", "${ENV_"))
        # Prefixing to avoid clashes
        config_dict = json.loads(config_str)

        # Rename keys back by removing the temporary prefix
        def remove_prefix(data: Any) -> Any:
            if isinstance(data, dict):
                return {k: remove_prefix(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [remove_prefix(item) for item in data]
            elif isinstance(data, str) and data.startswith("${ENV_"):
                env_var_name = data[6:-1]  # Extract original name
                # Return the actual env var value,
                # or None/empty string if not set
                return os.getenv(env_var_name, "")
            return data

        final_config: Dict[str, Any] = remove_prefix(config_dict)

        # Ensure 'logging' and 'api' are dictionaries,
        # defaulting if None or missing
        if final_config.get("logging") is None:
            final_config["logging"] = {}
        if final_config.get("api") is None:
            final_config["api"] = {}

        # Validate using Pydantic model
        try:
            validated_config = ConfigModel(**final_config)
            # Validate structure and capture validated data
            # Optionally convert back to dict if needed downstream, though
            # using the model is often better
            final_config = validated_config.model_dump()
        except ValidationError as e:
            print(f"Configuration validation error: {e}")
            # Decide how to handle validation errors
            raise

        # Ensure essential keys are present (example)
        # if "openai" not in final_config or "api_key" not
        # in final_config["openai"]:
        #     raise ValueError("Missing essential OpenAI configuration")

        return final_config

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            raise RuntimeError("Config not loaded")
        return self._config

    def get(self, key, default=None):
        """
        Retrieves a configuration value by key,
        returning a default if not found.

        Args:
            key (str): The dot-separated key or
            simple key of the configuration setting
                     (e.g., 'openai.api_key' or 'some_top_level_key').
            default (optional): The value to return if the key is not found.
            Defaults to None.

        Returns:
            Any: The configuration value or the provided default.
        """
        # Simple implementation for top-level keys;
        # can be extended for nested keys if needed
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
        Provides a developer-friendly
        string representation of the Config instance.

        Returns:
            str: Representation including the config file path
            (e.g., "Config(path/to/config.yaml)").
        """
        return f"Config(path={Path(__file__).parent.parent / 'config.yaml'})"
        # Added path= for clarity


# Singleton instance
config = Config().config

# --- Add validation or specific accessors for new config keys if needed ---
# Example (optional validation):
# if config.get("pipeline", {}).get("num_analysis_workers", 1) < 0:
#     raise ValueError("num_analysis_workers cannot be negative")

# Make Celery config easily accessible (optional)
# celery_config = config.get('celery', {})
# CELERY_BROKER_URL = celery_config.get('broker_url')
# CELERY_RESULT_BACKEND = celery_config.get('result_backend')
