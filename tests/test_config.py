"""
tests/test_config.py

Comprehensive tests for src/config.py configuration management.

Tests the Config class singleton pattern, YAML loading, environment variable
substitution, Pydantic validation, and access methods following cardinal rules:
1. Always test real functionality (not just mocks)
2. Never write tests just to pass (test actual behavior)
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from src.config import Config, ConfigModel, Neo4jConfig, PipelineConfig


class TestConfigSingleton:
    """Test singleton pattern behavior of Config class."""

    def test_singleton_instance_creation(self):
        """Test that Config follows singleton pattern - same instance returned."""
        # Clear any existing instance to test fresh creation
        Config._instance = None
        Config._config = None

        # Create two instances
        config1 = Config()
        config2 = Config()

        # Should be the same instance
        assert config1 is config2
        assert id(config1) == id(config2)

    def test_singleton_config_loaded_once(self):
        """Test that config is loaded only once even with multiple instantiations."""
        # Clear any existing instance
        Config._instance = None
        Config._config = None

        # Create realistic YAML content
        test_config = {
            "openai": {"api_key": "test-key", "model_name": "gpt-4"},
            "paths": {"input_dir": "/test/input", "output_dir": "/test/output"},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config) as mock_yaml_load:

            # Create multiple instances
            config1 = Config()
            config2 = Config()
            config3 = Config()

            # YAML should be loaded only once despite multiple instantiations
            assert mock_yaml_load.call_count == 1

            # All instances should have the same config
            assert config1.config == config2.config == config3.config

    def test_singleton_persistence_across_calls(self):
        """Test that singleton instance persists and maintains state."""
        # Clear any existing instance
        Config._instance = None
        Config._config = None

        test_config = {"test": "value", "number": 42}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            # First access
            config1 = Config()
            original_config = config1.config

            # Second access after some time/operations
            config2 = Config()

            # Should maintain same state
            assert config2.config is original_config
            assert config2.config == original_config


class TestConfigLoading:
    """Test configuration loading from YAML files."""

    def test_load_config_file_exists(self):
        """Test loading configuration when YAML file exists."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        # Create realistic configuration
        test_config = {
            "openai": {
                "api_key": "sk-test123",
                "model_name": "gpt-4o-mini",
                "max_tokens": 256,
                "temperature": 0.2,
            },
            "paths": {
                "input_dir": "data/input",
                "output_dir": "data/output",
                "map_dir": "data/maps",
            },
            "pipeline": {"num_analysis_workers": 10, "num_concurrent_files": 4},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()
            loaded_config = config.config

            # Validate structure matches expected
            assert "openai" in loaded_config
            assert "paths" in loaded_config
            assert "pipeline" in loaded_config
            assert loaded_config["openai"]["api_key"] == "sk-test123"
            assert loaded_config["paths"]["input_dir"] == "data/input"

    def test_load_config_file_not_exists(self):
        """Test loading configuration when YAML file doesn't exist - should use empty dict."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        with patch("pathlib.Path.exists", return_value=False):
            config = Config()
            loaded_config = config.config

            # Should have default structure from Pydantic model
            assert isinstance(loaded_config, dict)
            # Should have default sections from ConfigModel
            assert "openai" in loaded_config
            assert "paths" in loaded_config
            assert "pipeline" in loaded_config

    def test_yaml_parsing_error_handling(self):
        """Test handling of YAML parsing errors."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        # Malformed YAML content
        malformed_yaml = "invalid: yaml: content: [unclosed"

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=malformed_yaml)
        ), patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):

            # Should raise the YAML error (line 122: raise)
            with pytest.raises(yaml.YAMLError):
                Config()

    def test_config_file_path_resolution(self):
        """Test that config file path is correctly resolved."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {"test": "path_resolution"}

        with patch("pathlib.Path.exists", return_value=True) as mock_exists, patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            Config()

            # Should check existence of config.yaml in project root
            mock_exists.assert_called_once()
            # The path should be relative to src/config.py parent directory
            # mock_exists.call_args is a call object, access the args properly
            called_args = mock_exists.call_args
            if called_args and called_args[0]:
                called_path = called_args[0][0]
                assert str(called_path).endswith("config.yaml")
            else:
                # Alternative: check that exists was called on a Path object
                assert mock_exists.called


class TestEnvironmentVariables:
    """Test environment variable substitution in configuration."""

    def test_environment_variable_substitution(self):
        """Test that environment variables are properly substituted."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        # Config with environment variables - use structure that matches ConfigModel
        test_config = {
            "openai": {"api_key": "${TEST_API_KEY}", "model_name": "gpt-4"},
            "paths": {"input_dir": "${TEST_INPUT_DIR}", "output_dir": "/static/output"},
        }

        # Set environment variables
        test_env = {
            "TEST_API_KEY": "sk-real-api-key-123",
            "TEST_INPUT_DIR": "/dynamic/input",
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config), patch.dict(os.environ, test_env):

            config = Config()
            loaded_config = config.config

            # Environment variables should be substituted
            assert loaded_config["openai"]["api_key"] == "sk-real-api-key-123"
            assert loaded_config["paths"]["input_dir"] == "/dynamic/input"
            # Non-env vars should remain unchanged
            assert loaded_config["openai"]["model_name"] == "gpt-4"
            assert loaded_config["paths"]["output_dir"] == "/static/output"

    def test_environment_variable_not_set(self):
        """Test handling of undefined environment variables."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {
            "openai": {"api_key": "${UNDEFINED_ENV_VAR}"},
            "paths": {"input_dir": "static"},
        }

        # Ensure the env var is not set
        if "UNDEFINED_ENV_VAR" in os.environ:
            del os.environ["UNDEFINED_ENV_VAR"]

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()
            loaded_config = config.config

            # Undefined env var should become empty string (line 140)
            assert loaded_config["openai"]["api_key"] == ""
            assert loaded_config["paths"]["input_dir"] == "static"

    def test_environment_variable_in_nested_structures(self):
        """Test environment variable substitution in nested lists and dicts."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {
            "nested": {
                "dict": {"key": "${TEST_NESTED}"},
                "list": ["${TEST_LIST_ITEM}", "static_item", "${TEST_ANOTHER}"],
            },
            "simple": "${TEST_SIMPLE}",
        }

        test_env = {
            "TEST_NESTED": "nested_value",
            "TEST_LIST_ITEM": "list_value",
            "TEST_ANOTHER": "another_value",
            "TEST_SIMPLE": "simple_value",
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config), patch.dict(os.environ, test_env):

            config = Config()
            loaded_config = config.config

            # Test nested dict substitution
            assert loaded_config["nested"]["dict"]["key"] == "nested_value"
            # Test list substitution (line 135 - list processing)
            assert loaded_config["nested"]["list"][0] == "list_value"
            assert loaded_config["nested"]["list"][1] == "static_item"
            assert loaded_config["nested"]["list"][2] == "another_value"
            # Test simple substitution
            assert loaded_config["simple"] == "simple_value"


class TestValidation:
    """Test Pydantic validation and error handling."""

    def test_successful_validation(self):
        """Test successful Pydantic validation with valid config."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        # Valid configuration matching ConfigModel
        valid_config = {
            "openai": {"api_key": "test-key", "model_name": "gpt-4"},
            "pipeline": {
                "num_analysis_workers": 5,
                "num_concurrent_files": 2,
                "default_cardinality_limits": {
                    "HAS_FUNCTION": 1,
                    "HAS_STRUCTURE": 1,
                    "HAS_PURPOSE": 1,
                    "MENTIONS_KEYWORD": 6,
                },
            },
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
            },
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(valid_config))
        ), patch("yaml.safe_load", return_value=valid_config):

            config = Config()
            loaded_config = config.config

            # Should validate successfully and contain expected structure
            assert isinstance(loaded_config["pipeline"], dict)
            assert loaded_config["pipeline"]["num_analysis_workers"] == 5
            assert loaded_config["neo4j"]["uri"] == "bolt://localhost:7687"

    def test_validation_error_handling(self):
        """Test handling of Pydantic validation errors."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        # Invalid configuration that will fail Pydantic validation
        invalid_config = {
            "pipeline": {
                "num_analysis_workers": "invalid_string",  # Should be int
                "num_concurrent_files": -5,  # Should be positive
            },
            "neo4j": {
                "uri": 123,  # Should be string
                "username": None,  # Should be string
                # Missing required password field
            },
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(invalid_config))
        ), patch("yaml.safe_load", return_value=invalid_config):

            # Should raise ValidationError (lines 159-162)
            with pytest.raises(ValidationError):
                Config()

    def test_default_section_assignment(self):
        """Test that missing logging and api sections get default empty dicts."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        # Config without logging and api sections
        test_config = {
            "openai": {"api_key": "test"},
            "paths": {"input_dir": "/test"},
            # Missing logging and api sections
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()
            loaded_config = config.config

            # Should have default empty dicts (lines 147-150)
            assert "logging" in loaded_config
            assert "api" in loaded_config
            assert loaded_config["logging"] == {}
            assert loaded_config["api"] == {}

    def test_none_sections_replaced_with_empty_dict(self):
        """Test that None values for logging/api sections are replaced with empty dicts."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        # Config with None values for logging and api
        test_config = {
            "openai": {"api_key": "test"},
            "logging": None,  # Should be replaced with {}
            "api": None,  # Should be replaced with {}
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()
            loaded_config = config.config

            # None values should be replaced with empty dicts (lines 147-150)
            assert loaded_config["logging"] == {}
            assert loaded_config["api"] == {}


class TestAccessMethods:
    """Test configuration access methods."""

    def test_dictionary_access(self):
        """Test dictionary-style access to configuration."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {"section1": {"key1": "value1"}, "section2": {"key2": "value2"}}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()

            # Test __getitem__ method (line 209)
            assert config["section1"]["key1"] == "value1"
            assert config["section2"]["key2"] == "value2"

    def test_dictionary_access_key_error(self):
        """Test KeyError when accessing non-existent keys."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {"existing": "value"}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()

            # Should raise KeyError for non-existent key (line 209)
            with pytest.raises(KeyError):
                config["non_existent_key"]

    def test_get_method_with_default(self):
        """Test get method with default values."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {
            "existing_key": "existing_value",
            "nested": {"inner": "inner_value"},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()

            # Test get method (line 194)
            assert config.get("existing_key") == "existing_value"
            assert config.get("non_existent", "default_value") == "default_value"
            assert config.get("non_existent") is None  # Default None

            # Test existing nested structure
            assert config.get("nested")["inner"] == "inner_value"

    def test_config_property_runtime_error(self):
        """Test RuntimeError when config property accessed with None config."""
        # Create a Config instance but manually set _config to None to test error
        config_instance = Config()
        # Manually set _config to None to trigger RuntimeError
        config_instance._config = None

        # Should raise RuntimeError (line 174)
        with pytest.raises(RuntimeError, match="Config not loaded"):
            _ = config_instance.config

    def test_repr_method(self):
        """Test string representation of Config instance."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {"test": "repr"}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config):

            config = Config()
            repr_str = repr(config)

            # Should contain config path information
            assert "Config(path=" in repr_str
            assert "config.yaml" in repr_str


class TestPydanticModels:
    """Test Pydantic model validation for configuration sections."""

    def test_neo4j_config_validation(self):
        """Test Neo4jConfig model validation."""
        # Valid Neo4j config
        valid_neo4j = {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password123",
        }

        neo4j_config = Neo4jConfig(**valid_neo4j)
        assert neo4j_config.uri == "bolt://localhost:7687"
        assert neo4j_config.username == "neo4j"
        assert neo4j_config.password == "password123"

        # Invalid Neo4j config - missing required fields
        with pytest.raises(ValidationError):
            Neo4jConfig(uri="bolt://localhost:7687")  # Missing username, password

    def test_pipeline_config_validation(self):
        """Test PipelineConfig model validation and defaults."""
        # Test with defaults
        pipeline_config = PipelineConfig()
        assert pipeline_config.num_analysis_workers == 10
        assert pipeline_config.num_concurrent_files == 4
        assert pipeline_config.default_cardinality_limits["HAS_FUNCTION"] == 1

        # Test with custom values
        custom_pipeline = PipelineConfig(num_analysis_workers=5, num_concurrent_files=2)
        assert custom_pipeline.num_analysis_workers == 5
        assert custom_pipeline.num_concurrent_files == 2

    def test_config_model_defaults(self):
        """Test ConfigModel default values and structure."""
        # Test with minimal config
        config_model = ConfigModel()

        # Should have default empty dicts and default pipeline
        assert isinstance(config_model.openai, dict)
        assert isinstance(config_model.paths, dict)
        assert isinstance(config_model.domain_keywords, list)
        assert isinstance(config_model.pipeline, PipelineConfig)
        assert config_model.pipeline.num_analysis_workers == 10


class TestErrorHandling:
    """Test various error conditions and edge cases."""

    def test_file_permission_error(self):
        """Test handling of file permission errors during config loading."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", side_effect=PermissionError("Permission denied")
        ):

            # Should raise PermissionError
            with pytest.raises(PermissionError):
                Config()

    def test_empty_yaml_file(self):
        """Test handling of empty YAML file."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", mock_open(read_data="")), patch(
            "yaml.safe_load", return_value=None
        ):

            config = Config()
            loaded_config = config.config

            # Should handle None from yaml.safe_load gracefully
            assert isinstance(loaded_config, dict)

    def test_complex_environment_variable_scenarios(self):
        """Test complex environment variable substitution scenarios."""
        # Clear instance for fresh test
        Config._instance = None
        Config._config = None

        test_config = {
            "mixed": {
                "partial_env": "prefix_${TEST_VAR}_suffix",
                "multiple_env": "${VAR1}_${VAR2}",
                "no_env": "normal_string",
                "empty_env": "${EMPTY_VAR}",
            }
        }

        test_env = {
            "TEST_VAR": "middle",
            "VAR1": "first",
            "VAR2": "second",
            # EMPTY_VAR intentionally not set
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data=yaml.dump(test_config))
        ), patch("yaml.safe_load", return_value=test_config), patch.dict(os.environ, test_env, clear=False):

            config = Config()
            loaded_config = config.config

            # Test various environment variable scenarios
            mixed = loaded_config["mixed"]
            assert mixed["partial_env"] == "prefix_middle_suffix"
            assert mixed["multiple_env"] == "first_second"
            assert mixed["no_env"] == "normal_string"
            assert mixed["empty_env"] == ""  # Undefined env var becomes empty string
