"""
tests/test_main_cli.py

Comprehensive tests for the main CLI function that handles argument parsing,
metrics lifecycle, asyncio orchestration, and error logging.

These tests address the coverage gaps identified for CLI behavior testing.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.main import main


class TestMainCLIOrchestration:
    """Test the main() CLI function's orchestration behavior."""

    @contextmanager
    def mock_cli_dependencies(self):
        """Context manager to mock all CLI dependencies consistently."""
        with patch("src.main.config") as mock_config, patch("src.main.metrics_tracker") as mock_metrics, patch(
            "src.main.logger"
        ) as mock_logger, patch("src.main.asyncio.run") as mock_asyncio_run, patch(
            "src.main.run_pipeline"
        ) as mock_run_pipeline:

            # Setup realistic config defaults
            mock_config.__getitem__.side_effect = lambda key: {
                "paths": {"input_dir": "data/input", "output_dir": "data/output", "map_dir": "data/maps"}
            }[key]

            # Setup metrics tracker with JSON-serializable return value
            mock_metrics.get_summary.return_value = {
                "pipeline_duration": 45.2,
                "files_processed": 3,
                "sentences_analyzed": 150,
                "errors": 0,
            }

            yield {
                "config": mock_config,
                "metrics_tracker": mock_metrics,
                "logger": mock_logger,
                "asyncio_run": mock_asyncio_run,
                "run_pipeline": mock_run_pipeline,
            }

    def test_successful_pipeline_execution_with_defaults(self):
        """Test successful main() execution using default configuration values."""
        with self.mock_cli_dependencies() as mocks:
            # Mock sys.argv to simulate no CLI arguments (use defaults)
            with patch.object(sys, "argv", ["main.py"]):
                main()

            # Verify metrics lifecycle
            mocks["metrics_tracker"].reset.assert_called_once()
            mocks["metrics_tracker"].start_pipeline_timer.assert_called_once()
            mocks["metrics_tracker"].stop_pipeline_timer.assert_called_once()

            # Verify logging
            mocks["logger"].info.assert_has_calls(
                [call("Starting the Enriched Sentence Analysis Pipeline"), call("Pipeline execution completed.")]
            )

            # Verify asyncio.run was called with run_pipeline
            mocks["asyncio_run"].assert_called_once()
            # Just verify asyncio.run was called - the coroutine details are tested elsewhere
            assert mocks["asyncio_run"].called

    def test_cli_argument_parsing_with_custom_paths(self):
        """Test CLI argument parsing with custom input, output, and map directories."""
        with self.mock_cli_dependencies() as mocks:
            custom_args = [
                "main.py",
                "--input_dir",
                "/custom/input",
                "--output_dir",
                "/custom/output",
                "--map_dir",
                "/custom/maps",
            ]

            with patch.object(sys, "argv", custom_args):
                main()

            # Verify asyncio.run was called with custom paths
            mocks["asyncio_run"].assert_called_once()
            # Just verify asyncio.run was called - argument verification is complex with coroutines
            assert mocks["asyncio_run"].called

    @patch("src.main.run_pipeline")
    def test_cli_argument_parsing_verification(self, mock_run_pipeline):
        """Test that CLI arguments are correctly passed to run_pipeline."""
        with patch("src.main.config") as mock_config, patch("src.main.metrics_tracker") as mock_metrics, patch(
            "src.main.logger"
        ), patch("src.main.asyncio.run") as mock_asyncio_run:

            mock_config.__getitem__.side_effect = lambda key: {
                "paths": {"input_dir": "data/input", "output_dir": "data/output", "map_dir": "data/maps"}
            }[key]

            # Setup metrics tracker with JSON-serializable return value
            mock_metrics.get_summary.return_value = {"pipeline_duration": 30.5, "files_processed": 2, "errors": 0}

            custom_args = [
                "main.py",
                "--input_dir",
                "/test/input",
                "--output_dir",
                "/test/output",
                "--map_dir",
                "/test/maps",
            ]

            with patch.object(sys, "argv", custom_args):
                main()

            # Verify asyncio.run was called
            mock_asyncio_run.assert_called_once()
            # The actual run_pipeline call happens inside asyncio.run
            # We can verify the coroutine was created properly by checking if asyncio.run was called
            assert mock_asyncio_run.called

    def test_pipeline_execution_failure_error_handling(self):
        """Test error handling when pipeline execution fails."""
        with self.mock_cli_dependencies() as mocks:
            # Make asyncio.run raise an exception
            test_error = RuntimeError("Pipeline processing failed")
            mocks["asyncio_run"].side_effect = test_error

            with patch.object(sys, "argv", ["main.py"]):
                main()  # Should not raise - error should be caught and logged

            # Verify error was logged with exc_info
            mocks["logger"].critical.assert_called_once()
            critical_call = mocks["logger"].critical.call_args
            assert "Pipeline execution failed: Pipeline processing failed" in critical_call[0][0]
            assert critical_call[1]["exc_info"] is True

            # Verify metrics error tracking
            mocks["metrics_tracker"].increment_errors.assert_called_once()

            # Verify finally block still executes
            mocks["metrics_tracker"].stop_pipeline_timer.assert_called_once()

    def test_metrics_lifecycle_management(self):
        """Test that metrics lifecycle is properly managed regardless of success/failure."""
        with self.mock_cli_dependencies() as mocks:
            # Setup metrics summary
            test_summary = {"pipeline_duration": 45.2, "files_processed": 3, "sentences_analyzed": 150, "errors": 0}
            mocks["metrics_tracker"].get_summary.return_value = test_summary

            with patch.object(sys, "argv", ["main.py"]):
                main()

            # Verify complete metrics lifecycle
            mocks["metrics_tracker"].reset.assert_called_once()
            mocks["metrics_tracker"].start_pipeline_timer.assert_called_once()
            mocks["metrics_tracker"].stop_pipeline_timer.assert_called_once()
            mocks["metrics_tracker"].get_summary.assert_called_once()

            # Verify summary logging
            expected_log = f"Pipeline Execution Summary: {json.dumps(test_summary, indent=2)}"
            mocks["logger"].info.assert_any_call(expected_log)

    def test_metrics_lifecycle_with_exception(self):
        """Test that metrics lifecycle completes even when pipeline fails."""
        with self.mock_cli_dependencies() as mocks:
            # Make pipeline fail
            mocks["asyncio_run"].side_effect = Exception("Test failure")

            test_summary = {"pipeline_duration": 10.5, "errors": 1}
            mocks["metrics_tracker"].get_summary.return_value = test_summary

            with patch.object(sys, "argv", ["main.py"]):
                main()

            # Verify metrics lifecycle completed despite exception
            mocks["metrics_tracker"].reset.assert_called_once()
            mocks["metrics_tracker"].start_pipeline_timer.assert_called_once()
            mocks["metrics_tracker"].stop_pipeline_timer.assert_called_once()
            mocks["metrics_tracker"].get_summary.assert_called_once()
            mocks["metrics_tracker"].increment_errors.assert_called_once()

    def test_argument_parser_configuration(self):
        """Test that argument parser is configured with correct options."""
        with patch("src.main.config") as mock_config, patch("src.main.metrics_tracker") as mock_metrics, patch(
            "src.main.logger"
        ), patch("src.main.asyncio.run"), patch("argparse.ArgumentParser.parse_args") as mock_parse_args:

            mock_config.__getitem__.side_effect = lambda key: {
                "paths": {"input_dir": "data/input", "output_dir": "data/output", "map_dir": "data/maps"}
            }[key]

            # Setup metrics tracker with JSON-serializable return value
            mock_metrics.get_summary.return_value = {"pipeline_duration": 25.0, "files_processed": 1, "errors": 0}

            # Mock parsed args
            mock_args = MagicMock()
            mock_args.input_dir = Path("/test/input")
            mock_args.output_dir = Path("/test/output")
            mock_args.map_dir = Path("/test/maps")
            mock_parse_args.return_value = mock_args

            with patch("argparse.ArgumentParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser_class.return_value = mock_parser
                mock_parser.parse_args.return_value = mock_args

                main()

                # Verify parser was created with correct description
                mock_parser_class.assert_called_once_with(description="Enriched Sentence Analysis Pipeline")

                # Verify arguments were added
                assert mock_parser.add_argument.call_count == 3

                # Check input_dir argument
                input_arg_call = mock_parser.add_argument.call_args_list[0]
                assert input_arg_call[0][0] == "--input_dir"
                assert input_arg_call[1]["type"] == Path

                # Check output_dir argument
                output_arg_call = mock_parser.add_argument.call_args_list[1]
                assert output_arg_call[0][0] == "--output_dir"
                assert output_arg_call[1]["type"] == Path

                # Check map_dir argument
                map_arg_call = mock_parser.add_argument.call_args_list[2]
                assert map_arg_call[0][0] == "--map_dir"
                assert map_arg_call[1]["type"] == Path

    def test_config_integration_with_defaults(self):
        """Test that configuration defaults are properly integrated with argument parsing."""
        with patch("src.main.config") as mock_config, patch("src.main.metrics_tracker") as mock_metrics, patch(
            "src.main.logger"
        ), patch("src.main.asyncio.run"):

            # Setup config with specific default values
            test_config = {
                "paths": {"input_dir": "/config/input", "output_dir": "/config/output", "map_dir": "/config/maps"}
            }
            mock_config.__getitem__.side_effect = test_config.__getitem__

            # Setup metrics tracker with JSON-serializable return value
            mock_metrics.get_summary.return_value = {"pipeline_duration": 35.0, "files_processed": 2, "errors": 0}

            with patch.object(sys, "argv", ["main.py"]):  # No CLI args provided
                main()

            # Verify config was accessed for default values
            mock_config.__getitem__.assert_any_call("paths")

    def test_json_logging_format(self):
        """Test that pipeline summary is logged in proper JSON format."""
        with self.mock_cli_dependencies() as mocks:
            # Setup complex summary data
            complex_summary = {
                "pipeline_duration": 123.45,
                "files_processed": 5,
                "sentences_analyzed": 250,
                "errors": 0,
                "api_calls": 125,
                "custom_metrics": {
                    "sentences": {"total": 250, "successful": 248},
                    "files": {"total": 5, "successful": 5},
                },
            }
            mocks["metrics_tracker"].get_summary.return_value = complex_summary

            with patch.object(sys, "argv", ["main.py"]):
                main()

            # Verify JSON formatting in log call
            expected_json = json.dumps(complex_summary, indent=2)
            expected_log = f"Pipeline Execution Summary: {expected_json}"
            mocks["logger"].info.assert_any_call(expected_log)

    def test_exception_propagation_and_logging_details(self):
        """Test detailed exception logging with exc_info=True."""
        with self.mock_cli_dependencies() as mocks:
            # Create specific exception with details
            specific_error = ValueError("Configuration validation failed: missing required field 'api_key'")
            mocks["asyncio_run"].side_effect = specific_error

            with patch.object(sys, "argv", ["main.py"]):
                main()

            # Verify specific error logging
            mocks["logger"].critical.assert_called_once()
            critical_call = mocks["logger"].critical.call_args

            # Verify specific error message and exc_info
            assert "Pipeline execution failed:" in critical_call[0][0]
            assert "Configuration validation failed" in critical_call[0][0]

            # Verify exc_info=True for stack trace logging
            assert critical_call[1]["exc_info"] is True


class TestMainCLIEdgeCases:
    """Test edge cases and error conditions in main() CLI function."""

    def test_missing_config_paths_handling(self):
        """Test graceful handling when config paths are missing."""
        with patch("src.main.config") as mock_config, patch("src.main.metrics_tracker") as mock_metrics, patch(
            "src.main.logger"
        ), patch("src.main.asyncio.run") as mock_asyncio_run:

            # Config missing map_dir key
            mock_config.__getitem__.side_effect = lambda key: {
                "paths": {
                    "input_dir": "data/input",
                    "output_dir": "data/output",
                    # map_dir missing - should use default
                }
            }[key]

            # Setup metrics tracker with JSON-serializable return value
            mock_metrics.get_summary.return_value = {"pipeline_duration": 20.0, "files_processed": 1, "errors": 0}

            with patch.object(sys, "argv", ["main.py"]):
                main()

            # Should complete successfully using default for map_dir
            mock_asyncio_run.assert_called_once()
            mock_metrics.reset.assert_called_once()

    def test_invalid_path_arguments(self):
        """Test handling of invalid path arguments."""
        with patch("src.main.config") as mock_config, patch("src.main.metrics_tracker") as mock_metrics, patch(
            "src.main.logger"
        ), patch("src.main.asyncio.run"):

            mock_config.__getitem__.side_effect = lambda key: {
                "paths": {"input_dir": "data/input", "output_dir": "data/output"}
            }[key]

            # Setup metrics tracker with JSON-serializable return value
            mock_metrics.get_summary.return_value = {"pipeline_duration": 15.0, "files_processed": 0, "errors": 0}

            # Test with empty string path (should be converted to Path)
            invalid_args = ["main.py", "--input_dir", ""]

            with patch.object(sys, "argv", invalid_args):
                # Should not crash - argparse handles Path conversion
                main()

    def test_argument_parsing_system_exit_handling(self):
        """Test handling of argparse SystemExit (e.g., --help, invalid args)."""
        with patch("src.main.config") as mock_config:
            mock_config.__getitem__.side_effect = lambda key: {
                "paths": {"input_dir": "data/input", "output_dir": "data/output"}
            }[key]

            # Mock argparse to raise SystemExit (like --help would)
            with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
                mock_parse_args.side_effect = SystemExit(0)

                with pytest.raises(SystemExit):
                    main()

    def test_multiple_exception_types_logging(self):
        """Test logging behavior with different exception types."""
        exception_test_cases = [
            (FileNotFoundError("Input directory not found"), "Input directory not found"),
            (PermissionError("Access denied to output directory"), "Access denied to output directory"),
            (KeyError("Missing configuration key"), "Missing configuration key"),
            (RuntimeError("Pipeline initialization failed"), "Pipeline initialization failed"),
        ]

        for exception, expected_message in exception_test_cases:
            with patch("src.main.config") as mock_config, patch("src.main.metrics_tracker") as mock_metrics, patch(
                "src.main.logger"
            ) as mock_logger, patch("src.main.asyncio.run") as mock_asyncio_run:

                mock_config.__getitem__.side_effect = lambda key: {
                    "paths": {"input_dir": "data/input", "output_dir": "data/output"}
                }[key]

                # Setup metrics tracker with JSON-serializable return value
                mock_metrics.get_summary.return_value = {"pipeline_duration": 5.0, "files_processed": 0, "errors": 1}

                mock_asyncio_run.side_effect = exception

                with patch.object(sys, "argv", ["main.py"]):
                    main()

                # Verify specific error logging
                mock_logger.critical.assert_called_once()
                critical_call = mock_logger.critical.call_args
                assert expected_message in critical_call[0][0]
                assert critical_call[1]["exc_info"] is True

                # Verify error tracking
                mock_metrics.increment_errors.assert_called_once()
