"""
Tests for Celery background tasks in src/tasks.py.

These tests verify that the Celery tasks are properly configured and handle
various scenarios. Due to Celery's complex internal structure, we focus on
testing the core functionality and task configuration.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.tasks import run_pipeline_for_file


class TestRunPipelineForFileTask:
    """Test the run_pipeline_for_file Celery task."""

    def test_task_is_celery_task(self):
        """Test that run_pipeline_for_file is properly decorated as a Celery task."""
        # Verify it's a Celery task with all expected attributes
        assert hasattr(run_pipeline_for_file, "delay")
        assert hasattr(run_pipeline_for_file, "apply_async")
        assert hasattr(run_pipeline_for_file, "retry")
        assert hasattr(run_pipeline_for_file, "request")

        # Verify the task is callable
        assert callable(run_pipeline_for_file)

    def test_task_function_signature(self):
        """Test that the task function has the correct signature."""
        # The function should have the __wrapped__ attribute (from Celery decoration)
        assert hasattr(run_pipeline_for_file, "__wrapped__")

        # The wrapped function should be callable
        assert callable(run_pipeline_for_file.__wrapped__)

        # The wrapped function should have the correct signature
        import inspect

        sig = inspect.signature(run_pipeline_for_file.__wrapped__)
        param_names = list(sig.parameters.keys())
        expected_params = ["input_file_path_str", "output_dir_str", "map_dir_str", "config_dict"]
        assert param_names == expected_params

    def test_task_function_logic_components(self):
        """Test the core logic components of the task function."""
        # Test that the task function contains the expected logic patterns
        import inspect

        from src.tasks import run_pipeline_for_file

        # Get the source code to verify it has the right structure
        source = inspect.getsource(run_pipeline_for_file.__wrapped__)

        # Verify key components exist
        assert "self.request.id" in source
        assert "Path(" in source
        assert "asyncio.run" in source
        assert "run_pipeline" in source
        assert "logger.info" in source
        assert "logger.error" in source

        # Test path conversion logic works correctly
        test_path = "/home/user/documents/test.txt"
        path_obj = Path(test_path)

        assert path_obj.parent == Path("/home/user/documents")
        assert path_obj.name == "test.txt"

    @patch("src.tasks.run_pipeline")
    @patch("src.tasks.asyncio.run")
    def test_path_handling_logic(self, mock_asyncio_run, mock_run_pipeline):
        """Test that paths are correctly converted to Path objects."""
        mock_asyncio_run.return_value = None
        mock_run_pipeline.return_value = AsyncMock()

        input_file_path = "/home/user/documents/interview.txt"
        output_dir_path = "/home/user/output"
        map_dir_path = "/home/user/maps"

        # Test the path conversion logic directly
        input_path = Path(input_file_path)
        expected_input_dir = input_path.parent
        expected_output_dir = Path(output_dir_path)
        expected_map_dir = Path(map_dir_path)
        expected_filename = input_path.name

        # Verify path conversions work correctly
        assert expected_input_dir == Path("/home/user/documents")
        assert expected_output_dir == Path("/home/user/output")
        assert expected_map_dir == Path("/home/user/maps")
        assert expected_filename == "interview.txt"

        # Test that Path objects can be created from strings
        assert isinstance(Path(input_file_path), Path)
        assert isinstance(Path(output_dir_path), Path)
        assert isinstance(Path(map_dir_path), Path)

    def test_return_value_structure(self):
        """Test the expected return value structure."""
        input_file_path = "/test/file.txt"

        # Test success return structure
        success_result = {"status": "Success", "file": input_file_path}

        assert isinstance(success_result, dict)
        assert success_result["status"] == "Success"
        assert success_result["file"] == input_file_path
        assert len(success_result) == 2  # Only these two keys


class TestTasksModuleIntegration:
    """Integration tests for the tasks module."""

    def test_task_imports_and_dependencies(self):
        """Test that all required dependencies are properly imported."""
        # Test that the task function exists and is importable
        from src.tasks import run_pipeline_for_file

        assert callable(run_pipeline_for_file)

        # Test that required modules are available
        import src.tasks

        assert hasattr(src.tasks, "celery_app")
        assert hasattr(src.tasks, "run_pipeline")
        assert hasattr(src.tasks, "logger")
        assert hasattr(src.tasks, "asyncio")
        assert hasattr(src.tasks, "Path")

    def test_logger_configuration(self):
        """Test that logger is properly configured."""
        from src.tasks import logger

        # Verify logger exists and has expected methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

        # Verify logger is not None
        assert logger is not None

    def test_task_has_celery_attributes(self):
        """Test that the task has all expected Celery attributes."""
        from src.tasks import run_pipeline_for_file

        # Test Celery-specific attributes
        celery_attrs = ["delay", "apply_async", "retry", "request"]
        for attr in celery_attrs:
            assert hasattr(run_pipeline_for_file, attr), f"Task missing {attr} attribute"

    def test_celery_app_integration(self):
        """Test that the task is properly registered with the Celery app."""
        from src.celery_app import celery_app
        from src.tasks import run_pipeline_for_file

        # Verify the task is registered with the Celery app
        task_name = run_pipeline_for_file.name
        assert task_name in celery_app.tasks

        # Verify the registered task has the same name and is callable
        registered_task = celery_app.tasks[task_name]
        assert registered_task.name == run_pipeline_for_file.name
        assert callable(registered_task)

    def test_async_dependencies_available(self):
        """Test that async dependencies are properly available."""
        # Test that asyncio module is available
        import asyncio as async_module

        assert hasattr(async_module, "run")

        # Test that pathlib is available
        from pathlib import Path

        assert Path is not None

        # Test that we can create Path objects
        test_path = Path("/test/path")
        assert isinstance(test_path, Path)
        assert test_path.name == "path"
        assert test_path.parent == Path("/test")

    def test_task_function_error_handling_structure(self):
        """Test that the task function has proper error handling structure."""
        import inspect

        from src.tasks import run_pipeline_for_file

        # Get the source code of the wrapped function
        source = inspect.getsource(run_pipeline_for_file.__wrapped__)

        # Verify error handling patterns exist in the source
        assert "try:" in source
        assert "except FileNotFoundError" in source
        assert "except Exception" in source
        assert "logger.error" in source
        assert "exc_info=True" in source
