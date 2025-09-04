"""
tests/pipeline/test_run_pipeline.py

Tests for the main run_pipeline function that orchestrates the entire pipeline.

These tests focus on the run_pipeline function's behavior as the main entry point,
testing real orchestration logic rather than mock interactions.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline import run_pipeline


class TestRunPipeline:
    """Test the main run_pipeline function."""

    @pytest.mark.asyncio
    async def test_run_pipeline_creates_and_executes_orchestrator(self, tmp_path, realistic_config):
        """Test that run_pipeline creates orchestrator and calls execute with correct parameters."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        map_dir = tmp_path / "maps"

        # Create directories
        for directory in [input_dir, output_dir, map_dir]:
            directory.mkdir(exist_ok=True)

        specific_file = "test.txt"
        task_id = "test-pipeline-run"

        # Patch the orchestrator to avoid full pipeline execution in unit test
        with patch("src.pipeline.PipelineOrchestrator", autospec=True) as MockOrchestrator:
            mock_instance = MockOrchestrator.return_value
            mock_instance.execute = AsyncMock()

            await run_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                map_dir=map_dir,
                specific_file=specific_file,
                config_dict=realistic_config,
                task_id=task_id,
            )

            # Verify orchestrator was created with correct parameters
            MockOrchestrator.assert_called_once_with(
                input_dir=input_dir,
                output_dir=output_dir,
                map_dir=map_dir,
                config_dict=realistic_config,
                task_id=task_id,
            )

            # Verify execute was called with specific file
            mock_instance.execute.assert_awaited_once_with(specific_file=specific_file)

    @pytest.mark.asyncio
    async def test_run_pipeline_handles_orchestrator_initialization_error(self, tmp_path, realistic_config):
        """Test that run_pipeline properly handles orchestrator initialization errors."""
        input_dir = tmp_path / "nonexistent"  # This will cause FileNotFoundError

        with patch("src.pipeline.logger") as mock_logger:
            # Should raise the initialization error
            with pytest.raises(FileNotFoundError):
                await run_pipeline(
                    input_dir=input_dir,
                    config_dict=realistic_config,
                    task_id="test-init-error",
                )

            # Verify error was logged
            mock_logger.critical.assert_called()
            critical_call_args = mock_logger.critical.call_args[0][0]
            assert "Pipeline setup failed" in critical_call_args

    @pytest.mark.asyncio
    async def test_run_pipeline_with_minimal_parameters(self, tmp_path):
        """Test run_pipeline with only required parameters."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)

        with patch("src.pipeline.PipelineOrchestrator", autospec=True) as MockOrchestrator:
            mock_instance = MockOrchestrator.return_value
            mock_instance.execute = AsyncMock()

            await run_pipeline(input_dir=input_dir)

            # Should use defaults for optional parameters
            MockOrchestrator.assert_called_once_with(
                input_dir=input_dir,
                output_dir=None,
                map_dir=None,
                config_dict=None,
                task_id=None,
            )

            mock_instance.execute.assert_awaited_once_with(specific_file=None)

    @pytest.mark.asyncio
    async def test_run_pipeline_logs_start_and_finish(self, tmp_path):
        """Test that run_pipeline logs pipeline start and finish messages."""
        input_dir = tmp_path / "input"
        input_dir.mkdir(exist_ok=True)
        task_id = "test-logging"

        with patch("src.pipeline.PipelineOrchestrator", autospec=True) as MockOrchestrator:
            mock_instance = MockOrchestrator.return_value
            mock_instance.execute = AsyncMock()

            with patch("src.pipeline.logger") as mock_logger:
                await run_pipeline(input_dir=input_dir, task_id=task_id)

                # Check that start and finish messages were logged
                info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

                start_logged = any("Starting Pipeline Run" in msg and task_id in msg for msg in info_calls)
                finish_logged = any("Pipeline Run Finished" in msg and task_id in msg for msg in info_calls)

                assert start_logged, f"Start message not found in: {info_calls}"
                assert finish_logged, f"Finish message not found in: {info_calls}"

    @pytest.mark.asyncio
    async def test_run_pipeline_with_string_paths(self, tmp_path):
        """Test that run_pipeline accepts string paths and converts them properly."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        map_dir = tmp_path / "maps"

        for directory in [input_dir, output_dir, map_dir]:
            directory.mkdir(exist_ok=True)

        with patch("src.pipeline.PipelineOrchestrator", autospec=True) as MockOrchestrator:
            mock_instance = MockOrchestrator.return_value
            mock_instance.execute = AsyncMock()

            # Pass string paths instead of Path objects
            await run_pipeline(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                map_dir=str(map_dir),
            )

            # Orchestrator should receive the paths (as strings or Path objects)
            call_args = MockOrchestrator.call_args[1]
            assert str(call_args["input_dir"]) == str(input_dir)
            assert str(call_args["output_dir"]) == str(output_dir)
            assert str(call_args["map_dir"]) == str(map_dir)
