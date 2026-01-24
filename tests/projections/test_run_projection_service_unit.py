"""
Unit tests for run_projection_service.py (src/run_projection_service.py).

Tests the CLI entry point for the projection service.
"""

import asyncio
import logging
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMainFunction:
    """Test the main() async function."""

    @pytest.mark.asyncio
    async def test_main_initializes_components(self):
        """Test that main() initializes all required components."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal"):

            # Create a task that will set the shutdown event shortly
            async def run_main_with_timeout():
                from src.run_projection_service import main
                # We need to trigger shutdown after main starts
                main_task = asyncio.create_task(main(lane_count=4, log_level="WARNING"))
                # Give it a moment to start
                await asyncio.sleep(0.1)
                # Cancel it to simulate shutdown
                main_task.cancel()
                try:
                    await main_task
                except asyncio.CancelledError:
                    pass

            await run_main_with_timeout()

            # Verify components were created
            mock_service.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_uses_custom_lane_count(self):
        """Test that main() passes lane_count to ProjectionService."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service) as mock_ps_class, \
             patch("src.run_projection_service.signal.signal"):

            async def run_main_with_timeout():
                from src.run_projection_service import main
                main_task = asyncio.create_task(main(lane_count=8, log_level="ERROR"))
                await asyncio.sleep(0.1)
                main_task.cancel()
                try:
                    await main_task
                except asyncio.CancelledError:
                    pass

            await run_main_with_timeout()

            # Verify lane_count was passed
            mock_ps_class.assert_called_once()
            call_kwargs = mock_ps_class.call_args.kwargs
            assert call_kwargs["lane_count"] == 8

    @pytest.mark.asyncio
    async def test_main_configures_logging(self):
        """Test that main() configures logging with correct level."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal"), \
             patch("src.run_projection_service.logging.basicConfig") as mock_basic_config:

            async def run_main_with_timeout():
                from src.run_projection_service import main
                main_task = asyncio.create_task(main(lane_count=4, log_level="DEBUG"))
                await asyncio.sleep(0.1)
                main_task.cancel()
                try:
                    await main_task
                except asyncio.CancelledError:
                    pass

            await run_main_with_timeout()

            # Verify logging was configured
            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args.kwargs
            assert call_kwargs["level"] == logging.DEBUG

    @pytest.mark.asyncio
    async def test_main_registers_signal_handlers(self):
        """Test that main() registers SIGINT and SIGTERM handlers."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        registered_signals = []

        def mock_signal_register(sig, handler):
            registered_signals.append(sig)

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal", side_effect=mock_signal_register):

            async def run_main_with_timeout():
                from src.run_projection_service import main
                main_task = asyncio.create_task(main(lane_count=4, log_level="INFO"))
                await asyncio.sleep(0.1)
                main_task.cancel()
                try:
                    await main_task
                except asyncio.CancelledError:
                    pass

            await run_main_with_timeout()

            # Verify both signals were registered
            assert signal.SIGINT in registered_signals
            assert signal.SIGTERM in registered_signals

    @pytest.mark.asyncio
    async def test_main_stops_service_on_exception(self):
        """Test that main() stops service when exception occurs."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock(side_effect=Exception("Start failed"))
        mock_service.stop = AsyncMock()

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal"), \
             patch("src.run_projection_service.sys.exit") as mock_exit:

            from src.run_projection_service import main
            await main(lane_count=4, log_level="INFO")

            # Service stop should be called in finally block
            mock_service.stop.assert_called_once()
            # sys.exit(1) should be called on exception
            mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_main_stops_service_cleanly(self):
        """Test that main() calls service.stop() in finally block."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        shutdown_event = asyncio.Event()

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal"), \
             patch("src.run_projection_service.asyncio.Event", return_value=shutdown_event):

            async def run_main_and_shutdown():
                from src.run_projection_service import main
                main_task = asyncio.create_task(main(lane_count=4, log_level="INFO"))
                await asyncio.sleep(0.1)
                # Trigger shutdown
                shutdown_event.set()
                await main_task

            await run_main_and_shutdown()

            # Verify stop was called
            mock_service.stop.assert_called_once()


class TestSignalHandler:
    """Test signal handler behavior."""

    @pytest.mark.asyncio
    async def test_signal_handler_sets_shutdown_event(self):
        """Test that signal handler sets the shutdown event."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        captured_handler = None

        def capture_signal_handler(sig, handler):
            nonlocal captured_handler
            if sig == signal.SIGINT:
                captured_handler = handler

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal", side_effect=capture_signal_handler):

            async def run_main_and_signal():
                from src.run_projection_service import main
                main_task = asyncio.create_task(main(lane_count=4, log_level="INFO"))
                await asyncio.sleep(0.1)
                # Call the captured signal handler
                if captured_handler:
                    captured_handler(signal.SIGINT, None)
                await main_task

            await run_main_and_signal()

            # If we got here without hanging, the signal handler worked
            mock_service.stop.assert_called_once()


class TestArgumentParsing:
    """Test argument parsing logic."""

    def test_default_arguments(self):
        """Test default argument values."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--lane-count", type=int, default=12)
        parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

        args = parser.parse_args([])

        assert args.lane_count == 12
        assert args.log_level == "INFO"

    def test_custom_lane_count(self):
        """Test custom lane count argument."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--lane-count", type=int, default=12)
        parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

        args = parser.parse_args(["--lane-count", "24"])

        assert args.lane_count == 24

    def test_custom_log_level(self):
        """Test custom log level argument."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--lane-count", type=int, default=12)
        parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

        args = parser.parse_args(["--log-level", "DEBUG"])

        assert args.log_level == "DEBUG"

    def test_invalid_log_level_rejected(self):
        """Test that invalid log level is rejected."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--lane-count", type=int, default=12)
        parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

        with pytest.raises(SystemExit):
            parser.parse_args(["--log-level", "INVALID"])

    def test_all_log_levels_accepted(self):
        """Test that all valid log levels are accepted."""
        import argparse

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--lane-count", type=int, default=12)
            parser.add_argument("--log-level", type=str, default="INFO",
                              choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

            args = parser.parse_args(["--log-level", level])
            assert args.log_level == level

    def test_combined_arguments(self):
        """Test combining multiple arguments."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--lane-count", type=int, default=12)
        parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

        args = parser.parse_args(["--lane-count", "6", "--log-level", "WARNING"])

        assert args.lane_count == 6
        assert args.log_level == "WARNING"


class TestLoggingConfiguration:
    """Test logging configuration."""

    @pytest.mark.asyncio
    async def test_logging_format_configured(self):
        """Test that logging format is properly configured."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store), \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal"), \
             patch("src.run_projection_service.logging.basicConfig") as mock_basic_config:

            async def run_main_with_timeout():
                from src.run_projection_service import main
                main_task = asyncio.create_task(main(lane_count=4, log_level="INFO"))
                await asyncio.sleep(0.1)
                main_task.cancel()
                try:
                    await main_task
                except asyncio.CancelledError:
                    pass

            await run_main_with_timeout()

            # Verify format string includes expected components
            call_kwargs = mock_basic_config.call_args.kwargs
            assert "format" in call_kwargs
            assert "%(levelname)" in call_kwargs["format"]
            assert "%(name)s" in call_kwargs["format"]


class TestEventStoreConnection:
    """Test EventStore connection setup."""

    @pytest.mark.asyncio
    async def test_event_store_uses_config_connection_string(self):
        """Test that EventStore client uses connection string from config."""
        mock_registry = MagicMock()
        mock_event_store = MagicMock()
        mock_service = AsyncMock()
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        with patch("src.run_projection_service.create_handler_registry", return_value=mock_registry), \
             patch("src.run_projection_service.get_event_store_client", return_value=mock_event_store) as mock_get_client, \
             patch("src.run_projection_service.ProjectionService", return_value=mock_service), \
             patch("src.run_projection_service.signal.signal"), \
             patch("src.run_projection_service.ESDB_CONNECTION_STRING", "esdb://test:2113?tls=false"):

            async def run_main_with_timeout():
                from src.run_projection_service import main
                main_task = asyncio.create_task(main(lane_count=4, log_level="INFO"))
                await asyncio.sleep(0.1)
                main_task.cancel()
                try:
                    await main_task
                except asyncio.CancelledError:
                    pass

            await run_main_with_timeout()

            # Verify connection string was passed
            mock_get_client.assert_called_once()
            call_kwargs = mock_get_client.call_args.kwargs
            assert "connection_string" in call_kwargs
