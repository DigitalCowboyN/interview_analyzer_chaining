"""
Unit tests for the projection subscription manager.

Tests the SubscriptionManager class which manages persistent subscriptions
to EventStoreDB category streams.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.projections.subscription_manager import SubscriptionManager


class TestSubscriptionManagerInit:
    """Tests for SubscriptionManager initialization."""

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_init_uses_provided_event_store(self, mock_get_client):
        """When event_store is provided, it should be used directly."""
        mock_event_store = MagicMock()

        manager = SubscriptionManager(event_store=mock_event_store)

        assert manager.event_store is mock_event_store
        mock_get_client.assert_not_called()

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_init_uses_global_event_store_when_not_provided(self, mock_get_client):
        """When event_store is not provided, global client should be used."""
        mock_global_client = MagicMock()
        mock_get_client.return_value = mock_global_client

        manager = SubscriptionManager()

        assert manager.event_store is mock_global_client
        mock_get_client.assert_called_once()

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_init_stores_lane_manager(self, mock_get_client):
        """Lane manager should be stored for event routing."""
        mock_get_client.return_value = MagicMock()
        mock_lane_manager = MagicMock()

        manager = SubscriptionManager(lane_manager=mock_lane_manager)

        assert manager.lane_manager is mock_lane_manager

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_init_starts_with_empty_subscriptions(self, mock_get_client):
        """Subscriptions dict should be empty initially."""
        mock_get_client.return_value = MagicMock()

        manager = SubscriptionManager()

        assert manager.subscriptions == {}

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_init_starts_not_running(self, mock_get_client):
        """is_running should be False initially."""
        mock_get_client.return_value = MagicMock()

        manager = SubscriptionManager()

        assert manager.is_running is False


class TestSubscriptionManagerStart:
    """Tests for SubscriptionManager.start method."""

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {})
    async def test_start_sets_is_running_true(self, mock_get_client):
        """start should set is_running to True."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()

        await manager.start()

        assert manager.is_running is True

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "test_sub": {"stream": "$ce-Test", "group": "test-group", "allowlist": ["TestEvent"]}
    })
    async def test_start_creates_task_for_each_subscription(self, mock_get_client):
        """start should create an asyncio task for each subscription in config."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        # Mock _run_subscription to avoid actual subscription logic
        manager._run_subscription = AsyncMock()

        await manager.start()

        assert "test_sub" in manager.subscriptions
        assert isinstance(manager.subscriptions["test_sub"], asyncio.Task)

        # Cleanup
        await manager.stop()

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "sub1": {"stream": "$ce-Test1", "group": "group1", "allowlist": []},
        "sub2": {"stream": "$ce-Test2", "group": "group2", "allowlist": []},
    })
    async def test_start_creates_tasks_for_all_subscriptions(self, mock_get_client):
        """start should create tasks for all configured subscriptions."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        manager._run_subscription = AsyncMock()

        await manager.start()

        assert len(manager.subscriptions) == 2
        assert "sub1" in manager.subscriptions
        assert "sub2" in manager.subscriptions

        # Cleanup
        await manager.stop()

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {})
    async def test_start_warns_if_already_running(self, mock_get_client, caplog):
        """start should log warning if already running."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()

        await manager.start()

        with caplog.at_level(logging.WARNING):
            await manager.start()

        assert "already running" in caplog.text

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {})
    async def test_start_does_not_create_duplicate_tasks(self, mock_get_client):
        """start should not create new tasks if already running."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()

        await manager.start()
        initial_subscriptions = dict(manager.subscriptions)

        await manager.start()  # Second call

        assert manager.subscriptions == initial_subscriptions


class TestSubscriptionManagerStop:
    """Tests for SubscriptionManager.stop method."""

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {})
    async def test_stop_sets_is_running_false(self, mock_get_client):
        """stop should set is_running to False."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        await manager.start()

        await manager.stop()

        assert manager.is_running is False

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "test_sub": {"stream": "$ce-Test", "group": "test-group", "allowlist": []}
    })
    async def test_stop_cancels_all_tasks(self, mock_get_client):
        """stop should cancel all subscription tasks."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        manager._run_subscription = AsyncMock()
        await manager.start()

        task = manager.subscriptions["test_sub"]

        await manager.stop()

        assert task.cancelled()

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "sub1": {"stream": "$ce-Test1", "group": "group1", "allowlist": []},
        "sub2": {"stream": "$ce-Test2", "group": "group2", "allowlist": []},
    })
    async def test_stop_clears_subscriptions_dict(self, mock_get_client):
        """stop should clear the subscriptions dictionary."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        manager._run_subscription = AsyncMock()
        await manager.start()

        assert len(manager.subscriptions) == 2

        await manager.stop()

        assert manager.subscriptions == {}

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_stop_does_nothing_if_not_running(self, mock_get_client):
        """stop should do nothing if manager is not running."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()

        # Should not raise, should be a no-op
        await manager.stop()

        assert manager.is_running is False
        assert manager.subscriptions == {}

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "test_sub": {"stream": "$ce-Test", "group": "test-group", "allowlist": []}
    })
    async def test_stop_logs_stopped_subscriptions(self, mock_get_client, caplog):
        """stop should log when each subscription is stopped."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        manager._run_subscription = AsyncMock()
        await manager.start()

        with caplog.at_level(logging.INFO):
            await manager.stop()

        assert "Stopped subscription 'test_sub'" in caplog.text


class TestSubscriptionManagerGetStatus:
    """Tests for SubscriptionManager.get_status method."""

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_get_status_returns_correct_structure(self, mock_get_client):
        """get_status should return dict with expected keys."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()

        status = manager.get_status()

        assert "is_running" in status
        assert "subscription_count" in status
        assert "subscriptions" in status

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_get_status_shows_is_running_false_initially(self, mock_get_client):
        """get_status should show is_running=False when not started."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()

        status = manager.get_status()

        assert status["is_running"] is False

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {})
    async def test_get_status_shows_is_running_true_after_start(self, mock_get_client):
        """get_status should show is_running=True after start."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        await manager.start()

        status = manager.get_status()

        assert status["is_running"] is True

    @patch("src.projections.subscription_manager.get_event_store_client")
    def test_get_status_shows_zero_subscriptions_initially(self, mock_get_client):
        """get_status should show subscription_count=0 initially."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()

        status = manager.get_status()

        assert status["subscription_count"] == 0

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "sub1": {"stream": "$ce-Test1", "group": "group1", "allowlist": []},
        "sub2": {"stream": "$ce-Test2", "group": "group2", "allowlist": []},
    })
    async def test_get_status_shows_correct_subscription_count(self, mock_get_client):
        """get_status should show correct subscription count after start."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        manager._run_subscription = AsyncMock()
        await manager.start()

        status = manager.get_status()

        assert status["subscription_count"] == 2

        # Cleanup
        await manager.stop()

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "test_sub": {"stream": "$ce-Test", "group": "test-group", "allowlist": []}
    })
    async def test_get_status_includes_individual_subscription_status(self, mock_get_client):
        """get_status should include status for each subscription."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        manager._run_subscription = AsyncMock()
        await manager.start()

        status = manager.get_status()

        assert "test_sub" in status["subscriptions"]
        assert "running" in status["subscriptions"]["test_sub"]
        assert "cancelled" in status["subscriptions"]["test_sub"]

        # Cleanup
        await manager.stop()

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    @patch("src.projections.subscription_manager.SUBSCRIPTION_CONFIG", {
        "test_sub": {"stream": "$ce-Test", "group": "test-group", "allowlist": []}
    })
    async def test_get_status_shows_subscription_running(self, mock_get_client):
        """get_status should show running=True for active tasks."""
        mock_get_client.return_value = MagicMock()
        manager = SubscriptionManager()
        # Create a task that doesn't complete immediately
        async def long_running(sub_name, config):
            await asyncio.sleep(100)
        manager._run_subscription = long_running
        await manager.start()

        status = manager.get_status()

        # Task should still be running (not done)
        assert status["subscriptions"]["test_sub"]["running"] is True

        # Cleanup
        await manager.stop()


class TestEnsureSubscriptionExists:
    """Tests for SubscriptionManager._ensure_subscription_exists method."""

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_ensure_subscription_exists_checks_existing(self, mock_get_client):
        """Should check if subscription exists via get_subscription_info."""
        mock_event_store = MagicMock()
        mock_client = MagicMock()

        # Create a proper async context manager mock
        async_context_manager = MagicMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        async_context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_event_store.get_client.return_value = async_context_manager

        manager = SubscriptionManager(event_store=mock_event_store)

        await manager._ensure_subscription_exists("$ce-Test", "test-group")

        mock_client.get_subscription_info.assert_called_once_with(
            group_name="test-group",
            stream_name="$ce-Test",
        )

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_ensure_subscription_exists_creates_if_not_found(self, mock_get_client):
        """Should create subscription if NotFound is raised."""
        from esdbclient.exceptions import NotFound

        mock_event_store = MagicMock()
        mock_client = MagicMock()
        mock_client.get_subscription_info.side_effect = NotFound()

        # Create a proper async context manager mock
        async_context_manager = MagicMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        async_context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_event_store.get_client.return_value = async_context_manager

        manager = SubscriptionManager(event_store=mock_event_store)

        await manager._ensure_subscription_exists("$ce-Test", "test-group")

        mock_client.create_subscription_to_stream.assert_called_once_with(
            group_name="test-group",
            stream_name="$ce-Test",
            from_end=False,
        )

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_ensure_subscription_exists_does_not_create_if_exists(self, mock_get_client):
        """Should not create subscription if it already exists."""
        mock_event_store = MagicMock()
        mock_client = MagicMock()
        # get_subscription_info returns normally (subscription exists)
        mock_client.get_subscription_info.return_value = {"some": "info"}

        # Create a proper async context manager mock
        async_context_manager = MagicMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        async_context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_event_store.get_client.return_value = async_context_manager

        manager = SubscriptionManager(event_store=mock_event_store)

        await manager._ensure_subscription_exists("$ce-Test", "test-group")

        mock_client.create_subscription_to_stream.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_ensure_subscription_exists_raises_on_other_errors(self, mock_get_client):
        """Should raise if an error other than NotFound occurs."""
        mock_event_store = MagicMock()
        mock_client = MagicMock()
        mock_client.get_subscription_info.side_effect = RuntimeError("Connection failed")

        # Create a proper async context manager mock
        async_context_manager = MagicMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        async_context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_event_store.get_client.return_value = async_context_manager

        manager = SubscriptionManager(event_store=mock_event_store)

        with pytest.raises(RuntimeError, match="Connection failed"):
            await manager._ensure_subscription_exists("$ce-Test", "test-group")


class TestRunSubscription:
    """Tests for SubscriptionManager._run_subscription method.

    Note: Full loop behavior tests are complex due to the infinite while loop.
    These tests focus on error handling and cancellation scenarios that
    naturally break out of the loop.
    """

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_run_subscription_logs_connection_errors(self, mock_get_client, caplog):
        """Connection errors should be logged and trigger reconnection attempt."""
        mock_event_store = MagicMock()
        # Create a proper async context manager mock that raises on enter
        async_context_manager = MagicMock()
        async_context_manager.__aenter__ = AsyncMock(side_effect=RuntimeError("Connection lost"))
        async_context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_event_store.get_client.return_value = async_context_manager

        manager = SubscriptionManager(event_store=mock_event_store)
        manager._ensure_subscription_exists = AsyncMock()
        manager.is_running = True

        # Stop immediately after error is logged
        original_sleep = asyncio.sleep
        call_count = [0]
        async def patched_sleep(duration):
            call_count[0] += 1
            if call_count[0] >= 1:
                manager.is_running = False
            return await original_sleep(0)  # Don't actually sleep

        with patch("asyncio.sleep", patched_sleep):
            with caplog.at_level(logging.ERROR):
                config = {"stream": "$ce-Test", "group": "test-group"}
                await manager._run_subscription("test_sub", config)

        assert "Error in subscription" in caplog.text
        assert "Connection lost" in caplog.text

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_run_subscription_stops_on_cancelled_error(self, mock_get_client, caplog):
        """CancelledError should stop the subscription cleanly."""
        mock_event_store = MagicMock()
        # Create a proper async context manager mock that raises CancelledError
        async_context_manager = MagicMock()
        async_context_manager.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError())
        async_context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_event_store.get_client.return_value = async_context_manager

        manager = SubscriptionManager(event_store=mock_event_store)
        manager._ensure_subscription_exists = AsyncMock()
        manager.is_running = True

        with caplog.at_level(logging.INFO):
            config = {"stream": "$ce-Test", "group": "test-group"}
            await manager._run_subscription("test_sub", config)

        assert "cancelled" in caplog.text

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_run_subscription_calls_ensure_subscription_exists(self, mock_get_client):
        """_run_subscription should ensure subscription exists before connecting."""
        mock_event_store = MagicMock()
        # Raise CancelledError to break out of the loop cleanly
        async_context_manager = MagicMock()
        async_context_manager.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError())
        async_context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_event_store.get_client.return_value = async_context_manager

        manager = SubscriptionManager(event_store=mock_event_store)
        manager._ensure_subscription_exists = AsyncMock()
        manager.is_running = True

        config = {"stream": "$ce-Test", "group": "test-group"}
        await manager._run_subscription("test_sub", config)

        manager._ensure_subscription_exists.assert_called_with("$ce-Test", "test-group")

    @pytest.mark.asyncio
    @patch("src.projections.subscription_manager.get_event_store_client")
    async def test_run_subscription_stops_when_is_running_false(self, mock_get_client):
        """_run_subscription should exit immediately if is_running is False."""
        mock_event_store = MagicMock()
        manager = SubscriptionManager(event_store=mock_event_store)
        manager._ensure_subscription_exists = AsyncMock()
        manager.is_running = False  # Already stopped

        config = {"stream": "$ce-Test", "group": "test-group"}
        # Should return immediately without doing anything
        await manager._run_subscription("test_sub", config)

        # Should not have tried to ensure subscription exists
        manager._ensure_subscription_exists.assert_not_called()
