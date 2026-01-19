"""
Unit tests for the projection service orchestrator.

Tests the ProjectionService class and get_projection_service singleton.
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.projections.projection_service import (
    ProjectionService,
    get_projection_service,
)


class TestProjectionServiceInit:
    """Tests for ProjectionService initialization."""

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_init_uses_provided_event_store(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """When event_store is provided, it should be used directly."""
        mock_event_store = MagicMock()

        service = ProjectionService(event_store=mock_event_store)

        assert service.event_store is mock_event_store
        mock_get_client.assert_not_called()

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_init_uses_global_event_store_when_not_provided(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """When event_store is not provided, global client should be used."""
        mock_global_client = MagicMock()
        mock_get_client.return_value = mock_global_client

        service = ProjectionService()

        assert service.event_store is mock_global_client
        mock_get_client.assert_called_once()

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_init_creates_lane_manager_with_registry(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """LaneManager should be created with provided handler_registry."""
        mock_get_client.return_value = MagicMock()
        mock_registry = MagicMock()

        ProjectionService(handler_registry=mock_registry)

        mock_lane_mgr.assert_called_once()
        call_kwargs = mock_lane_mgr.call_args[1]
        assert call_kwargs["handler_registry"] is mock_registry

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_init_creates_lane_manager_with_lane_count(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """LaneManager should be created with specified lane_count."""
        mock_get_client.return_value = MagicMock()

        ProjectionService(lane_count=8)

        mock_lane_mgr.assert_called_once()
        call_kwargs = mock_lane_mgr.call_args[1]
        assert call_kwargs["lane_count"] == 8

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_init_creates_lane_manager_with_default_lane_count(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """LaneManager should use default lane_count of 12 if not specified."""
        mock_get_client.return_value = MagicMock()

        ProjectionService()

        mock_lane_mgr.assert_called_once()
        call_kwargs = mock_lane_mgr.call_args[1]
        assert call_kwargs["lane_count"] == 12

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_init_creates_subscription_manager(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """SubscriptionManager should be created with event_store and lane_manager."""
        mock_event_store = MagicMock()
        mock_lane_instance = MagicMock()
        mock_lane_mgr.return_value = mock_lane_instance

        ProjectionService(event_store=mock_event_store)

        mock_sub_mgr.assert_called_once()
        call_kwargs = mock_sub_mgr.call_args[1]
        assert call_kwargs["event_store"] is mock_event_store
        assert call_kwargs["lane_manager"] is mock_lane_instance

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_init_starts_not_running(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """Service should start with is_running=False."""
        mock_get_client.return_value = MagicMock()

        service = ProjectionService()

        assert service.is_running is False
        assert service.started_at is None


class TestProjectionServiceStart:
    """Tests for ProjectionService.start method."""

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_start_sets_is_running_true(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """start should set is_running to True."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()

        service = ProjectionService()
        await service.start()

        assert service.is_running is True

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_start_sets_started_at_timestamp(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """start should set started_at timestamp."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()

        service = ProjectionService()
        before = datetime.now(timezone.utc)
        await service.start()
        after = datetime.now(timezone.utc)

        assert service.started_at is not None
        assert before <= service.started_at <= after

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_start_starts_lane_manager_before_subscription_manager(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """Lanes should be started before subscriptions."""
        mock_get_client.return_value = MagicMock()
        call_order = []

        async def lane_start():
            call_order.append("lane_manager")

        async def sub_start():
            call_order.append("subscription_manager")

        mock_lane_mgr.return_value.start = lane_start
        mock_sub_mgr.return_value.start = sub_start

        service = ProjectionService()
        await service.start()

        assert call_order == ["lane_manager", "subscription_manager"]

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_start_warns_if_already_running(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client, caplog
    ):
        """start should log warning if already running."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()

        service = ProjectionService()
        await service.start()

        with caplog.at_level(logging.WARNING):
            await service.start()

        assert "already running" in caplog.text

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", False)
    async def test_start_does_nothing_when_disabled(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client, caplog
    ):
        """start should not start anything when service is disabled."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()

        service = ProjectionService()

        with caplog.at_level(logging.WARNING):
            await service.start()

        assert "disabled" in caplog.text
        assert service.is_running is False
        mock_lane_mgr.return_value.start.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_start_stops_on_error(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """start should stop and re-raise if an error occurs."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock(side_effect=RuntimeError("Lane start failed"))
        mock_sub_mgr.return_value.stop = AsyncMock()
        mock_lane_mgr.return_value.stop = AsyncMock()

        service = ProjectionService()

        with pytest.raises(RuntimeError, match="Lane start failed"):
            await service.start()

        assert service.is_running is False


class TestProjectionServiceStop:
    """Tests for ProjectionService.stop method."""

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_stop_sets_is_running_false(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """stop should set is_running to False."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock()
        mock_lane_mgr.return_value.stop = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.stop = AsyncMock()

        service = ProjectionService()
        await service.start()
        await service.stop()

        assert service.is_running is False

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_stop_stops_subscriptions_before_lanes(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """Subscriptions should be stopped before lanes."""
        mock_get_client.return_value = MagicMock()
        call_order = []

        mock_lane_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()

        async def sub_stop():
            call_order.append("subscription_manager")

        async def lane_stop():
            call_order.append("lane_manager")

        mock_sub_mgr.return_value.stop = sub_stop
        mock_lane_mgr.return_value.stop = lane_stop

        service = ProjectionService()
        await service.start()
        await service.stop()

        assert call_order == ["subscription_manager", "lane_manager"]

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    async def test_stop_does_nothing_if_not_running(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """stop should do nothing if service is not running."""
        mock_get_client.return_value = MagicMock()
        mock_sub_mgr.return_value.stop = AsyncMock()
        mock_lane_mgr.return_value.stop = AsyncMock()

        service = ProjectionService()

        await service.stop()  # Should not raise

        mock_sub_mgr.return_value.stop.assert_not_called()
        mock_lane_mgr.return_value.stop.assert_not_called()


class TestProjectionServiceGetHealthStatus:
    """Tests for ProjectionService.get_health_status method."""

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_health_status_returns_correct_structure(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_health_status should return dict with expected keys."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.get_status.return_value = {"lanes": []}
        mock_sub_mgr.return_value.get_status.return_value = {"subscriptions": {}}

        service = ProjectionService()
        status = service.get_health_status()

        assert "status" in status
        assert "is_running" in status
        assert "started_at" in status
        assert "uptime_seconds" in status
        assert "lanes" in status
        assert "subscriptions" in status
        assert "parked_events" in status

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_health_status_shows_unhealthy_when_not_running(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_health_status should show unhealthy when service is not running."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.get_status.return_value = {"lanes": []}
        mock_sub_mgr.return_value.get_status.return_value = {"subscriptions": {}}

        service = ProjectionService()
        status = service.get_health_status()

        assert status["status"] == "unhealthy"
        assert status["is_running"] is False

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_get_health_status_shows_healthy_when_running(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_health_status should show healthy when all components are running."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()
        mock_lane_mgr.return_value.get_status.return_value = {
            "lanes": [{"is_running": True}]
        }
        mock_sub_mgr.return_value.get_status.return_value = {
            "subscriptions": {"main": {"running": True}}
        }

        service = ProjectionService()
        await service.start()
        status = service.get_health_status()

        assert status["status"] == "healthy"
        assert status["is_running"] is True

    @pytest.mark.asyncio
    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    @patch("src.projections.projection_service.ENABLE_PROJECTION_SERVICE", True)
    async def test_get_health_status_calculates_uptime(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_health_status should calculate uptime when running."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.start = AsyncMock()
        mock_sub_mgr.return_value.start = AsyncMock()
        mock_lane_mgr.return_value.get_status.return_value = {"lanes": []}
        mock_sub_mgr.return_value.get_status.return_value = {"subscriptions": {}}

        service = ProjectionService()
        await service.start()
        await asyncio.sleep(0.01)  # Small delay
        status = service.get_health_status()

        assert status["uptime_seconds"] is not None
        assert status["uptime_seconds"] >= 0

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_health_status_includes_lane_status(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_health_status should include lane manager status."""
        mock_get_client.return_value = MagicMock()
        lane_status = {
            "lane_count": 2,
            "lanes": [
                {"id": 0, "is_running": True},
                {"id": 1, "is_running": True},
            ],
        }
        mock_lane_mgr.return_value.get_status.return_value = lane_status
        mock_sub_mgr.return_value.get_status.return_value = {"subscriptions": {}}

        service = ProjectionService()
        status = service.get_health_status()

        assert status["lanes"] == lane_status

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_health_status_includes_subscription_status(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_health_status should include subscription manager status."""
        mock_get_client.return_value = MagicMock()
        mock_lane_mgr.return_value.get_status.return_value = {"lanes": []}
        sub_status = {"subscription_count": 2, "subscriptions": {"main": {"running": True}}}
        mock_sub_mgr.return_value.get_status.return_value = sub_status

        service = ProjectionService()
        status = service.get_health_status()

        assert status["subscriptions"] == sub_status


class TestGetProjectionService:
    """Tests for the global projection service singleton accessor."""

    def setup_method(self):
        """Reset global state before each test."""
        import src.projections.projection_service as service_module
        service_module._projection_service = None

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_projection_service_returns_projection_service_instance(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_projection_service should return a ProjectionService instance."""
        mock_get_client.return_value = MagicMock()

        service = get_projection_service()

        assert isinstance(service, ProjectionService)

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_projection_service_returns_same_instance_on_multiple_calls(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """get_projection_service should return the same singleton instance."""
        mock_get_client.return_value = MagicMock()

        service1 = get_projection_service()
        service2 = get_projection_service()

        assert service1 is service2

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_projection_service_passes_handler_registry_on_first_call(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """handler_registry should be passed on first call only."""
        mock_get_client.return_value = MagicMock()
        mock_registry = MagicMock()

        service = get_projection_service(handler_registry=mock_registry)

        assert service.handler_registry is mock_registry

    @patch("src.projections.projection_service.get_event_store_client")
    @patch("src.projections.projection_service.LaneManager")
    @patch("src.projections.projection_service.SubscriptionManager")
    @patch("src.projections.projection_service.ParkedEventsManager")
    def test_get_projection_service_ignores_params_on_subsequent_calls(
        self, mock_parked_mgr, mock_sub_mgr, mock_lane_mgr, mock_get_client
    ):
        """Parameters on subsequent calls should be ignored."""
        mock_get_client.return_value = MagicMock()
        mock_registry1 = MagicMock()
        mock_registry2 = MagicMock()

        service1 = get_projection_service(handler_registry=mock_registry1)
        service2 = get_projection_service(handler_registry=mock_registry2)

        # Second call's registry should be ignored
        assert service1 is service2
        assert service2.handler_registry is mock_registry1
