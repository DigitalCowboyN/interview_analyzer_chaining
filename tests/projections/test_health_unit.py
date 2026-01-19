"""
Unit tests for the projection health module.

Tests the get_health_status and format_health_status functions.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.projections.health import get_health_status, format_health_status


class TestGetHealthStatus:
    """Tests for get_health_status async function."""

    @pytest.mark.asyncio
    async def test_get_health_status_returns_service_health_status(self):
        """get_health_status should return the projection service's health status."""
        mock_service = MagicMock()
        expected_status = {
            "status": "healthy",
            "is_running": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mock_service.get_health_status.return_value = expected_status

        with patch("src.projections.health.get_projection_service", return_value=mock_service):
            result = await get_health_status()

        assert result == expected_status
        mock_service.get_health_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_health_status_returns_error_status_on_exception(self):
        """get_health_status should return error status when service raises exception."""
        with patch(
            "src.projections.health.get_projection_service",
            side_effect=RuntimeError("Service not initialized"),
        ):
            result = await get_health_status()

        assert result["status"] == "error"
        assert "Service not initialized" in result["error"]
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_get_health_status_logs_error_on_exception(self, caplog):
        """get_health_status should log the error when an exception occurs."""
        with patch(
            "src.projections.health.get_projection_service",
            side_effect=ValueError("Test error message"),
        ):
            import logging

            with caplog.at_level(logging.ERROR):
                await get_health_status()

        assert "Failed to get health status" in caplog.text
        assert "Test error message" in caplog.text

    @pytest.mark.asyncio
    async def test_get_health_status_error_includes_iso_timestamp(self):
        """Error status should include properly formatted ISO timestamp."""
        with patch(
            "src.projections.health.get_projection_service",
            side_effect=Exception("Test"),
        ):
            result = await get_health_status()

        # Verify timestamp is valid ISO format
        timestamp = result["timestamp"]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed is not None


class TestFormatHealthStatusBasic:
    """Tests for format_health_status basic fields."""

    def test_format_health_status_shows_status_uppercase(self):
        """Status should be displayed in uppercase."""
        status = {"status": "healthy"}

        result = format_health_status(status)

        assert "Status: HEALTHY" in result

    def test_format_health_status_shows_unknown_for_missing_status(self):
        """Missing status should display as UNKNOWN."""
        status = {}

        result = format_health_status(status)

        assert "Status: UNKNOWN" in result

    def test_format_health_status_shows_running_true(self):
        """Should show Running: True when service is running."""
        status = {"status": "healthy", "is_running": True}

        result = format_health_status(status)

        assert "Running: True" in result

    def test_format_health_status_shows_running_false(self):
        """Should show Running: False when service is not running."""
        status = {"status": "stopped", "is_running": False}

        result = format_health_status(status)

        assert "Running: False" in result

    def test_format_health_status_shows_running_false_when_missing(self):
        """Missing is_running should default to False."""
        status = {"status": "unknown"}

        result = format_health_status(status)

        assert "Running: False" in result

    def test_format_health_status_shows_started_at(self):
        """Should show started_at when present."""
        started = "2026-01-18T10:30:00+00:00"
        status = {"status": "healthy", "started_at": started}

        result = format_health_status(status)

        assert f"Started: {started}" in result

    def test_format_health_status_omits_started_at_when_missing(self):
        """Should not show Started line when started_at is missing."""
        status = {"status": "healthy"}

        result = format_health_status(status)

        assert "Started:" not in result


class TestFormatHealthStatusUptime:
    """Tests for format_health_status uptime formatting."""

    def test_format_health_status_formats_uptime_hours_minutes_seconds(self):
        """Uptime should be formatted as hours, minutes, seconds."""
        status = {"status": "healthy", "uptime_seconds": 3661}  # 1h 1m 1s

        result = format_health_status(status)

        assert "Uptime: 1h 1m 1s" in result

    def test_format_health_status_omits_uptime_when_zero(self):
        """Zero uptime should be omitted (0 is falsy)."""
        status = {"status": "healthy", "uptime_seconds": 0}

        result = format_health_status(status)

        # 0 is falsy in Python, so uptime won't be shown
        assert "Uptime:" not in result

    def test_format_health_status_formats_uptime_large_value(self):
        """Large uptime values should format correctly."""
        status = {"status": "healthy", "uptime_seconds": 90061}  # 25h 1m 1s

        result = format_health_status(status)

        assert "Uptime: 25h 1m 1s" in result

    def test_format_health_status_omits_uptime_when_missing(self):
        """Should not show Uptime line when uptime_seconds is missing."""
        status = {"status": "healthy"}

        result = format_health_status(status)

        assert "Uptime:" not in result

    def test_format_health_status_omits_uptime_when_falsy(self):
        """Should not show Uptime line when uptime_seconds is falsy (0 is truthy enough)."""
        # Note: 0 is falsy in Python, so uptime of 0 won't show
        status = {"status": "healthy", "uptime_seconds": None}

        result = format_health_status(status)

        assert "Uptime:" not in result


class TestFormatHealthStatusLanes:
    """Tests for format_health_status lane information."""

    def test_format_health_status_shows_lane_count(self):
        """Should show total lane count."""
        status = {
            "status": "healthy",
            "lanes": {"lane_count": 4, "total_events_processed": 0, "total_events_failed": 0},
        }

        result = format_health_status(status)

        assert "Lanes: 4" in result

    def test_format_health_status_shows_total_events_processed(self):
        """Should show total events processed across all lanes."""
        status = {
            "status": "healthy",
            "lanes": {
                "lane_count": 2,
                "total_events_processed": 1500,
                "total_events_failed": 0,
            },
        }

        result = format_health_status(status)

        assert "Total Events Processed: 1500" in result

    def test_format_health_status_shows_total_events_failed(self):
        """Should show total events failed across all lanes."""
        status = {
            "status": "healthy",
            "lanes": {
                "lane_count": 2,
                "total_events_processed": 1500,
                "total_events_failed": 25,
            },
        }

        result = format_health_status(status)

        assert "Total Events Failed: 25" in result

    def test_format_health_status_shows_individual_lane_status(self):
        """Should show status for each individual lane."""
        status = {
            "status": "healthy",
            "lanes": {
                "lane_count": 2,
                "total_events_processed": 200,
                "total_events_failed": 5,
                "lanes": [
                    {"id": 0, "queue_depth": 10, "events_processed": 100, "events_failed": 2},
                    {"id": 1, "queue_depth": 5, "events_processed": 100, "events_failed": 3},
                ],
            },
        }

        result = format_health_status(status)

        assert "Lane 0: Queue=10, Processed=100, Failed=2" in result
        assert "Lane 1: Queue=5, Processed=100, Failed=3" in result

    def test_format_health_status_omits_lanes_when_missing(self):
        """Should not show Lanes section when lanes is missing."""
        status = {"status": "healthy"}

        result = format_health_status(status)

        assert "Lanes:" not in result

    def test_format_health_status_omits_lanes_when_empty(self):
        """Should not show Lanes section when lanes is empty dict."""
        status = {"status": "healthy", "lanes": {}}

        result = format_health_status(status)

        assert "Lanes:" not in result


class TestFormatHealthStatusSubscriptions:
    """Tests for format_health_status subscription information."""

    def test_format_health_status_shows_subscription_count(self):
        """Should show total subscription count."""
        status = {
            "status": "healthy",
            "subscriptions": {"subscription_count": 2, "subscriptions": {}},
        }

        result = format_health_status(status)

        assert "Subscriptions: 2" in result

    def test_format_health_status_shows_running_subscription(self):
        """Should show Running for active subscriptions."""
        status = {
            "status": "healthy",
            "subscriptions": {
                "subscription_count": 1,
                "subscriptions": {"main-subscription": {"running": True}},
            },
        }

        result = format_health_status(status)

        assert "main-subscription: Running" in result

    def test_format_health_status_shows_stopped_subscription(self):
        """Should show Stopped for inactive subscriptions."""
        status = {
            "status": "healthy",
            "subscriptions": {
                "subscription_count": 1,
                "subscriptions": {"backup-subscription": {"running": False}},
            },
        }

        result = format_health_status(status)

        assert "backup-subscription: Stopped" in result

    def test_format_health_status_shows_multiple_subscriptions(self):
        """Should show all subscriptions with their status."""
        status = {
            "status": "healthy",
            "subscriptions": {
                "subscription_count": 2,
                "subscriptions": {
                    "primary": {"running": True},
                    "secondary": {"running": False},
                },
            },
        }

        result = format_health_status(status)

        assert "primary: Running" in result
        assert "secondary: Stopped" in result

    def test_format_health_status_omits_subscriptions_when_missing(self):
        """Should not show Subscriptions section when subscriptions is missing."""
        status = {"status": "healthy"}

        result = format_health_status(status)

        assert "Subscriptions:" not in result


class TestFormatHealthStatusParkedEvents:
    """Tests for format_health_status parked events information."""

    def test_format_health_status_shows_parked_events_header(self):
        """Should show Parked Events header when parked events exist."""
        status = {"status": "healthy", "parked_events": {"Interview": 5}}

        result = format_health_status(status)

        assert "Parked Events:" in result

    def test_format_health_status_shows_parked_events_by_aggregate_type(self):
        """Should show count of parked events per aggregate type."""
        status = {
            "status": "healthy",
            "parked_events": {"Interview": 5, "Sentence": 10},
        }

        result = format_health_status(status)

        assert "Interview: 5" in result
        assert "Sentence: 10" in result

    def test_format_health_status_omits_parked_events_when_missing(self):
        """Should not show Parked Events section when parked_events is missing."""
        status = {"status": "healthy"}

        result = format_health_status(status)

        assert "Parked Events:" not in result

    def test_format_health_status_omits_parked_events_when_empty(self):
        """Should not show Parked Events section when parked_events is empty."""
        status = {"status": "healthy", "parked_events": {}}

        result = format_health_status(status)

        assert "Parked Events:" not in result


class TestFormatHealthStatusComplete:
    """Tests for format_health_status with complete status data."""

    def test_format_health_status_complete_status(self):
        """Should format a complete status with all sections."""
        status = {
            "status": "healthy",
            "is_running": True,
            "started_at": "2026-01-18T10:00:00+00:00",
            "uptime_seconds": 7265,  # 2h 1m 5s
            "lanes": {
                "lane_count": 2,
                "total_events_processed": 500,
                "total_events_failed": 3,
                "lanes": [
                    {"id": 0, "queue_depth": 0, "events_processed": 250, "events_failed": 1},
                    {"id": 1, "queue_depth": 2, "events_processed": 250, "events_failed": 2},
                ],
            },
            "subscriptions": {
                "subscription_count": 1,
                "subscriptions": {"main": {"running": True}},
            },
            "parked_events": {"Interview": 2, "Sentence": 1},
        }

        result = format_health_status(status)

        # Verify all sections are present
        assert "Status: HEALTHY" in result
        assert "Running: True" in result
        assert "Started: 2026-01-18T10:00:00+00:00" in result
        assert "Uptime: 2h 1m 5s" in result
        assert "Lanes: 2" in result
        assert "Total Events Processed: 500" in result
        assert "Total Events Failed: 3" in result
        assert "Lane 0:" in result
        assert "Lane 1:" in result
        assert "Subscriptions: 1" in result
        assert "main: Running" in result
        assert "Parked Events:" in result
        assert "Interview: 2" in result
        assert "Sentence: 1" in result

    def test_format_health_status_error_status(self):
        """Should format an error status properly."""
        status = {
            "status": "error",
            "error": "Connection failed",
            "timestamp": "2026-01-18T10:30:00+00:00",
        }

        result = format_health_status(status)

        assert "Status: ERROR" in result
        assert "Running: False" in result
