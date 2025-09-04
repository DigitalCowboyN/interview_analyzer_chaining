"""
Tests for the Celery application configuration in src/celery_app.py.

These tests verify that the Celery app is properly configured with correct
broker settings, serialization, and task discovery.
"""

import os
from unittest.mock import patch

from celery import Celery

from src.celery_app import celery_app


class TestCeleryAppConfiguration:
    """Test the Celery application configuration and initialization."""

    def test_celery_app_instance_creation(self):
        """Test that celery_app is a valid Celery instance."""
        assert isinstance(celery_app, Celery)
        assert celery_app.main == "interview_analyzer"

    def test_default_broker_url_from_environment(self):
        """Test that broker URL defaults to Redis when env var not set."""
        # Test the default value when CELERY_BROKER_URL is not set
        with patch.dict(os.environ, {}, clear=True):
            # Import here to get fresh environment
            # Re-import to test the environment variable logic
            import importlib

            from src import celery_app as celery_module

            importlib.reload(celery_module)

            expected_default = "redis://redis:6379/0"
            assert celery_module.redis_url == expected_default

    def test_custom_broker_url_from_environment(self):
        """Test that broker URL is read from CELERY_BROKER_URL environment variable."""
        custom_broker_url = "redis://custom-redis:6380/1"

        with patch.dict(os.environ, {"CELERY_BROKER_URL": custom_broker_url}):
            # Re-import to test the environment variable logic
            import importlib

            from src import celery_app as celery_module

            importlib.reload(celery_module)

            assert celery_module.redis_url == custom_broker_url

    def test_celery_broker_configuration(self):
        """Test that the Celery app is configured with the correct broker."""
        # The broker should be set to the redis_url
        broker_url = celery_app.conf.broker_url
        assert broker_url is not None
        assert "redis://" in broker_url

    def test_celery_backend_configuration(self):
        """Test that the Celery app is configured with the correct result backend."""
        # The backend should be set to the redis_url
        result_backend = celery_app.conf.result_backend
        assert result_backend is not None
        assert "redis://" in result_backend

    def test_celery_task_includes(self):
        """Test that the Celery app includes the correct task modules."""
        includes = celery_app.conf.include
        assert "src.tasks" in includes

    def test_celery_serialization_configuration(self):
        """Test that Celery is configured with correct serialization settings."""
        conf = celery_app.conf

        # Test serialization settings
        assert conf.task_serializer == "json"
        assert conf.result_serializer == "json"
        assert "json" in conf.accept_content

    def test_celery_timezone_configuration(self):
        """Test that Celery is configured with correct timezone settings."""
        conf = celery_app.conf

        assert conf.timezone == "UTC"
        assert conf.enable_utc is True

    def test_celery_result_expiration_configuration(self):
        """Test that Celery is configured with correct result expiration."""
        conf = celery_app.conf

        assert conf.result_expires == 3600  # 1 hour

    def test_celery_app_name_consistency(self):
        """Test that the Celery app name matches the project name."""
        assert celery_app.main == "interview_analyzer"

    def test_celery_app_has_start_method(self):
        """Test that the celery app has a start method for direct execution."""
        # This tests that the celery_app instance has the start method
        # which is called in the if __name__ == "__main__" block
        assert hasattr(celery_app, "start")
        assert callable(celery_app.start)

        # Test that we can call start method (though we won't actually start it)
        # This verifies the method exists and is properly configured
        assert celery_app.start is not None

    def test_celery_configuration_update(self):
        """Test that the celery app configuration is properly updated."""
        # Verify that conf.update was called with our settings
        conf = celery_app.conf

        # Check all the configuration values that should have been set
        expected_config = {
            "result_expires": 3600,
            "task_serializer": "json",
            "accept_content": ["json"],
            "result_serializer": "json",
            "timezone": "UTC",
            "enable_utc": True,
        }

        for key, expected_value in expected_config.items():
            actual_value = getattr(conf, key)
            assert actual_value == expected_value, f"Config {key} should be {expected_value}, got {actual_value}"


class TestCeleryAppIntegration:
    """Integration tests for Celery app functionality."""

    def test_celery_app_can_discover_tasks(self):
        """Test that the Celery app can discover tasks from included modules."""
        # This tests that the include configuration works
        # The tasks should be discoverable (even if we can't import the actual tasks module)

        includes = celery_app.conf.include
        assert len(includes) > 0
        assert "src.tasks" in includes

    def test_celery_app_configuration_is_complete(self):
        """Test that all required Celery configuration is set."""
        conf = celery_app.conf

        # Check that essential configuration is not None/empty
        assert conf.broker_url is not None
        assert conf.result_backend is not None
        assert conf.task_serializer is not None
        assert conf.result_serializer is not None
        assert conf.timezone is not None

        # Check that include list is not empty
        assert len(conf.include) > 0

    def test_celery_app_ready_for_worker_startup(self):
        """Test that the Celery app is properly configured for worker startup."""
        # Verify the app has all necessary attributes for a worker to start
        assert hasattr(celery_app, "conf")
        assert hasattr(celery_app, "main")
        assert hasattr(celery_app, "start")

        # Verify broker connection can be established (configuration-wise)
        broker_url = celery_app.conf.broker_url
        assert broker_url.startswith("redis://")

        # Verify the broker URL has the expected format
        assert "://" in broker_url
        assert broker_url.count(":") >= 2  # redis://host:port format
