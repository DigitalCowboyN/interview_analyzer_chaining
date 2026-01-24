"""
Unit tests for src/utils/environment.py.

Tests environment detection, connection configuration, and service availability checks.
"""

import os
import socket
from unittest.mock import MagicMock, patch, mock_open

import pytest
import requests

from src.utils.environment import (
    detect_environment,
    get_neo4j_connection_config,
    get_neo4j_test_connection_config,
    is_neo4j_available,
    get_available_neo4j_config,
    is_service_externally_managed,
    check_neo4j_test_ready,
    is_eventstore_externally_managed,
    check_eventstore_ready,
)


class TestDetectEnvironment:
    """Test environment detection logic."""

    def test_detects_ci_from_CI_variable(self):
        """Test that CI env var is detected."""
        with patch.dict(os.environ, {"CI": "true"}, clear=False):
            with patch("os.path.exists", return_value=False):
                assert detect_environment() == "ci"

    def test_detects_ci_from_CONTINUOUS_INTEGRATION(self):
        """Test CONTINUOUS_INTEGRATION env var is detected."""
        with patch.dict(os.environ, {"CONTINUOUS_INTEGRATION": "true"}, clear=False):
            with patch("os.path.exists", return_value=False):
                # Clear other CI vars to ensure this one is detected
                env_copy = {k: v for k, v in os.environ.items()
                           if k not in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "BUILDKITE")}
                env_copy["CONTINUOUS_INTEGRATION"] = "true"
                with patch.dict(os.environ, env_copy, clear=True):
                    assert detect_environment() == "ci"

    def test_detects_ci_from_GITHUB_ACTIONS(self):
        """Test GITHUB_ACTIONS env var is detected."""
        env = {"GITHUB_ACTIONS": "true"}
        with patch.dict(os.environ, env, clear=True):
            with patch("os.path.exists", return_value=False):
                assert detect_environment() == "ci"

    def test_detects_ci_from_GITLAB_CI(self):
        """Test GITLAB_CI env var is detected."""
        env = {"GITLAB_CI": "true"}
        with patch.dict(os.environ, env, clear=True):
            with patch("os.path.exists", return_value=False):
                assert detect_environment() == "ci"

    def test_detects_ci_from_JENKINS_URL(self):
        """Test JENKINS_URL env var is detected."""
        env = {"JENKINS_URL": "http://jenkins.example.com"}
        with patch.dict(os.environ, env, clear=True):
            with patch("os.path.exists", return_value=False):
                assert detect_environment() == "ci"

    def test_detects_ci_from_BUILDKITE(self):
        """Test BUILDKITE env var is detected."""
        env = {"BUILDKITE": "true"}
        with patch.dict(os.environ, env, clear=True):
            with patch("os.path.exists", return_value=False):
                assert detect_environment() == "ci"

    def test_detects_docker_from_dockerenv_file(self):
        """Test detection of Docker from /.dockerenv file."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                assert detect_environment() == "docker"

    def test_detects_docker_from_cgroup_docker(self):
        """Test detection of Docker from /proc/1/cgroup containing 'docker'."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                cgroup_content = "12:memory:/docker/abc123\n"
                with patch("builtins.open", mock_open(read_data=cgroup_content)):
                    assert detect_environment() == "docker"

    def test_detects_docker_from_cgroup_containerd(self):
        """Test detection of Docker from /proc/1/cgroup containing 'containerd'."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                cgroup_content = "12:memory:/containerd/abc123\n"
                with patch("builtins.open", mock_open(read_data=cgroup_content)):
                    assert detect_environment() == "docker"

    def test_detects_docker_from_hostname_pattern(self):
        """Test detection of Docker from 12-char alphanumeric hostname."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                # Make cgroup file not found
                with patch("builtins.open", side_effect=FileNotFoundError):
                    with patch("socket.gethostname", return_value="abc123def456"):
                        assert detect_environment() == "docker"

    def test_detects_host_as_default(self):
        """Test that host is detected as default."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                with patch("builtins.open", side_effect=FileNotFoundError):
                    with patch("socket.gethostname", return_value="my-development-machine"):
                        assert detect_environment() == "host"

    def test_cgroup_permission_error_handled(self):
        """Test that PermissionError reading cgroup is handled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                with patch("builtins.open", side_effect=PermissionError):
                    with patch("socket.gethostname", return_value="my-machine"):
                        assert detect_environment() == "host"

    def test_ci_takes_priority_over_docker(self):
        """Test that CI detection takes priority over Docker."""
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            with patch("os.path.exists", return_value=True):  # Would detect docker
                assert detect_environment() == "ci"


class TestGetNeo4jConnectionConfig:
    """Test Neo4j connection configuration."""

    def test_env_vars_take_priority(self):
        """Test that explicit env vars override defaults."""
        env = {
            "NEO4J_URI": "bolt://custom:9999",
            "NEO4J_USER": "customuser",
            "NEO4J_PASSWORD": "custompass",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="host"):
                config = get_neo4j_connection_config()
                assert config["uri"] == "bolt://custom:9999"
                assert config["username"] == "customuser"
                assert config["password"] == "custompass"
                assert "environment_variables" in config["source"]

    def test_env_vars_with_empty_password(self):
        """Test that empty password is valid."""
        env = {
            "NEO4J_URI": "bolt://custom:9999",
            "NEO4J_USER": "customuser",
            "NEO4J_PASSWORD": "",  # Empty is valid
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="host"):
                config = get_neo4j_connection_config()
                assert config["password"] == ""

    def test_docker_defaults(self):
        """Test Docker environment defaults."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                config = get_neo4j_connection_config()
                assert config["uri"] == "bolt://neo4j:7687"
                assert config["username"] == "neo4j"
                assert config["source"] == "docker_defaults"

    def test_ci_defaults(self):
        """Test CI environment defaults."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="ci"):
                config = get_neo4j_connection_config()
                assert config["uri"] == "bolt://localhost:7687"
                assert config["username"] == "neo4j"
                assert config["source"] == "ci_defaults"

    def test_host_defaults(self):
        """Test host environment defaults."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="host"):
                config = get_neo4j_connection_config()
                assert config["uri"] == "bolt://localhost:7687"
                assert config["username"] == "neo4j"
                assert config["source"] == "host_defaults"

    def test_partial_env_vars_uses_defaults(self):
        """Test that partial env vars fall back to defaults."""
        # Only URI set, missing user and password
        env = {"NEO4J_URI": "bolt://custom:9999"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="host"):
                config = get_neo4j_connection_config()
                # Should use host defaults since not all env vars present
                assert config["source"] == "host_defaults"


class TestGetNeo4jTestConnectionConfig:
    """Test Neo4j TEST database connection configuration."""

    def test_test_env_vars_take_priority(self):
        """Test that test-specific env vars override defaults."""
        env = {
            "NEO4J_TEST_URI": "bolt://test-custom:9999",
            "NEO4J_TEST_USER": "testuser",
            "NEO4J_TEST_PASSWORD": "testpass",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="host"):
                config = get_neo4j_test_connection_config()
                assert config["uri"] == "bolt://test-custom:9999"
                assert config["username"] == "testuser"
                assert config["password"] == "testpass"
                assert "test_environment_variables" in config["source"]

    def test_docker_test_defaults(self):
        """Test Docker environment test defaults."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                config = get_neo4j_test_connection_config()
                assert config["uri"] == "bolt://neo4j-test:7687"
                assert config["username"] == "neo4j"
                assert config["password"] == "testpassword"
                assert config["source"] == "docker_test_defaults"

    def test_host_test_defaults(self):
        """Test host environment test defaults (port 7688)."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="host"):
                config = get_neo4j_test_connection_config()
                assert config["uri"] == "bolt://localhost:7688"
                assert config["username"] == "neo4j"
                assert config["password"] == "testpassword"
                assert config["source"] == "host_test_defaults"

    def test_ci_test_defaults(self):
        """Test CI environment test defaults."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="ci"):
                config = get_neo4j_test_connection_config()
                assert config["uri"] == "bolt://localhost:7688"
                assert config["source"] == "ci_test_defaults"


class TestIsNeo4jAvailable:
    """Test Neo4j availability check."""

    def test_available_when_socket_connects(self):
        """Test returns True when socket connects successfully."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("socket.socket", return_value=mock_socket):
            result = is_neo4j_available("bolt://localhost:7687")
            assert result is True
            mock_socket.connect_ex.assert_called_once_with(("localhost", 7687))
            mock_socket.close.assert_called_once()

    def test_unavailable_when_socket_fails(self):
        """Test returns False when socket fails to connect."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 111  # Connection refused

        with patch("socket.socket", return_value=mock_socket):
            result = is_neo4j_available("bolt://localhost:7687")
            assert result is False

    def test_parses_uri_with_port(self):
        """Test correctly parses URI with explicit port."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("socket.socket", return_value=mock_socket):
            is_neo4j_available("bolt://myhost:9999")
            mock_socket.connect_ex.assert_called_once_with(("myhost", 9999))

    def test_parses_uri_without_port_uses_default(self):
        """Test uses default port 7687 when not specified."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("socket.socket", return_value=mock_socket):
            is_neo4j_available("bolt://myhost")
            mock_socket.connect_ex.assert_called_once_with(("myhost", 7687))

    def test_invalid_uri_format_returns_false(self):
        """Test returns False for non-bolt URI."""
        result = is_neo4j_available("http://localhost:7687")
        assert result is False

    def test_socket_error_returns_false(self):
        """Test returns False on socket error."""
        with patch("socket.socket", side_effect=socket.error("Connection error")):
            result = is_neo4j_available("bolt://localhost:7687")
            assert result is False

    def test_value_error_returns_false(self):
        """Test returns False on invalid port number."""
        result = is_neo4j_available("bolt://localhost:notaport")
        assert result is False

    def test_respects_timeout_parameter(self):
        """Test that timeout is passed to socket."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("socket.socket", return_value=mock_socket):
            is_neo4j_available("bolt://localhost:7687", timeout=10.0)
            mock_socket.settimeout.assert_called_once_with(10.0)


class TestGetAvailableNeo4jConfig:
    """Test getting available Neo4j configuration."""

    def test_returns_primary_config_when_available(self):
        """Test returns primary config when Neo4j is available."""
        with patch("src.utils.environment.get_neo4j_connection_config") as mock_config:
            mock_config.return_value = {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "test",
                "source": "host_defaults",
            }
            with patch("src.utils.environment.is_neo4j_available", return_value=True):
                config = get_available_neo4j_config(test_mode=False)
                assert config is not None
                assert config["uri"] == "bolt://localhost:7687"

    def test_returns_none_when_all_unavailable(self):
        """Test returns None when no configs are available."""
        with patch("src.utils.environment.get_neo4j_connection_config") as mock_config:
            mock_config.return_value = {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "test",
            }
            with patch("src.utils.environment.is_neo4j_available", return_value=False):
                with patch("src.utils.environment.detect_environment", return_value="ci"):
                    config = get_available_neo4j_config(test_mode=False)
                    assert config is None

    def test_test_mode_uses_test_config(self):
        """Test that test_mode=True uses test configuration."""
        with patch("src.utils.environment.get_neo4j_test_connection_config") as mock_config:
            mock_config.return_value = {
                "uri": "bolt://localhost:7688",
                "username": "neo4j",
                "password": "testpassword",
                "source": "test_defaults",
            }
            with patch("src.utils.environment.is_neo4j_available", return_value=True):
                config = get_available_neo4j_config(test_mode=True)
                assert config is not None
                assert config["uri"] == "bolt://localhost:7688"

    def test_tries_fallback_configs_for_host(self):
        """Test fallback configs are tried for host environment."""
        call_count = [0]

        def mock_available(uri, timeout=5.0):
            call_count[0] += 1
            # First call (primary) fails, second call (fallback 127.0.0.1) succeeds
            return call_count[0] >= 2

        with patch("src.utils.environment.get_neo4j_connection_config") as mock_config:
            mock_config.return_value = {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "test",
            }
            with patch("src.utils.environment.is_neo4j_available", side_effect=mock_available):
                with patch("src.utils.environment.detect_environment", return_value="host"):
                    config = get_available_neo4j_config(test_mode=False)
                    assert config is not None
                    assert config["source"] == "fallback_host"

    def test_tries_fallback_configs_for_docker(self):
        """Test fallback configs are tried for docker environment."""
        call_count = [0]

        def mock_available(uri, timeout=5.0):
            call_count[0] += 1
            return call_count[0] >= 2

        with patch("src.utils.environment.get_neo4j_connection_config") as mock_config:
            mock_config.return_value = {
                "uri": "bolt://neo4j:7687",
                "username": "neo4j",
                "password": "test",
            }
            with patch("src.utils.environment.is_neo4j_available", side_effect=mock_available):
                with patch("src.utils.environment.detect_environment", return_value="docker"):
                    config = get_available_neo4j_config(test_mode=False)
                    assert config is not None
                    assert config["source"] == "fallback_docker"

    def test_config_func_exception_handled(self):
        """Test that exceptions from config function are handled."""
        with patch("src.utils.environment.get_neo4j_connection_config", side_effect=Exception("Config error")):
            with patch("src.utils.environment.detect_environment", return_value="ci"):
                config = get_available_neo4j_config(test_mode=False)
                # Should return None since exception was caught
                assert config is None


class TestIsServiceExternallyManaged:
    """Test service externally managed detection."""

    def test_returns_false_for_host_environment(self):
        """Test that host environment always returns False."""
        with patch("src.utils.environment.detect_environment", return_value="host"):
            result = is_service_externally_managed("neo4j-test", 7687)
            assert result is False

    def test_respects_MANAGE_TEST_SERVICES_false(self):
        """Test MANAGE_TEST_SERVICES=false means externally managed."""
        env = {"MANAGE_TEST_SERVICES": "false"}
        with patch.dict(os.environ, env, clear=False):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                result = is_service_externally_managed("neo4j-test", 7687)
                assert result is True  # Externally managed

    def test_respects_MANAGE_TEST_SERVICES_true(self):
        """Test MANAGE_TEST_SERVICES=true means NOT externally managed."""
        env = {"MANAGE_TEST_SERVICES": "true"}
        with patch.dict(os.environ, env, clear=False):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                result = is_service_externally_managed("neo4j-test", 7687)
                assert result is False  # Not externally managed

    def test_respects_MANAGE_TEST_SERVICES_zero(self):
        """Test MANAGE_TEST_SERVICES=0 means externally managed."""
        env = {"MANAGE_TEST_SERVICES": "0"}
        with patch.dict(os.environ, env, clear=False):
            with patch("src.utils.environment.detect_environment", return_value="ci"):
                result = is_service_externally_managed("neo4j-test", 7687)
                assert result is True

    def test_respects_MANAGE_TEST_SERVICES_no(self):
        """Test MANAGE_TEST_SERVICES=no means externally managed."""
        env = {"MANAGE_TEST_SERVICES": "no"}
        with patch.dict(os.environ, env, clear=False):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                result = is_service_externally_managed("neo4j-test", 7687)
                assert result is True

    def test_checks_socket_in_docker_when_no_env_var(self):
        """Test that socket is checked in docker when MANAGE_TEST_SERVICES not set."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0  # Connection succeeds

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                with patch("socket.socket", return_value=mock_socket):
                    result = is_service_externally_managed("neo4j-test", 7687)
                    assert result is True  # Service is accessible

    def test_socket_failure_means_not_managed(self):
        """Test that socket failure means not externally managed."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 111  # Connection refused

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                with patch("socket.socket", return_value=mock_socket):
                    result = is_service_externally_managed("neo4j-test", 7687)
                    assert result is False

    def test_socket_error_handled(self):
        """Test that socket.error is handled gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                with patch("socket.socket", side_effect=socket.error("Error")):
                    result = is_service_externally_managed("neo4j-test", 7687)
                    assert result is False

    def test_gaierror_handled(self):
        """Test that socket.gaierror (DNS failure) is handled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                with patch("socket.socket", side_effect=socket.gaierror("DNS error")):
                    result = is_service_externally_managed("neo4j-test", 7687)
                    assert result is False


class TestCheckNeo4jTestReady:
    """Test Neo4j test readiness check."""

    def test_returns_false_when_bolt_unavailable(self):
        """Test returns False when Bolt is not available."""
        with patch("src.utils.environment.get_neo4j_test_connection_config") as mock_config:
            mock_config.return_value = {"uri": "bolt://localhost:7688"}
            with patch("src.utils.environment.is_neo4j_available", return_value=False):
                result = check_neo4j_test_ready()
                assert result is False

    def test_checks_http_port_for_docker(self):
        """Test checks HTTP port 7474 for docker environment."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("src.utils.environment.get_neo4j_test_connection_config") as mock_config:
            mock_config.return_value = {"uri": "bolt://neo4j-test:7687"}
            with patch("src.utils.environment.is_neo4j_available", return_value=True):
                with patch("src.utils.environment.detect_environment", return_value="docker"):
                    with patch("socket.socket", return_value=mock_socket):
                        result = check_neo4j_test_ready()
                        assert result is True
                        mock_socket.connect_ex.assert_called_with(("neo4j-test", 7474))

    def test_checks_http_port_for_host(self):
        """Test checks HTTP port 7475 for host environment."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("src.utils.environment.get_neo4j_test_connection_config") as mock_config:
            mock_config.return_value = {"uri": "bolt://localhost:7688"}
            with patch("src.utils.environment.is_neo4j_available", return_value=True):
                with patch("src.utils.environment.detect_environment", return_value="host"):
                    with patch("socket.socket", return_value=mock_socket):
                        result = check_neo4j_test_ready()
                        assert result is True
                        mock_socket.connect_ex.assert_called_with(("localhost", 7475))

    def test_returns_true_when_http_check_fails_but_bolt_works(self):
        """Test returns True if HTTP fails but Bolt is available."""
        with patch("src.utils.environment.get_neo4j_test_connection_config") as mock_config:
            mock_config.return_value = {"uri": "bolt://localhost:7688"}
            with patch("src.utils.environment.is_neo4j_available", return_value=True):
                with patch("src.utils.environment.detect_environment", return_value="host"):
                    with patch("socket.socket", side_effect=Exception("HTTP error")):
                        result = check_neo4j_test_ready()
                        assert result is True

    def test_respects_timeout_parameter(self):
        """Test that timeout is respected."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("src.utils.environment.get_neo4j_test_connection_config") as mock_config:
            mock_config.return_value = {"uri": "bolt://localhost:7688"}
            with patch("src.utils.environment.is_neo4j_available", return_value=True) as mock_avail:
                with patch("src.utils.environment.detect_environment", return_value="host"):
                    with patch("socket.socket", return_value=mock_socket):
                        check_neo4j_test_ready(timeout=10.0)
                        mock_avail.assert_called_once_with("bolt://localhost:7688", timeout=10.0)


class TestIsEventstoreExternallyManaged:
    """Test EventStoreDB externally managed detection."""

    def test_returns_false_for_host_environment(self):
        """Test that host environment always returns False."""
        with patch("src.utils.environment.detect_environment", return_value="host"):
            result = is_eventstore_externally_managed("eventstore", 2113)
            assert result is False

    def test_respects_MANAGE_TEST_SERVICES_false(self):
        """Test MANAGE_TEST_SERVICES=false means externally managed."""
        env = {"MANAGE_TEST_SERVICES": "false"}
        with patch.dict(os.environ, env, clear=False):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                result = is_eventstore_externally_managed("eventstore", 2113)
                assert result is True

    def test_respects_MANAGE_TEST_SERVICES_true(self):
        """Test MANAGE_TEST_SERVICES=true means NOT externally managed."""
        env = {"MANAGE_TEST_SERVICES": "true"}
        with patch.dict(os.environ, env, clear=False):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                result = is_eventstore_externally_managed("eventstore", 2113)
                assert result is False

    def test_checks_socket_in_ci_when_no_env_var(self):
        """Test that socket is checked in CI when MANAGE_TEST_SERVICES not set."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="ci"):
                with patch("socket.socket", return_value=mock_socket):
                    result = is_eventstore_externally_managed("eventstore", 2113)
                    assert result is True

    def test_socket_gaierror_handled(self):
        """Test that socket.gaierror is handled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                with patch("socket.socket", side_effect=socket.gaierror("DNS error")):
                    result = is_eventstore_externally_managed("eventstore", 2113)
                    assert result is False


class TestCheckEventstoreReady:
    """Test EventStoreDB readiness check."""

    def test_returns_false_when_socket_unavailable(self):
        """Test returns False when socket connection fails."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 111  # Connection refused

        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", return_value=mock_socket):
                result = check_eventstore_ready()
                assert result is False

    def test_checks_health_endpoint_for_docker(self):
        """Test checks health endpoint with docker service name."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("src.utils.environment.detect_environment", return_value="docker"):
            with patch("socket.socket", return_value=mock_socket):
                with patch("requests.get", return_value=mock_response) as mock_get:
                    result = check_eventstore_ready()
                    assert result is True
                    mock_get.assert_called_once_with(
                        "http://eventstore:2113/health/live",
                        timeout=5.0
                    )

    def test_checks_health_endpoint_for_host(self):
        """Test checks health endpoint with localhost."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", return_value=mock_socket):
                with patch("requests.get", return_value=mock_response) as mock_get:
                    result = check_eventstore_ready()
                    assert result is True
                    mock_get.assert_called_once_with(
                        "http://localhost:2113/health/live",
                        timeout=5.0
                    )

    def test_returns_true_when_health_returns_204(self):
        """Test returns True when health endpoint returns 204."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", return_value=mock_socket):
                with patch("requests.get", return_value=mock_response):
                    result = check_eventstore_ready()
                    assert result is True

    def test_returns_false_when_health_returns_other_status(self):
        """Test returns False when health returns non-204 status."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", return_value=mock_socket):
                with patch("requests.get", return_value=mock_response):
                    result = check_eventstore_ready()
                    assert result is False

    def test_returns_true_when_http_fails_but_socket_works(self):
        """Test returns True if HTTP health fails but TCP is available."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", return_value=mock_socket):
                with patch("requests.get", side_effect=requests.RequestException("Error")):
                    result = check_eventstore_ready()
                    assert result is True

    def test_socket_error_returns_false(self):
        """Test socket.error returns False."""
        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", side_effect=socket.error("Error")):
                result = check_eventstore_ready()
                assert result is False

    def test_socket_gaierror_returns_false(self):
        """Test socket.gaierror (DNS failure) returns False."""
        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", side_effect=socket.gaierror("DNS error")):
                result = check_eventstore_ready()
                assert result is False

    def test_respects_timeout_parameter(self):
        """Test that timeout is passed to socket and requests."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0

        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("src.utils.environment.detect_environment", return_value="host"):
            with patch("socket.socket", return_value=mock_socket):
                with patch("requests.get", return_value=mock_response) as mock_get:
                    check_eventstore_ready(timeout=15.0)
                    mock_socket.settimeout.assert_called_with(15.0)
                    mock_get.assert_called_with(
                        "http://localhost:2113/health/live",
                        timeout=15.0
                    )
