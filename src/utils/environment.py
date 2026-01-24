"""
src/utils/environment.py

Environment detection utilities for determining runtime context
and providing appropriate configuration values.
"""

import os
import socket
from typing import Dict, Optional

import requests  # noqa: E402


def detect_environment() -> str:
    """
    Detect the current runtime environment.

    Returns:
        str: One of 'docker', 'ci', 'host'
    """
    # Check for CI environment variables
    ci_indicators = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_URL",
        "BUILDKITE",
    ]
    if any(os.getenv(var) for var in ci_indicators):
        return "ci"

    # Check for Docker environment
    # Method 1: Check for .dockerenv file
    if os.path.exists("/.dockerenv"):
        return "docker"

    # Method 2: Check if we're in a container via cgroup
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup_content = f.read()
            if "docker" in cgroup_content or "containerd" in cgroup_content:
                return "docker"
    except (FileNotFoundError, PermissionError):
        pass

    # Method 3: Check hostname patterns common in containers
    hostname = socket.gethostname()
    if len(hostname) == 12 and hostname.isalnum():  # Docker default hostname pattern
        return "docker"

    # Default to host environment
    return "host"


def get_neo4j_connection_config() -> Dict[str, str]:
    """
    Get Neo4j connection configuration based on current environment.

    Returns:
        Dict[str, str]: Connection configuration with uri, username, password

    Raises:
        ValueError: If no valid configuration is found
    """
    environment = detect_environment()

    # Priority order: ENV_VARS > Environment-specific defaults > Config fallback

    # 1. Check environment variables first (highest priority)
    env_uri = os.getenv("NEO4J_URI")
    env_user = os.getenv("NEO4J_USER")
    env_password = os.getenv("NEO4J_PASSWORD")

    if env_uri and env_user and env_password is not None:
        return {
            "uri": env_uri,
            "username": env_user,
            "password": env_password,
            "source": f"environment_variables_{environment}",
        }

    # 2. Environment-specific defaults
    if environment == "docker":
        # Inside Docker container - use service names
        return {
            "uri": "bolt://neo4j:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD", "defaultpassword"),
            "source": "docker_defaults",
        }
    elif environment == "ci":
        # CI environment - typically localhost with specific ports
        return {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD", "testpassword"),
            "source": "ci_defaults",
        }
    else:  # host environment
        # Host development - localhost with Docker-exposed ports
        return {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD", "defaultpassword"),
            "source": "host_defaults",
        }


def get_neo4j_test_connection_config() -> Dict[str, str]:
    """
    Get Neo4j TEST database connection configuration.

    Returns:
        Dict[str, str]: Test database connection configuration
    """
    environment = detect_environment()

    # Check for test-specific environment variables
    test_uri = os.getenv("NEO4J_TEST_URI")
    test_user = os.getenv("NEO4J_TEST_USER")
    test_password = os.getenv("NEO4J_TEST_PASSWORD")

    if test_uri and test_user and test_password is not None:
        return {
            "uri": test_uri,
            "username": test_user,
            "password": test_password,
            "source": f"test_environment_variables_{environment}",
        }

    # Environment-specific test defaults
    if environment == "docker":
        # Inside Docker - use test service name
        return {
            "uri": "bolt://neo4j-test:7687",
            "username": "neo4j",
            "password": "testpassword",
            "source": "docker_test_defaults",
        }
    else:  # host or ci
        # Host/CI - use localhost with test port
        return {
            "uri": "bolt://localhost:7688",  # Test service exposed on 7688
            "username": "neo4j",
            "password": "testpassword",
            "source": f"{environment}_test_defaults",
        }


def is_neo4j_available(uri: str, timeout: float = 5.0) -> bool:
    """
    Check if Neo4j service is available at the given URI.

    Args:
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
        timeout: Connection timeout in seconds

    Returns:
        bool: True if service is reachable, False otherwise
    """
    try:
        # Extract host and port from URI
        if uri.startswith("bolt://"):
            host_port = uri[7:]  # Remove "bolt://"
            if ":" in host_port:
                host, port_str = host_port.split(":", 1)
                port = int(port_str)
            else:
                host = host_port
                port = 7687  # Default Neo4j port
        else:
            return False

        # Try to establish a TCP connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        return result == 0

    except (ValueError, socket.error):
        return False


def get_available_neo4j_config(test_mode: bool = False) -> Optional[Dict[str, str]]:
    """
    Get the first available Neo4j configuration by testing connectivity.

    Args:
        test_mode: If True, get test database configuration

    Returns:
        Optional[Dict[str, str]]: First working configuration, or None if none work
    """
    config_func = get_neo4j_test_connection_config if test_mode else get_neo4j_connection_config

    try:
        config = config_func()
        if is_neo4j_available(config["uri"]):
            return config
    except Exception:
        pass

    # If primary config doesn't work, try alternative URIs based on environment
    environment = detect_environment()
    fallback_configs = []

    if test_mode:
        if environment == "host":
            fallback_configs = [
                {
                    "uri": "bolt://localhost:7688",
                    "username": "neo4j",
                    "password": "testpassword",
                },
                {
                    "uri": "bolt://127.0.0.1:7688",
                    "username": "neo4j",
                    "password": "testpassword",
                },
            ]
        elif environment == "docker":
            fallback_configs = [
                {
                    "uri": "bolt://neo4j-test:7687",
                    "username": "neo4j",
                    "password": "testpassword",
                },
            ]
    else:
        if environment == "host":
            fallback_configs = [
                {
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": os.getenv("NEO4J_PASSWORD", "defaultpassword"),
                },
                {
                    "uri": "bolt://127.0.0.1:7687",
                    "username": "neo4j",
                    "password": os.getenv("NEO4J_PASSWORD", "defaultpassword"),
                },
            ]
        elif environment == "docker":
            fallback_configs = [
                {
                    "uri": "bolt://neo4j:7687",
                    "username": "neo4j",
                    "password": os.getenv("NEO4J_PASSWORD", "defaultpassword"),
                },
            ]

    # Test fallback configurations
    for config in fallback_configs:
        if is_neo4j_available(config["uri"]):
            config["source"] = f"fallback_{environment}"
            return config

    return None


def is_service_externally_managed(service_name: str = "neo4j-test", port: int = 7687) -> bool:
    """
    Check if a service is externally managed (e.g., by docker-compose).

    This is determined by checking if:
    1. We're running inside a container
    2. The service is already accessible on the network

    Args:
        service_name: Name of the service to check (default: neo4j-test)
        port: Port to check (default: 7687 for Neo4j)

    Returns:
        bool: True if service is externally managed, False otherwise
    """
    environment = detect_environment()

    # If we're not in a container, services are NOT externally managed
    # (they need to be started by the test runner)
    if environment not in ("docker", "ci"):
        return False

    # Check if explicitly overridden
    manage_services = os.getenv("MANAGE_TEST_SERVICES")
    if manage_services is not None:
        # "false", "0", "no" = externally managed (don't start)
        # "true", "1", "yes" = NOT externally managed (do start)
        return manage_services.lower() in ("false", "0", "no")

    # In container: Check if service is accessible
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((service_name, port))
        sock.close()

        # If we can connect, it's externally managed
        return result == 0
    except (socket.error, socket.gaierror):
        # If we can't even resolve the name or connect, it's not managed
        return False


def check_neo4j_test_ready(timeout: float = 5.0) -> bool:
    """
    Check if neo4j-test service is ready to accept connections.

    This performs a more thorough check than just network connectivity,
    attempting to verify the service is actually responding.

    Args:
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if service is ready, False otherwise
    """
    import time

    config = get_neo4j_test_connection_config()
    uri = config["uri"]

    # First check basic network connectivity
    if not is_neo4j_available(uri, timeout=timeout):
        return False

    # Additional check: Try to get the HTTP interface
    # Neo4j exposes HTTP on port 7474 when ready
    try:
        environment = detect_environment()
        if environment == "docker":
            http_host = "neo4j-test"
            http_port = 7474
        else:
            http_host = "localhost"
            http_port = 7475  # Test service HTTP port

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((http_host, http_port))
        sock.close()

        return result == 0
    except Exception:  # noqa: E722
        # If HTTP check fails, but Bolt works, that's still OK
        return True


def is_eventstore_externally_managed(service_name: str = "eventstore", port: int = 2113) -> bool:
    """
    Check if EventStoreDB service is externally managed (e.g., by docker-compose).

    This is determined by checking if:
    1. We're running inside a container
    2. The service is already accessible on the network

    Args:
        service_name: Name of the service to check (default: eventstore)
        port: Port to check (default: 2113 for EventStoreDB HTTP)

    Returns:
        bool: True if service is externally managed, False otherwise
    """
    environment = detect_environment()

    # If we're not in a container, services are NOT externally managed
    # (they need to be started by the test runner)
    if environment not in ("docker", "ci"):
        return False

    # Check if explicitly overridden
    manage_services = os.getenv("MANAGE_TEST_SERVICES")
    if manage_services is not None:
        # "false", "0", "no" = externally managed (don't start)
        # "true", "1", "yes" = NOT externally managed (do start)
        return manage_services.lower() in ("false", "0", "no")

    # In container: Check if service is accessible
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((service_name, port))
        sock.close()

        # If we can connect, it's externally managed
        return result == 0
    except (socket.error, socket.gaierror):
        # If we can't even resolve the name or connect, it's not managed
        return False


def check_eventstore_ready(timeout: float = 5.0) -> bool:
    """
    Check if EventStoreDB service is ready to accept connections.

    This checks the HTTP health endpoint to verify the service is operational.

    Args:
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if service is ready, False otherwise
    """
    environment = detect_environment()

    # Determine the correct host based on environment
    if environment == "docker":
        http_host = "eventstore"
        http_port = 2113
    else:
        http_host = "localhost"
        http_port = 2113

    # First check basic network connectivity
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((http_host, http_port))
        sock.close()

        if result != 0:
            return False
    except (socket.error, socket.gaierror):
        return False

    # Additional check: Try to access the health endpoint
    # EventStoreDB exposes /health/live for liveness checks
    try:
        url = f"http://{http_host}:{http_port}/health/live"
        response = requests.get(url, timeout=timeout)

        # EventStoreDB returns 204 No Content when healthy
        return response.status_code == 204
    except Exception:  # noqa: E722
        # If HTTP check fails, but TCP works, assume it's ready enough
        # (some versions might not have health endpoint)
        return True
