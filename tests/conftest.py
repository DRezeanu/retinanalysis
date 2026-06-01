"""Shared pytest helpers for retinanalysis tests."""

from __future__ import annotations

import subprocess

import pytest


EXPECTED_DB_CONTAINER = "test_database-db-1"


def running_docker_container_names() -> set[str]:
    """Return names of currently running Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("Docker is not installed or is not available on PATH.")
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"Could not inspect running Docker containers: {exc}")

    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def require_expected_test_database_container() -> str:
    """Validate that DB tests target the known local migration test database."""
    names = running_docker_container_names()

    suspicious_db_names = {
        name
        for name in names
        if (
            "database" in name.lower()
            or "datajoint" in name.lower()
            or name.endswith("-db-1")
        )
    }

    if EXPECTED_DB_CONTAINER not in names:
        pytest.skip(
            f"Expected DataJoint test container {EXPECTED_DB_CONTAINER!r} is not running. "
            f"Running containers: {sorted(names)!r}"
        )

    unexpected_db_names = suspicious_db_names - {EXPECTED_DB_CONTAINER}
    if unexpected_db_names:
        pytest.fail(
            "Refusing to run database tests because unexpected database-like "
            f"containers are running: {sorted(unexpected_db_names)!r}"
        )

    return EXPECTED_DB_CONTAINER


@pytest.fixture
def test_database_container() -> str:
    """Require and return the expected local DataJoint test DB container name."""
    return require_expected_test_database_container()
