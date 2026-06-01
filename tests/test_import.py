"""Import and lazy-loading smoke tests for retinanalysis."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def test_import_retinanalysis_smoke() -> None:
    """The package should be importable through the lab-facing convenience API."""
    import retinanalysis as ra

    assert ra is not None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Current retinanalysis import eagerly imports retinanalysis.config.schema; "
        "the DataJoint migration should make schema loading lazy."
    ),
)
def test_plain_import_does_not_load_schema_module() -> None:
    """Future target: plain package import should not load the schema module."""
    code = textwrap.dedent(
        """
        import sys
        import retinanalysis  # noqa: F401
        raise SystemExit('retinanalysis.config.schema' in sys.modules)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
