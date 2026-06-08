"""Opt-in destructive smoke tests for DataJoint migration helpers.

These tests intentionally mutate the local migration test database. They are
kept separate from the normal DB smoke tests and require an explicit environment
variable opt-in in addition to the Docker container safety gate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = [pytest.mark.db, pytest.mark.destructive_db, pytest.mark.migration_fixture]

EXP_NAME = "20260401C"
FIXTURE_ROOT_ENV_VAR = "RETINANALYSIS_MIGRATION_FIXTURE_ROOT"
EXPECTED_SORTING_CHUNKS = 7
EXPECTED_CELL_TYPE_FILES = 3
EXPECTED_SORTED_CELL_TYPES = 1463
EXPECTED_CELL_TYPE_FILE_NAME = "dragos_kilosort2.5.classification.txt"


def require_destructive_db_opt_in() -> None:
    """Skip unless this run explicitly opted into destructive DB checks."""
    if os.environ.get("RETINANALYSIS_RUN_DESTRUCTIVE_DB_TESTS") != "1":
        pytest.skip(
            "Set RETINANALYSIS_RUN_DESTRUCTIVE_DB_TESTS=1 to run destructive "
            "DataJoint test-database checks."
        )


def require_migration_fixture_root() -> Path:
    """Return the explicitly configured migration fixture root, or skip."""
    fixture_root = os.environ.get(FIXTURE_ROOT_ENV_VAR)
    if not fixture_root:
        pytest.skip(
            f"Set {FIXTURE_ROOT_ENV_VAR} to run migration-fixture destructive checks."
        )

    path = Path(fixture_root).expanduser()
    if not path.exists():
        pytest.skip(f"DataJoint migration fixture is not mounted: {path}")
    return path


def configure_fixture_paths(monkeypatch: pytest.MonkeyPatch, fixture_root: Path):
    """Point population helpers at the isolated migration fixture."""
    from retinanalysis.utils import database_pop

    monkeypatch.setattr(database_pop, "DATA_DIR", str(fixture_root / "sorted"))
    monkeypatch.setattr(database_pop, "ANALYSIS_DIR", str(fixture_root / "analysis"))
    return database_pop


def populate_fixture_database(ra, fixture_root: Path) -> int:
    """Populate the test DB from the isolated 20260401C fixture."""
    return ra.populate_database(
        username="test_user",
        h5_dir=str(fixture_root / "h5"),
        meta_dir=str(fixture_root / "meta"),
        tags_dir=str(fixture_root / "tags"),
    )


def assert_refreshed_fixture_counts(ra) -> None:
    """Assert the fixture DB has the known post-population counts."""
    exp_q = ra.schema.Experiment() & {"exp_name": EXP_NAME}
    assert len(exp_q) == 1
    exp_id = exp_q.fetch1("id")

    assert len(ra.schema.SortingChunk() & {"experiment_id": exp_id}) == EXPECTED_SORTING_CHUNKS
    assert len(ra.schema.CellTypeFile()) == EXPECTED_CELL_TYPE_FILES
    assert len(ra.schema.SortedCellType()) == EXPECTED_SORTED_CELL_TYPES


def ensure_fixture_database_populated(ra, fixture_root: Path) -> None:
    """Populate the fixture if this destructive test starts from an empty DB."""
    if len(ra.schema.Experiment() & {"exp_name": EXP_NAME}) == 0:
        assert populate_fixture_database(ra, fixture_root) == 1

    assert_refreshed_fixture_counts(ra)


def test_reload_celltypefiles_purge_and_refresh_fixture_database(
    test_database_container: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise destructive helpers, then restore the fresh fixture DB state."""
    require_destructive_db_opt_in()

    import retinanalysis as ra
    from retinanalysis.utils import database_utils

    fixture_root = require_migration_fixture_root()
    database_pop = configure_fixture_paths(monkeypatch, fixture_root)
    ensure_fixture_database_populated(ra, fixture_root)

    exp_id = (ra.schema.Experiment() & {"exp_name": EXP_NAME}).fetch1("id")
    chunk_ids = (ra.schema.SortingChunk() & {"experiment_id": exp_id}).to_arrays("id")

    database_pop.reload_celltypefiles([EXP_NAME])

    cell_type_files = ra.schema.CellTypeFile() & [{"chunk_id": int(chunk_id)} for chunk_id in chunk_ids]
    assert len(cell_type_files) == EXPECTED_CELL_TYPE_FILES
    assert set(cell_type_files.to_arrays("file_name")) == {EXPECTED_CELL_TYPE_FILE_NAME}

    database_utils.purge_database()

    assert len(ra.schema.Experiment()) == 0
    assert len(ra.schema.SortingChunk()) == 0
    assert len(ra.schema.CellTypeFile()) == 0
    assert len(ra.schema.SortedCellType()) == 0

    assert populate_fixture_database(ra, fixture_root) == 1
    assert_refreshed_fixture_counts(ra)
