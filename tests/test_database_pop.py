"""Tests for database population helpers that do not touch MySQL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from retinanalysis.utils import database_pop


TABLE_NAMES = (
    "Experiment",
    "Animal",
    "Preparation",
    "Cell",
    "EpochGroup",
    "EpochBlock",
    "Epoch",
    "Response",
    "Stimulus",
    "Protocol",
    "Tags",
    "SortingChunk",
    "SortedCell",
    "CellTypeFile",
    "SortedCellType",
)

FIXTURE_ROOT_ENV_VAR = "RETINANALYSIS_MIGRATION_FIXTURE_ROOT"
FIXTURE_EXP_NAME = "20260401C"
FIXTURE_SS_VERSION = "kilosort2.5"
FIXTURE_CHUNK_MAP = {
    "chirp": ["data014"],
    "chunk1": ["data000", "data001", "data002", "data003"],
    "chunk2": ["data012", "data013"],
    "chunk3": ["data015", "data016", "data017"],
    "defocus1": ["data004", "data005", "data006", "data007", "data008", "data009"],
    "defocus2": ["data018", "data019", "data020", "data021", "data022", "data023"],
    "nat_images": ["data010", "data011"],
}
FIXTURE_ANALYSIS_CHUNKS = {"chunk1", "chunk2", "chunk3"}


@pytest.fixture
def restore_database_pop_globals():
    """Restore database_pop table globals after each test."""
    sentinel = object()
    names = (*TABLE_NAMES, "db", "table_dict")
    before = {name: getattr(database_pop, name, sentinel) for name in names}

    yield

    for name, value in before.items():
        if value is sentinel:
            database_pop.__dict__.pop(name, None)
        else:
            setattr(database_pop, name, value)


def make_fake_schema_source() -> tuple[SimpleNamespace, dict[str, object]]:
    """Create a schema-like object with fake table attributes."""
    tables = {name: object() for name in TABLE_NAMES}
    return SimpleNamespace(**tables), tables


def require_migration_fixture_root() -> Path:
    """Return the explicitly configured migration fixture root, or skip."""
    fixture_root = os.environ.get(FIXTURE_ROOT_ENV_VAR)
    if not fixture_root:
        pytest.skip(
            f"Set {FIXTURE_ROOT_ENV_VAR} to run migration-fixture preflight checks."
        )

    path = Path(fixture_root).expanduser()
    if not path.exists():
        pytest.skip(f"DataJoint migration fixture is not mounted: {path}")
    return path


def test_configure_tables_binds_schema_like_source(restore_database_pop_globals) -> None:
    """configure_tables should bind table globals without touching DataJoint."""
    schema_source, tables = make_fake_schema_source()

    table_dict = database_pop.configure_tables(schema_source)

    assert database_pop.db is schema_source
    for name, table in tables.items():
        assert getattr(database_pop, name) is table

    assert table_dict["experiment"] is tables["Experiment"]
    assert table_dict["animal"] is tables["Animal"]
    assert table_dict["preparation"] is tables["Preparation"]
    assert table_dict["cell"] is tables["Cell"]
    assert table_dict["epoch_group"] is tables["EpochGroup"]
    assert table_dict["epoch_block"] is tables["EpochBlock"]
    assert table_dict["epoch"] is tables["Epoch"]
    assert table_dict["response"] is tables["Response"]
    assert table_dict["stimulus"] is tables["Stimulus"]
    assert table_dict["tags"] is tables["Tags"]


def test_fill_tables_keeps_existing_compatibility_path(restore_database_pop_globals) -> None:
    """fill_tables should still work for callers that set database_pop.db."""
    schema_source, tables = make_fake_schema_source()
    database_pop.db = schema_source

    database_pop.fill_tables()

    assert database_pop.Experiment is tables["Experiment"]
    assert database_pop.CellTypeFile is tables["CellTypeFile"]
    assert database_pop.table_dict["experiment"] is tables["Experiment"]


def test_configure_tables_rejects_missing_schema_source(restore_database_pop_globals) -> None:
    """A missing schema source should fail explicitly rather than half-binding globals."""
    with pytest.raises(ValueError, match="schema_source cannot be None"):
        database_pop.configure_tables(None)


@pytest.mark.integration
@pytest.mark.migration_fixture
def test_datajoint_migration_fixture_preflight(monkeypatch) -> None:
    """Validate an explicitly configured migration fixture without touching MySQL."""
    fixture_root = require_migration_fixture_root()

    h5_dir = fixture_root / "h5"
    meta_dir = fixture_root / "meta"
    tags_dir = fixture_root / "tags"
    sorted_dir = fixture_root / "sorted"
    analysis_dir = fixture_root / "analysis"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fixture preflight should not create or convert files")

    monkeypatch.setattr(database_pop, "gen_tags", fail_if_called)
    monkeypatch.setattr(database_pop, "parse_data", fail_if_called)
    monkeypatch.setattr(database_pop, "DATA_DIR", str(sorted_dir))
    monkeypatch.setattr(database_pop, "ANALYSIS_DIR", str(analysis_dir))

    meta_list = database_pop.gen_meta_list(str(h5_dir), str(meta_dir), str(tags_dir))
    assert meta_list == [
        [
            str(meta_dir / f"{FIXTURE_EXP_NAME}.json"),
            str(h5_dir / f"{FIXTURE_EXP_NAME}.h5"),
            str(tags_dir / f"{FIXTURE_EXP_NAME}.json"),
        ]
    ]

    with (meta_dir / f"{FIXTURE_EXP_NAME}.json").open() as f:
        meta = json.load(f)
    assert meta["rig_type"] == "MEA"
    assert meta["uuid"]
    assert len(meta["animals"]) == 1

    experiment_sorted_dir = sorted_dir / FIXTURE_EXP_NAME
    actual_chunk_dirs = {
        path.name
        for path in experiment_sorted_dir.iterdir()
        if path.is_dir() and not path.name.startswith("data")
    }
    assert actual_chunk_dirs == set(FIXTURE_CHUNK_MAP)

    for chunk_name, datafiles in FIXTURE_CHUNK_MAP.items():
        chunk_file = experiment_sorted_dir / f"{FIXTURE_EXP_NAME}_{chunk_name}.txt"
        assert chunk_file.exists()
        assert chunk_file.read_text().split() == datafiles

        cluster_file = experiment_sorted_dir / chunk_name / FIXTURE_SS_VERSION / "cluster_KSLabel.tsv"
        assert cluster_file.exists()
        with cluster_file.open() as f:
            cluster_rows = [line for line in f if line.strip() and not line.startswith("cluster_id")]
        assert cluster_rows

    all_mapped_datafiles = {datafile for datafiles in FIXTURE_CHUNK_MAP.values() for datafile in datafiles}
    assert len(all_mapped_datafiles) == 24
    assert "data009" in all_mapped_datafiles

    actual_data_dirs = {path.name for path in experiment_sorted_dir.iterdir() if path.is_dir() and path.name.startswith("data")}
    assert actual_data_dirs.issubset(all_mapped_datafiles)
    assert "data009" in actual_data_dirs

    actual_analysis_chunks = {path.name for path in (analysis_dir / FIXTURE_EXP_NAME).iterdir() if path.is_dir()}
    assert actual_analysis_chunks == FIXTURE_ANALYSIS_CHUNKS
