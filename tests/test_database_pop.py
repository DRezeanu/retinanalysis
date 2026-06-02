"""Unit tests for database population helpers that do not touch MySQL."""

from __future__ import annotations

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
