"""Unit tests for database utility wiring that do not touch MySQL."""

from __future__ import annotations

from retinanalysis.utils import database_utils


def test_populate_database_uses_canonical_schema_module(monkeypatch) -> None:
    """populate_database should pass config/schema.py's module to append_data."""
    schema_source = object()
    captured = {}

    def fake_get_schema_module():
        return schema_source

    def fake_append_data(h5_dir, meta_dir, tags_dir, username, db_param):
        captured["h5_dir"] = h5_dir
        captured["meta_dir"] = meta_dir
        captured["tags_dir"] = tags_dir
        captured["username"] = username
        captured["db_param"] = db_param

    monkeypatch.setattr(database_utils, "get_schema_module", fake_get_schema_module)
    monkeypatch.setattr(database_utils.database_pop, "append_data", fake_append_data)

    database_utils.populate_database(
        username="test_user",
        h5_dir="/tmp/h5",
        meta_dir="/tmp/meta",
        tags_dir="/tmp/tags",
    )

    assert captured == {
        "h5_dir": "/tmp/h5",
        "meta_dir": "/tmp/meta",
        "tags_dir": "/tmp/tags",
        "username": "test_user",
        "db_param": schema_source,
    }
