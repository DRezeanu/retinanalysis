"""Lazy accessors for DataJoint-backed database objects.

This module must stay safe to import without a running DataJoint/MySQL
server. Importing it should not import :mod:`retinanalysis.config.schema`;
the schema module is loaded only on first attribute access through
``schema`` or by an explicit call to ``get_schema_module``.
"""

from __future__ import annotations

from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Any


@cache
def get_schema_module() -> ModuleType:
    """Import and cache the DataJoint schema module on first use."""
    return import_module("retinanalysis.schema")


class LazySchema:
    """Proxy for lazy access to ``retinanalysis.config.schema`` attributes."""

    def __getattr__(self, name: str) -> Any:
        schema_module = get_schema_module()
        return getattr(schema_module, name)

    def __dir__(self) -> list[str]:
        schema_module = get_schema_module()
        return sorted(set(dir(type(self))) | set(dir(schema_module)))

    def __repr__(self) -> str:
        return "<LazySchema proxy for retinanalysis.schema>"


schema = LazySchema()

__all__ = ["LazySchema", "get_schema_module", "schema"]
