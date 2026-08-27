"""Connection helpers for the single staged database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import SCHEMA, VIEWS

DEFAULT_PATH = Path("data/shgk.sqlite3")


def connect(path: str | Path = DEFAULT_PATH, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        if not path.is_file():
            raise FileNotFoundError(f"Database not found: {path}")
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(path: str | Path = DEFAULT_PATH) -> None:
    """Create tables and views if they are missing. Safe to call repeatedly."""
    with connect(path) as connection:
        connection.executescript(SCHEMA)
        connection.executescript(VIEWS)
