"""SQLite connection factory and authoritative schema."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

BUSY_TIMEOUT_MS = 5_000
RECORD_TABLES = (
    "people",
    "person_photos",
    "clothing_items",
    "clothing_photos",
    "generations",
    "generation_clothing",
)
_SCHEMA = """
BEGIN;
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    names_json TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS person_photos (
    id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    relative_path TEXT UNIQUE,
    media_type TEXT NOT NULL CHECK(media_type='image/jpeg'),
    width INTEGER NOT NULL CHECK(width>0), height INTEGER NOT NULL CHECK(height>0),
    size_bytes INTEGER NOT NULL CHECK(size_bytes>=0),
    is_current INTEGER NOT NULL CHECK(is_current IN(0,1)), created_at TEXT NOT NULL,
    superseded_at TEXT,
    CHECK(is_current=0 OR relative_path IS NOT NULL)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_current_photo_per_person
    ON person_photos(person_id) WHERE is_current=1;
CREATE TABLE IF NOT EXISTS clothing_items (
    id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id) ON DELETE RESTRICT,
    name TEXT NOT NULL CHECK(length(trim(name, ' ' || char(9) || char(10) || char(13))) > 0),
    category TEXT NOT NULL CHECK(
        length(trim(category, ' ' || char(9) || char(10) || char(13))) > 0
    ),
    description TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', current_photo_id TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    FOREIGN KEY(current_photo_id, id) REFERENCES clothing_photos(id, clothing_item_id)
        DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE IF NOT EXISTS clothing_photos (
    id TEXT PRIMARY KEY,
    clothing_item_id TEXT NOT NULL REFERENCES clothing_items(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL UNIQUE, width INTEGER NOT NULL CHECK(width>0),
    height INTEGER NOT NULL CHECK(height>0), size_bytes INTEGER NOT NULL CHECK(size_bytes>=0),
    created_at TEXT NOT NULL, UNIQUE(id, clothing_item_id)
);
CREATE INDEX IF NOT EXISTS clothing_photos_item
    ON clothing_photos(clothing_item_id,created_at,id);
CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(
        status IN ('submission_unknown','processing','ready','failed')
    ),
    prompt TEXT NOT NULL,
    person_photo_id TEXT NOT NULL REFERENCES person_photos(id) ON DELETE RESTRICT,
    model TEXT NOT NULL CHECK(model='flux-2-pro'), provider_request_id TEXT, polling_url TEXT,
    result_relative_path TEXT, error_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generation_clothing (
    generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK(position>=0),
    clothing_item_id TEXT NOT NULL REFERENCES clothing_items(id) ON DELETE RESTRICT,
    clothing_photo_id TEXT NOT NULL, snapshot_name TEXT NOT NULL,
    PRIMARY KEY(generation_id, position),
    FOREIGN KEY(clothing_photo_id, clothing_item_id)
        REFERENCES clothing_photos(id, clothing_item_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS generation_clothing_item
    ON generation_clothing(clothing_item_id);
INSERT OR IGNORE INTO people
    (id,names_json,created_at,updated_at)
VALUES
    ('me','["me"]',strftime('%Y-%m-%dT%H:%M:%fZ','now'),strftime('%Y-%m-%dT%H:%M:%fZ','now'));
COMMIT;
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._unit_connection: ContextVar[sqlite3.Connection | None] = ContextVar(
            "unit_connection", default=None
        )

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.read_connection() as c:
            c.executescript(_SCHEMA)

    def record_counts(self) -> dict[str, int]:
        """Return current row counts for the application data tables."""
        with self.read_connection() as connection:
            return {
                table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in RECORD_TABLES
            }

    def connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        _restrict_database_permissions(self.path)
        c.row_factory = sqlite3.Row
        c.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Generator[sqlite3.Connection, None, None]:
        active = self._unit_connection.get()
        if active is not None:
            yield active
            return
        c = self.connect()
        try:
            c.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield c
        except BaseException:
            c.rollback()
            raise
        else:
            c.commit()
        finally:
            c.close()

    @contextmanager
    def unit_of_work(self) -> Generator[sqlite3.Connection, None, None]:
        """Share one immediate SQLite transaction with nested repository calls.

        This synchronous context is intentionally not suitable for an await boundary:
        SQLite connections are owned by the calling thread and the transaction holds
        the writer lock until the outermost context exits.
        """
        active = self._unit_connection.get()
        if active is not None:
            yield active
            return
        c = self.connect()
        token = self._unit_connection.set(c)
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
        except BaseException:
            c.rollback()
            raise
        else:
            c.commit()
        finally:
            self._unit_connection.reset(token)
            c.close()

    @contextmanager
    def read_connection(self) -> Generator[sqlite3.Connection, None, None]:
        active = self._unit_connection.get()
        if active is not None:
            yield active
            return
        c = self.connect()
        try:
            yield c
        finally:
            c.close()


def _restrict_database_permissions(path: Path) -> None:
    if os.name == "posix" and path.exists():
        path.chmod(0o600)
