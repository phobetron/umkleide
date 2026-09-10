import sqlite3
from pathlib import Path

import pytest

from umkleide.config import DATABASE_FILENAME
from umkleide.database import RECORD_TABLES, Database
from umkleide.models import JPEG_MEDIA_TYPE


def test_schema_initialization_is_idempotent_and_enforces_relational_references(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / DATABASE_FILENAME)
    database.initialize()
    database.initialize()

    expected_counts = dict.fromkeys(RECORD_TABLES, 0)
    expected_counts["people"] = 1
    assert database.record_counts() == expected_counts

    with database.read_connection() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        default = connection.execute(
            "SELECT id,names_json,description FROM people WHERE id='me'"
        ).fetchone()
        assert tuple(default) == ("me", '["me"]', None)

    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO people VALUES(?,?,?,?,?)",
            ("profile", '["Person"]', "Usually wears size M", "now", "now"),
        )
        connection.execute(
            "INSERT INTO person_photos VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "person",
                "profile",
                "person/person.jpg",
                JPEG_MEDIA_TYPE,
                1,
                1,
                1,
                1,
                "now",
                None,
            ),
        )
        connection.execute(
            "INSERT INTO clothing_items VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "coat",
                "profile",
                "Coat",
                "outerwear",
                None,
                "{}",
                "coat-v1",
                "now",
                "now",
            ),
        )
        connection.execute(
            "INSERT INTO clothing_photos VALUES(?,?,?,?,?,?,?)",
            ("coat-v1", "coat", "clothing/coat/coat-v1.jpg", 1, 1, 1, "now"),
        )
        connection.execute(
            "INSERT INTO generations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                "look",
                "ready",
                "prompt",
                "person",
                "flux-2-pro",
                None,
                None,
                None,
                None,
                "now",
                "now",
            ),
        )
        connection.execute(
            "INSERT INTO generation_clothing VALUES(?,?,?,?,?)",
            ("look", 0, "coat", "coat-v1", "Coat"),
        )

    with database.transaction() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO generation_clothing VALUES(?,?,?,?,?)",
                ("look", 1, "coat", "missing-photo", "Coat"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM person_photos WHERE id='person'")
