"""SQLite persistence for immutable media and bounded offset pages."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from .database import Database
from .models import (
    DEFAULT_PERSON_ID,
    JPEG_MEDIA_TYPE,
    ClothingItem,
    ClothingPhoto,
    ClothingProvenance,
    CursorPage,
    Generation,
    GenerationStatus,
    Person,
    PersonPhoto,
    utc_timestamp,
)


class RepositoryError(ValueError):
    pass


class UnsetType:
    """Public sentinel type for fields intentionally omitted from an update."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()
Row = sqlite3.Row
Model = TypeVar("Model")
TextUpdate = str | None | UnsetType
MetadataUpdate = Mapping[str, object] | None | UnsetType
IntegerUpdate = int | UnsetType


@dataclass(frozen=True)
class MediaDeletionPlan:
    """Rows remain authoritative until every path in this plan is unlinked."""

    target_id: str
    paths: tuple[str, ...]
    person_photos: int = 0
    clothing_items: int = 0
    generations: int = 0


@dataclass(frozen=True)
class MaintenanceCandidate:
    kind: str
    record_id: str
    relative_path: str | None


_MAX_CURSOR_DIGITS = 10
_CLOTHING_SELECT = (
    "SELECT ci.id, ci.person_id, ci.name, ci.category, ci.description, ci.metadata_json, "
    "ci.created_at, "
    "ci.updated_at, cp.id AS photo_id, "
    "cp.relative_path AS current_photo_relative_path, cp.width AS current_photo_width, "
    "cp.height AS current_photo_height, cp.size_bytes AS current_photo_size_bytes"
)


def new_id() -> str:
    return str(uuid.uuid4())


def _cursor(v: str | None) -> int:
    if v is None:
        return 0
    if not isinstance(v, str) or not v.isdecimal() or len(v) > _MAX_CURSOR_DIGITS:
        raise RepositoryError("invalid cursor")
    offset = int(v)
    if offset > 2_147_483_647:
        raise RepositoryError("invalid cursor")
    return offset


class Repository:
    def __init__(self, database: Database) -> None:
        self.database = database

    @contextmanager
    def unit_of_work(self) -> Generator[None, None, None]:
        """Make nested repository mutations one atomic synchronous transaction."""
        with self.database.unit_of_work():
            yield

    def create_person(
        self,
        names: tuple[str, ...],
        path: str,
        width: int,
        height: int,
        size: int,
        description: str | None = None,
    ) -> Person:
        now = utc_timestamp()
        person_id, photo_id = new_id(), new_id()
        person = Person(
            id=person_id,
            names=names,
            description=description,
            current_photo_id=photo_id,
            photo_relative_path=path,
            width=width,
            height=height,
            size_bytes=size,
            created_at=now,
            updated_at=now,
        )
        with self.database.transaction() as c:
            c.execute(
                "INSERT INTO people (id,names_json,created_at,updated_at,description) "
                "VALUES(?,?,?,?,?)",
                (person_id, json.dumps(names), now, now, description),
            )
            c.execute(
                "INSERT INTO person_photos VALUES(?,?,?,?,?,?,?,1,?,NULL)",
                (photo_id, person_id, path, JPEG_MEDIA_TYPE, width, height, size, now),
            )
        return person

    def get_person(self, person_id: str) -> Person:
        return self._one(
            "SELECT p.id,p.names_json,p.description,p.created_at,p.updated_at,"
            "ph.id AS current_photo_id,"
            "ph.relative_path AS photo_relative_path,ph.width,ph.height,ph.size_bytes "
            "FROM people p LEFT JOIN person_photos ph "
            "ON ph.person_id=p.id AND ph.is_current=1 WHERE p.id=?",
            (person_id,),
            _person_from_profile_row,
            "person was not found",
        )

    def list_people(self, *, limit: int = 50, cursor: str | None = None) -> CursorPage[Person]:
        if not 1 <= limit <= 100:
            raise RepositoryError("limit must be between 1 and 100")
        offset = _cursor(cursor)
        with self.database.read_connection() as c:
            rows = c.execute(
                "SELECT p.id,p.names_json,p.description,p.created_at,p.updated_at,"
                "ph.id AS current_photo_id,"
                "ph.relative_path AS photo_relative_path,ph.width,ph.height,ph.size_bytes "
                "FROM people p LEFT JOIN person_photos ph "
                "ON ph.person_id=p.id AND ph.is_current=1 "
                "ORDER BY CASE WHEN p.id='me' THEN 0 ELSE 1 END,p.created_at,p.id LIMIT ? OFFSET ?",
                (limit + 1, offset),
            ).fetchall()
        return CursorPage(
            items=tuple(_person_from_profile_row(row) for row in rows[:limit]),
            next_cursor=str(offset + limit) if len(rows) > limit else None,
        )

    def update_person(
        self,
        person_id: str,
        *,
        names: tuple[str, ...] | None | UnsetType = UNSET,
        description: str | None | UnsetType = UNSET,
        photo_id: str | UnsetType = UNSET,
        photo_relative_path: str | UnsetType = UNSET,
        width: int | UnsetType = UNSET,
        height: int | UnsetType = UNSET,
        size_bytes: int | UnsetType = UNSET,
    ) -> Person:
        old = self.get_person(person_id)
        updated_names = old.names if names is UNSET else names
        updated_description = old.description if description is UNSET else description
        if updated_names is None:
            raise RepositoryError("person names cannot be null")
        updated_names = cast(tuple[str, ...], updated_names)
        if person_id == DEFAULT_PERSON_ID and "me" not in {
            name.casefold() for name in updated_names
        }:
            raise RepositoryError("the default person's names must include 'me'")
        now = utc_timestamp()
        replacing_photo = photo_id is not UNSET
        if replacing_photo and any(
            value is UNSET for value in (photo_relative_path, width, height, size_bytes)
        ):
            raise RepositoryError("a replacement person photo needs complete metadata")
        with self.database.transaction(immediate=replacing_photo) as c:
            c.execute(
                "UPDATE people SET names_json=?,description=?,updated_at=? WHERE id=?",
                (json.dumps(updated_names), updated_description, now, person_id),
            )
            if replacing_photo:
                c.execute(
                    "UPDATE person_photos SET is_current=0,superseded_at=? "
                    "WHERE person_id=? AND is_current=1",
                    (now, person_id),
                )
                c.execute(
                    "INSERT INTO person_photos VALUES(?,?,?,?,?,?,?,1,?,NULL)",
                    (
                        photo_id,
                        person_id,
                        photo_relative_path,
                        JPEG_MEDIA_TYPE,
                        width,
                        height,
                        size_bytes,
                        now,
                    ),
                )
        return self.get_person(person_id)

    def plan_person_deletion(self, person_id: str) -> MediaDeletionPlan:
        if person_id == DEFAULT_PERSON_ID:
            raise RepositoryError("the default person 'me' cannot be deleted")
        self.get_person(person_id)
        with self.database.read_connection() as connection:
            rows = connection.execute(
                "SELECT relative_path FROM person_photos WHERE person_id=? "
                "AND relative_path IS NOT NULL "
                "UNION ALL SELECT cp.relative_path FROM clothing_photos cp JOIN clothing_items ci "
                "ON ci.id=cp.clothing_item_id WHERE ci.person_id=? "
                "UNION ALL SELECT g.result_relative_path FROM generations g JOIN person_photos p "
                "ON p.id=g.person_photo_id WHERE p.person_id=? "
                "AND g.result_relative_path IS NOT NULL",
                (person_id, person_id, person_id),
            ).fetchall()
            people = connection.execute(
                "SELECT count(*) FROM person_photos WHERE person_id=?", (person_id,)
            ).fetchone()[0]
            clothing = connection.execute(
                "SELECT count(*) FROM clothing_items WHERE person_id=?", (person_id,)
            ).fetchone()[0]
            generations = connection.execute(
                "SELECT count(*) FROM generations g JOIN person_photos p ON p.id=g.person_photo_id "
                "WHERE p.person_id=?",
                (person_id,),
            ).fetchone()[0]
        return MediaDeletionPlan(
            person_id, tuple(row[0] for row in rows), people, clothing, generations
        )

    def commit_person_deletion(self, person_id: str) -> None:
        if person_id == DEFAULT_PERSON_ID:
            raise RepositoryError("the default person 'me' cannot be deleted")
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM generations WHERE person_photo_id IN "
                "(SELECT id FROM person_photos WHERE person_id=?)",
                (person_id,),
            )
            connection.execute(
                "DELETE FROM generation_clothing WHERE clothing_item_id IN "
                "(SELECT id FROM clothing_items WHERE person_id=?)",
                (person_id,),
            )
            connection.execute("DELETE FROM clothing_items WHERE person_id=?", (person_id,))
            connection.execute("DELETE FROM people WHERE id=?", (person_id,))

    def set_person_photo(
        self, person_id: str, path: str, width: int, height: int, size: int
    ) -> PersonPhoto:
        photo_id = new_id()
        self.update_person(
            person_id,
            photo_id=photo_id,
            photo_relative_path=path,
            width=width,
            height=height,
            size_bytes=size,
        )
        return self.get_person_photo_by_id(photo_id)

    def get_person_photo(self, person_id: str) -> PersonPhoto:
        return self._one(
            "SELECT * FROM person_photos WHERE person_id=? AND is_current=1",
            (person_id,),
            _person_photo_from_row,
            "no current person photo is set",
        )

    def get_person_photo_by_id(self, i: str) -> PersonPhoto:
        return self._one(
            "SELECT * FROM person_photos WHERE id=?",
            (i,),
            _person_photo_from_row,
            "person photo was not found",
        )

    def reclaimable_person_photo_bytes(self, photo_id: str) -> int:
        """Return reclaim credit only when no retained generation references the photo."""
        with self.database.read_connection() as connection:
            row = connection.execute(
                "SELECT size_bytes FROM person_photos WHERE id=? AND NOT EXISTS "
                "(SELECT 1 FROM generations WHERE person_photo_id=person_photos.id)",
                (photo_id,),
            ).fetchone()
        return int(row["size_bytes"]) if row is not None else 0

    def create_clothing_item(
        self,
        person_id: str,
        item_id: str,
        name: str,
        category: str,
        description: str | None,
        metadata: Mapping[str, object] | None,
        width: int,
        height: int,
        size_bytes: int,
        *,
        photo_id: str | None = None,
        photo_relative_path: str | None = None,
    ) -> ClothingItem:
        now = utc_timestamp()
        self.get_person(person_id)
        pid = photo_id or new_id()
        path = photo_relative_path or f"clothing/{item_id}/{pid}.jpg"
        x = ClothingItem(
            id=item_id,
            person_id=person_id,
            name=name,
            category=category,
            description=description,
            metadata=dict(metadata or {}),
            photo_id=pid,
            photo_relative_path=path,
            width=width,
            height=height,
            size_bytes=size_bytes,
            created_at=now,
            updated_at=now,
        )
        with self.database.transaction() as c:
            c.execute(
                "INSERT INTO clothing_items "
                "(id,person_id,name,category,description,metadata_json,current_photo_id,created_at,"
                "updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    item_id,
                    person_id,
                    name,
                    category,
                    description,
                    json.dumps(x.metadata),
                    pid,
                    now,
                    now,
                ),
            )
            c.execute(
                "INSERT INTO clothing_photos VALUES(?,?,?,?,?,?,?)",
                (pid, item_id, path, width, height, size_bytes, now),
            )
        return x

    def get_clothing_item(self, i: str) -> ClothingItem:
        return self._one(
            _CLOTHING_SELECT + " FROM clothing_items ci JOIN clothing_photos cp "
            "ON cp.id=ci.current_photo_id WHERE ci.id=?",
            (i,),
            _clothing_item_from_row,
            "clothing item was not found",
        )

    def get_clothing_photo(self, i: str) -> ClothingPhoto:
        return self._one(
            "SELECT * FROM clothing_photos WHERE id=?",
            (i,),
            _clothing_photo_from_row,
            "clothing photo was not found",
        )

    def reclaimable_clothing_photo_bytes(self, photo_id: str) -> int:
        """Return reclaim credit only when no retained generation references the photo."""
        with self.database.read_connection() as connection:
            row = connection.execute(
                "SELECT size_bytes FROM clothing_photos WHERE id=? AND NOT EXISTS "
                "(SELECT 1 FROM generation_clothing WHERE clothing_photo_id=clothing_photos.id)",
                (photo_id,),
            ).fetchone()
        return int(row["size_bytes"]) if row is not None else 0

    def list_clothing_items(
        self,
        person_id: str,
        category: str | None = None,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> CursorPage[ClothingItem]:
        return self._items(person_id, category, None, {}, limit, cursor)

    def find_clothing_items(
        self,
        person_id: str,
        query: str | None = None,
        category: str | None = None,
        metadata: Mapping[str, object] | None = None,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> CursorPage[ClothingItem]:
        m = dict(metadata or {})
        return self._items(
            person_id,
            category,
            query,
            m,
            limit,
            cursor,
        )

    def _items(
        self,
        person_id: str,
        category: str | None,
        needle: str | None,
        metadata: dict[str, object],
        limit: int,
        cursor: str | None,
    ) -> CursorPage[ClothingItem]:
        if not 1 <= limit <= 100:
            raise RepositoryError("limit must be between 1 and 100")
        offset = _cursor(cursor)
        where = ["ci.person_id=?"]
        v: list[Any] = [person_id]
        if category:
            where += ["lower(ci.category)=lower(?)"]
            v += [category]
        if needle:
            where += ["(lower(ci.name) LIKE ? OR lower(coalesce(ci.description,'')) LIKE ?)"]
            v += ["%" + needle.casefold() + "%"] * 2
        sql = (
            _CLOTHING_SELECT
            + " FROM clothing_items ci JOIN clothing_photos cp ON cp.id=ci.current_photo_id WHERE "
            + " AND ".join(where)
            + " ORDER BY ci.created_at,ci.id LIMIT ? OFFSET ?"
        )
        if not metadata:
            with self.database.read_connection() as c:
                rows = c.execute(sql, [*v, limit + 1, offset]).fetchall()
            return CursorPage(
                items=tuple(_clothing_item_from_row(row) for row in rows[:limit]),
                next_cursor=str(offset + limit) if len(rows) > limit else None,
            )

        # Metadata filtering happens in Python.  The offset therefore tracks every
        # SQL row examined, rather than just matching rows, so a later page cannot
        # skip or duplicate a metadata match.
        got: list[ClothingItem] = []
        scanned = offset
        while True:
            with self.database.read_connection() as c:
                rows = c.execute(sql, [*v, limit + 1, scanned]).fetchall()
            if not rows:
                return CursorPage(items=tuple(got))
            for row in rows:
                item = _clothing_item_from_row(row)
                if all(item.metadata.get(key) == value for key, value in metadata.items()):
                    if len(got) == limit:
                        return CursorPage(items=tuple(got), next_cursor=str(scanned))
                    got.append(item)
                scanned += 1
            if len(rows) < limit + 1:
                return CursorPage(items=tuple(got))

    def update_clothing_item(
        self,
        i: str,
        *,
        name: TextUpdate = UNSET,
        category: TextUpdate = UNSET,
        description: TextUpdate = UNSET,
        metadata: MetadataUpdate = UNSET,
        width: IntegerUpdate = UNSET,
        height: IntegerUpdate = UNSET,
        size_bytes: IntegerUpdate = UNSET,
        photo_id: str | UnsetType = UNSET,
        photo_relative_path: str | UnsetType = UNSET,
    ) -> ClothingItem:
        old = self.get_clothing_item(i)
        now = utc_timestamp()
        d = {
            "name": old.name if name is UNSET else name,
            "category": old.category if category is UNSET else category,
            "description": old.description if description is UNSET else description,
            "metadata": (
                old.metadata
                if isinstance(metadata, UnsetType)
                else None
                if metadata is None
                else dict(metadata)
            ),
            "width": old.width if width is UNSET else width,
            "height": old.height if height is UNSET else height,
            "size_bytes": old.size_bytes if size_bytes is UNSET else size_bytes,
            "photo_id": old.photo_id if photo_id is UNSET else photo_id,
            "photo_relative_path": old.photo_relative_path
            if photo_relative_path is UNSET
            else photo_relative_path,
        }
        if d["name"] is None or d["category"] is None or d["metadata"] is None:
            raise RepositoryError("required clothing fields cannot be null")
        x = ClothingItem(
            id=i,
            person_id=old.person_id,
            created_at=old.created_at,
            updated_at=now,
            **d,
        )
        with self.database.transaction() as c:
            if photo_id is not UNSET:
                c.execute(
                    "INSERT INTO clothing_photos VALUES(?,?,?,?,?,?,?)",
                    (x.photo_id, i, x.photo_relative_path, x.width, x.height, x.size_bytes, now),
                )
            c.execute(
                "UPDATE clothing_items SET name=?, category=?, description=?, metadata_json=?, "
                "current_photo_id=?, updated_at=? WHERE id=?",
                (
                    x.name,
                    x.category,
                    x.description,
                    json.dumps(x.metadata),
                    x.photo_id,
                    now,
                    i,
                ),
            )
        return x

    def plan_clothing_deletion(self, clothing_item_id: str) -> MediaDeletionPlan:
        self.get_clothing_item(clothing_item_id)
        with self.database.read_connection() as connection:
            paths = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT relative_path FROM clothing_photos WHERE clothing_item_id=?",
                    (clothing_item_id,),
                )
            )
        return MediaDeletionPlan(clothing_item_id, paths, clothing_items=1)

    def commit_clothing_deletion(self, clothing_item_id: str) -> None:
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM generation_clothing WHERE clothing_item_id=?", (clothing_item_id,)
            )
            connection.execute("DELETE FROM clothing_items WHERE id=?", (clothing_item_id,))

    def plan_generation_deletion(self, generation_id: str) -> MediaDeletionPlan:
        generation = self.get_generation(generation_id)
        if generation.status in {GenerationStatus.SUBMISSION_UNKNOWN, GenerationStatus.PROCESSING}:
            raise RepositoryError("an active generation cannot be deleted")
        return MediaDeletionPlan(
            target_id=generation_id,
            paths=(generation.result_relative_path,) if generation.result_relative_path else (),
            generations=1,
        )

    def commit_generation_deletion(self, generation_id: str) -> None:
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM generations WHERE id=? AND status IN ('ready','failed')",
                (generation_id,),
            )

    def maintenance_candidates(
        self, cutoff: str, *, limit: int = 100
    ) -> tuple[MaintenanceCandidate, ...]:
        """Return retryable file-first cleanup work without mutating catalog rows."""
        if not 1 <= limit <= 500:
            raise RepositoryError("limit must be between 1 and 500")
        with self.database.read_connection() as connection:
            rows = connection.execute(
                "SELECT 'person_photo' AS kind,p.id,p.relative_path,p.created_at "
                "FROM person_photos p WHERE p.is_current=0 AND p.relative_path IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM generations g WHERE g.person_photo_id=p.id) "
                "UNION ALL "
                "SELECT 'clothing_photo',cp.id,cp.relative_path,cp.created_at "
                "FROM clothing_photos cp JOIN clothing_items ci ON ci.id=cp.clothing_item_id "
                "WHERE ci.current_photo_id<>cp.id AND NOT EXISTS "
                "(SELECT 1 FROM generation_clothing gc "
                "WHERE gc.clothing_photo_id=cp.id) "
                "UNION ALL "
                "SELECT 'generation',g.id,g.result_relative_path,g.created_at FROM generations g "
                "WHERE g.status IN ('failed','submission_unknown') AND g.created_at<? "
                "ORDER BY created_at,id LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return tuple(
            MaintenanceCandidate(row["kind"], row["id"], row["relative_path"]) for row in rows
        )

    def commit_maintenance_candidate(self, candidate: MaintenanceCandidate) -> None:
        with self.database.transaction(immediate=True) as connection:
            if candidate.kind == "person_photo":
                connection.execute(
                    "DELETE FROM person_photos WHERE id=? AND is_current=0 "
                    "AND NOT EXISTS (SELECT 1 FROM generations WHERE person_photo_id=?)",
                    (candidate.record_id, candidate.record_id),
                )
            elif candidate.kind == "clothing_photo":
                connection.execute(
                    "DELETE FROM clothing_photos WHERE id=? AND id<>(SELECT current_photo_id "
                    "FROM clothing_items WHERE id=clothing_photos.clothing_item_id) "
                    "AND NOT EXISTS (SELECT 1 FROM generation_clothing WHERE clothing_photo_id=?)",
                    (candidate.record_id, candidate.record_id),
                )
            elif candidate.kind == "generation":
                connection.execute(
                    "DELETE FROM generations WHERE id=? "
                    "AND status IN ('failed','submission_unknown')",
                    (candidate.record_id,),
                )
            else:  # pragma: no cover - repository-created candidates are exhaustive
                raise RepositoryError("unknown maintenance candidate")

    def managed_media_paths(self) -> frozenset[str]:
        """Return all catalog-owned media paths for interrupted-write orphan recovery."""
        with self.database.read_connection() as connection:
            rows = connection.execute(
                "SELECT relative_path FROM person_photos WHERE relative_path IS NOT NULL "
                "UNION SELECT relative_path FROM clothing_photos "
                "UNION SELECT result_relative_path FROM generations "
                "WHERE result_relative_path IS NOT NULL "
                "UNION SELECT 'generations/' || id || '.jpg' FROM generations "
                "WHERE status='processing'"
            ).fetchall()
        return frozenset(row["relative_path"] for row in rows)

    def create_generation(self, g: Generation) -> None:
        with self.database.transaction() as c:
            c.execute(
                "INSERT INTO generations "
                "(id,status,prompt,person_photo_id,model,provider_request_id,polling_url,"
                "result_relative_path,error_json,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                _genvals(g),
            )
            c.executemany(
                "INSERT INTO generation_clothing "
                "(generation_id,position,clothing_item_id,clothing_photo_id,snapshot_name) "
                "VALUES(?,?,?,?,?)",
                [
                    (
                        g.id,
                        position,
                        ref.clothing_item_id,
                        ref.clothing_photo_id,
                        ref.name,
                    )
                    for position, ref in enumerate(g.selected_clothing)
                ],
            )

    def get_generation(self, i: str) -> Generation:
        with self.database.read_connection() as c:
            row = c.execute(
                "SELECT g.*,p.person_id FROM generations g JOIN person_photos p "
                "ON p.id=g.person_photo_id WHERE g.id=?",
                (i,),
            ).fetchone()
            if row is None:
                raise RepositoryError("generation was not found")
            return _generation_from_row(row, self._generation_clothing(c, (i,)).get(i, ()))

    def list_generations(
        self,
        person_id: str,
        status: GenerationStatus | None = None,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> CursorPage[Generation]:
        if not 1 <= limit <= 100:
            raise RepositoryError("limit must be between 1 and 100")
        offset = _cursor(cursor)
        where = ["p.person_id=?"]
        v = [person_id]
        if status:
            where += ["g.status=?"]
            v += [status.value]
        sql = (
            "SELECT g.*,p.person_id FROM generations g JOIN person_photos p "
            "ON p.id=g.person_photo_id"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY g.created_at DESC,g.id DESC LIMIT ? OFFSET ?"
        )
        v += [limit + 1, offset]
        with self.database.read_connection() as c:
            rows = c.execute(sql, v).fetchall()
            clothing = self._generation_clothing(c, tuple(row["id"] for row in rows[:limit]))
        return CursorPage(
            items=tuple(
                _generation_from_row(row, clothing.get(row["id"], ())) for row in rows[:limit]
            ),
            next_cursor=str(offset + limit) if len(rows) > limit else None,
        )

    def processing_generation_ids(self) -> tuple[str, ...]:
        with self.database.read_connection() as connection:
            rows = connection.execute(
                "SELECT id FROM generations WHERE status='processing' ORDER BY updated_at,id"
            ).fetchall()
        return tuple(row["id"] for row in rows)

    def update_generation(
        self,
        i: str,
        *,
        status: GenerationStatus,
        provider_request_id: str | None | UnsetType = UNSET,
        polling_url: str | None | UnsetType = UNSET,
        result_relative_path: str | None | UnsetType = UNSET,
        error: Mapping[str, object] | None = None,
    ) -> Generation:
        old = self.get_generation(i)
        x = old.model_copy(
            update={
                "status": status,
                "provider_request_id": old.provider_request_id
                if provider_request_id is UNSET
                else provider_request_id,
                "polling_url": old.polling_url if polling_url is UNSET else polling_url,
                "result_relative_path": old.result_relative_path
                if result_relative_path is UNSET
                else result_relative_path,
                "error": dict(error) if error else None,
                "updated_at": utc_timestamp(),
            }
        )
        with self.database.transaction() as c:
            c.execute(
                "UPDATE generations SET status=?, provider_request_id=?, polling_url=?, "
                "result_relative_path=?, error_json=?, updated_at=? WHERE id=?",
                (
                    x.status.value,
                    x.provider_request_id,
                    x.polling_url,
                    x.result_relative_path,
                    json.dumps(x.error) if x.error else None,
                    x.updated_at,
                    i,
                ),
            )
        return x

    @staticmethod
    def _generation_clothing(
        c: sqlite3.Connection, generation_ids: tuple[str, ...]
    ) -> dict[str, tuple[ClothingProvenance, ...]]:
        if not generation_ids:
            return {}
        placeholders = ",".join("?" for _ in generation_ids)
        rows = c.execute(
            "SELECT generation_id,clothing_item_id,clothing_photo_id,snapshot_name "
            f"FROM generation_clothing WHERE generation_id IN ({placeholders}) "
            "ORDER BY generation_id,position",
            generation_ids,
        )
        result: dict[str, list[ClothingProvenance]] = {}
        for row in rows:
            result.setdefault(row["generation_id"], []).append(
                ClothingProvenance(
                    clothing_item_id=row["clothing_item_id"],
                    clothing_photo_id=row["clothing_photo_id"],
                    name=row["snapshot_name"],
                )
            )
        return {generation_id: tuple(refs) for generation_id, refs in result.items()}

    def _one(self, sql: str, v: tuple[Any, ...], fn: Callable[[Row], Model], msg: str) -> Model:
        with self.database.read_connection() as c:
            r = c.execute(sql, v).fetchone()
        if r is None:
            raise RepositoryError(msg)
        return fn(r)


def _person_photo_from_row(r: Row) -> PersonPhoto:
    return PersonPhoto(
        id=r["id"],
        person_id=r["person_id"],
        relative_path=r["relative_path"],
        media_type=r["media_type"],
        width=r["width"],
        height=r["height"],
        size_bytes=r["size_bytes"],
        is_current=bool(r["is_current"]),
        created_at=r["created_at"],
        superseded_at=r["superseded_at"],
    )


def _clothing_photo_from_row(r: Row) -> ClothingPhoto:
    return ClothingPhoto(**dict(r))


def _clothing_item_from_row(r: Row) -> ClothingItem:
    return ClothingItem(
        id=r["id"],
        person_id=r["person_id"],
        name=r["name"],
        category=r["category"],
        description=r["description"],
        metadata=json.loads(r["metadata_json"]),
        photo_id=r["photo_id"],
        photo_relative_path=r["current_photo_relative_path"],
        width=r["current_photo_width"],
        height=r["current_photo_height"],
        size_bytes=r["current_photo_size_bytes"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def _generation_from_row(r: Row, selected_clothing: tuple[ClothingProvenance, ...]) -> Generation:
    return Generation(
        id=r["id"],
        person_id=r["person_id"],
        status=GenerationStatus(r["status"]),
        prompt=r["prompt"],
        person_photo_id=r["person_photo_id"],
        selected_clothing=selected_clothing,
        model=r["model"],
        provider_request_id=r["provider_request_id"],
        polling_url=r["polling_url"],
        result_relative_path=r["result_relative_path"],
        error=json.loads(r["error_json"]) if r["error_json"] else None,
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def _genvals(g: Generation) -> tuple[Any, ...]:
    return (
        g.id,
        g.status.value,
        g.prompt,
        g.person_photo_id,
        g.model,
        g.provider_request_id,
        g.polling_url,
        g.result_relative_path,
        json.dumps(g.error) if g.error else None,
        g.created_at,
        g.updated_at,
    )


def _person_from_profile_row(r: Row) -> Person:
    return Person(
        id=r["id"],
        names=tuple(json.loads(r["names_json"])),
        description=r["description"],
        current_photo_id=r["current_photo_id"],
        photo_relative_path=r["photo_relative_path"],
        width=r["width"],
        height=r["height"],
        size_bytes=r["size_bytes"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )
