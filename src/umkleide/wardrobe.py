"""The single gateway for durable wardrobe and managed-media mutations."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Literal, TypeAlias

from .api import DataUriPhotoSource, ImportPhotoSource, PhotoSource, UrlPhotoSource
from .locking import media_lock
from .models import (
    CleanupResult,
    ClothingItem,
    Generation,
    GenerationStatus,
    Person,
    PersonPhoto,
    utc_now,
    utc_timestamp,
)
from .repositories import UNSET, Repository, RepositoryError, UnsetType, new_id
from .storage import MediaStorage, PreparedImage, StorageError

PhotoReference: TypeAlias = tuple[Literal["person", "clothing"], str]

_MAINTENANCE_INTERVAL_SECONDS = 5.0
_MAINTENANCE_BATCH_SIZE = 100
_MAINTENANCE_PASSES = 4
PersonNamesUpdate = tuple[str, ...] | None | UnsetType
TextUpdate = str | None | UnsetType
MetadataUpdate = Mapping[str, object] | None | UnsetType


class MediaQuotaError(StorageError):
    """The managed-media quota cannot accommodate this write."""


class WardrobeService:
    """Serialize catalog and media writes across local server processes."""

    def __init__(
        self,
        repository: Repository,
        storage: MediaStorage,
        *,
        quota_bytes: int,
        import_root: Path | str,
    ) -> None:
        if quota_bytes < 0:
            raise ValueError("quota_bytes must be non-negative")
        self.repository = repository
        self.storage = storage
        self.quota_bytes = quota_bytes
        self.import_root = import_root
        self._lock = media_lock(storage.root)
        self._batch_depth = 0
        self._staged_paths: list[str] = []
        self._replacements: set[PhotoReference] = set()
        self._last_maintenance_at = 0.0

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        with self._lock:
            yield

    @contextmanager
    def batch(self) -> Generator[None, None, None]:
        """Atomically group synchronous catalog writes and their staged media.

        Do not cross an await or provider/network call while this context is active.
        """
        with self._mutation():
            outer = self._batch_depth == 0
            if outer:
                self._maintain_best_effort(force=True)
                self._staged_paths = []
                self._replacements = set()
            self._batch_depth += 1
            try:
                with self.repository.unit_of_work():
                    yield
                    if outer:
                        self._ensure_quota(0)
            except BaseException:
                if outer:
                    self._delete_staged_paths()
                raise
            finally:
                self._batch_depth -= 1
                if outer:
                    self._staged_paths = []
                    self._replacements = set()
            if outer:
                self._maintain_best_effort(force=True)

    def prepare_user_image(self, source: PhotoSource) -> PreparedImage:
        if isinstance(source, DataUriPhotoSource):
            return self.storage.prepare_user_image(source.data)
        if isinstance(source, ImportPhotoSource):
            return self.storage.prepare_imported_image(self.import_root, source.path)
        if isinstance(source, UrlPhotoSource):
            return self.storage.prepare_remote_image(source.url)
        raise TypeError("unsupported photo source")  # pragma: no cover

    def prepare_provider_image(self, payload: bytes) -> PreparedImage:
        return self.storage.prepare_provider_image(payload)

    def create_person(
        self, *, names: tuple[str, ...], prepared: PreparedImage, description: str | None = None
    ) -> Person:
        with self._mutation():
            self._maintain_best_effort()
            path = self._store(f"person/{new_id()}.jpg", prepared)
            try:
                return self.repository.create_person(
                    names, path, prepared.width, prepared.height, prepared.size_bytes, description
                )
            except BaseException:
                self._rollback_staged_path(path)
                raise
            finally:
                self._maintain_best_effort(force=True)

    def set_person_photo(self, person_id: str, prepared: PreparedImage) -> PersonPhoto:
        person = self.update_person(person_id, prepared=prepared)
        return self.repository.get_person_photo_by_id(person.current_photo_id or "")

    def update_person(
        self,
        person_id: str,
        *,
        names: PersonNamesUpdate = UNSET,
        description: TextUpdate = UNSET,
        prepared: PreparedImage | None = None,
    ) -> Person:
        with self._mutation():
            self._maintain_best_effort()
            path: str | None = None
            try:
                if prepared is not None:
                    old = self.repository.get_person(person_id)
                    path = self._store(
                        f"person/{new_id()}.jpg",
                        prepared,
                        replaces=("person", old.current_photo_id) if old.current_photo_id else None,
                    )
                    return self.repository.update_person(
                        person_id,
                        names=names,
                        description=description,
                        photo_id=new_id(),
                        photo_relative_path=path,
                        width=prepared.width,
                        height=prepared.height,
                        size_bytes=prepared.size_bytes,
                    )
                return self.repository.update_person(
                    person_id, names=names, description=description
                )
            except BaseException:
                if path is not None:
                    self._rollback_staged_path(path)
                raise
            finally:
                self._maintain_best_effort(force=True)

    def delete_person(self, person_id: str) -> CleanupResult:
        with self._mutation():
            plan = self.repository.plan_person_deletion(person_id)
            deleted = self._unlink_all(plan.paths)
            self.repository.commit_person_deletion(plan.target_id)
            self.storage.remove_empty_clothing_directories()
            self._maintain_best_effort(force=True)
            return CleanupResult(
                clothing_items_deleted=plan.clothing_items,
                person_photos_deleted=plan.person_photos,
                generations_deleted=plan.generations,
                bytes_deleted=deleted,
                has_more=False,
            )

    def add_clothing(
        self,
        *,
        person_id: str,
        name: str,
        category: str,
        prepared: PreparedImage,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ClothingItem:
        with self._mutation():
            self._maintain_best_effort()
            item_id, photo_id = new_id(), new_id()
            path = self._store(f"clothing/{item_id}/{photo_id}.jpg", prepared)
            try:
                return self.repository.create_clothing_item(
                    person_id,
                    item_id,
                    name,
                    category,
                    description,
                    metadata,
                    prepared.width,
                    prepared.height,
                    prepared.size_bytes,
                    photo_id=photo_id,
                    photo_relative_path=path,
                )
            except BaseException:
                self._rollback_staged_path(path)
                raise
            finally:
                self._maintain_best_effort(force=True)

    def update_clothing(
        self,
        clothing_item_id: str,
        *,
        name: TextUpdate = UNSET,
        category: TextUpdate = UNSET,
        description: TextUpdate = UNSET,
        metadata: MetadataUpdate = UNSET,
        prepared: PreparedImage | None = None,
    ) -> ClothingItem:
        with self._mutation():
            self._maintain_best_effort()
            path: str | None = None
            try:
                if prepared is not None:
                    old = self.repository.get_clothing_item(clothing_item_id)
                    photo_id = new_id()
                    path = self._store(
                        f"clothing/{clothing_item_id}/{photo_id}.jpg",
                        prepared,
                        replaces=("clothing", old.photo_id),
                    )
                    return self.repository.update_clothing_item(
                        clothing_item_id,
                        name=name,
                        category=category,
                        description=description,
                        metadata=metadata,
                        width=prepared.width,
                        height=prepared.height,
                        size_bytes=prepared.size_bytes,
                        photo_id=photo_id,
                        photo_relative_path=path,
                    )
                return self.repository.update_clothing_item(
                    clothing_item_id,
                    name=name,
                    category=category,
                    description=description,
                    metadata=metadata,
                )
            except BaseException:
                if path is not None:
                    self._rollback_staged_path(path)
                raise
            finally:
                self._maintain_best_effort(force=True)

    def delete_clothing(self, clothing_item_id: str) -> CleanupResult:
        with self._mutation():
            plan = self.repository.plan_clothing_deletion(clothing_item_id)
            deleted = self._unlink_all(plan.paths)
            self.repository.commit_clothing_deletion(plan.target_id)
            self.storage.remove_empty_clothing_directories()
            self._maintain_best_effort(force=True)
            return CleanupResult(
                clothing_items_deleted=1,
                person_photos_deleted=0,
                generations_deleted=0,
                bytes_deleted=deleted,
                has_more=False,
            )

    def delete_generation(self, generation_id: str) -> CleanupResult:
        """Delete terminal user generation content after its result file is unlinked."""
        with self._mutation():
            plan = self.repository.plan_generation_deletion(generation_id)
            deleted = self._unlink_all(plan.paths)
            self.repository.commit_generation_deletion(plan.target_id)
            self._maintain_best_effort(force=True)
            return CleanupResult(
                clothing_items_deleted=0,
                person_photos_deleted=0,
                generations_deleted=1,
                bytes_deleted=deleted,
                has_more=False,
            )

    def save_generation_result(self, generation_id: str, prepared: PreparedImage) -> Generation:
        with self._mutation():
            self._maintain_best_effort()
            generation = self.repository.get_generation(generation_id)
            if generation.status is not GenerationStatus.PROCESSING:
                return generation
            path = self._store(f"generations/{generation_id}.jpg", prepared)
            try:
                return self.repository.update_generation(
                    generation_id, status=GenerationStatus.READY, result_relative_path=path
                )
            except BaseException:
                self._rollback_staged_path(path)
                raise
            finally:
                self._maintain_best_effort(force=True)

    def maintain(self, *, force: bool = False) -> CleanupResult:
        """Best-effort retention; failed unlinks leave rows for a later retry."""
        with self._mutation():
            return self._maintain_best_effort(force=force)

    def _maintain_best_effort(self, *, force: bool = False) -> CleanupResult:
        if self._batch_depth:
            return _empty_cleanup()
        now = monotonic()
        if not force and now - self._last_maintenance_at < _MAINTENANCE_INTERVAL_SECONDS:
            return _empty_cleanup()
        self._last_maintenance_at = now
        cutoff = utc_timestamp(utc_now() - timedelta(days=30))
        people = clothing = generations = deleted = 0
        has_more = False
        try:
            for _ in range(_MAINTENANCE_PASSES):
                candidates = self.repository.maintenance_candidates(
                    cutoff, limit=_MAINTENANCE_BATCH_SIZE
                )
                if not candidates:
                    break
                for candidate in candidates:
                    if candidate.relative_path:
                        deleted += self.storage.delete_image_with_size(candidate.relative_path)
                    self.repository.commit_maintenance_candidate(candidate)
                    if candidate.kind == "person_photo":
                        people += 1
                    elif candidate.kind == "clothing_photo":
                        clothing += 1
                    else:
                        generations += 1
                has_more = len(candidates) == _MAINTENANCE_BATCH_SIZE
            deleted += self._remove_orphans()
            self.storage.remove_empty_clothing_directories()
        except (OSError, sqlite3.Error, StorageError, RepositoryError):
            has_more = True
        return CleanupResult(
            clothing_items_deleted=clothing,
            person_photos_deleted=people,
            generations_deleted=generations,
            bytes_deleted=deleted,
            has_more=has_more,
        )

    def _store(
        self, path: str, prepared: PreparedImage, *, replaces: PhotoReference | None = None
    ) -> str:
        self._ensure_quota(prepared.size_bytes, replaces=replaces)
        self.storage.store(path, prepared)
        if self._batch_depth:
            self._staged_paths.append(path)
            if replaces is not None:
                self._replacements.add(replaces)
        return path

    def _ensure_quota(self, incoming_bytes: int, *, replaces: PhotoReference | None = None) -> None:
        replacements = self._replacements | ({replaces} if replaces is not None else set())
        reclaimable = sum(
            self.repository.reclaimable_person_photo_bytes(photo_id)
            if kind == "person"
            else self.repository.reclaimable_clothing_photo_bytes(photo_id)
            for kind, photo_id in replacements
        )
        if self.storage.managed_usage() + incoming_bytes - reclaimable > self.quota_bytes:
            raise MediaQuotaError(
                "managed-media quota is full; delete unneeded clothing or outfits"
            )

    def _rollback_staged_path(self, path: str) -> None:
        try:
            self.storage.delete_image(path)
        finally:
            if path in self._staged_paths:
                self._staged_paths.remove(path)

    def _delete_staged_paths(self) -> None:
        for path in reversed(self._staged_paths):
            try:
                self.storage.delete_image(path)
            except StorageError:
                pass

    def _unlink_all(self, paths: tuple[str, ...]) -> int:
        return sum(self.storage.delete_image_with_size(path) for path in paths)

    def _remove_orphans(self, *, limit: int = _MAINTENANCE_BATCH_SIZE) -> int:
        """Remove bounded stale JPEGs and interrupted atomic-write temps at maintenance time."""
        known = self.repository.managed_media_paths()
        removed = 0
        examined = 0
        for directory_name in ("person", "clothing", "generations"):
            directory = self.storage.root / directory_name
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if examined >= limit:
                    return removed
                if not path.is_file():
                    continue
                relative_path = path.relative_to(self.storage.root).as_posix()
                if path.suffix.lower() == ".jpg" and relative_path in known:
                    continue
                if path.suffix.lower() != ".jpg" and not path.name.startswith(".write-"):
                    continue
                examined += 1
                removed += path.stat().st_size
                path.unlink()
        return removed


def _empty_cleanup() -> CleanupResult:
    return CleanupResult(
        clothing_items_deleted=0,
        person_photos_deleted=0,
        generations_deleted=0,
        bytes_deleted=0,
        has_more=False,
    )
