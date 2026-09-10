from __future__ import annotations

from pathlib import Path

import pytest

from umkleide.config import DATABASE_FILENAME
from umkleide.database import Database
from umkleide.models import ClothingProvenance, Generation, GenerationStatus
from umkleide.repositories import Repository, RepositoryError
from umkleide.storage import MediaStorage, PreparedImage
from umkleide.wardrobe import WardrobeService


@pytest.fixture
def wardrobe(tmp_path: Path) -> WardrobeService:
    database = Database(tmp_path / DATABASE_FILENAME)
    database.initialize()
    return WardrobeService(
        Repository(database),
        MediaStorage(tmp_path / "media"),
        quota_bytes=1_000,
        import_root=tmp_path / "imports",
    )


def _prepared(size: int = 10) -> PreparedImage:
    return PreparedImage(b"x" * size, 1, 1)


def _generation(
    person_id: str,
    person_photo_id: str,
    generation_id: str,
    *,
    status: GenerationStatus,
    path: str | None,
    references: tuple[ClothingProvenance, ...] = (),
):
    return Generation(
        id=generation_id,
        person_id=person_id,
        status=status,
        prompt="outfit",
        person_photo_id=person_photo_id,
        selected_clothing=references,
        result_relative_path=path,
        created_at="2020-01-01T00:00:00Z",
        updated_at="2020-01-01T00:00:00Z",
    )


def test_prepared_media_is_persisted_and_rolled_back_on_clothing_update_failure(
    wardrobe: WardrobeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    person = wardrobe.set_person_photo("me", _prepared())
    item = wardrobe.add_clothing(
        person_id="me", name="Coat", category="outerwear", prepared=_prepared()
    )
    updated = wardrobe.update_clothing(item.id, prepared=_prepared(11))
    assert person.relative_path is not None
    assert wardrobe.storage.read_bytes(person.relative_path) == b"x" * 10
    assert updated.size_bytes == 11
    assert wardrobe.storage.read_bytes(updated.photo_relative_path) == b"x" * 11

    old_paths = set((wardrobe.storage.root / "clothing").rglob("*.jpg"))

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RepositoryError("database failed")

    monkeypatch.setattr(wardrobe.repository, "update_clothing_item", fail)
    with pytest.raises(RepositoryError, match="database failed"):
        wardrobe.update_clothing(item.id, prepared=_prepared(12))
    assert set((wardrobe.storage.root / "clothing").rglob("*.jpg")) == old_paths


def test_batch_rolls_back_catalog_and_staged_media(wardrobe: WardrobeService) -> None:
    with pytest.raises(RuntimeError):
        with wardrobe.batch():
            wardrobe.add_clothing(
                person_id="me", name="Coat", category="outerwear", prepared=_prepared(10)
            )
            raise RuntimeError("abort batch")

    assert wardrobe.repository.database.record_counts()["clothing_items"] == 0
    assert wardrobe.storage.managed_usage() == 0


def test_quota_refuses_media_when_full(tmp_path: Path) -> None:
    database = Database(tmp_path / DATABASE_FILENAME)
    database.initialize()
    repository = Repository(database)
    wardrobe = WardrobeService(
        repository,
        MediaStorage(tmp_path / "media"),
        quota_bytes=20,
        import_root=tmp_path / "imports",
    )
    wardrobe.set_person_photo("me", _prepared(1))
    item = wardrobe.add_clothing(
        person_id="me", name="Kept", category="top", prepared=_prepared(19)
    )

    with pytest.raises(ValueError, match="quota"):
        wardrobe.add_clothing(
            person_id="me", name="Too much", category="top", prepared=_prepared(1)
        )

    assert repository.get_clothing_item(item.id).id == item.id


def test_replacement_credit_requires_an_unreferenced_old_photo(wardrobe: WardrobeService) -> None:
    repository = wardrobe.repository
    wardrobe.quota_bytes = 10
    person = wardrobe.set_person_photo("me", _prepared(10))
    repository.create_generation(
        _generation("me", person.id, "kept", status=GenerationStatus.READY, path=None)
    )

    with pytest.raises(ValueError, match="quota"):
        wardrobe.set_person_photo("me", _prepared(10))
    assert repository.get_person_photo("me").id == person.id
    assert wardrobe.storage.managed_usage() == 10


def test_replacement_credit_allows_unreferenced_photo_at_quota(wardrobe: WardrobeService) -> None:
    wardrobe.quota_bytes = 10
    old = wardrobe.set_person_photo("me", _prepared(10))
    replacement = wardrobe.set_person_photo("me", _prepared(10))

    assert replacement.id != old.id
    assert wardrobe.storage.managed_usage() == 10
    with pytest.raises(RepositoryError):
        wardrobe.repository.get_person_photo_by_id(old.id)


def test_batch_rolls_back_mixed_replacement_credit_when_input_is_referenced(
    wardrobe: WardrobeService,
) -> None:
    wardrobe.quota_bytes = 20
    person = wardrobe.set_person_photo("me", _prepared(10))
    item = wardrobe.add_clothing(
        person_id="me", name="Coat", category="outerwear", prepared=_prepared(10)
    )
    wardrobe.repository.create_generation(
        _generation("me", person.id, "kept", status=GenerationStatus.READY, path=None)
    )

    with pytest.raises(ValueError, match="quota"):
        with wardrobe.batch():
            wardrobe.update_clothing(item.id, prepared=_prepared(10))
            wardrobe.set_person_photo("me", _prepared(10))

    assert wardrobe.repository.get_clothing_item(item.id).photo_id == item.photo_id
    assert wardrobe.repository.get_person_photo("me").id == person.id
    assert wardrobe.storage.managed_usage() == 20


def test_automatic_maintenance_prunes_only_unreferenced_and_expired_media(
    wardrobe: WardrobeService,
) -> None:
    storage, repository = wardrobe.storage, wardrobe.repository
    for path in (
        "person/free.jpg",
        "person/old.jpg",
        "person/current.jpg",
        "generations/ready.jpg",
        "generations/pending.jpg",
    ):
        storage.store(path, _prepared(3))
    repository.set_person_photo("me", "person/free.jpg", 1, 1, 3)
    old = repository.set_person_photo("me", "person/old.jpg", 1, 1, 3)
    current = repository.set_person_photo("me", "person/current.jpg", 1, 1, 3)
    repository.create_generation(
        _generation(
            "me",
            old.id,
            "ready",
            status=GenerationStatus.READY,
            path="generations/ready.jpg",
        )
    )
    repository.create_generation(
        _generation(
            "me",
            old.id,
            "pending",
            status=GenerationStatus.PROCESSING,
            path="generations/pending.jpg",
        )
    )
    with repository.database.transaction() as connection:
        connection.execute("UPDATE person_photos SET created_at='2020-01-01'")

    first = wardrobe.maintain()
    second = wardrobe.maintain()

    assert first.bytes_deleted == 3
    assert not first.has_more
    assert second.bytes_deleted == 0
    assert not (storage.root / "person/free.jpg").exists()
    assert (storage.root / "person/old.jpg").exists()
    assert repository.get_generation("ready").status is GenerationStatus.READY
    assert current.relative_path is not None
    assert (storage.root / current.relative_path).exists()
    assert (storage.root / "generations/pending.jpg").exists()


def test_automatic_maintenance_deletes_expired_failed_generation_without_result_media(
    wardrobe: WardrobeService,
) -> None:
    repository = wardrobe.repository
    person = wardrobe.set_person_photo("me", _prepared(1))
    repository.create_generation(
        _generation("me", person.id, "failed", status=GenerationStatus.FAILED, path=None)
    )

    cleanup = wardrobe.maintain(force=True)

    assert cleanup.generations_deleted == 1
    assert cleanup.bytes_deleted == 0
    with pytest.raises(RepositoryError):
        repository.get_generation("failed")


def test_delete_clothing_removes_media_and_generation_references(
    wardrobe: WardrobeService,
) -> None:
    storage, repository = wardrobe.storage, wardrobe.repository
    person = wardrobe.set_person_photo("me", _prepared(1))
    item = wardrobe.add_clothing(
        person_id="me", name="Coat", category="outerwear", prepared=_prepared(10)
    )
    repository.create_generation(
        Generation(
            id="look",
            person_id="me",
            status=GenerationStatus.READY,
            prompt="outfit",
            person_photo_id=person.id,
            selected_clothing=(
                ClothingProvenance(
                    clothing_item_id=item.id,
                    clothing_photo_id=item.photo_id,
                    name=item.name,
                ),
            ),
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
    )
    deleted = wardrobe.delete_clothing(item.id)

    assert deleted.clothing_items_deleted == 1
    assert deleted.bytes_deleted == 10
    with pytest.raises(RepositoryError):
        repository.get_clothing_item(item.id)
    assert repository.get_generation("look").selected_clothing == ()
    assert not (storage.root / item.photo_relative_path).exists()


def test_delete_person_removes_their_data_but_keeps_other_people(
    wardrobe: WardrobeService,
) -> None:
    storage, repository = wardrobe.storage, wardrobe.repository
    with pytest.raises(RepositoryError, match="cannot be deleted"):
        wardrobe.delete_person("me")
    me_photo = wardrobe.set_person_photo("me", _prepared(1))
    kept = wardrobe.add_clothing(person_id="me", name="Kept", category="top", prepared=_prepared(2))
    susan = wardrobe.create_person(names=("Susan",), prepared=_prepared(3))
    susan_photo = repository.get_person_photo(susan.id)
    borrowed = wardrobe.add_clothing(
        person_id=susan.id, name="Borrowed", category="top", prepared=_prepared(4)
    )
    storage.store("generations/susan.jpg", _prepared(5))
    repository.create_generation(
        _generation(
            susan.id,
            susan_photo.id,
            "susan-look",
            status=GenerationStatus.READY,
            path="generations/susan.jpg",
        )
    )
    repository.create_generation(
        _generation(
            "me",
            me_photo.id,
            "borrowed-look",
            status=GenerationStatus.READY,
            path=None,
            references=(
                ClothingProvenance(
                    clothing_item_id=borrowed.id,
                    clothing_photo_id=borrowed.photo_id,
                    name=borrowed.name,
                ),
            ),
        )
    )

    deleted = wardrobe.delete_person(susan.id)

    assert (deleted.person_photos_deleted, deleted.clothing_items_deleted) == (1, 1)
    assert (deleted.generations_deleted, deleted.bytes_deleted) == (1, 12)
    with pytest.raises(RepositoryError):
        repository.get_person(susan.id)
    with pytest.raises(RepositoryError):
        repository.get_clothing_item(borrowed.id)
    with pytest.raises(RepositoryError):
        repository.get_generation("susan-look")
    assert repository.get_person("me").id == "me"
    assert repository.get_clothing_item(kept.id).id == kept.id
    assert repository.get_generation("borrowed-look").selected_clothing == ()


def test_combined_person_update_rolls_back_text_when_photo_exceeds_quota(wardrobe):
    person = wardrobe.set_person_photo("me", _prepared(10))
    wardrobe.quota_bytes = 10
    with pytest.raises(ValueError, match="quota"):
        wardrobe.update_person(
            "me", names=("me", "Changed"), description="Changed", prepared=_prepared(11)
        )
    current = wardrobe.repository.get_person("me")
    assert current.names == ("me",)
    assert current.description is None
    assert current.current_photo_id == person.id


def test_partial_file_deletion_can_be_retried(wardrobe, monkeypatch):
    from umkleide.storage import StorageError

    item = wardrobe.add_clothing(person_id="me", name="Coat", category="top", prepared=_prepared())
    original = wardrobe.storage.delete_image_with_size
    monkeypatch.setattr(
        wardrobe.storage,
        "delete_image_with_size",
        lambda path: (_ for _ in ()).throw(StorageError("cannot unlink")),
    )
    with pytest.raises(StorageError):
        wardrobe.delete_clothing(item.id)
    assert wardrobe.repository.get_clothing_item(item.id).id == item.id
    assert (wardrobe.storage.root / item.photo_relative_path).exists()
    monkeypatch.setattr(wardrobe.storage, "delete_image_with_size", original)
    assert wardrobe.delete_clothing(item.id).clothing_items_deleted == 1
    assert not (wardrobe.storage.root / item.photo_relative_path).exists()


def test_maintenance_recovers_orphans_without_removing_owned_media(wardrobe):
    person = wardrobe.set_person_photo("me", _prepared())
    wardrobe.storage.store("person/orphan.jpg", _prepared(3))
    temporary = wardrobe.storage.root / "person/.write-interrupted"
    temporary.write_bytes(b"temp")
    wardrobe.maintain(force=True)
    assert not temporary.exists()
    assert not (wardrobe.storage.root / "person/orphan.jpg").exists()
    assert person.relative_path is not None
    assert wardrobe.storage.read_bytes(person.relative_path) == _prepared().data


def test_expired_unknown_request_releases_its_superseded_photo(wardrobe):
    old = wardrobe.set_person_photo("me", _prepared())
    wardrobe.repository.create_generation(
        _generation("me", old.id, "unknown", status=GenerationStatus.SUBMISSION_UNKNOWN, path=None)
    )
    wardrobe.repository.set_person_photo("me", "person/new.jpg", 1, 1, 10)
    wardrobe.storage.store("person/new.jpg", _prepared())
    wardrobe.maintain(force=True)
    with pytest.raises(RepositoryError):
        wardrobe.repository.get_generation("unknown")
    with pytest.raises(RepositoryError):
        wardrobe.repository.get_person_photo_by_id(old.id)
    assert wardrobe.storage.read_bytes("person/new.jpg") == _prepared().data


def test_replacement_credit_applies_to_whole_local_batch(wardrobe):
    wardrobe.quota_bytes = 15
    wardrobe.set_person_photo("me", _prepared(10))
    with wardrobe.batch():
        wardrobe.set_person_photo("me", _prepared(10))
        wardrobe.add_clothing(person_id="me", name="Coat", category="top", prepared=_prepared(5))
    assert wardrobe.storage.managed_usage() == 15
