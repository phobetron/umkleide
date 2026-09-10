from __future__ import annotations

from pathlib import Path

import pytest

from umkleide.config import DATABASE_FILENAME
from umkleide.database import Database
from umkleide.models import DEFAULT_PERSON_ID, ClothingProvenance, Generation, GenerationStatus
from umkleide.repositories import UNSET, Repository, RepositoryError


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    database = Database(tmp_path / DATABASE_FILENAME)
    database.initialize()
    return Repository(database)


def _item(
    repository: Repository,
    person_id: str,
    item_id: str,
    *,
    metadata: dict[str, str] | None = None,
):
    return repository.create_clothing_item(
        person_id, item_id, item_id.title(), "top", "original", metadata or {}, 2, 3, 4
    )


def _at(repository: Repository, item_id: str, timestamp: str) -> None:
    with repository.database.transaction() as connection:
        connection.execute(
            "UPDATE clothing_items SET created_at=?, updated_at=? WHERE id=?",
            (timestamp, timestamp, item_id),
        )


def _generation(
    generation_id: str,
    person_id: str,
    person_photo_id: str,
    references: tuple[ClothingProvenance, ...],
) -> Generation:
    return Generation(
        id=generation_id,
        person_id=person_id,
        status=GenerationStatus.READY,
        prompt="outfit",
        person_photo_id=person_photo_id,
        selected_clothing=references,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def test_repository_preserves_photo_history_and_relational_generation_pages(
    repository: Repository,
) -> None:
    person = repository.set_person_photo("me", "person/person.jpg", 1, 1, 1)
    first = _item(repository, "me", "first", metadata={"colour": "blue"})
    second = _item(repository, "me", "second", metadata={"colour": "blue"})
    _item(repository, "me", "third", metadata={"colour": "red"})
    for index, item_id in enumerate(("first", "second", "third"), start=1):
        _at(repository, item_id, f"2026-01-0{index}T00:00:00Z")

    replacement = repository.update_clothing_item(
        "first",
        photo_id="first-v2",
        photo_relative_path="clothing/first/first-v2.jpg",
        width=5,
        height=6,
        size_bytes=7,
    )
    repository.create_generation(
        _generation(
            "look-1",
            "me",
            person.id,
            (
                ClothingProvenance(
                    clothing_item_id="second", clothing_photo_id=second.photo_id, name="Second"
                ),
                ClothingProvenance(
                    clothing_item_id="first", clothing_photo_id=first.photo_id, name="First"
                ),
            ),
        )
    )
    repository.create_generation(_generation("look-2", "me", person.id, ()))

    assert repository.get_clothing_photo(first.photo_id).relative_path == first.photo_relative_path
    assert repository.get_clothing_item("first").photo_id == replacement.photo_id
    assert [
        reference.name for reference in repository.get_generation("look-1").selected_clothing
    ] == [
        "Second",
        "First",
    ]
    generations = repository.list_generations("me", limit=1)
    assert [generation.id for generation in generations.items] == ["look-2"]
    assert [
        generation.id
        for generation in repository.list_generations(
            "me", limit=1, cursor=generations.next_cursor
        ).items
    ] == ["look-1"]

    repository.commit_clothing_deletion(repository.plan_clothing_deletion("third").target_id)
    assert [item.id for item in repository.list_clothing_items("me", limit=10).items] == [
        "first",
        "second",
    ]
    with pytest.raises(RepositoryError):
        repository.get_clothing_item("third")
    matches = repository.find_clothing_items("me", metadata={"colour": "blue"}, limit=1)
    assert [item.id for item in matches.items] == ["first"]
    assert [
        item.id
        for item in repository.find_clothing_items(
            "me",
            metadata={"colour": "blue"},
            limit=1,
            cursor=matches.next_cursor,
        ).items
    ] == ["second"]

    repository.commit_clothing_deletion(repository.plan_clothing_deletion("first").target_id)
    assert [
        ref.clothing_item_id for ref in repository.get_generation("look-1").selected_clothing
    ] == ["second"]


def test_repository_update_merges_omitted_fields_and_clears_explicit_null(
    repository: Repository,
) -> None:
    _item(repository, "me", "coat")

    unchanged = repository.update_clothing_item("coat", name="Renamed", description=UNSET)
    cleared = repository.update_clothing_item("coat", description=None)

    assert unchanged.description == "original"
    assert cleared.description is None


def test_unit_of_work_rolls_back_nested_repository_mutations(repository: Repository) -> None:
    with pytest.raises(RuntimeError):
        with repository.unit_of_work():
            _item(repository, "me", "rolled-back")
            raise RuntimeError("abort batch")

    with pytest.raises(RepositoryError):
        repository.get_clothing_item("rolled-back")


def test_processing_generation_scan_uses_global_updated_at_order(
    repository: Repository,
) -> None:
    person = repository.set_person_photo("me", "person/person.jpg", 1, 1, 1)
    for generation_id, updated_at in (
        ("late", "2026-01-03T00:00:00Z"),
        ("early-b", "2026-01-01T00:00:00Z"),
        ("early-a", "2026-01-01T00:00:00Z"),
    ):
        generation = _generation(generation_id, "me", person.id, ()).model_copy(
            update={
                "status": GenerationStatus.PROCESSING,
                "polling_url": "https://poll.example/x",
                "updated_at": updated_at,
            }
        )
        repository.create_generation(generation)
    assert repository.processing_generation_ids() == ("early-a", "early-b", "late")


def test_people_have_independent_current_photos_and_wardrobe_searches(
    repository: Repository,
) -> None:
    with pytest.raises(RepositoryError, match="include 'me'"):
        repository.update_person(DEFAULT_PERSON_ID, names=("Charles",))
    alice = repository.create_person(
        ("Alice",), "person/alice-1.jpg", 1, 1, 1, "170 cm; usually wears size 38"
    )
    bob = repository.create_person(("Bob",), "person/bob-1.jpg", 1, 1, 1)
    alice_first_photo = repository.get_person_photo(alice.id)
    bob_photo = repository.get_person_photo(bob.id)
    repository.set_person_photo(alice.id, "person/alice-2.jpg", 2, 2, 2)
    alice_coat = _item(repository, alice.id, "alice-coat")
    bob_coat = _item(repository, bob.id, "bob-coat")
    updated_alice = repository.update_person(
        alice.id, description="170 cm; straight build; prefers relaxed fits"
    )

    assert [person.names for person in repository.list_people().items] == [
        (DEFAULT_PERSON_ID,),
        ("Alice",),
        ("Bob",),
    ]
    assert repository.get_person_photo(alice.id).id != alice_first_photo.id
    assert repository.get_person_photo(bob.id).id == bob_photo.id
    assert updated_alice.description == "170 cm; straight build; prefers relaxed fits"
    assert repository.get_person(alice.id).description == updated_alice.description
    assert [item.id for item in repository.list_clothing_items(alice.id).items] == [alice_coat.id]
    assert [item.id for item in repository.list_clothing_items(bob.id).items] == [bob_coat.id]
    assert repository.get_clothing_item(bob_coat.id).person_id == bob.id
