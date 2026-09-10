from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from umkleide.api import ClothingIdSelector, DataUriPhotoSource, NewClothingItemInput
from umkleide.bfl import (
    MAX_REFERENCES,
    BFLJobState,
    BFLPollResult,
    BFLProviderError,
    BFLRetryableDownloadError,
    BFLSubmission,
    BFLSubmissionRejected,
    BFLSubmissionUncertain,
)
from umkleide.database import Database
from umkleide.generations import (
    GENERATION_POLL_INTERVAL_SECONDS,
    GenerationError,
    GenerationService,
)
from umkleide.models import (
    DEFAULT_PERSON_ID,
    ClothingProvenance,
    Generation,
    GenerationStatus,
    utc_timestamp,
)
from umkleide.repositories import Repository, RepositoryError
from umkleide.storage import (
    JPEG_DATA_URI_PREFIX,
    PROVIDER_IMAGE_SIZE,
    MediaStorage,
    PreparedImage,
    StorageError,
)
from umkleide.wardrobe import WardrobeService

TEST_QUOTA_BYTES = 100_000_000
DEFAULT_SUBMISSION = BFLSubmission("request-1", "https://poll.example/request-1")
RESULT_URL = "https://result.example/a"
PERSON_DESCRIPTION = "Usually wears size M; straight build"
OUTFIT_PROMPT = "portrait"


class FakeProvider:
    def __init__(
        self,
        *,
        submission: BFLSubmission | Exception | None = None,
        poll: BFLPollResult | Exception | None = None,
        download: bytes | Exception = b"provider-result",
    ) -> None:
        self.submission = submission or DEFAULT_SUBMISSION
        self.poll_result = poll or BFLPollResult(BFLJobState.PROCESSING)
        self.download = download
        self.submit_calls = self.poll_calls = self.download_calls = 0
        self.submitted_requests: list[Any] = []
        self.poll_started = asyncio.Event()
        self.release_poll: asyncio.Event | None = None

    async def submit_flux_2_pro(self, request: Any) -> BFLSubmission:
        self.submit_calls += 1
        self.submitted_requests.append(request)
        if isinstance(self.submission, Exception):
            raise self.submission
        return self.submission

    async def poll(self, polling_url: str) -> BFLPollResult:
        self.poll_calls += 1
        self.poll_started.set()
        if self.release_poll is not None:
            await self.release_poll.wait()
        if isinstance(self.poll_result, Exception):
            raise self.poll_result
        return self.poll_result

    async def download_result(self, result_url: str) -> bytes:
        self.download_calls += 1
        if isinstance(self.download, Exception):
            raise self.download
        return self.download


def _photo_source() -> DataUriPhotoSource:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="JPEG")
    return DataUriPhotoSource(
        type="data_uri",
        data=JPEG_DATA_URI_PREFIX + base64.b64encode(output.getvalue()).decode("ascii"),
    )


def _provider_image_bytes(size: tuple[int, int] = PROVIDER_IMAGE_SIZE) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "green").save(output, format="JPEG")
    return output.getvalue()


def _services(
    tmp_path: Path,
    provider: FakeProvider | None,
    *,
    now: Callable[[], datetime] | None = None,
    quota_bytes: int = TEST_QUOTA_BYTES,
):
    database = Database(tmp_path / "x.sqlite3")
    database.initialize()
    repository = Repository(database)
    storage = MediaStorage(tmp_path / "media")
    wardrobe = WardrobeService(
        repository,
        storage,
        quota_bytes=quota_bytes,
        import_root=tmp_path / "imports",
    )
    service = GenerationService(repository, wardrobe, provider, **({"now": now} if now else {}))
    return repository, wardrobe, service


async def _submit(service: GenerationService, **kwargs: Any):
    """Exercise the approved-generation path without hiding its two phases."""
    prepared = await service.prepare_outfit_photo(**kwargs)
    return await service.submit_prepared(prepared)


def _seed_person_and_item(repository: Repository, wardrobe: WardrobeService):
    source = _photo_source()
    person = wardrobe.set_person_photo(DEFAULT_PERSON_ID, wardrobe.prepare_user_image(source))
    item = wardrobe.add_clothing(
        person_id=DEFAULT_PERSON_ID,
        name="red shirt",
        category="top",
        prepared=wardrobe.prepare_user_image(source),
    )
    return source, person, item


def _processing_generation(repository: Repository, *, updated_at: datetime) -> Generation:
    person = repository.set_person_photo(DEFAULT_PERSON_ID, "person/person.jpg", 1, 1, 1)
    repository.create_clothing_item(
        DEFAULT_PERSON_ID,
        "clothing-item",
        "shirt",
        "top",
        None,
        {},
        1,
        1,
        1,
        photo_id="clothing-photo",
    )
    generation = Generation(
        id="generation",
        person_id=DEFAULT_PERSON_ID,
        status=GenerationStatus.PROCESSING,
        prompt=OUTFIT_PROMPT,
        person_photo_id=person.id,
        selected_clothing=(
            ClothingProvenance(
                clothing_item_id="clothing-item", clothing_photo_id="clothing-photo", name="shirt"
            ),
        ),
        polling_url="https://poll.example/generation",
        created_at=utc_timestamp(updated_at),
        updated_at=utc_timestamp(updated_at),
    )
    repository.create_generation(generation)
    return generation


@pytest.mark.parametrize(
    "case",
    ["empty", "missing_provider", "missing_person", "too_many", "unresolved", "duplicate"],
)
async def test_preflight_failures_have_no_durable_or_provider_effect(
    tmp_path: Path, case: str
) -> None:
    provider = None if case == "missing_provider" else FakeProvider()
    repository, wardrobe, service = _services(tmp_path, provider)
    source = _photo_source()
    item = None
    person_id = "missing"
    if case not in {"missing_person", "unresolved"}:
        _, _, item = _seed_person_and_item(repository, wardrobe)
        person_id = item.person_id
    elif case == "unresolved":
        wardrobe.set_person_photo(DEFAULT_PERSON_ID, wardrobe.prepare_user_image(source))
        person_id = DEFAULT_PERSON_ID

    new_item = NewClothingItemInput(type="new", name="unwanted", category="top", photo=source)
    selector = ClothingIdSelector(type="id", clothing_item_id=item.id if item else "missing")
    items = {
        "empty": [],
        "missing_provider": [new_item],
        "missing_person": [new_item],
        "too_many": [selector] * MAX_REFERENCES,
        "unresolved": [selector],
        "duplicate": [selector, selector],
    }[case]
    catalog_before = (
        repository.list_clothing_items(person_id).items if person_id != "missing" else ()
    )

    with pytest.raises((GenerationError, RepositoryError)):
        await _submit(service, person_id=person_id, prompt=OUTFIT_PROMPT, items=items)

    if person_id != "missing":
        assert repository.list_generations(person_id).items == ()
        assert repository.list_clothing_items(person_id).items == catalog_before
    assert provider is None or provider.submit_calls == 0


async def test_later_malformed_new_image_leaves_catalog_and_generation_unchanged(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    repository, _, service = _services(tmp_path, provider)
    source, person, item = _seed_person_and_item(repository, service.wardrobe)
    malformed = DataUriPhotoSource(type="data_uri", data=JPEG_DATA_URI_PREFIX + "eA==")

    with pytest.raises(StorageError):
        await _submit(
            service,
            person_id=item.person_id,
            prompt=OUTFIT_PROMPT,
            person_photo=service.wardrobe.prepare_user_image(source),
            items=[
                NewClothingItemInput(type="new", name="first", category="top", photo=source),
                NewClothingItemInput(type="new", name="broken", category="top", photo=malformed),
            ],
        )

    assert repository.get_person_photo(item.person_id).id == person.id
    assert repository.list_clothing_items(item.person_id).items == (item,)
    assert repository.list_generations(item.person_id).items == ()
    assert provider.submit_calls == 0


async def test_local_record_failure_rolls_back_new_clothing_and_person_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    repository, wardrobe, service = _services(tmp_path, provider)
    source, person, item = _seed_person_and_item(repository, wardrobe)

    def fail_record(*_args: object) -> None:
        raise RepositoryError("database write failed")

    monkeypatch.setattr(repository, "create_generation", fail_record)
    with pytest.raises(RepositoryError, match="database write failed"):
        await _submit(
            service,
            person_id=item.person_id,
            prompt=OUTFIT_PROMPT,
            person_photo=wardrobe.prepare_user_image(source),
            items=[NewClothingItemInput(type="new", name="new", category="top", photo=source)],
        )

    assert repository.get_person_photo(item.person_id).id == person.id
    assert repository.list_clothing_items(item.person_id).items == (item,)
    assert provider.submit_calls == 0


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (DEFAULT_SUBMISSION, GenerationStatus.PROCESSING),
        (BFLSubmissionRejected("rejected"), GenerationStatus.FAILED),
        (BFLSubmissionUncertain("transport interrupted"), GenerationStatus.SUBMISSION_UNKNOWN),
    ],
)
async def test_submission_outcomes_are_durably_distinguished(
    tmp_path: Path, outcome: BFLSubmission | Exception, status: GenerationStatus
) -> None:
    provider = FakeProvider(submission=outcome)
    repository, wardrobe, service = _services(tmp_path, provider)
    _, person, item = _seed_person_and_item(repository, wardrobe)

    generation = await _submit(
        service,
        person_id=item.person_id,
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
    )

    stored = repository.get_generation(generation.id)
    assert (generation.status, stored.status) == (status, status)
    assert stored.person_photo_id == person.id
    assert stored.provider_request_id == (
        DEFAULT_SUBMISSION.request_id if status is GenerationStatus.PROCESSING else None
    )
    assert stored.polling_url == (
        DEFAULT_SUBMISSION.polling_url if status is GenerationStatus.PROCESSING else None
    )


async def test_supplied_person_photo_becomes_current_and_is_used_for_generation(
    tmp_path: Path,
) -> None:
    description = PERSON_DESCRIPTION
    provider = FakeProvider(submission=BFLSubmissionRejected("rejected"))
    repository, wardrobe, service = _services(tmp_path, provider)
    source = _photo_source()
    wardrobe.set_person_photo(DEFAULT_PERSON_ID, wardrobe.prepare_user_image(source))
    item = wardrobe.add_clothing(
        person_id=DEFAULT_PERSON_ID,
        name="red shirt",
        category="top",
        prepared=wardrobe.prepare_user_image(source),
    )

    generation = await _submit(
        service,
        person_id=DEFAULT_PERSON_ID,
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
        person_photo=wardrobe.prepare_user_image(source),
        person_description=description,
    )

    current = repository.get_person_photo(DEFAULT_PERSON_ID)
    assert generation.person_photo_id == current.id
    assert repository.get_generation(generation.id).person_photo_id == current.id
    assert repository.get_person(DEFAULT_PERSON_ID).description == description
    assert "## Person — reference image 1\n\n**Role:**" in provider.submitted_requests[0].prompt
    assert f"> {description}" in provider.submitted_requests[0].prompt


async def test_first_run_accepts_a_supplied_person_photo_and_new_clothing(tmp_path: Path) -> None:
    provider = FakeProvider(submission=BFLSubmissionRejected("rejected"))
    repository, wardrobe, service = _services(tmp_path, provider)
    source = _photo_source()

    generation = await _submit(
        service,
        person_id=DEFAULT_PERSON_ID,
        prompt=OUTFIT_PROMPT,
        person_photo=wardrobe.prepare_user_image(source),
        items=[NewClothingItemInput(type="new", name="first shirt", category="top", photo=source)],
    )

    assert generation.status is GenerationStatus.FAILED
    assert repository.get_person_photo(DEFAULT_PERSON_ID).id == generation.person_photo_id
    assert (
        repository.get_clothing_item(generation.selected_clothing[0].clothing_item_id).name
        == "first shirt"
    )
    assert provider.submit_calls == 1


async def test_missing_stored_reference_rolls_back_all_local_generation_changes(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    repository, wardrobe, service = _services(tmp_path, provider)
    source, person, item = _seed_person_and_item(repository, wardrobe)
    wardrobe.storage.delete_image(item.photo_relative_path)

    with pytest.raises(StorageError):
        await _submit(
            service,
            person_id=item.person_id,
            prompt=OUTFIT_PROMPT,
            person_photo=wardrobe.prepare_user_image(source),
            items=[
                ClothingIdSelector(type="id", clothing_item_id=item.id),
                NewClothingItemInput(type="new", name="new", category="top", photo=source),
            ],
        )

    assert repository.get_person_photo(item.person_id).id == person.id
    assert repository.list_clothing_items(item.person_id).items == (item,)
    assert repository.list_generations(item.person_id).items == ()
    assert provider.submit_calls == 0


async def test_definitive_rejection_retains_exact_provenance_after_item_rename(
    tmp_path: Path,
) -> None:
    repository, wardrobe, service = _services(
        tmp_path, FakeProvider(submission=BFLSubmissionRejected("rejected"))
    )
    _, person, item = _seed_person_and_item(repository, wardrobe)

    generation = await _submit(
        service,
        person_id=item.person_id,
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
    )
    repository.update_clothing_item(item.id, name="renamed")

    stored = repository.get_generation(generation.id)
    assert stored.status is GenerationStatus.FAILED
    assert stored.person_photo_id == person.id
    assert stored.selected_clothing == (
        ClothingProvenance(
            clothing_item_id=item.id, clothing_photo_id=item.photo_id, name=item.name
        ),
    )


async def test_person_can_wear_clothing_owned_by_another_person(tmp_path: Path) -> None:
    repository, wardrobe, service = _services(
        tmp_path, FakeProvider(submission=BFLSubmissionRejected("rejected"))
    )
    source = _photo_source()
    alice = wardrobe.create_person(names=("Alice",), prepared=wardrobe.prepare_user_image(source))
    susan = wardrobe.create_person(
        names=("Susan",),
        prepared=wardrobe.prepare_user_image(source),
    )
    coat = wardrobe.add_clothing(
        person_id=susan.id,
        name="Susan's coat",
        category="outerwear",
        prepared=wardrobe.prepare_user_image(source),
    )

    generation = await _submit(
        service,
        person_id=alice.id,
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=coat.id)],
    )

    assert generation.person_id == alice.id
    assert generation.selected_clothing[0].clothing_item_id == coat.id
    assert repository.get_clothing_item(coat.id).person_id == susan.id


async def test_new_item_is_persisted_when_submission_is_definitively_rejected(
    tmp_path: Path,
) -> None:
    name = "new coat"
    description = "Size M with a boxy, oversized fit"
    repository, wardrobe, service = _services(
        tmp_path, FakeProvider(submission=BFLSubmissionRejected("rejected"))
    )
    source, _, item = _seed_person_and_item(repository, wardrobe)

    generation = await _submit(
        service,
        person_id=item.person_id,
        prompt=OUTFIT_PROMPT,
        items=[
            NewClothingItemInput(
                type="new",
                name=name,
                category="outerwear",
                photo=source,
                description=description,
            )
        ],
    )

    created = repository.get_clothing_item(generation.selected_clothing[0].clothing_item_id)
    assert generation.status is GenerationStatus.FAILED
    assert (created.name, created.photo_id) == (
        name,
        generation.selected_clothing[0].clothing_photo_id,
    )
    assert created.description == description
    assert (
        repository.get_generation(generation.id).selected_clothing[0].clothing_item_id == created.id
    )


async def test_submission_includes_descriptions_and_keeps_reference_images_in_order(
    tmp_path: Path,
) -> None:
    person_description = PERSON_DESCRIPTION
    base_description = "Close-fitting through the torso"
    outer_name = "outer garment"
    outer_description = "Relaxed fit with dropped shoulders"
    instructions = "Standing outside"
    provider = FakeProvider(submission=BFLSubmissionRejected("rejected"))
    repository, wardrobe, service = _services(tmp_path, provider)
    source = _photo_source()
    wardrobe.set_person_photo(DEFAULT_PERSON_ID, wardrobe.prepare_user_image(source))
    wardrobe.update_person(DEFAULT_PERSON_ID, description=person_description)
    base = wardrobe.add_clothing(
        person_id=DEFAULT_PERSON_ID,
        name="base garment",
        category="top",
        prepared=wardrobe.prepare_user_image(source),
        description=base_description,
    )

    generation = await _submit(
        service,
        person_id=DEFAULT_PERSON_ID,
        prompt=instructions,
        items=[
            ClothingIdSelector(type="id", clothing_item_id=base.id),
            NewClothingItemInput(
                type="new",
                name=outer_name,
                category="outerwear",
                photo=source,
                description=outer_description,
            ),
        ],
    )

    request = provider.submitted_requests[0]
    person = repository.get_person_photo(DEFAULT_PERSON_ID)
    assert person.relative_path is not None
    assert [reference.name for reference in generation.selected_clothing] == [base.name, outer_name]
    assert request.reference_images == (
        wardrobe.storage.managed_image_data_uri(person.relative_path),
        *(
            wardrobe.storage.managed_image_data_uri(
                repository.get_clothing_item(reference.clothing_item_id).photo_relative_path
            )
            for reference in generation.selected_clothing
        ),
    )
    prompt_parts = (
        "# Outfit edit request",
        "## Person — reference image 1",
        f"> {person_description}",
        "## Garments",
        f"### Reference image 2 — {base.name}",
        f"> {base_description}",
        f"### Reference image 3 — {outer_name}",
        f"> {outer_description}",
        "## Additional agent context",
        f"> {instructions}",
    )
    assert all(part in request.prompt for part in prompt_parts)
    assert [request.prompt.index(part) for part in prompt_parts] == sorted(
        request.prompt.index(part) for part in prompt_parts
    )


async def test_provider_submission_does_not_hold_wardrobe_mutation_lock(tmp_path: Path) -> None:
    provider = FakeProvider(submission=BFLSubmissionRejected("rejected"))
    repository, wardrobe, service = _services(tmp_path, provider)
    _, _, item = _seed_person_and_item(repository, wardrobe)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_submit = provider.submit_flux_2_pro

    async def blocked_submit(request: Any) -> BFLSubmission:
        entered.set()
        await release.wait()
        return await original_submit(request)

    provider.submit_flux_2_pro = blocked_submit  # type: ignore[method-assign]
    submission = asyncio.create_task(
        _submit(
            service,
            person_id=item.person_id,
            prompt=OUTFIT_PROMPT,
            items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.wait_for(
        asyncio.to_thread(
            wardrobe.add_clothing,
            person_id=item.person_id,
            name="concurrent",
            category="top",
            prepared=wardrobe.prepare_user_image(_photo_source()),
        ),
        timeout=1,
    )
    release.set()
    await submission


async def test_concurrent_generation_reads_share_one_provider_poll(tmp_path: Path) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    provider = FakeProvider()
    provider.release_poll = asyncio.Event()
    repository, _, service = _services(tmp_path, provider, now=lambda: base + timedelta(seconds=5))
    generation = _processing_generation(repository, updated_at=base)

    first = asyncio.create_task(service.advance_generation(generation.id))
    await asyncio.wait_for(provider.poll_started.wait(), timeout=1)
    second = asyncio.create_task(service.advance_generation(generation.id))
    await asyncio.sleep(0)
    provider.release_poll.set()

    first_result, second_result = await asyncio.gather(first, second)
    assert provider.poll_calls == 1
    assert first_result.status is second_result.status is GenerationStatus.PROCESSING


async def test_generation_snapshot_does_not_advance_provider_work(tmp_path: Path) -> None:
    repository, _, service = _services(tmp_path, FakeProvider())
    generation = _processing_generation(repository, updated_at=datetime.now(timezone.utc))
    snapshot = await service.get_generation_snapshot(generation.id)
    assert snapshot == generation
    assert isinstance(service.provider, FakeProvider)
    assert service.provider.poll_calls == service.provider.download_calls == 0


async def test_prepare_has_no_catalog_side_effects_until_submission(tmp_path: Path) -> None:
    repository, wardrobe, service = _services(tmp_path, FakeProvider())
    source = _photo_source()
    prepared = await service.prepare_outfit_photo(
        person_id=DEFAULT_PERSON_ID,
        prompt=OUTFIT_PROMPT,
        person_photo=source,
        items=[NewClothingItemInput(type="new", name="shirt", category="top", photo=source)],
    )
    assert prepared.person_photo is not None
    assert len(prepared.items) == 1
    assert repository.get_person(DEFAULT_PERSON_ID).current_photo_id is None
    assert repository.list_clothing_items(DEFAULT_PERSON_ID).items == ()
    assert repository.list_generations(DEFAULT_PERSON_ID).items == ()


async def test_submission_rejects_a_stale_prepared_catalog(tmp_path: Path) -> None:
    repository, wardrobe, service = _services(tmp_path, FakeProvider())
    _, _, item = _seed_person_and_item(repository, wardrobe)
    prepared = await service.prepare_outfit_photo(
        person_id=DEFAULT_PERSON_ID,
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
    )
    wardrobe.update_clothing(item.id, description="new description")
    with pytest.raises(GenerationError, match="changed after approval"):
        await service.submit_prepared(prepared)
    assert repository.list_generations(DEFAULT_PERSON_ID).items == ()


async def test_prepared_request_rejects_clothing_changed_during_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, wardrobe, service = _services(tmp_path, FakeProvider())
    _, _, item = _seed_person_and_item(repository, wardrobe)
    original_get = repository.get_clothing_item
    original_update = repository.update_clothing_item
    calls = 0

    def change_after_read(clothing_item_id: str):
        nonlocal calls
        record = original_get(clothing_item_id)
        if calls == 0:
            calls += 1
            original_update(clothing_item_id, description="changed during inspection")
        return record

    monkeypatch.setattr(repository, "get_clothing_item", change_after_read)
    prepared = await service.prepare_outfit_photo(
        person_id=DEFAULT_PERSON_ID,
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
    )
    with pytest.raises(GenerationError, match="changed after approval"):
        await service.submit_prepared(prepared)
    assert repository.list_generations(DEFAULT_PERSON_ID).items == ()


async def test_cancelled_submit_still_persists_provider_receipt(tmp_path: Path) -> None:
    provider = FakeProvider()
    repository, wardrobe, service = _services(tmp_path, provider)
    _, _, item = _seed_person_and_item(repository, wardrobe)
    release = asyncio.Event()
    original = provider.submit_flux_2_pro

    async def blocked(request: Any) -> BFLSubmission:
        await release.wait()
        return await original(request)

    provider.submit_flux_2_pro = blocked  # type: ignore[method-assign]
    prepared = await service.prepare_outfit_photo(
        person_id=DEFAULT_PERSON_ID,
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
    )
    submission = asyncio.create_task(service.submit_prepared(prepared))
    await asyncio.sleep(0)
    submission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submission
    release.set()
    await service.drain_submissions()
    generation = repository.list_generations(DEFAULT_PERSON_ID).items[0]
    assert generation.status is GenerationStatus.PROCESSING
    assert generation.provider_request_id == DEFAULT_SUBMISSION.request_id


@pytest.mark.parametrize("transient", [False, True], ids=["processing", "transient_failure"])
async def test_poll_cadence_is_persisted_across_service_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transient: bool
) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = [base]
    monkeypatch.setattr("umkleide.repositories.utc_timestamp", lambda: utc_timestamp(clock[0]))
    provider = FakeProvider(
        poll=(
            BFLProviderError("temporarily unavailable")
            if transient
            else BFLPollResult(BFLJobState.PROCESSING)
        )
    )
    repository, wardrobe, service = _services(tmp_path, provider, now=lambda: clock[0])
    generation = _processing_generation(repository, updated_at=base - timedelta(seconds=5))

    first = await service.advance_generation(generation.id)
    persisted = repository.get_generation(generation.id)
    restarted = GenerationService(repository, wardrobe, provider, now=lambda: clock[0])
    clock[0] += timedelta(seconds=GENERATION_POLL_INTERVAL_SECONDS - 1)
    await restarted.advance_generation(generation.id)
    clock[0] += timedelta(seconds=1)
    await restarted.advance_generation(generation.id)

    assert first.status is GenerationStatus.PROCESSING
    assert first.error == (
        {"code": "retrieval_unavailable", "message": "temporarily unavailable"}
        if transient
        else None
    )
    assert persisted.updated_at == utc_timestamp(base)
    assert repository.get_generation(generation.id).error == (
        {"code": "retrieval_unavailable", "message": "temporarily unavailable"}
        if transient
        else None
    )
    assert provider.poll_calls == 2


async def test_ready_provider_result_is_normalized_and_persisted(tmp_path: Path) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload = _provider_image_bytes()
    provider = FakeProvider(
        poll=BFLPollResult(BFLJobState.READY, result_url=RESULT_URL),
        download=payload,
    )
    repository, wardrobe, service = _services(
        tmp_path, provider, now=lambda: base + timedelta(seconds=5)
    )
    generation = _processing_generation(repository, updated_at=base)

    result = await service.advance_generation(generation.id)

    assert result.status is GenerationStatus.READY
    assert result.result_relative_path == f"generations/{generation.id}.jpg"
    assert result.result_relative_path is not None
    with Image.open(wardrobe.storage.root / result.result_relative_path) as image:
        assert (image.format, image.mode, image.size) == ("JPEG", "RGB", PROVIDER_IMAGE_SIZE)


@pytest.mark.parametrize(
    ("poll", "download", "quota_bytes", "message", "download_calls"),
    [
        (
            BFLPollResult(BFLJobState.FAILED, error="content moderated"),
            b"",
            TEST_QUOTA_BYTES,
            "content moderated",
            0,
        ),
        (
            BFLPollResult(BFLJobState.READY, result_url=RESULT_URL),
            BFLProviderError("delivery unavailable"),
            TEST_QUOTA_BYTES,
            "delivery unavailable",
            1,
        ),
        (
            BFLPollResult(BFLJobState.READY, result_url=RESULT_URL),
            b"not an image",
            TEST_QUOTA_BYTES,
            "valid supported image",
            1,
        ),
        (
            BFLPollResult(BFLJobState.READY, result_url=RESULT_URL),
            _provider_image_bytes((PROVIDER_IMAGE_SIZE[0] - 1, PROVIDER_IMAGE_SIZE[1])),
            TEST_QUOTA_BYTES,
            f"{PROVIDER_IMAGE_SIZE[0]}x{PROVIDER_IMAGE_SIZE[1]}",
            1,
        ),
    ],
    ids=["terminal", "download", "invalid_image", "wrong_size"],
)
async def test_terminal_and_result_failures_are_durably_failed(
    tmp_path: Path,
    poll: BFLPollResult,
    download: bytes | Exception,
    quota_bytes: int,
    message: str,
    download_calls: int,
) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    provider = FakeProvider(poll=poll, download=download)
    repository, wardrobe, service = _services(
        tmp_path,
        provider,
        now=lambda: base + timedelta(seconds=5),
        quota_bytes=quota_bytes,
    )
    generation = _processing_generation(repository, updated_at=base)

    result = await service.advance_generation(generation.id)
    stored = repository.get_generation(generation.id)

    assert (result.status, stored.status) == (GenerationStatus.FAILED, GenerationStatus.FAILED)
    assert stored.error is not None
    assert message in stored.error["message"]
    assert "https://" not in stored.error["message"]
    assert stored.result_relative_path is None
    assert provider.download_calls == download_calls
    assert not (wardrobe.storage.root / f"generations/{generation.id}.jpg").exists()


async def test_download_recovery_after_restart_reuses_submission_and_obeys_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    monkeypatch.setattr("umkleide.repositories.utc_timestamp", lambda: utc_timestamp(clock[0]))
    monkeypatch.setattr("umkleide.generations.utc_timestamp", lambda: utc_timestamp(clock[0]))
    monkeypatch.setattr("umkleide.wardrobe.utc_now", lambda: clock[0])
    provider = FakeProvider(
        poll=BFLPollResult(BFLJobState.READY, result_url=RESULT_URL),
        download=BFLRetryableDownloadError("delivery temporarily unavailable"),
    )
    repository, wardrobe, service = _services(tmp_path, provider, now=lambda: clock[0])
    _, _, item = _seed_person_and_item(repository, wardrobe)
    generation = await _submit(
        service,
        person_id="me",
        prompt=OUTFIT_PROMPT,
        items=[ClothingIdSelector(type="id", clothing_item_id=item.id)],
    )
    clock[0] += timedelta(seconds=5)
    result = await service.advance_generation(generation.id)
    assert result.status is GenerationStatus.PROCESSING
    assert result.error is not None and result.error["code"] == "retrieval_unavailable"
    assert repository.get_generation(generation.id).error == result.error

    provider.download = _provider_image_bytes()
    restarted = GenerationService(repository, wardrobe, provider, now=lambda: clock[0])
    clock[0] += timedelta(seconds=4)
    assert (await restarted.advance_generation(generation.id)).error == result.error
    assert provider.poll_calls == 1
    clock[0] += timedelta(seconds=1)
    ready = await restarted.advance_generation(generation.id)
    assert ready.status is GenerationStatus.READY and ready.error is None
    assert ready.provider_request_id == generation.provider_request_id
    assert repository.get_generation(generation.id).error is None
    assert provider.submit_calls == 1
    assert provider.poll_calls == provider.download_calls == 2
    assert ready.result_relative_path is not None
    assert wardrobe.storage.read_bytes(ready.result_relative_path) == (
        wardrobe.prepare_provider_image(provider.download).data
    )


async def test_result_can_be_retrieved_after_freeing_quota(tmp_path: Path) -> None:
    provider = FakeProvider(
        poll=BFLPollResult(BFLJobState.READY, result_url=RESULT_URL),
        download=_provider_image_bytes(),
    )
    repository, wardrobe, service = _services(
        tmp_path, provider, now=lambda: datetime.now(timezone.utc) + timedelta(seconds=5)
    )
    generation = _processing_generation(repository, updated_at=datetime.now(timezone.utc))
    wardrobe.quota_bytes = wardrobe.prepare_provider_image(_provider_image_bytes()).size_bytes
    filler = wardrobe.add_clothing(
        person_id="me", name="filler", category="top", prepared=PreparedImage(b"x", 1, 1)
    )
    blocked = await service.advance_generation(generation.id)
    assert blocked.status is GenerationStatus.PROCESSING
    assert blocked.error is not None and blocked.error["code"] == "storage_unavailable"
    wardrobe.delete_clothing(filler.id)
    ready = await service.advance_generation(generation.id)
    assert ready.status is GenerationStatus.READY and ready.error is None
    assert provider.submit_calls == 0
    assert wardrobe.storage.managed_usage() == wardrobe.quota_bytes


@pytest.mark.parametrize("error", [OSError("private-path secret-key")])
async def test_storage_errors_remain_recoverable_and_sanitized(tmp_path: Path, monkeypatch, error):
    provider = FakeProvider(
        poll=BFLPollResult(BFLJobState.READY, result_url=RESULT_URL),
        download=_provider_image_bytes(),
    )
    repository, wardrobe, service = _services(tmp_path, provider)
    generation = _processing_generation(
        repository, updated_at=datetime.now(timezone.utc) - timedelta(seconds=5)
    )

    def fail(*_args):
        raise error

    monkeypatch.setattr(wardrobe, "save_generation_result", fail)
    result = await service.advance_generation(generation.id)
    assert result.status is GenerationStatus.PROCESSING
    assert result.error is not None and result.error["code"] == "storage_unavailable"
    assert "secret-key" not in result.error["message"]
    assert repository.get_generation(generation.id).error == result.error


async def test_retained_result_is_reused_before_polling_even_when_quota_is_full(tmp_path: Path):
    repository, wardrobe, service = _services(tmp_path, FakeProvider(), quota_bytes=1)
    generation = _processing_generation(repository, updated_at=datetime.now(timezone.utc))
    prepared = wardrobe.prepare_provider_image(_provider_image_bytes())
    wardrobe.storage.store(f"generations/{generation.id}.jpg", prepared)

    result = await service.advance_generation(generation.id)

    assert result.status is GenerationStatus.READY
    assert isinstance(service.provider, FakeProvider)
    assert service.provider.poll_calls == service.provider.download_calls == 0


def _process_poll(root: str, started, release, results) -> None:
    class BlockingProvider(FakeProvider):
        async def poll(self, polling_url: str) -> BFLPollResult:
            self.poll_calls += 1
            started.set()
            assert await asyncio.to_thread(release.wait, 10)
            return BFLPollResult(BFLJobState.READY, result_url=RESULT_URL)

    async def run() -> None:
        provider = BlockingProvider(download=_provider_image_bytes())
        _, _, service = _services(Path(root), provider)
        generation = await service.advance_generation("generation")
        results.put((generation.status.value, provider.poll_calls, provider.download_calls))

    asyncio.run(run())


async def test_processes_coordinate_a_generation_without_duplicate_provider_calls(tmp_path: Path):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    started, release = context.Event(), context.Event()
    results = context.Queue()
    provider = FakeProvider(poll=BFLProviderError("must not poll concurrently"))
    repository, wardrobe, service = _services(tmp_path, provider)
    _processing_generation(
        repository, updated_at=datetime.now(timezone.utc) - timedelta(seconds=10)
    )
    process = context.Process(target=_process_poll, args=(str(tmp_path), started, release, results))
    process.start()
    try:
        assert await asyncio.to_thread(started.wait, 10)
        # Make cadence eligible to prove the process lock, rather than only the timer.
        with repository.database.transaction() as connection:
            connection.execute("UPDATE generations SET updated_at='2026-01-01T00:00:00Z'")
        generation = await service.advance_generation("generation")
        assert generation.status is GenerationStatus.PROCESSING
        assert provider.poll_calls == provider.download_calls == 0
        release.set()
        outcome = await asyncio.to_thread(results.get, True, 10)
        assert outcome == ("ready", 1, 1)
        generation = await service.advance_generation("generation")
        assert generation.status is GenerationStatus.READY
        assert provider.poll_calls == provider.download_calls == 0
        assert generation.result_relative_path is not None
        assert wardrobe.storage.read_bytes(generation.result_relative_path)
    finally:
        release.set()
        await asyncio.to_thread(process.join, 10)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)
        results.close()
    assert process.exitcode == 0


async def test_cancelled_generation_check_releases_process_poll_lock(tmp_path: Path):
    provider = FakeProvider()
    provider.release_poll = asyncio.Event()
    repository, _, service = _services(tmp_path, provider)
    _processing_generation(
        repository, updated_at=datetime.now(timezone.utc) - timedelta(seconds=10)
    )
    first = asyncio.create_task(service.advance_generation("generation"))
    await asyncio.wait_for(provider.poll_started.wait(), 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    next_provider = FakeProvider(poll=BFLPollResult(BFLJobState.FAILED, error="terminal"))
    _, _, next_service = _services(tmp_path, next_provider)
    with repository.database.transaction() as connection:
        connection.execute("UPDATE generations SET updated_at='2026-01-01T00:00:00Z'")
    generation = await asyncio.wait_for(next_service.advance_generation("generation"), 5)
    assert generation.status is GenerationStatus.FAILED
    assert next_provider.poll_calls == 1
