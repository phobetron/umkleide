"""Durable FLUX.2 photo generation lifecycle."""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from weakref import WeakValueDictionary

from filelock import Timeout

from .api import (
    ClothingIdSelector,
    ClothingMetadataSelector,
    NewClothingItemInput,
    OutfitItem,
    PhotoSource,
)
from .bfl import (
    MAX_REFERENCES,
    BFLJobState,
    BFLPollResult,
    BFLProviderError,
    BFLRetryableDownloadError,
    BFLSubmission,
    BFLSubmissionRejected,
    BFLSubmissionUncertain,
    Flux2ProRequest,
)
from .locking import generation_lock
from .models import (
    ClothingItem,
    ClothingProvenance,
    Generation,
    GenerationStatus,
    Person,
    utc_now,
    utc_timestamp,
)
from .repositories import Repository, RepositoryError
from .storage import JPEG_DATA_URI_PREFIX, PreparedImage, StorageError
from .wardrobe import MediaQuotaError, WardrobeService
from .workers import run_blocking

GENERATION_POLL_INTERVAL_SECONDS = 5.0


class Provider(Protocol):
    async def submit_flux_2_pro(self, request: Flux2ProRequest) -> BFLSubmission: ...

    async def poll(self, polling_url: str) -> BFLPollResult: ...
    async def download_result(self, result_url: str) -> bytes: ...


class GenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedClothing:
    item: NewClothingItemInput
    image: PreparedImage


@dataclass(frozen=True, slots=True)
class PreparedGenerationRequest:
    """A validated, immutable plan which has not yet changed local state."""

    person: Person
    person_photo: PreparedImage | None
    person_description: str | None
    items: tuple[ClothingItem | PreparedClothing, ...]
    request: Flux2ProRequest


class GenerationService:
    def __init__(
        self,
        repository: Repository,
        wardrobe: WardrobeService,
        provider: Provider | None,
        *,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self.repository, self.wardrobe, self.provider = repository, wardrobe, provider
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._now = now
        self._accepting = True
        self._submissions: set[asyncio.Task[Generation]] = set()

    def _provider(self) -> Provider:
        if self.provider is None:
            raise GenerationError("configure Umkleide or set BFL_API_KEY before contacting BFL")
        return self.provider

    def close_intake(self) -> None:
        self._accepting = False

    async def drain_submissions(self, grace_seconds: float = 35.0) -> None:
        if grace_seconds < 0:
            raise ValueError("grace_seconds must be non-negative")
        tasks = tuple(self._submissions)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=grace_seconds)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def prepare_outfit_photo(
        self,
        *,
        person_id: str,
        prompt: str,
        items: Sequence[OutfitItem],
        person_photo: PhotoSource | PreparedImage | None = None,
        person_description: str | None = None,
    ) -> PreparedGenerationRequest:
        """Perform all non-mutating validation and image preparation before approval."""
        if not prompt.strip():
            raise GenerationError("prompt must be non-empty")
        if not 1 <= len(items) <= MAX_REFERENCES - 1:
            raise GenerationError(f"between 1 and {MAX_REFERENCES - 1} clothing items are required")
        self._provider()
        plan = await run_blocking(
            self._inspect_request, person_id, prompt, tuple(items), person_photo, person_description
        )
        return plan

    async def submit_prepared(self, prepared: PreparedGenerationRequest) -> Generation:
        if not self._accepting:
            raise GenerationError("generation submissions are shutting down")
        # Own the entire post-approval path before it can be cancelled by a client.
        task = asyncio.create_task(self._commit_submit_and_classify(prepared))
        self._submissions.add(task)
        task.add_done_callback(self._forget_submission)
        return await asyncio.shield(task)

    def _forget_submission(self, task: asyncio.Task[Generation]) -> None:
        self._submissions.discard(task)
        if not task.cancelled():
            try:
                task.exception()
            except Exception:
                pass

    async def _commit_submit_and_classify(self, prepared: PreparedGenerationRequest) -> Generation:
        generation = await run_blocking(self._commit_prepared, prepared)
        try:
            receipt = await self._provider().submit_flux_2_pro(prepared.request)
            generation = await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.PROCESSING,
                provider_request_id=receipt.request_id,
                polling_url=receipt.polling_url,
            )
        except BFLSubmissionRejected as exc:
            generation = await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.FAILED,
                error={"message": str(exc)},
            )
        except (BFLSubmissionUncertain, BFLProviderError, OSError, ValueError) as exc:
            generation = await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.SUBMISSION_UNKNOWN,
                error={"message": str(exc)},
            )
        return generation

    async def get_generation_snapshot(self, generation_id: str) -> Generation:
        return await run_blocking(self.repository.get_generation, generation_id)

    async def advance_generation(self, generation_id: str) -> Generation:
        lock = self._locks.get(generation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[generation_id] = lock
        async with lock:
            process_lock = generation_lock(self.wardrobe.storage.root, generation_id)
            acquired = False
            try:
                try:
                    await run_blocking(process_lock.acquire)
                    acquired = True
                except Timeout:
                    return await self.get_generation_snapshot(generation_id)
                return await self._poll_generation(generation_id)
            finally:
                if acquired:
                    await run_blocking(process_lock.release)

    async def _poll_generation(self, generation_id: str) -> Generation:
        await run_blocking(self.wardrobe.maintain)
        generation = await run_blocking(self.repository.get_generation, generation_id)
        if generation.status is not GenerationStatus.PROCESSING:
            return generation
        output = f"generations/{generation.id}.jpg"
        try:
            await run_blocking(self.wardrobe.storage.read_bytes, output)
        except StorageError:
            pass
        else:
            return await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.READY,
                result_relative_path=output,
            )
        if not generation.polling_url:
            return await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.FAILED,
                error={
                    "code": "result_unavailable",
                    "message": "generation is missing its polling URL",
                },
            )
        polling_url = generation.polling_url
        updated_at = datetime.fromisoformat(generation.updated_at.replace("Z", "+00:00"))
        if self._now().astimezone(timezone.utc) - updated_at < timedelta(
            seconds=GENERATION_POLL_INTERVAL_SECONDS
        ):
            return generation
        # Touch durable state before awaiting the provider.  This makes the
        # cadence survive process restarts and throttles transient failures.
        generation = await run_blocking(
            self.repository.update_generation, generation.id, status=GenerationStatus.PROCESSING
        )
        try:
            poll = await self._provider().poll(polling_url)
        except BFLProviderError as exc:
            return await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.PROCESSING,
                error={"code": "retrieval_unavailable", "message": str(exc)[:500]},
            )
        if poll.state is BFLJobState.PROCESSING:
            return generation
        if poll.state is BFLJobState.FAILED:
            return await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.FAILED,
                error={"code": "provider_failed", "message": poll.error or "BFL generation failed"},
            )
        if poll.result_url is None:
            return await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.FAILED,
                error={
                    "code": "result_unavailable",
                    "message": "BFL generation result did not include a download URL",
                },
            )
        try:
            payload = await self._provider().download_result(poll.result_url)
            prepared = await run_blocking(self.wardrobe.prepare_provider_image, payload)
        except BFLRetryableDownloadError as exc:
            return await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.PROCESSING,
                error={"code": "retrieval_unavailable", "message": str(exc)[:500]},
            )
        except (BFLProviderError, StorageError, ValueError) as exc:
            return await run_blocking(
                self.repository.update_generation,
                generation.id,
                status=GenerationStatus.FAILED,
                error={"code": "result_unavailable", "message": str(exc)[:500]},
            )
        try:
            return await run_blocking(self.wardrobe.save_generation_result, generation.id, prepared)
        except MediaQuotaError as exc:
            message = str(exc)
        except (OSError, StorageError):
            message = "generated image could not be stored; check local storage and permissions"
        return await run_blocking(
            self.repository.update_generation,
            generation.id,
            status=GenerationStatus.PROCESSING,
            error={"code": "storage_unavailable", "message": message},
        )

    def _resolve_item(self, selector: ClothingIdSelector | ClothingMetadataSelector):
        if isinstance(selector, ClothingIdSelector):
            item = self.repository.get_clothing_item(selector.clothing_item_id)
        elif isinstance(selector, ClothingMetadataSelector):
            page = self.repository.find_clothing_items(
                selector.owner_person_id,
                selector.query,
                selector.category,
                selector.metadata,
                limit=1,
            )
            item = page.items[0] if page.items else None
            if item is None:
                raise GenerationError("clothing selector did not match an item")
        else:
            raise TypeError("clothing selectors must be typed selector models")
        return item

    def _inspect_request(
        self,
        person_id: str,
        prompt: str,
        items: tuple[OutfitItem, ...],
        person_photo: PhotoSource | PreparedImage | None,
        person_description: str | None,
    ) -> PreparedGenerationRequest:
        """Inspect durable input and decode images, without taking a mutation lock."""
        try:
            person = self.repository.get_person(person_id)
        except RepositoryError as exc:
            raise GenerationError("person was not found") from exc
        if person_photo is None:
            try:
                current_person = self.repository.get_person_photo(person_id)
            except RepositoryError as exc:
                raise GenerationError("a current person photo is required") from exc
            if current_person.relative_path is None:
                raise GenerationError("the current person photo bytes are unavailable")
        else:
            current_person = None
        resolved_items: list[ClothingItem | NewClothingItemInput] = []
        for item in items:
            if isinstance(item, NewClothingItemInput):
                try:
                    self.repository.get_person(item.owner_person_id)
                except RepositoryError as exc:
                    raise GenerationError("clothing owner was not found") from exc
                resolved_items.append(item)
            else:
                record = self._resolve_item(item)
                resolved_items.append(record)
        existing = [item for item in resolved_items if isinstance(item, ClothingItem)]
        if len({item.id for item in existing}) != len(existing):
            raise GenerationError("clothing selectors resolved to duplicate items")
        # All cheap catalog checks are complete.  Now validate the actual image bytes.
        if current_person is not None:
            assert current_person.relative_path is not None
            self.wardrobe.storage.prepare_user_image_bytes(
                self.wardrobe.storage.read_bytes(current_person.relative_path)
            )
        for reference in existing:
            self.wardrobe.storage.prepare_user_image_bytes(
                self.wardrobe.storage.read_bytes(reference.photo_relative_path)
            )
        prepared_person = (
            person_photo
            if isinstance(person_photo, PreparedImage)
            else self.wardrobe.prepare_user_image(person_photo)
            if person_photo is not None
            else None
        )
        prepared_items: list[ClothingItem | PreparedClothing] = [
            PreparedClothing(item, self.wardrobe.prepare_user_image(item.photo))
            if isinstance(item, NewClothingItemInput)
            else item
            for item in resolved_items
        ]
        # A cheap advisory admission check; submit repeats it under the media lock.
        incoming = sum(
            item.image.size_bytes for item in prepared_items if isinstance(item, PreparedClothing)
        ) + (prepared_person.size_bytes if prepared_person is not None else 0)
        reclaim = 0
        if prepared_person is not None and person.current_photo_id:
            reclaim = self.repository.reclaimable_person_photo_bytes(person.current_photo_id)
        if self.wardrobe.storage.managed_usage() + incoming - reclaim > self.wardrobe.quota_bytes:
            raise MediaQuotaError(
                "managed-media quota is full; delete unneeded clothing or outfits"
            )
        assembled = _prompt(
            prompt,
            person_description if person_description is not None else person.description,
            tuple(
                item.item if isinstance(item, PreparedClothing) else item for item in prepared_items
            ),
        )
        stored_refs: list[str] = []
        if current_person is not None:
            assert current_person.relative_path is not None
            stored_refs.append(
                self.wardrobe.storage.managed_image_data_uri(current_person.relative_path)
            )
        else:
            assert prepared_person is not None
            stored_refs.append(
                JPEG_DATA_URI_PREFIX + base64.b64encode(prepared_person.data).decode("ascii")
            )
        for item in prepared_items:
            if isinstance(item, PreparedClothing):
                stored_refs.append(
                    JPEG_DATA_URI_PREFIX + base64.b64encode(item.image.data).decode("ascii")
                )
            else:
                stored_refs.append(
                    self.wardrobe.storage.managed_image_data_uri(item.photo_relative_path)
                )
        return PreparedGenerationRequest(
            person=person,
            person_photo=prepared_person,
            person_description=person_description,
            items=tuple(prepared_items),
            request=Flux2ProRequest(assembled, tuple(stored_refs)),
        )

    def _commit_prepared(self, prepared: PreparedGenerationRequest) -> Generation:
        """Revalidate an approved plan while holding the sole catalog/media writer lock."""
        with self.wardrobe.batch():
            try:
                person = self.repository.get_person(prepared.person.id)
            except RepositoryError as exc:
                raise GenerationError("prepared person no longer exists") from exc
            if (person.updated_at, person.current_photo_id) != (
                prepared.person.updated_at,
                prepared.person.current_photo_id,
            ):
                raise GenerationError("catalog changed after approval; prepare the outfit again")
            for item in prepared.items:
                if isinstance(item, PreparedClothing):
                    try:
                        self.repository.get_person(item.item.owner_person_id)
                    except RepositoryError as exc:
                        raise GenerationError("clothing owner changed after approval") from exc
                else:
                    try:
                        current = self.repository.get_clothing_item(item.id)
                    except RepositoryError as exc:
                        raise GenerationError("selected clothing changed after approval") from exc
                    if (current.updated_at, current.photo_id) != (item.updated_at, item.photo_id):
                        raise GenerationError("selected clothing changed after approval")
            if prepared.person_photo is not None or prepared.person_description is not None:
                self.wardrobe.update_person(
                    prepared.person.id,
                    prepared=prepared.person_photo,
                    description=prepared.person_description
                    if prepared.person_description is not None
                    else person.description,
                )
            current_person = self.repository.get_person_photo(prepared.person.id)
            resolved: list[ClothingItem] = []
            for item in prepared.items:
                if isinstance(item, PreparedClothing):
                    created = self.wardrobe.add_clothing(
                        person_id=item.item.owner_person_id,
                        name=item.item.name,
                        category=item.item.category,
                        prepared=item.image,
                        description=item.item.description,
                        metadata=item.item.metadata,
                    )
                    resolved.append(created)
                else:
                    resolved.append(item)
            if len({item.id for item in resolved}) != len(resolved):
                raise GenerationError("clothing selectors resolved to duplicate items")
            now = utc_timestamp()
            generation = Generation(
                id=str(uuid.uuid4()),
                person_id=prepared.person.id,
                status=GenerationStatus.SUBMISSION_UNKNOWN,
                prompt=prepared.request.prompt,
                person_photo_id=current_person.id,
                selected_clothing=tuple(
                    ClothingProvenance(
                        clothing_item_id=ref.id,
                        clothing_photo_id=ref.photo_id,
                        name=ref.name,
                    )
                    for ref in resolved
                ),
                created_at=now,
                updated_at=now,
            )
            self.repository.create_generation(generation)
            return generation


def _prompt(
    prompt: str,
    person_description: str | None,
    resolved: Sequence[ClothingItem | NewClothingItemInput],
) -> str:
    lines = [
        "# Outfit edit request",
        "",
        "Each description applies only to the person or garment in its own section. Additional",
        "agent context contains outfit instructions, not properties of the person or a garment,",
        "unless it explicitly identifies the target whose properties it changes.",
        "",
        "## Person — reference image 1",
        "",
        "**Role:** Edit reference image 1 as the base photograph. Keep its sole person's exact",
        "identity, face, complexion, hair, facial hair, pose, visible skin, and footwear or bare",
        "feet unless the additional agent context explicitly requests a change. Preserve that",
        "person's body exactly, including their height, build, body shape, proportions,",
        "silhouette, and physical volume.",
        "",
        "**Description:**",
        "",
        _markdown_quote(person_description),
        "",
        "## Garment reference rules",
        "",
        "Later references supply only the appearance of their named garments. Source every person,",
        "face, skin, anatomical feature, pose, and background exclusively from reference image 1.",
        "Keep the body shape, proportions, size, build, and physical volume of the person in image",
        "1. Keep their footwear and accessories unless a named garment is itself footwear or an",
        "accessory, or the additional agent context explicitly requests a change.",
        "Treat every article of clothing worn in reference image 1 as source material to remove",
        "and replace, never as part of the requested outfit. Reconstruct the subject's clothing",
        "from the named garment references instead of layering them over the original clothing.",
        "Do not retain or reveal any part of the original clothing at garment boundaries, through",
        "fabric, or anywhere else in the result, unless the additional agent context explicitly",
        "requests it.",
        "",
        "## Garments",
    ]
    for index, reference in enumerate(resolved, start=2):
        lines += [
            "",
            f"### Reference image {index} — {_markdown_heading(reference.name)}",
            "",
            "**Role:** Use this reference only for the appearance of the named garment.",
            "",
            "**Description:**",
            "",
            _markdown_quote(reference.description),
        ]
    lines += [
        "",
        "## Composition instructions",
        "",
        "Dress the subject in all referenced garments as one coherent outfit, inferring intended",
        "fit, fastening, overlap, tucked or untucked styling, and natural layer order from the",
        "source images unless the additional agent context requests otherwise. Whether a garment",
        "is shown on a model or mannequin, on a hanger, or laid flat, reconstruct it as a real",
        "worn garment fitted to the unchanged body in image 1. Fill and drape the fabric naturally",
        "around that body using plausible volume, folds, tension, and gravity.",
        "",
        "## Additional agent context",
        "",
        _markdown_quote(prompt),
    ]
    return "\n".join(lines)


def _markdown_quote(value: str | None) -> str:
    """Render free text as one Markdown block quote within its assigned section."""

    if value is None or not value.strip():
        return "_No description provided._"
    return "\n".join(">" if not line else f"> {line}" for line in value.strip().splitlines())


def _markdown_heading(value: str) -> str:
    """Keep a clothing name on one literal Markdown heading line."""

    heading = " ".join(value.split())
    return heading.replace("\\", "\\\\").replace("#", "\\#")
