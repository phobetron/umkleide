"""MCP tools and resources for the local wardrobe application."""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Annotated, Any, ParamSpec, TypeVar

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.types import Annotations, CallToolResult, ContentBlock, ImageContent, ToolAnnotations
from pydantic import Field

from .api import (
    ClothingItemPatch,
    ClothingItemResult,
    DiagnosticsResult,
    GenerationImageFileResult,
    GenerationResult,
    GenerationWaitSeconds,
    OutfitItem,
    Page,
    PersonPatch,
    PersonResult,
    PhotoSource,
    RecordCounts,
)
from .application import AppState
from .config import AppConfig
from .guidance import (
    CLOTHING_DESCRIPTION_GUIDANCE,
    OUTFIT_PROMPT_GUIDANCE,
    PERSON_DESCRIPTION_GUIDANCE,
)
from .models import DEFAULT_PERSON_ID, JPEG_MEDIA_TYPE, CleanupResult, Generation, GenerationStatus
from .presentation import generation_data as present_generation
from .presentation import item_data, person_data
from .repositories import UNSET, RepositoryError
from .storage import StorageError
from .workers import run_blocking

P = ParamSpec("P")
T = TypeVar("T")


def _tool_annotations(
    title: str,
    *,
    read_only: bool,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = False,
) -> ToolAnnotations:
    """Build explicit MCP hints used by hosts when presenting tools."""
    return ToolAnnotations(
        title=title,
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


ASSISTANT_RESOURCE = Annotations(audience=["assistant"])
# Some MCP hosts reject floating-point priority tokens in tool-result annotations.
USER_MEDIA = Annotations(audience=["user", "assistant"])

_APPROVAL_FORM_SCHEMA: dict[str, object] = {"type": "object", "properties": {}}


async def _require_approval(ctx: Context, message: str, refusal_message: str) -> None:
    """Require the user to accept an MCP approval form."""
    try:
        response = await ctx.session.elicit_form(
            message=message,
            requestedSchema=_APPROVAL_FORM_SCHEMA,
            related_request_id=ctx.request_id,
        )
    except Exception as exc:
        raise ValueError(refusal_message) from exc
    if response.action != "accept":
        raise ValueError(refusal_message)


def create_server(*, app: AppState, config: AppConfig) -> FastMCP:
    """Register the MCP interface over an explicitly owned application."""
    cfg = config

    def in_worker(function: Callable[P, T]) -> Callable[P, Awaitable[T]]:
        @wraps(function)
        async def run(*args: P.args, **kwargs: P.kwargs) -> T:
            def perform() -> T:
                app.wardrobe.maintain()
                return function(*args, **kwargs)

            return await run_blocking(perform)

        return run

    mcp = FastMCP(
        "Umkleide",
        instructions=(
            f"Local image import directory: {json.dumps(str(cfg.import_root))}. Place "
            "local image files there and pass their relative paths using photo source type "
            "'import'. Use Umkleide instead of a general-purpose image generator whenever the "
            "user asks for a virtual try-on or an outfit image involving a person and supplied "
            "or cataloged clothing photos. Do not substitute a generic image generator or a "
            "manually constructed MCP client for this workflow. The default person is 'me'; "
            "list people to resolve other names and aliases. Clothing is organized by owner, "
            "but any person may wear items from any wardrobe. Catalog and find clothing to plan "
            "outfits. Before creating or updating a person, examine the supplied or current photo "
            "and infer a thorough fit-relevant description covering build, body shape, frame, "
            "shoulders, chest-waist-hip proportions, torso and limb proportions, posture, sizes, "
            "height, fit preferences, and how clothing sits on the body. Before creating or "
            "updating clothing, examine the supplied or current item photo and infer a thorough "
            "description covering garment type, size, cut, silhouette, proportions, length, "
            "sleeves, neckline or collar, rise or waist position, leg shape, ease, fabric "
            "structure, weight, drape, stretch, closures, intended fit, and layering behavior. "
            "When a clothing photo shows the item worn by a model, infer and include an exact "
            "description of how it fits the model, including where it is fitted, relaxed, "
            "oversized, cropped, long, taut, or loose and where hems and sleeves fall. Store these "
            "details in the corresponding person or item description. "
            "create_outfit_photo can use cataloged or "
            "newly supplied clothing. When "
            "using it to supply a new person photo or new items in one request, pass the gathered "
            "person details in person_description, put each new garment's fit details in its "
            "description. The selected person's description and every selected or new item's "
            "description are automatically included in the BFL prompt. Do not repeat those "
            "descriptions in prompt; use prompt only for additional outfit instructions. "
            "It sends the selected photos and prompt to Black Forest Labs, may incur a charge, and "
            "requires MCP form approval. create_outfit_photo waits up to 60 seconds by default "
            "while the server polls BFL internally, returning a generation record and, when "
            "ready, the exact retained native JPEG in the same result. Only if the response is "
            "still processing, call get_generation with wait_seconds=60 using the same ID. "
            "Use bounded waits rather than a manual polling loop. Processing "
            "includes waiting "
            "to retrieve and retain a provider result locally. If error.code is "
            "retrieval_unavailable, repeat that same-ID bounded wait. If error.code is "
            "storage_unavailable, show the error and wait for the user to resolve local storage "
            "before checking again. Never automatically resubmit "
            "a submission_unknown generation because the provider may already have accepted it. "
            "For failed, result_unavailable, or submission_unknown jobs, report the result and do "
            "not submit again automatically. If a transport timeout occurs before a generation ID "
            "is returned, inspect generation history before retrying. Both stdio and HTTP servers "
            "continue retrieval while their process remains alive. "
            "When a generation is ready, its response already contains native MCP image content; "
            "present it as the final user-facing deliverable. get_generation_image_file also "
            "returns the exact retained JPEG path for file-backed rendering without copying or "
            "modifying it. Use the image tools to inspect retained person and clothing photos. The "
            "durable resource URI remains available from get_generation."
        ),
    )

    def generation_data(generation: Generation) -> GenerationResult:
        photo = app.repository.get_person_photo_by_id(generation.person_photo_id)
        return present_generation(generation, photo)

    def generation_result(generation: Generation) -> CallToolResult:
        """Return structured generation data and the exact retained JPEG when ready."""
        data = generation_data(generation)
        content: list[ContentBlock] = []
        if (
            generation.status is GenerationStatus.READY
            and generation.result_relative_path is not None
        ):
            try:
                payload = app.storage.read_bytes(generation.result_relative_path)
            except StorageError as exc:
                raise ValueError(
                    "retained generation image is unavailable; correct local storage before "
                    "checking again"
                ) from exc
            else:
                content.append(
                    ImageContent(
                        type="image",
                        data=base64.b64encode(payload).decode("ascii"),
                        mimeType=JPEG_MEDIA_TYPE,
                        annotations=USER_MEDIA,
                    )
                )
        return CallToolResult(content=content, structuredContent=data.model_dump(mode="json"))

    async def wait_for_generation(
        generation: Generation, wait_seconds: int, ctx: Context | None
    ) -> Generation:
        """Observe persisted work through the process-owned retriever."""
        if wait_seconds == 0 or generation.status is not GenerationStatus.PROCESSING:
            return generation
        retriever = app.retriever
        if ctx is not None:
            await ctx.report_progress(0, 1, "Waiting for the retained outfit image")
        result = await retriever.wait_for_result(generation.id, timeout=float(wait_seconds))
        if ctx is not None and result.status is not GenerationStatus.PROCESSING:
            await ctx.report_progress(1, 1, "Outfit image retrieval finished")
        return result

    def image_data(relative_path: str | None, unavailable_message: str) -> Image:
        if relative_path is None:
            raise ValueError(unavailable_message)
        try:
            return Image(data=app.storage.read_bytes(relative_path), format="jpeg")
        except StorageError as exc:
            raise ValueError(unavailable_message) from exc

    @mcp.tool(
        title="Add a person",
        description=(
            "Create a person profile with one or more names or phrases and an initial photo. "
            "Examples for one person include 'Susan', 'my wife', and 'the lady'. Before calling, "
            "examine the supplied photo and capture a thorough fit-relevant description covering "
            "overall build and body shape, frame, shoulder width and slope, chest-waist-hip "
            "proportions, torso and limb proportions, posture, clothing sizes, height, fit "
            "preferences, and how clothing sits on the body."
        ),
        annotations=_tool_annotations("Add a person", read_only=False),
    )
    @in_worker
    def add_person(
        names: list[str],
        photo: PhotoSource,
        description: Annotated[str | None, Field(description=PERSON_DESCRIPTION_GUIDANCE)] = None,
    ) -> PersonResult:
        wardrobe = app.wardrobe
        try:
            return person_data(
                wardrobe.create_person(
                    names=tuple(names),
                    prepared=wardrobe.prepare_user_image(photo),
                    description=description,
                )
            )
        except (StorageError, RepositoryError, sqlite3.Error) as exc:
            raise ValueError(str(exc)) from exc

    @mcp.tool(
        title="List people",
        description=(
            "List person profiles, including all names and aliases. The built-in default profile "
            "is 'me' and is used when a person or clothing owner is omitted."
        ),
        annotations=_tool_annotations("List people", read_only=True, idempotent=True),
    )
    @in_worker
    def list_people(limit: int = 50, cursor: str | None = None) -> Page[PersonResult]:
        page = app.repository.list_people(limit=limit, cursor=cursor)
        return Page[PersonResult](
            items=[person_data(x) for x in page.items], next_cursor=page.next_cursor
        )

    @mcp.tool(
        title="Get a person",
        description="Return one named person profile and its current photo identifier.",
        annotations=_tool_annotations("Get a person", read_only=True, idempotent=True),
    )
    @in_worker
    def get_person(person_id: str = DEFAULT_PERSON_ID) -> PersonResult:
        return person_data(app.repository.get_person(person_id))

    @mcp.tool(
        title="Get Umkleide diagnostics",
        description=(
            "Return local data and import directories, BFL configuration status, media quota "
            "usage, and database row counts. The BFL API key itself is never returned."
        ),
        annotations=_tool_annotations("Get Umkleide diagnostics", read_only=True, idempotent=True),
    )
    @in_worker
    def get_diagnostics() -> DiagnosticsResult:
        current = app
        usage = current.storage.managed_usage()
        return DiagnosticsResult(
            data_root=str(cfg.data_root),
            import_root=str(cfg.import_root),
            bfl_api_key_source=current.bfl_api_key_source,
            media_usage_bytes=usage,
            media_quota_bytes=cfg.media_quota_bytes,
            media_available_bytes=max(0, cfg.media_quota_bytes - usage),
            records=RecordCounts.model_validate(current.repository.database.record_counts()),
        )

    @mcp.tool(
        title="Update a person",
        description=(
            "Change a person's names, fit-relevant description, current photo, or any combination. "
            "Before calling, examine the current or supplied photo and capture a thorough updated "
            "description covering overall build and body shape, frame, shoulder width and slope, "
            "chest-waist-hip proportions, torso and limb proportions, posture, clothing sizes, "
            "height, fit preferences, and how clothing sits on the body. A replaced photo remains "
            "available while retained generation history refers to it."
        ),
        annotations=_tool_annotations("Update a person", read_only=False, destructive=True),
    )
    @in_worker
    def update_person(changes: PersonPatch, person_id: str = DEFAULT_PERSON_ID) -> PersonResult:
        fields = changes.model_fields_set
        photo = changes.photo
        prepared = None
        if "photo" in fields:
            assert photo is not None
            prepared = app.wardrobe.prepare_user_image(photo)
        names = tuple(changes.names) if changes.names is not None else UNSET
        description = changes.description if "description" in fields else UNSET
        try:
            return person_data(
                app.wardrobe.update_person(
                    person_id, names=names, description=description, prepared=prepared
                )
            )
        except (StorageError, RepositoryError, sqlite3.Error) as exc:
            raise ValueError(str(exc)) from exc

    @mcp.tool(
        title="Delete a person",
        description=(
            "Permanently delete a non-default person, their photos, owned clothing, and outfit "
            "generation history. The built-in 'me' profile cannot be deleted."
        ),
        annotations=_tool_annotations(
            "Delete a person", read_only=False, destructive=True, idempotent=True
        ),
    )
    async def delete_person(person_id: str, ctx: Context) -> CleanupResult:
        if person_id == DEFAULT_PERSON_ID:
            raise ValueError("the default person 'me' cannot be deleted")
        person = await run_blocking(app.repository.get_person, person_id)
        await _require_approval(
            ctx,
            f"Permanently delete {person.names[0]}, all of their person photos, owned clothing "
            "and clothing photos, and outfit generation history? Clothing references will also "
            "be removed from other retained generation records. This cannot be undone.",
            "person deletion was not approved",
        )
        return await run_blocking(app.wardrobe.delete_person, person_id)

    @mcp.tool(
        title="Get a person photo image",
        description=(
            "Return an exact retained person photo as native MCP image content using the stable "
            "person photo ID returned by person-photo or generation tools."
        ),
        annotations=_tool_annotations("Get a person photo image", read_only=True, idempotent=True),
        structured_output=False,
    )
    @in_worker
    def get_person_photo_image(person_photo_id: str) -> Image:
        photo = app.repository.get_person_photo_by_id(person_photo_id)
        return image_data(photo.relative_path, "person photo image is unavailable")

    @mcp.tool(
        title="Add clothing to the wardrobe",
        description=(
            "Permanently catalog one clothing item with one supplied photo and searchable "
            "details in its owner's wardrobe. Ownership helps lookup and never restricts who can "
            "wear the item. The owner defaults to 'me'. Before calling, examine the supplied photo "
            "and put a thorough description of garment type, size, cut, silhouette, proportions, "
            "length, sleeves, neckline or collar, rise or waist position, leg shape, ease, fabric "
            "structure, weight, drape, stretch, closures, intended fit, and layering behavior in "
            "description. When the item is worn by a model, infer and include an exact description "
            "of how it fits the model, including where it is fitted, relaxed, oversized, cropped, "
            "long, taut, or loose and where hems and sleeves fall."
        ),
        annotations=_tool_annotations(
            "Add clothing to the wardrobe", read_only=False, destructive=False
        ),
    )
    @in_worker
    def add_clothing_item(
        name: str,
        category: str,
        photo: PhotoSource,
        owner_person_id: str = DEFAULT_PERSON_ID,
        description: Annotated[str | None, Field(description=CLOTHING_DESCRIPTION_GUIDANCE)] = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClothingItemResult:
        wardrobe = app.wardrobe
        return item_data(
            wardrobe.add_clothing(
                person_id=owner_person_id,
                name=name,
                category=category,
                prepared=wardrobe.prepare_user_image(photo),
                description=description,
                metadata=metadata,
            )
        )

    @mcp.tool(
        title="List wardrobe items",
        description=(
            "List active cataloged clothing, optionally restricted to a category. Use to browse "
            "the wardrobe or compare possible outfit combinations."
        ),
        annotations=_tool_annotations("List wardrobe items", read_only=True, idempotent=True),
    )
    @in_worker
    def list_clothing_items(
        owner_person_id: str = DEFAULT_PERSON_ID,
        category: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[ClothingItemResult]:
        page = app.repository.list_clothing_items(
            owner_person_id, category, limit=limit, cursor=cursor
        )
        return Page[ClothingItemResult](
            items=[item_data(x) for x in page.items], next_cursor=page.next_cursor
        )

    @mcp.tool(
        title="Find matching clothing",
        description=(
            "Search active wardrobe items by name or description text, category, and exact "
            "metadata values. Use to resolve natural requests such as 'my blue jacket' before "
            "planning or generating an outfit."
        ),
        annotations=_tool_annotations("Find matching clothing", read_only=True, idempotent=True),
    )
    @in_worker
    def find_clothing_items(
        owner_person_id: str = DEFAULT_PERSON_ID,
        query: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[ClothingItemResult]:
        page = app.repository.find_clothing_items(
            owner_person_id, query, category, metadata, limit=limit, cursor=cursor
        )
        return Page[ClothingItemResult](
            items=[item_data(x) for x in page.items], next_cursor=page.next_cursor
        )

    @mcp.tool(
        title="Get a clothing item",
        description=(
            "Return one active clothing item's searchable details and immutable photo ID. Use "
            "get_clothing_photo_image with that photo ID to inspect the image."
        ),
        annotations=_tool_annotations("Get a clothing item", read_only=True, idempotent=True),
    )
    @in_worker
    def get_clothing_item(clothing_item_id: str) -> ClothingItemResult:
        return item_data(app.repository.get_clothing_item(clothing_item_id))

    @mcp.tool(
        title="Get a clothing photo image",
        description=(
            "Return an exact immutable clothing photo as native MCP image content using the "
            "photo ID returned by clothing or generation tools."
        ),
        annotations=_tool_annotations(
            "Get a clothing photo image", read_only=True, idempotent=True
        ),
        structured_output=False,
    )
    @in_worker
    def get_clothing_photo_image(clothing_photo_id: str) -> Image:
        photo = app.repository.get_clothing_photo(clothing_photo_id)
        return image_data(photo.relative_path, "clothing photo image is unavailable")

    @mcp.tool(
        title="Update a clothing item",
        description=(
            "Change an active clothing item's name, category, description, metadata, or single "
            "photo. Before calling, examine the current or supplied photo and record a thorough "
            "description of garment type, size, cut, silhouette, proportions, length, sleeves, "
            "neckline or collar, rise or waist position, leg shape, ease, fabric structure, "
            "weight, drape, stretch, closures, intended fit, and layering behavior. When the item "
            "is worn by a model, infer and include an exact description of how it fits the model, "
            "including where it is fitted, relaxed, oversized, cropped, long, taut, or loose and "
            "where hems and sleeves fall. Omitted fields remain unchanged; supplying a photo "
            "replaces the prior image."
        ),
        annotations=_tool_annotations("Update a clothing item", read_only=False, destructive=True),
    )
    @in_worker
    def update_clothing_item(
        clothing_item_id: str,
        changes: ClothingItemPatch,
    ) -> ClothingItemResult:
        wardrobe = app.wardrobe
        fields = changes.model_fields_set
        photo = changes.photo
        if "photo" in fields:
            assert photo is not None
            prepared = wardrobe.prepare_user_image(photo)
        else:
            prepared = None
        try:
            return item_data(
                wardrobe.update_clothing(
                    clothing_item_id,
                    name=changes.name if changes.name is not None else UNSET,
                    category=changes.category if changes.category is not None else UNSET,
                    description=changes.description if "description" in fields else UNSET,
                    metadata=changes.metadata if changes.metadata is not None else UNSET,
                    prepared=prepared,
                )
            )
        except (StorageError, RepositoryError, sqlite3.Error) as exc:
            raise ValueError(str(exc)) from exc

    @mcp.tool(
        title="Delete clothing from the wardrobe",
        description=(
            "Permanently delete one clothing item and all of its photo versions. References to "
            "the item are removed from retained generation records. This cannot be undone."
        ),
        annotations=_tool_annotations(
            "Delete clothing from the wardrobe", read_only=False, destructive=True
        ),
    )
    async def delete_clothing_item(clothing_item_id: str, ctx: Context) -> CleanupResult:
        item = await run_blocking(app.repository.get_clothing_item, clothing_item_id)
        await _require_approval(
            ctx,
            f"Permanently delete '{item.name}' and all of its photo versions? References to it "
            "will also be removed from retained generation records. This cannot be undone.",
            "clothing deletion was not approved",
        )
        return await run_blocking(app.wardrobe.delete_clothing, clothing_item_id)

    @mcp.tool(
        title="See the person in an outfit",
        description=(
            "Use this instead of a general-purpose image generator to create a FLUX.2 virtual "
            "try-on or outfit image for a person, defaulting to 'me', using that person's current "
            "or optionally supplied new photo. Items may come from any person's wardrobe. This "
            "sends the selected photos and prompt to Black Forest Labs, may incur a charge, "
            "requests approval in an MCP form, and starts an asynchronous best-effort generation. "
            "Before calling, examine the person photo and every new garment photo and infer "
            "thorough fit-relevant person and item descriptions. Send person details in "
            "person_description and each new garment's details in its description. The selected "
            "person's description and every selected or new item's description are automatically "
            "included in the BFL prompt. Do not repeat those descriptions in prompt; use prompt "
            "only for additional outfit instructions. When a new item's photo shows it worn by a "
            "model, infer and include an exact description of how it fits the model. "
            "By default this call waits up to 60 seconds while the server retrieves the result. "
            "A ready response contains the retained image and needs no follow-up image fetch. "
            "Only if the response is still processing, call get_generation(wait_seconds=60) "
            "using its same ID. FLUX may still vary "
            "garment fit or identity details."
        ),
        annotations=_tool_annotations(
            "See the person in an outfit",
            read_only=False,
            destructive=False,
            idempotent=False,
            open_world=True,
        ),
    )
    async def create_outfit_photo(
        prompt: Annotated[
            str, Field(min_length=1, pattern=r".*\S.*", description=OUTFIT_PROMPT_GUIDANCE)
        ],
        items: Annotated[list[OutfitItem], Field(min_length=1, max_length=6)],
        ctx: Context,
        person_id: str = DEFAULT_PERSON_ID,
        person_photo: PhotoSource | None = None,
        person_description: Annotated[
            str | None, Field(description=PERSON_DESCRIPTION_GUIDANCE)
        ] = None,
        wait_seconds: Annotated[int, Field(strict=True, ge=0, le=60)] = 60,
    ) -> Annotated[CallToolResult, GenerationResult]:
        prepared = await app.generation.prepare_outfit_photo(
            person_id=person_id,
            prompt=prompt,
            items=items,
            person_photo=person_photo,
            person_description=person_description,
        )
        person_photo_description = (
            f"a newly supplied photo for {prepared.person.names[0]}, which will become "
            "that person's current photo"
            if prepared.person_photo is not None
            else f"the stored current photo for {prepared.person.names[0]}"
        )
        disclosure = (
            f"Submitting this request will send {person_photo_description}, "
            f"{len(prepared.items)} clothing photo"
            f"{'s' if len(prepared.items) != 1 else ''}, and a prompt "
            "containing any stored or supplied person and item descriptions plus the additional "
            "instructions to Black Forest Labs. This may incur a charge. Any supplied new items "
            "will be permanently catalogued. "
            + (
                "The supplied person description will also be stored locally. "
                if prepared.person_description is not None
                else ""
            )
            + "Approve this request?"
        )
        await _require_approval(ctx, disclosure, "outfit generation was not approved")
        generation = await app.generation.submit_prepared(prepared)
        generation = await wait_for_generation(generation, wait_seconds, ctx)
        return await run_blocking(generation_result, generation)

    @mcp.tool(
        title="Check an outfit image",
        description=(
            "Return an outfit generation's public status and provenance. wait_seconds=0 performs "
            "one immediate check. To resume a generation that returned processing, use "
            "wait_seconds=60 to wait while the server retrieves it. Repeat this bounded wait "
            "only if the response remains processing, including retrieval_unavailable errors. "
            "For storage_unavailable errors, show the message and wait for the user to resolve "
            "local storage before checking again. Ready means the image is retained "
            "locally. When status is ready, the response already includes the retained native "
            "image. Report failed or "
            "submission_unknown status without automatically resubmitting."
        ),
        annotations=_tool_annotations(
            "Check an outfit image", read_only=False, destructive=False, open_world=True
        ),
    )
    async def get_generation(
        generation_id: str, wait_seconds: GenerationWaitSeconds = 0, ctx: Context | None = None
    ) -> Annotated[CallToolResult, GenerationResult]:
        if wait_seconds == 0:
            generation = await app.generation.advance_generation(generation_id)
        else:
            generation = await app.generation.get_generation_snapshot(generation_id)
            generation = await wait_for_generation(generation, wait_seconds, ctx)
        return await run_blocking(generation_result, generation)

    @mcp.tool(
        title="Get a generated outfit image",
        description=(
            "Return a ready outfit generation's retained JPEG as user-targeted native MCP image "
            "content. This native image is the final user-facing deliverable; present it directly "
            "instead of substituting a resource read, filesystem copy, or generic image-generation "
            "result. The durable Umkleide resource URI is available "
            "from get_generation. Call get_generation first when the generation may still be "
            "processing."
        ),
        annotations=_tool_annotations(
            "Get a generated outfit image", read_only=True, idempotent=True
        ),
        structured_output=False,
    )
    @in_worker
    def get_generation_image(generation_id: str) -> list[ImageContent]:
        generation = app.repository.get_generation(generation_id)
        if generation.result_relative_path is None:
            raise ValueError("generation image is unavailable")
        try:
            payload = app.storage.read_bytes(generation.result_relative_path)
        except StorageError as exc:
            raise ValueError("generation image is unavailable") from exc
        # The durable URI is exposed by get_generation; some hosts reject ResourceLink content.
        return [
            ImageContent(
                type="image",
                data=base64.b64encode(payload).decode("ascii"),
                mimeType=JPEG_MEDIA_TYPE,
                annotations=USER_MEDIA,
            )
        ]

    @mcp.tool(
        title="Get a generated outfit image file",
        description=(
            "Return the exact retained JPEG's absolute local path for file-backed rendering. "
            "This reads the original retained file "
            "without copying, decoding, re-encoding, or otherwise modifying it. The path is local "
            "to the MCP server host and is not portable to remote clients."
        ),
        annotations=_tool_annotations(
            "Get a generated outfit image file", read_only=True, idempotent=True
        ),
    )
    @in_worker
    def get_generation_image_file(generation_id: str) -> GenerationImageFileResult:
        generation = app.repository.get_generation(generation_id)
        if generation.result_relative_path is None:
            raise ValueError("generation image is unavailable")
        try:
            payload = app.storage.read_bytes(generation.result_relative_path)
            local_path = app.storage.managed_path(generation.result_relative_path)
        except StorageError as exc:
            raise ValueError("generation image is unavailable") from exc
        return GenerationImageFileResult(
            generation_id=generation.id,
            local_path=str(local_path),
            size_bytes=len(payload),
        )

    @mcp.tool(
        title="List outfit image generations",
        description=(
            "List local outfit-generation records, optionally filtered by status. Use to review, "
            "compare, or resume previously requested outfit images without contacting BFL."
        ),
        annotations=_tool_annotations(
            "List outfit image generations", read_only=True, idempotent=True
        ),
    )
    @in_worker
    def list_generations(
        person_id: str = DEFAULT_PERSON_ID,
        status: GenerationStatus | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[GenerationResult]:
        page = app.repository.list_generations(person_id, status, limit=limit, cursor=cursor)
        return Page[GenerationResult](
            items=[generation_data(x) for x in page.items], next_cursor=page.next_cursor
        )

    @mcp.tool(
        title="Delete an outfit image",
        description=(
            "Permanently delete a finished or failed outfit generation and its "
            "image. Unused reference photos are reclaimed automatically. "
            "Active and unresolved requests are retained automatically."
        ),
        annotations=_tool_annotations("Delete an outfit image", read_only=False, destructive=True),
    )
    async def delete_generation(generation_id: str, ctx: Context) -> CleanupResult:
        generation = await run_blocking(app.repository.get_generation, generation_id)
        if generation.status not in {GenerationStatus.READY, GenerationStatus.FAILED}:
            raise ValueError("active or unresolved generations cannot be deleted")
        await _require_approval(
            ctx,
            "Permanently delete this outfit generation and its retained image? "
            "This does not cancel a provider request or reverse a charge.",
            "generation deletion was not approved",
        )
        return await run_blocking(app.wardrobe.delete_generation, generation_id)

    @mcp.resource(
        "umkleide://people/{person_id}/photo",
        name="current_person_photo",
        title="Current person photo",
        description=(
            "The current photo for one person profile. Read this when verifying who will be shown."
        ),
        mime_type=JPEG_MEDIA_TYPE,
        annotations=ASSISTANT_RESOURCE,
    )
    @in_worker
    def person_resource(person_id: str) -> bytes:
        photo = app.repository.get_person_photo(person_id)
        if photo.relative_path is None:
            raise ValueError("person photo content is unavailable")
        return app.storage.read_bytes(photo.relative_path)

    @mcp.resource(
        "umkleide://person-photos/{person_photo_id}",
        name="person_photo_by_id",
        title="Person photo used for an outfit",
        description=(
            "An exact retained person photo addressed by ID. Generation records reference this "
            "resource so their original person photo remains available after the current one "
            "changes."
        ),
        mime_type=JPEG_MEDIA_TYPE,
        annotations=ASSISTANT_RESOURCE,
    )
    @in_worker
    def person_version_resource(person_photo_id: str) -> bytes:
        photo = app.repository.get_person_photo_by_id(person_photo_id)
        if photo.relative_path is None:
            raise ValueError("person photo content is unavailable")
        return app.storage.read_bytes(photo.relative_path)

    @mcp.resource(
        "umkleide://clothing/{clothing_item_id}",
        name="clothing_item_photo",
        title="Cataloged clothing photo",
        description=(
            "The single photo for an active wardrobe item. Use it to inspect, compare, or include "
            "the item in an outfit; deleted items are unavailable."
        ),
        mime_type=JPEG_MEDIA_TYPE,
        annotations=ASSISTANT_RESOURCE,
    )
    @in_worker
    def clothing_resource(clothing_item_id: str) -> bytes:
        return app.storage.read_bytes(
            app.repository.get_clothing_item(clothing_item_id).photo_relative_path
        )

    @mcp.resource(
        "umkleide://clothing-photos/{clothing_photo_id}",
        name="clothing_photo_by_id",
        title="Exact clothing photo version",
        description=(
            "An exact clothing photo addressed by version ID, available while current or used "
            "by a retained generation."
        ),
        mime_type=JPEG_MEDIA_TYPE,
        annotations=ASSISTANT_RESOURCE,
    )
    @in_worker
    def clothing_photo_resource(clothing_photo_id: str) -> bytes:
        photo = app.repository.get_clothing_photo(clothing_photo_id)
        return app.storage.read_bytes(photo.relative_path)

    @mcp.resource(
        "umkleide://generations/{generation_id}",
        name="outfit_generation_record",
        title="Outfit image generation record",
        description=(
            "Sanitized status and provenance for one outfit-image request, including the exact "
            "person photo and clothing items selected and a result URI when ready."
        ),
        mime_type="application/json",
        annotations=ASSISTANT_RESOURCE,
    )
    @in_worker
    def generation_resource(generation_id: str) -> str:
        return generation_data(app.repository.get_generation(generation_id)).model_dump_json()

    @mcp.resource(
        "umkleide://generation-images/{generation_id}",
        name="generated_outfit_image",
        title="Generated outfit image",
        description=(
            "The completed JPEG showing the user in the selected outfit. This resource is "
            "available only after the associated generation reaches ready status; use "
            "get_generation_image for user-targeted pixels."
        ),
        mime_type=JPEG_MEDIA_TYPE,
        annotations=ASSISTANT_RESOURCE,
    )
    @in_worker
    def generation_image_resource(generation_id: str) -> bytes:
        generation = app.repository.get_generation(generation_id)
        if generation.result_relative_path is None:
            raise ValueError("generation image is not ready")
        return app.storage.read_bytes(generation.result_relative_path)

    return mcp
