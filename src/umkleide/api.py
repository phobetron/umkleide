"""Typed, sanitized MCP input and output contracts."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .credentials import BflCredentialSource
from .guidance import CLOTHING_DESCRIPTION_GUIDANCE, PERSON_DESCRIPTION_GUIDANCE
from .models import DEFAULT_PERSON_ID, JPEG_MEDIA_TYPE, GenerationStatus

_DATA_URI_PATTERN = (
    r"^data:image/(?:jpeg|png|webp);base64,"
    r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{4})$"
)
_NONBLANK_PATTERN = r".*\S.*"
_MAX_DATA_URI_LENGTH = 27_962_051


class ApiModel(BaseModel):
    """Strict JSON-serializable values exposed at the MCP boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


PageItem = TypeVar("PageItem")


class Page(ApiModel, Generic[PageItem]):
    """A bounded page of MCP results with a decimal offset cursor."""

    items: list[PageItem]
    next_cursor: str | None = None


class DataUriPhotoSource(ApiModel):
    type: Literal["data_uri"]
    data: str = Field(
        max_length=_MAX_DATA_URI_LENGTH,
        pattern=_DATA_URI_PATTERN,
        description=(
            "Base64 JPEG, PNG, or WebP image data URI. Prefer another source for large images."
        ),
    )


class ImportPhotoSource(ApiModel):
    type: Literal["import"]
    path: str = Field(
        min_length=1,
        pattern=_NONBLANK_PATTERN,
        description="Relative path below the server's configured image import directory.",
    )

    @field_validator("path")
    @classmethod
    def require_contained_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if value.startswith(("/", "\\")) or "\\" in value or ".." in candidate.parts:
            raise ValueError("path must be relative and contained")
        return value


class UrlPhotoSource(ApiModel):
    type: Literal["url"]
    url: str = Field(
        min_length=9,
        max_length=2048,
        pattern=r"^https://",
        description="Public HTTPS URL from which Umkleide should retrieve the image.",
    )


PhotoSource: TypeAlias = Annotated[
    DataUriPhotoSource | ImportPhotoSource | UrlPhotoSource,
    Field(discriminator="type"),
]


class ClothingIdSelector(ApiModel):
    type: Literal["id"] = Field(description="Selects one stored clothing item by its identifier.")
    clothing_item_id: str = Field(min_length=1, description="Stored clothing item identifier.")


class ClothingMetadataSelector(ApiModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "anyOf": [
                {"required": ["query"]},
                {"required": ["category"]},
                {"required": ["metadata"]},
            ]
        },
    )
    type: Literal["metadata"] = Field(
        description="Selects the first stored item matching supplied criteria."
    )
    owner_person_id: str = Field(
        default=DEFAULT_PERSON_ID,
        description="Person whose wardrobe to search; defaults to the 'me' profile.",
    )
    query: str | None = Field(
        default=None,
        min_length=1,
        pattern=_NONBLANK_PATTERN,
        description="Optional nonblank name or description text.",
    )
    category: str | None = Field(
        default=None,
        min_length=1,
        pattern=_NONBLANK_PATTERN,
        description="Optional nonblank clothing category.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        min_length=1,
        description="Optional nonempty exact metadata key/value criteria.",
    )

    @field_validator("query", "category")
    @classmethod
    def trim_criteria(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def requires_a_criterion(self) -> ClothingMetadataSelector:
        if not (self.query or self.category or self.metadata):
            raise ValueError("metadata selector requires query, category, or metadata")
        return self


class NewClothingItemInput(ApiModel):
    type: Literal["new"] = Field(
        description="Permanently catalogs a new clothing item for this request."
    )
    owner_person_id: str = Field(
        default=DEFAULT_PERSON_ID,
        description="Person whose wardrobe will own this item; defaults to 'me'.",
    )
    name: str = Field(
        min_length=1, pattern=_NONBLANK_PATTERN, description="Nonblank clothing name."
    )
    category: str = Field(
        min_length=1, pattern=_NONBLANK_PATTERN, description="Nonblank clothing category."
    )
    photo: PhotoSource = Field(description="Photo of this permanently stored clothing item.")
    description: str | None = Field(
        default=None,
        description=(CLOTHING_DESCRIPTION_GUIDANCE),
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional clothing metadata.")

    @field_validator("name", "category")
    @classmethod
    def trim_required_strings(cls, value: str) -> str:
        return value.strip()


class ClothingItemPatch(ApiModel):
    """Patch body whose ``model_fields_set`` distinguishes omission from null."""

    name: str | None = Field(default=None, min_length=1, pattern=_NONBLANK_PATTERN)
    category: str | None = Field(default=None, min_length=1, pattern=_NONBLANK_PATTERN)
    description: str | None = Field(
        default=None,
        description=(
            CLOTHING_DESCRIPTION_GUIDANCE
            + " Examine the current or supplied photo. Explicit null clears the description."
        ),
    )
    metadata: dict[str, Any] | None = None
    photo: PhotoSource | None = None

    @model_validator(mode="after")
    def reject_required_nulls(self) -> ClothingItemPatch:
        for field in ("name", "category", "metadata", "photo"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class PersonPatch(ApiModel):
    """Person update whose fields must include at least one actual change."""

    names: list[str] | None = Field(default=None, min_length=1)
    description: str | None = Field(
        default=None,
        description=(
            PERSON_DESCRIPTION_GUIDANCE
            + " Examine the current or supplied photo. Explicit null clears the description."
        ),
    )
    photo: PhotoSource | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> PersonPatch:
        if not self.model_fields_set:
            raise ValueError("person update requires names, description, or photo")
        for field in ("names", "photo"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self

    @field_validator("names")
    @classmethod
    def normalize_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        names = [name.strip() for name in value]
        if any(not name for name in names):
            raise ValueError("names must be nonblank")
        return names


OutfitItem = Annotated[
    ClothingIdSelector | ClothingMetadataSelector | NewClothingItemInput,
    Field(discriminator="type", description="A stored or new durable clothing reference."),
]


class PersonResult(ApiModel):
    person_id: str = Field(description="Stable identifier of the person profile.")
    names: list[str] = Field(description="Names and phrases that identify this person.")
    description: str | None = Field(
        description="Stored fit-relevant size, build, body, proportion, and preference details."
    )
    person_photo_id: str | None = Field(description="Current person photo identifier, if set.")
    width: int | None = Field(default=None, ge=1, description="JPEG width when a photo is set.")
    height: int | None = Field(default=None, ge=1, description="JPEG height when a photo is set.")
    size_bytes: int | None = Field(default=None, ge=0, description="JPEG size when a photo is set.")
    resource_uri: str | None = Field(
        default=None,
        description=(
            "Current-photo resource URI when a photo is set; use get_person_photo_image with "
            "person_photo_id for native image content."
        ),
    )


class ClothingItemResult(ApiModel):
    owner_person_id: str = Field(description="Person whose wardrobe owns this item.")
    clothing_item_id: str = Field(description="Stable identifier of the clothing item.")
    name: str = Field(description="Human-readable clothing name.")
    category: str = Field(description="Clothing category.")
    description: str | None = Field(description="Optional clothing description.")
    metadata: dict[str, Any] = Field(description="Stored metadata for clothing lookup.")
    width: int = Field(ge=1, description="Normalized JPEG width in pixels.")
    height: int = Field(ge=1, description="Normalized JPEG height in pixels.")
    size_bytes: int = Field(ge=0, description="Normalized JPEG byte size.")
    resource_uri: str = Field(
        description=(
            "MCP resource URI for stable reference; use get_clothing_photo_image "
            "with clothing_photo_id for native image content."
        ),
    )
    clothing_photo_id: str = Field(description="Current immutable photo ID.")


class ResolvedClothingReference(ApiModel):
    clothing_item_id: str = Field(description="Identifier of the selected clothing item.")
    clothing_photo_id: str = Field(
        description="Immutable clothing photo identifier used for this generation."
    )
    name: str = Field(description="Name of the selected clothing item.")
    resource_uri: str = Field(
        description=(
            "MCP resource URI for the exact immutable clothing image; use "
            "get_clothing_photo_image with clothing_photo_id for native image content."
        )
    )


class GenerationResult(ApiModel):
    generation_id: str = Field(description="Stable identifier of the generation job.")
    person_id: str = Field(description="Identifier of the person shown in this generation.")
    status: GenerationStatus = Field(description="Current local generation state.")
    prompt: str = Field(description="Exact assembled prompt submitted for this generation.")
    person_photo_id: str = Field(
        description="Identifier of the person photo used for this request."
    )
    person_photo_resource_uri: str | None = Field(
        default=None,
        description=(
            "Stable MCP resource URI for the selected person photo, omitted after cleanup; use "
            "get_person_photo_image with person_photo_id for native image content."
        ),
    )
    selected_clothing: list[ResolvedClothingReference] = Field(
        description="Stored clothing references selected for this request."
    )
    image_resource_uri: str | None = Field(
        default=None,
        description=(
            "Stable generated-image resource URI for a retained ready result; use "
            "get_generation_image with generation_id for native image content."
        ),
    )
    error: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Sanitized error detail, if any. A processing job may have code "
            "'retrieval_unavailable' (repeat get_generation with wait_seconds=60) or "
            "'storage_unavailable' "
            "(resolve local storage before checking again), plus a message."
        ),
    )
    created_at: str = Field(description="UTC creation timestamp.")
    updated_at: str = Field(description="UTC last-update timestamp.")


GenerationWaitSeconds = Annotated[
    int,
    Field(
        default=0,
        strict=True,
        ge=0,
        le=60,
        description=(
            "Seconds to wait for local retrieval after checking a generation. Zero performs "
            "one immediate check; positive values observe the process-owned retriever."
        ),
    ),
]


class GenerationImageFileResult(ApiModel):
    generation_id: str = Field(description="Stable identifier of the ready generation job.")
    local_path: str = Field(
        description=(
            "Absolute path to the exact retained JPEG on the MCP server host. Local clients can "
            "use this for file-backed rendering without copying or modifying the image."
        )
    )
    media_type: Literal["image/jpeg"] = Field(
        default=JPEG_MEDIA_TYPE, description="Media type of the retained generation image."
    )
    size_bytes: int = Field(ge=0, description="Size of the retained JPEG in bytes.")


class RecordCounts(ApiModel):
    people: int = Field(ge=0, description="Person profile rows, including the default profile.")
    person_photos: int = Field(ge=0, description="Current and retained person-photo rows.")
    clothing_items: int = Field(ge=0, description="Active clothing-item rows.")
    clothing_photos: int = Field(ge=0, description="Current and retained clothing-photo rows.")
    generations: int = Field(ge=0, description="Outfit-generation rows.")
    generation_clothing: int = Field(
        ge=0, description="Clothing references attached to generation rows."
    )


class DiagnosticsResult(ApiModel):
    data_root: str = Field(description="Resolved private application-data directory.")
    import_root: str = Field(description="Resolved directory for relative-path image imports.")
    bfl_api_key_source: BflCredentialSource = Field(
        description="Active BFL credential source: environment, stored, or none."
    )
    media_usage_bytes: int = Field(ge=0, description="Bytes used by retained managed JPEGs.")
    media_quota_bytes: int = Field(ge=1, description="Configured managed-media quota in bytes.")
    media_available_bytes: int = Field(
        ge=0, description="Remaining managed-media capacity before reaching the quota."
    )
    records: RecordCounts = Field(description="Current SQLite row counts by application table.")
