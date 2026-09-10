"""Typed persistent rows, pagination, and immutable generation provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StringEnum(str, Enum):
    """String-valued enum suitable for SQLite and JSON."""


class GenerationStatus(StringEnum):
    SUBMISSION_UNKNOWN = "submission_unknown"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


DEFAULT_PERSON_ID = "me"
JPEG_MEDIA_TYPE = "image/jpeg"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: datetime | None = None) -> str:
    return (
        (utc_now() if value is None else value.astimezone(timezone.utc))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class PersistentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _relative(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(x in {"", ".", ".."} for x in value.split("/"))
    ):
        raise ValueError("must be a non-empty, POSIX relative path")
    return value


class PersonPhoto(PersistentModel):
    id: str
    person_id: str
    relative_path: str | None
    media_type: Literal["image/jpeg"] = JPEG_MEDIA_TYPE
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    size_bytes: int = Field(ge=0)
    is_current: bool
    created_at: str
    superseded_at: str | None = None
    _path = field_validator("relative_path")(_relative)


class Person(PersistentModel):
    id: str
    names: tuple[str, ...] = Field(min_length=1)
    description: str | None = None
    current_photo_id: str | None = None
    photo_relative_path: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    size_bytes: int | None = Field(default=None, ge=0)
    created_at: str
    updated_at: str
    _path = field_validator("photo_relative_path")(_relative)

    @field_validator("names")
    @classmethod
    def valid_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(name.strip() for name in value)
        if any(not name for name in names):
            raise ValueError("names must be nonblank")
        return names


class ClothingPhoto(PersistentModel):
    id: str
    clothing_item_id: str
    relative_path: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    size_bytes: int = Field(ge=0)
    created_at: str
    _path = field_validator("relative_path")(_relative)


class ClothingItem(PersistentModel):
    id: str
    person_id: str
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    photo_id: str
    photo_relative_path: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    size_bytes: int = Field(ge=0)
    created_at: str
    updated_at: str
    _path = field_validator("photo_relative_path")(_relative)

    @field_validator("name", "category")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain at least one non-whitespace character")
        return value


class ClothingProvenance(PersistentModel):
    clothing_item_id: str
    clothing_photo_id: str
    name: str


class Generation(PersistentModel):
    id: str
    person_id: str
    status: GenerationStatus
    prompt: str
    person_photo_id: str
    selected_clothing: tuple[ClothingProvenance, ...]
    model: Literal["flux-2-pro"] = "flux-2-pro"
    provider_request_id: str | None = None
    polling_url: str | None = None
    result_relative_path: str | None = None
    error: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    _path = field_validator("result_relative_path", mode="before")(_relative)


PageItem = TypeVar("PageItem")


class CursorPage(PersistentModel, Generic[PageItem]):
    items: tuple[PageItem, ...]
    next_cursor: str | None = None


class CleanupResult(PersistentModel):
    clothing_items_deleted: int = Field(ge=0)
    person_photos_deleted: int = Field(ge=0)
    generations_deleted: int = Field(ge=0)
    bytes_deleted: int = Field(ge=0)
    has_more: bool
