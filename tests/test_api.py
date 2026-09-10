import pytest
from pydantic import TypeAdapter, ValidationError

from umkleide.api import (
    ClothingItemPatch,
    DataUriPhotoSource,
    GenerationWaitSeconds,
    ImportPhotoSource,
    PersonPatch,
    PhotoSource,
    UrlPhotoSource,
)


def test_generation_wait_seconds_is_a_bounded_strict_integer() -> None:
    adapter = TypeAdapter(GenerationWaitSeconds)
    assert adapter.validate_python(0) == 0
    assert adapter.validate_python(60) == 60
    for invalid in (-1, 61, 0.0, True, "5"):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)


def test_photo_sources_are_a_strict_three_variant_union() -> None:
    adapter = TypeAdapter(PhotoSource)
    valid_sources = (
        ({"type": "data_uri", "data": "data:image/png;base64,AA=="}, DataUriPhotoSource),
        ({"type": "import", "path": "person.jpg"}, ImportPhotoSource),
        ({"type": "url", "url": "https://example.com/person.jpg"}, UrlPhotoSource),
    )
    for value, kind in valid_sources:
        assert isinstance(adapter.validate_python(value), kind)
    for invalid in (
        {"data": "data:image/png;base64,AA=="},
        {"type": "import", "path": "/tmp/person.jpg"},
        {"type": "file", "path": "person.jpg"},
        {"type": "url", "url": "http://example.com/person.jpg"},
    ):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)


def test_clothing_patch_preserves_omission_and_rejects_null_required_changes() -> None:
    assert ClothingItemPatch().model_fields_set == set()
    assert ClothingItemPatch(description=None).model_fields_set == {"description"}
    for field in ("name", "category", "metadata", "photo"):
        with pytest.raises(ValidationError, match="cannot be null"):
            ClothingItemPatch.model_validate({field: None})


def test_person_patch_accepts_description_updates_and_explicit_clearing() -> None:
    assert PersonPatch(description="Usually wears size M").model_fields_set == {"description"}
    assert PersonPatch(description=None).model_fields_set == {"description"}
    with pytest.raises(ValidationError, match="requires names, description, or photo"):
        PersonPatch()
