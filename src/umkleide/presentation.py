"""Convert stored records to public MCP results."""

from .api import ClothingItemResult, GenerationResult, PersonResult, ResolvedClothingReference
from .models import ClothingItem, Generation, Person, PersonPhoto


def person_data(person: Person) -> PersonResult:
    return PersonResult(
        person_id=person.id,
        names=list(person.names),
        description=person.description,
        person_photo_id=person.current_photo_id,
        width=person.width,
        height=person.height,
        size_bytes=person.size_bytes,
        resource_uri=(
            f"umkleide://people/{person.id}/photo" if person.current_photo_id is not None else None
        ),
    )


def item_data(item: ClothingItem) -> ClothingItemResult:
    return ClothingItemResult(
        owner_person_id=item.person_id,
        clothing_item_id=item.id,
        name=item.name,
        category=item.category,
        description=item.description,
        metadata=item.metadata,
        width=item.width,
        height=item.height,
        size_bytes=item.size_bytes,
        resource_uri=f"umkleide://clothing/{item.id}",
        clothing_photo_id=item.photo_id,
    )


def generation_data(generation: Generation, person_photo: PersonPhoto) -> GenerationResult:
    selected = [
        ResolvedClothingReference(
            clothing_item_id=reference.clothing_item_id,
            clothing_photo_id=reference.clothing_photo_id,
            name=reference.name,
            resource_uri=f"umkleide://clothing-photos/{reference.clothing_photo_id}",
        )
        for reference in generation.selected_clothing
    ]
    person_photo_id = generation.person_photo_id
    return GenerationResult(
        generation_id=generation.id,
        person_id=generation.person_id,
        status=generation.status,
        prompt=generation.prompt,
        person_photo_id=person_photo_id,
        person_photo_resource_uri=(
            f"umkleide://person-photos/{person_photo_id}"
            if person_photo.relative_path is not None
            else None
        ),
        selected_clothing=selected,
        image_resource_uri=(
            f"umkleide://generation-images/{generation.id}"
            if generation.result_relative_path is not None
            else None
        ),
        error=generation.error,
        created_at=generation.created_at,
        updated_at=generation.updated_at,
    )
