import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import Literal

import pytest
from mcp.server.fastmcp import Context
from mcp.types import CallToolResult, ElicitResult, ImageContent, TextContent
from PIL import Image

import umkleide.config as config_module
from umkleide.api import (
    ClothingIdSelector,
    DataUriPhotoSource,
    NewClothingItemInput,
    PersonPatch,
    UrlPhotoSource,
)
from umkleide.application import AppState
from umkleide.bfl import BFLJobState, BFLPollResult, BFLSubmission, Flux2ProRequest
from umkleide.config import AppConfig
from umkleide.credentials import BflCredentialSource
from umkleide.database import Database
from umkleide.generations import GenerationService, Provider
from umkleide.models import (
    DEFAULT_PERSON_ID,
    JPEG_MEDIA_TYPE,
    ClothingProvenance,
    Generation,
    GenerationStatus,
    utc_timestamp,
)
from umkleide.repositories import Repository, RepositoryError, new_id
from umkleide.retrieval import GenerationRetriever
from umkleide.server import create_server
from umkleide.storage import JPEG_DATA_URI_PREFIX, PROVIDER_IMAGE_SIZE, MediaStorage, StorageError
from umkleide.wardrobe import WardrobeService

EXPECTED_TOOLS = {
    "add_person",
    "get_diagnostics",
    "list_people",
    "get_person",
    "update_person",
    "delete_person",
    "get_person_photo_image",
    "add_clothing_item",
    "list_clothing_items",
    "find_clothing_items",
    "get_clothing_item",
    "get_clothing_photo_image",
    "update_clothing_item",
    "delete_clothing_item",
    "create_outfit_photo",
    "get_generation",
    "get_generation_image",
    "get_generation_image_file",
    "list_generations",
    "delete_generation",
}
GENERATION_ID = "generation"
GENERATION_PATH = f"generations/{GENERATION_ID}.jpg"
FIRST_PERSON_PATH = "person/first.jpg"


def _server(config: AppConfig, *, provider: Provider | None = None):
    database = Database(config.database_path)
    database.initialize()
    repository = Repository(database)
    storage = MediaStorage(config.data_root)
    wardrobe = WardrobeService(
        repository, storage, quota_bytes=config.media_quota_bytes, import_root=config.import_root
    )
    generation = GenerationService(repository, wardrobe, provider)
    retriever = GenerationRetriever(generation)
    app = AppState(
        repository=repository,
        storage=storage,
        wardrobe=wardrobe,
        generation=generation,
        provider=provider,
        bfl_api_key_source=BflCredentialSource.NONE,
        retriever=retriever,
    )
    return create_server(app=app, config=config)


def _tool(server, name: str):
    tool = server._tool_manager.get_tool(name)
    assert tool is not None
    return tool.fn


def _call(server, name: str, *args, **kwargs):
    return asyncio.run(_tool(server, name)(*args, **kwargs))


def _jpeg() -> bytes:
    image = Image.new("RGB", (2, 3), "steelblue")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _source() -> DataUriPhotoSource:
    return DataUriPhotoSource(
        type="data_uri", data=JPEG_DATA_URI_PREFIX + base64.b64encode(_jpeg()).decode()
    )


def _set_elicitation_response(monkeypatch, response: ElicitResult | Exception) -> dict[str, object]:
    captured: dict[str, object] = {}

    class Session:
        async def elicit_form(
            self,
            *,
            message: str,
            requestedSchema: dict[str, object],
            related_request_id: object,
        ) -> ElicitResult:
            captured.update(
                message=message,
                schema=requestedSchema,
                related_request_id=related_request_id,
            )
            if isinstance(response, Exception):
                raise response
            return response

    session = Session()
    monkeypatch.setattr(Context, "session", property(lambda _ctx: session))
    monkeypatch.setattr(Context, "request_id", property(lambda _ctx: "request-id"))
    return captured


def _seed_generation(config: AppConfig):
    config.ensure_data_root()
    database = Database(config.database_path)
    database.initialize()
    repository = Repository(database)
    storage = MediaStorage(config.data_root)
    first_person = storage.prepare_user_image(_source().data)
    storage.store(FIRST_PERSON_PATH, first_person)
    person = repository.set_person_photo(
        DEFAULT_PERSON_ID,
        FIRST_PERSON_PATH,
        first_person.width,
        first_person.height,
        first_person.size_bytes,
    )
    item_id, photo_id = new_id(), new_id()
    clothing_path = f"clothing/{item_id}/{photo_id}.jpg"
    clothing = storage.prepare_user_image(_source().data)
    storage.store(clothing_path, clothing)
    item = repository.create_clothing_item(
        DEFAULT_PERSON_ID,
        item_id,
        "Jacket",
        "outerwear",
        None,
        {},
        clothing.width,
        clothing.height,
        clothing.size_bytes,
        photo_id=photo_id,
        photo_relative_path=clothing_path,
    )
    result = storage.prepare_user_image(_source().data)
    storage.store(GENERATION_PATH, result)
    now = utc_timestamp()
    repository.create_generation(
        Generation(
            id=GENERATION_ID,
            person_id=DEFAULT_PERSON_ID,
            status=GenerationStatus.READY,
            prompt="portrait",
            person_photo_id=person.id,
            selected_clothing=(
                ClothingProvenance(
                    clothing_item_id=item.id, clothing_photo_id=item.photo_id, name=item.name
                ),
            ),
            result_relative_path=GENERATION_PATH,
            created_at=now,
            updated_at=now,
        )
    )
    replacement = storage.prepare_user_image(_source().data)
    storage.store("person/current.jpg", replacement)
    repository.set_person_photo(
        DEFAULT_PERSON_ID,
        "person/current.jpg",
        replacement.width,
        replacement.height,
        replacement.size_bytes,
    )
    return person, item


def test_mcp_discovers_the_supported_tools_resources_and_input_contracts(tmp_path: Path) -> None:
    server = _server(AppConfig(tmp_path))
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(tools) == EXPECTED_TOOLS
    assert asyncio.run(server.list_prompts()) == []
    assert {
        resource.name
        for resource in [
            *asyncio.run(server.list_resources()),
            *asyncio.run(server.list_resource_templates()),
        ]
    } == {
        "current_person_photo",
        "person_photo_by_id",
        "clothing_item_photo",
        "clothing_photo_by_id",
        "outfit_generation_record",
        "generated_outfit_image",
    }

    photo_schema = tools["add_person"].inputSchema["properties"]["photo"]
    assert photo_schema["discriminator"]["propertyName"] == "type"
    assert set(photo_schema["discriminator"]["mapping"]) == {
        "data_uri",
        "import",
        "url",
    }
    add_person = tools["add_person"].inputSchema
    assert add_person["required"] == ["names", "photo"]
    assert set(add_person["properties"]) == {"names", "photo", "description"}
    add = tools["add_clothing_item"].inputSchema
    assert add["required"] == ["name", "category", "photo"]
    assert set(add["properties"]) == {
        "owner_person_id",
        "name",
        "category",
        "photo",
        "description",
        "metadata",
    }
    update = tools["update_clothing_item"].inputSchema
    assert set(update["properties"]) == {"clothing_item_id", "changes"}
    outfit = tools["create_outfit_photo"].inputSchema
    assert outfit["required"] == ["prompt", "items"]
    assert {"person_photo", "person_description", "wait_seconds"} <= set(outfit["properties"])
    assert outfit["properties"]["items"]["items"]["discriminator"]["propertyName"] == "type"
    assert outfit["properties"]["items"]["minItems"] == 1
    assert outfit["properties"]["items"]["maxItems"] == 6
    assert outfit["properties"]["wait_seconds"]["maximum"] == 60
    assert tools["get_generation"].inputSchema["properties"]["wait_seconds"]["maximum"] == 60
    assert tools["get_generation"].outputSchema == tools["create_outfit_photo"].outputSchema
    image_file_tool = tools["get_generation_image_file"]
    assert image_file_tool.outputSchema is not None
    assert set(image_file_tool.outputSchema["properties"]) == {
        "generation_id",
        "local_path",
        "media_type",
        "size_bytes",
    }
    for image_tool in (
        "get_person_photo_image",
        "get_clothing_photo_image",
        "get_generation_image",
    ):
        assert tools[image_tool].outputSchema is None
        annotations = tools[image_tool].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.idempotentHint is True
    image_file_annotations = image_file_tool.annotations
    assert image_file_annotations is not None
    assert image_file_annotations.readOnlyHint is True
    assert image_file_annotations.idempotentHint is True
    for tool_name in ("delete_person", "delete_clothing_item", "delete_generation"):
        annotations = tools[tool_name].annotations
        assert annotations is not None and annotations.destructiveHint is True
    annotations = tools["get_person"].annotations
    assert annotations is not None and annotations.readOnlyHint is True
    annotations = tools["create_outfit_photo"].annotations
    assert annotations is not None and annotations.openWorldHint is True


def test_mcp_instructions_advertise_the_import_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config_module.tempfile, "gettempdir", lambda: str(tmp_path))
    import_root = tmp_path / "umkleide-imports"
    server = _server(config=AppConfig(data_root=tmp_path / "data"))
    instructions = server._mcp_server.create_initialization_options().instructions

    assert instructions is not None
    assert str(import_root.resolve()) in instructions[:512]
    assert "photo source type 'import'" in instructions[:512]
    assert "response already contains native MCP image content" in instructions
    assert "Never automatically resubmit a submission_unknown generation" in instructions


def test_default_person_and_aliases_are_exposed_to_the_agent(tmp_path: Path) -> None:
    server = _server(config=AppConfig(data_root=tmp_path / "data"))

    default = _call(server, "get_person")
    assert (default.person_id, default.names) == (DEFAULT_PERSON_ID, [DEFAULT_PERSON_ID])
    assert default.person_photo_id is None

    updated = _call(
        server,
        "update_person",
        PersonPatch(
            names=[DEFAULT_PERSON_ID, "Charles"],
            description="Usually wears size M; prefers relaxed fits",
            photo=_source(),
        ),
    )
    assert updated.person_id == DEFAULT_PERSON_ID
    assert updated.names == [DEFAULT_PERSON_ID, "Charles"]
    assert updated.description == "Usually wears size M; prefers relaxed fits"
    assert updated.person_photo_id is not None

    susan = _call(
        server,
        "add_person",
        ["Susan", "my wife", "the lady"],
        _source(),
        "Usually wears size 38; straight build",
    )
    assert susan.names == ["Susan", "my wife", "the lady"]
    assert susan.description == "Usually wears size 38; straight build"
    assert [person.person_id for person in _call(server, "list_people").items] == [
        DEFAULT_PERSON_ID,
        susan.person_id,
    ]


def test_diagnostics_report_paths_provider_status_usage_and_record_counts(tmp_path: Path) -> None:
    config = AppConfig(data_root=tmp_path / "data", media_quota_bytes=9999)
    server = _server(config=config)
    person = _call(server, "add_person", ["Person"], _source())

    result = _call(server, "get_diagnostics")

    assert result.data_root == str(config.data_root)
    assert result.import_root == str(config.import_root)
    assert result.bfl_api_key_source == "none"
    assert result.media_usage_bytes == person.size_bytes
    assert result.media_quota_bytes == 9999
    assert result.media_available_bytes == 9999 - person.size_bytes
    assert result.records.model_dump() == {
        "people": 2,
        "person_photos": 1,
        "clothing_items": 0,
        "clothing_photos": 0,
        "generations": 0,
        "generation_clothing": 0,
    }


def test_generation_preflight_rejects_missing_credential_before_elicitation(
    tmp_path: Path, monkeypatch
) -> None:
    prompt = "A formal outfit"
    person_description = "Usually wears size M; straight build"
    server = _server(config=AppConfig(data_root=tmp_path / "data"))
    elicitation = _set_elicitation_response(monkeypatch, ElicitResult(action="accept"))
    person = _call(server, "add_person", ["Person"], _source())
    item = ClothingIdSelector(type="id", clothing_item_id="jacket")

    with pytest.raises(Exception, match="configure Umkleide"):
        asyncio.run(
            _tool(server, "create_outfit_photo")(
                person_id=person.person_id,
                prompt=prompt,
                items=[item],
                ctx=Context(fastmcp=server),
                person_photo=_source(),
                person_description=person_description,
            )
        )
    assert elicitation == {}


def test_declined_clothing_deletion_keeps_the_item(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig(data_root=tmp_path / "data")
    server = _server(config=config)
    _set_elicitation_response(monkeypatch, ElicitResult(action="decline"))
    item = _call(server, "add_clothing_item", "Coat", "outerwear", _source())

    with pytest.raises(ValueError, match="approved"):
        asyncio.run(
            _tool(server, "delete_clothing_item")(item.clothing_item_id, Context(fastmcp=server))
        )

    assert Repository(Database(config.database_path)).get_clothing_item(item.clothing_item_id)


@pytest.mark.parametrize(
    ("tool_name", "argument_name", "identifier", "relative_path"),
    [
        ("get_person_photo_image", "person_photo_id", "person", FIRST_PERSON_PATH),
        (
            "get_clothing_photo_image",
            "clothing_photo_id",
            "clothing",
            "clothing",
        ),
        (
            "get_generation_image",
            "generation_id",
            GENERATION_ID,
            GENERATION_PATH,
        ),
    ],
)
def test_image_tools_return_native_jpeg_content(
    tmp_path: Path,
    tool_name: str,
    argument_name: str,
    identifier: str,
    relative_path: str,
) -> None:
    config = AppConfig(data_root=tmp_path / "data")
    person, item = _seed_generation(config)
    server = _server(config=config)
    if identifier == "person":
        identifier = person.id
    elif identifier == "clothing":
        identifier = item.photo_id
        relative_path = item.photo_relative_path
    elif tool_name == "get_generation_image":
        output = io.BytesIO()
        Image.effect_noise(PROVIDER_IMAGE_SIZE, 64).convert("RGB").save(output, format="JPEG")
        storage = MediaStorage(config.data_root)
        storage.store(relative_path, storage.prepare_provider_image(output.getvalue()))

    content = asyncio.run(server.call_tool(tool_name, {argument_name: identifier}))

    assert isinstance(content, list)
    assert len(content) == 1
    image = content[0]
    assert isinstance(image, ImageContent)
    assert image.mimeType == JPEG_MEDIA_TYPE
    if tool_name == "get_generation_image":
        assert len(image.data) > 100_000
        assert image.annotations is not None
        assert image.annotations.audience == ["user", "assistant"]
        assert image.annotations.priority is None
    assert base64.b64decode(image.data, validate=True) == MediaStorage(config.data_root).read_bytes(
        relative_path
    )


def test_generation_image_file_returns_the_exact_retained_local_jpeg(tmp_path: Path) -> None:
    config = AppConfig(data_root=tmp_path / "data")
    _seed_generation(config)
    server = _server(config=config)

    result = _call(server, "get_generation_image_file", GENERATION_ID)
    expected_path = (config.data_root / GENERATION_PATH).resolve()

    assert result.generation_id == GENERATION_ID
    assert result.local_path == str(expected_path)
    assert result.media_type == JPEG_MEDIA_TYPE
    assert result.size_bytes == expected_path.stat().st_size
    assert expected_path.read_bytes() == MediaStorage(config.data_root).read_bytes(GENERATION_PATH)

    tool_result = asyncio.run(
        server.call_tool("get_generation_image_file", {"generation_id": GENERATION_ID})
    )
    assert isinstance(tool_result, tuple)
    content, structured = tool_result
    assert isinstance(content, list)
    assert isinstance(structured, dict)
    assert len(content) == 1
    assert isinstance(content[0], TextContent)
    assert json.loads(content[0].text) == result.model_dump()
    assert structured == result.model_dump()


def test_ready_generation_result_includes_structured_data_and_exact_retained_jpeg(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_root=tmp_path / "data")
    _seed_generation(config)
    server = _server(config=config)

    result = _call(server, "get_generation", GENERATION_ID)

    assert isinstance(result, CallToolResult)
    assert result.structuredContent is not None
    assert result.structuredContent["generation_id"] == GENERATION_ID
    assert len(result.content) == 1
    image = result.content[0]
    assert isinstance(image, ImageContent)
    assert image.mimeType == JPEG_MEDIA_TYPE
    assert base64.b64decode(image.data, validate=True) == MediaStorage(config.data_root).read_bytes(
        GENERATION_PATH
    )


@pytest.mark.parametrize("unavailable", ["missing", "unreadable"])
def test_ready_generation_with_unavailable_retained_image_fails(
    tmp_path: Path, monkeypatch, unavailable: str
) -> None:
    config = AppConfig(data_root=tmp_path / "data")
    _seed_generation(config)
    if unavailable == "missing":
        (config.data_root / GENERATION_PATH).unlink()
    else:
        monkeypatch.setattr(
            MediaStorage,
            "read_bytes",
            lambda _self, _path: (_ for _ in ()).throw(StorageError("read failed")),
        )
    server = _server(config=config)
    with pytest.raises(ValueError, match="retained generation image is unavailable"):
        _call(server, "get_generation", GENERATION_ID)


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_declining_generation_deletion_keeps_image(
    tmp_path: Path, monkeypatch, action: Literal["decline", "cancel"]
):
    config = AppConfig(tmp_path)
    _seed_generation(config)
    server = _server(config)
    _set_elicitation_response(monkeypatch, ElicitResult(action=action))
    with pytest.raises(ValueError, match="not approved"):
        _call(server, "delete_generation", GENERATION_ID, Context(fastmcp=server))
    assert (tmp_path / GENERATION_PATH).exists()
    assert _call(server, "get_generation_image_file", GENERATION_ID).generation_id == GENERATION_ID


def test_deleting_generation_reclaims_unused_reference_and_keeps_catalog(
    tmp_path: Path, monkeypatch
):
    config = AppConfig(tmp_path)
    person, item = _seed_generation(config)
    server = _server(config)
    _set_elicitation_response(monkeypatch, ElicitResult(action="accept"))
    result = _call(server, "delete_generation", GENERATION_ID, Context(fastmcp=server))
    assert result.generations_deleted == 1
    assert not (tmp_path / GENERATION_PATH).exists()
    assert not (tmp_path / FIRST_PERSON_PATH).exists()
    assert _call(server, "get_person").person_photo_id != person.id
    assert _call(server, "get_clothing_item", item.id).clothing_item_id == item.id
    assert _call(server, "list_generations").items == []
    with pytest.raises(RepositoryError, match="not found"):
        _call(server, "get_generation_image_file", GENERATION_ID)


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_generation_approval_precedes_catalog_changes(
    tmp_path: Path, monkeypatch, action: Literal["decline", "cancel"]
):
    config = AppConfig(tmp_path)

    class ProviderStub:
        async def submit_flux_2_pro(self, request: Flux2ProRequest) -> BFLSubmission:
            return BFLSubmission("request", "https://example.com/poll")

        async def poll(self, polling_url: str) -> BFLPollResult:
            return BFLPollResult(BFLJobState.PROCESSING)

        async def download_result(self, result_url: str) -> bytes:
            return b""

    server = _server(config, provider=ProviderStub())
    elicitation = _set_elicitation_response(monkeypatch, ElicitResult(action=action))

    async def must_not_generate(*args, **kwargs):
        pytest.fail("generation ran without approval")

    monkeypatch.setattr(GenerationService, "submit_prepared", must_not_generate)
    with pytest.raises(ValueError, match="not approved"):
        _call(
            server,
            "create_outfit_photo",
            prompt="An outfit",
            items=[
                NewClothingItemInput(type="new", name="Coat", category="outerwear", photo=_source())
            ],
            person_photo=_source(),
            ctx=Context(fastmcp=server),
        )
    assert elicitation["schema"] == {"type": "object", "properties": {}}
    assert _call(server, "get_person").person_photo_id is None
    assert _call(server, "list_clothing_items").items == []
    assert _call(server, "list_generations").items == []
    assert list(config.data_root.rglob("*.jpg")) == []


async def test_slow_image_preparation_does_not_block_other_tools(tmp_path: Path, monkeypatch):
    server = _server(AppConfig(tmp_path))
    started = asyncio.Event()
    loop = asyncio.get_running_loop()
    prepared = MediaStorage(tmp_path).prepare_user_image(_source().data)

    def slow_image(self, url):
        loop.call_soon_threadsafe(started.set)
        time.sleep(0.2)
        return prepared

    monkeypatch.setattr(MediaStorage, "prepare_remote_image", slow_image)
    creation = asyncio.create_task(
        _tool(server, "add_person")(
            ["Susan"], UrlPhotoSource(type="url", url="https://example.com/photo.jpg")
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    person = await asyncio.wait_for(_tool(server, "get_person")(), timeout=0.1)
    assert person.person_id == DEFAULT_PERSON_ID
    assert not creation.done()
    await creation
