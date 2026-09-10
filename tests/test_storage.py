from __future__ import annotations

import base64
import io
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

import umkleide.storage as storage_module
from umkleide.models import JPEG_MEDIA_TYPE
from umkleide.storage import JPEG_DATA_URI_PREFIX, PROVIDER_IMAGE_SIZE, MediaStorage, StorageError


def _image_bytes(
    image_format: str = "JPEG",
    *,
    size: tuple[int, int] = (12, 8),
    orientation: int | None = None,
    mode: str = "RGB",
) -> bytes:
    image = Image.new(mode, size, (30, 90, 180, 120) if "A" in mode else "royalblue")
    output = io.BytesIO()
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        kwargs["exif"] = exif
    image.save(output, format=image_format, **kwargs)
    return output.getvalue()


def _data_uri(payload: bytes, media_type: str = JPEG_MEDIA_TYPE) -> str:
    return f"data:{media_type};base64," + base64.b64encode(payload).decode("ascii")


@pytest.fixture
def storage(tmp_path: Path) -> MediaStorage:
    return MediaStorage(tmp_path / "data")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("data:text/plain;base64,SGVsbG8=", "base64 JPEG"),
        ("data:image/gif;base64,SGVsbG8=", "base64 JPEG"),
        (JPEG_DATA_URI_PREFIX + "not base64", "base64 JPEG"),
        (_data_uri(b"not an image"), "valid supported image"),
    ],
)
def test_prepare_user_image_only_accepts_allowed_image_data_uris(
    storage: MediaStorage, source: str, message: str
) -> None:
    with pytest.raises(StorageError, match=message):
        storage.prepare_user_image(source)


def test_prepare_user_image_enforces_byte_and_pixel_limits(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="byte limit"):
        MediaStorage(tmp_path / "bytes", max_bytes=10).prepare_user_image(_data_uri(_image_bytes()))
    with pytest.raises(StorageError, match="pixel limit"):
        MediaStorage(tmp_path / "pixels", max_pixels=20).prepare_user_image(
            _data_uri(_image_bytes())
        )


def test_managed_path_returns_only_contained_absolute_jpeg_paths(storage: MediaStorage) -> None:
    expected = (storage.root / "generations/result.jpg").resolve()

    assert storage.managed_path("generations/result.jpg") == expected
    with pytest.raises(StorageError, match="contained managed JPEG"):
        storage.managed_path("../result.jpg")


def test_prepare_imported_image_is_confined_to_configured_root(
    storage: MediaStorage, tmp_path: Path
) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    (import_root / "photo.png").write_bytes(_image_bytes("PNG"))
    outside = tmp_path / "outside.png"
    outside.write_bytes(_image_bytes("PNG"))

    assert storage.prepare_imported_image(import_root, "photo.png").width == 12
    with pytest.raises(StorageError, match="relative"):
        storage.prepare_imported_image(import_root, str(outside))
    (import_root / "escape.png").symlink_to(outside)
    with pytest.raises(StorageError, match="escapes"):
        storage.prepare_imported_image(import_root, "escape.png")


def test_prepare_remote_image_fetches_public_https_and_rejects_private_hosts(
    storage: MediaStorage, monkeypatch
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=_image_bytes("PNG"))
    )
    real_client = httpx.Client
    monkeypatch.setattr(storage, "_require_public_https_url", lambda _url: None)
    monkeypatch.setattr(
        storage_module.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    assert storage.prepare_remote_image("https://example.com/photo.png").height == 8
    with pytest.raises(StorageError, match="public addresses"):
        MediaStorage._require_public_https_url("https://127.0.0.1/photo.png")


@pytest.mark.parametrize(
    "path",
    ["../outside.jpg", "trash/a.jpg", "person/a.png", "/tmp/a.jpg", "person/a\\b.jpg"],
)
def test_managed_operations_reject_paths_outside_contained_jpeg_layout(
    storage: MediaStorage, path: str
) -> None:
    with pytest.raises(StorageError, match="contained managed JPEG"):
        storage.read_bytes(path)


def test_normalization_transposes_exif_removes_metadata_and_converts_to_jpeg(
    storage: MediaStorage,
) -> None:
    prepared = storage.prepare_user_image(
        _data_uri(_image_bytes("PNG", size=(12, 8), orientation=6, mode="RGBA"), "image/png")
    )
    storage.store("clothing/blue-coat.jpg", prepared)

    assert (prepared.width, prepared.height) == (8, 12)
    with Image.open(storage.root / "clothing/blue-coat.jpg") as image:
        assert (image.format, image.mode, image.size) == ("JPEG", "RGB", (8, 12))
        assert image.getexif().get(274) is None


def test_prepare_provider_image_requires_exact_flux_dimensions(storage: MediaStorage) -> None:
    prepared = storage.prepare_provider_image(
        _image_bytes("PNG", size=PROVIDER_IMAGE_SIZE, mode="RGBA")
    )

    assert (prepared.width, prepared.height) == PROVIDER_IMAGE_SIZE
    width, height = PROVIDER_IMAGE_SIZE
    with pytest.raises(StorageError, match=f"{width}x{height}"):
        storage.prepare_provider_image(_image_bytes(size=(width - 1, height)))


def test_store_is_atomic_and_keeps_media_private(storage: MediaStorage) -> None:
    prepared = storage.prepare_user_image(_data_uri(_image_bytes()))
    storage.store("person/current.jpg", prepared)

    destination = storage.root / "person/current.jpg"
    assert destination.read_bytes() == prepared.data
    assert not list(destination.parent.glob(".write-*"))
    if os.name == "posix":
        for path in (storage.root, destination.parent):
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_store_without_posix_permission_apis(tmp_path: Path, monkeypatch) -> None:
    portable_os = SimpleNamespace(
        name="nt", fdopen=os.fdopen, fsync=os.fsync, replace=os.replace
    )
    monkeypatch.setattr(storage_module, "os", portable_os)
    storage = MediaStorage(tmp_path / "data")
    prepared = storage.prepare_user_image_bytes(_image_bytes())

    storage.store("person/photo.jpg", prepared)

    assert storage.read_bytes("person/photo.jpg") == prepared.data


@pytest.mark.parametrize("failure_point", ["fsync", "replace"])
def test_failed_atomic_write_keeps_destination_and_removes_temporary_file(
    storage: MediaStorage, monkeypatch, failure_point: str
) -> None:
    original = storage.prepare_user_image_bytes(_image_bytes())
    replacement = storage.prepare_user_image_bytes(_image_bytes(size=(20, 20)))
    storage.store("person/photo.jpg", original)

    def fail(*_args):
        raise OSError("injected write failure")

    monkeypatch.setattr(storage_module.os, failure_point, fail)
    with pytest.raises(OSError, match="injected"):
        storage.store("person/photo.jpg", replacement)

    assert storage.read_bytes("person/photo.jpg") == original.data
    assert not list((storage.root / "person").glob(".write-*"))


def test_managed_image_data_uri_reads_stored_bytes_and_delete_is_idempotent(
    storage: MediaStorage,
) -> None:
    prepared = storage.prepare_user_image(_data_uri(_image_bytes()))
    storage.store("person/current.jpg", prepared)

    assert storage.managed_image_data_uri("person/current.jpg") == _data_uri(prepared.data)
    storage.delete_image("person/current.jpg")
    storage.delete_image("person/current.jpg")

    assert not (storage.root / "person/current.jpg").exists()


def test_empty_clothing_directories_are_removed_without_touching_retained_media(
    storage: MediaStorage,
) -> None:
    prepared = storage.prepare_user_image(_data_uri(_image_bytes()))
    storage.store("clothing/retained/photo.jpg", prepared)
    stale = storage.root / "clothing/stale"
    stale.mkdir()

    storage.remove_empty_clothing_directories()

    assert not stale.exists()
    assert (storage.root / "clothing/retained/photo.jpg").exists()

    storage.delete_image("clothing/retained/photo.jpg")
    storage.remove_empty_clothing_directories()

    assert not (storage.root / "clothing/retained").exists()
