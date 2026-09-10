"""Small, private filesystem storage for Umkleide's local JPEG images."""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import os
import re
import socket
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from .bfl import FLUX_2_HEIGHT, FLUX_2_WIDTH
from .config import MANAGED_DIRECTORIES
from .models import JPEG_MEDIA_TYPE

DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_PIXELS = 32_000_000
JPEG_DATA_URI_PREFIX = f"data:{JPEG_MEDIA_TYPE};base64,"
PROVIDER_IMAGE_SIZE = (FLUX_2_WIDTH, FLUX_2_HEIGHT)
_DATA_URI = re.compile(
    r"^data:image/(?:jpeg|png|webp);base64,"
    r"((?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{4}))$"
)
_MANAGED_DIRECTORIES = frozenset(MANAGED_DIRECTORIES)
_SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
_MAX_REDIRECTS = 3


class StorageError(ValueError):
    """An image source or managed-storage operation could not be completed."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A validated, metadata-free JPEG ready for one persistent write."""

    data: bytes
    width: int
    height: int

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class MediaStorage:
    """Store normalized JPEGs below one private local data directory."""

    def __init__(
        self,
        data_root: Path | str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        if max_bytes <= 0 or max_pixels <= 0:
            raise ValueError("max_bytes and max_pixels must be positive")
        self.root = Path(data_root).expanduser().resolve()
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        self._private_directory(self.root)
        for name in _MANAGED_DIRECTORIES:
            self._private_directory(self.root / name)

    def prepare_user_image(self, data_uri: str) -> PreparedImage:
        """Decode a user data URI once into a normalized JPEG."""
        return self.prepare_user_image_bytes(self._read_data_uri(data_uri))

    def prepare_user_image_bytes(self, payload: bytes) -> PreparedImage:
        """Normalize bounded JPEG, PNG, or WebP bytes supplied by a user."""
        if len(payload) > self.max_bytes:
            raise StorageError("source image exceeds byte limit")
        image = self._decode_image(payload)
        return PreparedImage(self._encode_jpeg(image), image.width, image.height)

    def prepare_imported_image(self, import_root: Path | str, relative_path: str) -> PreparedImage:
        """Read an image from one explicitly configured, contained import path."""
        candidate = Path(relative_path)
        if (
            not relative_path
            or candidate.is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise StorageError("import path must be relative to the image import directory")
        root = Path(import_root).expanduser().resolve()
        try:
            path = (root / candidate).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError) as exc:
            raise StorageError(
                "import path is unavailable or escapes the import directory"
            ) from exc
        return self.prepare_user_image_bytes(self._read_bounded_file(path))

    def prepare_remote_image(self, url: str) -> PreparedImage:
        """Retrieve one bounded image from a public HTTPS URL."""
        current = url
        try:
            with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
                for redirect_count in range(_MAX_REDIRECTS + 1):
                    self._require_public_https_url(current)
                    with client.stream("GET", current) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location or redirect_count == _MAX_REDIRECTS:
                                raise StorageError("image URL has too many or invalid redirects")
                            current = urljoin(current, location)
                            continue
                        response.raise_for_status()
                        payload = bytearray()
                        for chunk in response.iter_bytes():
                            payload.extend(chunk)
                            if len(payload) > self.max_bytes:
                                raise StorageError("source image exceeds byte limit")
                        return self.prepare_user_image_bytes(bytes(payload))
        except StorageError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise StorageError("image URL could not be retrieved") from exc
        raise StorageError("image URL could not be retrieved")  # pragma: no cover

    def prepare_provider_image(self, payload: bytes) -> PreparedImage:
        """Decode a provider result once, requiring the FLUX output dimensions."""
        image = self._decode_image(payload)
        if image.size != PROVIDER_IMAGE_SIZE:
            width, height = PROVIDER_IMAGE_SIZE
            raise StorageError(f"provider image must be exactly {width}x{height} pixels")
        return PreparedImage(self._encode_jpeg(image), image.width, image.height)

    def managed_usage(self) -> int:
        """Return actual bytes below the fixed DB-managed media directories."""
        total = 0
        for name in _MANAGED_DIRECTORIES:
            directory = self.root / name
            for path in directory.rglob("*.jpg"):
                if path.is_file():
                    total += path.stat().st_size
        return total

    def store(self, relative_path: str, prepared: PreparedImage) -> None:
        """Atomically persist a previously prepared normalized JPEG."""
        self._atomic_write(self._managed_path(relative_path), prepared.data)

    def read_bytes(self, relative_path: str) -> bytes:
        """Read one contained managed JPEG."""
        path = self._managed_path(relative_path)
        try:
            if not path.is_file():
                raise StorageError("managed image does not exist")
            if path.stat().st_size > self.max_bytes:
                raise StorageError("managed image exceeds byte limit")
            payload = path.read_bytes()
        except OSError as exc:
            raise StorageError("managed image could not be read") from exc
        if len(payload) > self.max_bytes:
            raise StorageError("managed image exceeds byte limit")
        return payload

    def managed_path(self, relative_path: str) -> Path:
        """Return the absolute path for one contained managed JPEG."""
        return self._managed_path(relative_path)

    def delete_image(self, relative_path: str) -> None:
        """Remove one managed JPEG; a missing image is already deleted."""
        path = self._managed_path(relative_path)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise StorageError("managed image could not be deleted") from exc

    def delete_image_with_size(self, relative_path: str) -> int:
        """Delete one DB-owned path and return bytes actually removed."""
        path = self._managed_path(relative_path)
        try:
            size = path.stat().st_size
            path.unlink()
            return size
        except FileNotFoundError:
            return 0
        except OSError as exc:
            raise StorageError("managed image could not be deleted") from exc

    def remove_empty_clothing_directories(self) -> None:
        """Remove empty per-item directories left after clothing-photo deletion."""
        clothing_root = self.root / "clothing"
        try:
            directories = tuple(path for path in clothing_root.iterdir() if path.is_dir())
            for directory in directories:
                try:
                    directory.rmdir()
                except OSError:
                    pass
        except OSError as exc:
            raise StorageError("empty clothing directories could not be cleaned up") from exc

    def managed_image_data_uri(self, relative_path: str) -> str:
        """Return one managed JPEG as an inline data URI for a provider request."""
        payload = self.read_bytes(relative_path)
        return JPEG_DATA_URI_PREFIX + base64.b64encode(payload).decode("ascii")

    def _read_data_uri(self, data_uri: str) -> bytes:
        if not isinstance(data_uri, str):
            raise StorageError("source must be a base64 image data URI")
        match = _DATA_URI.fullmatch(data_uri)
        if match is None:
            raise StorageError("source must be a base64 JPEG, PNG, or WebP data URI")
        try:
            payload = base64.b64decode(match.group(1), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise StorageError("source contains invalid base64 data") from exc
        if len(payload) > self.max_bytes:
            raise StorageError("source image exceeds byte limit")
        return payload

    def _read_bounded_file(self, path: Path) -> bytes:
        try:
            if not path.is_file():
                raise StorageError("import path is not a regular file")
            with path.open("rb") as handle:
                payload = handle.read(self.max_bytes + 1)
        except StorageError:
            raise
        except OSError as exc:
            raise StorageError("import image could not be read") from exc
        if len(payload) > self.max_bytes:
            raise StorageError("source image exceeds byte limit")
        return payload

    @staticmethod
    def _require_public_https_url(url: str) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise StorageError("image URL must be a public HTTPS URL without credentials")
        try:
            port = parsed.port or 443
            addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        except (OSError, ValueError) as exc:
            raise StorageError("image URL host could not be resolved") from exc
        if not addresses:
            raise StorageError("image URL host could not be resolved")
        for address in addresses:
            ip = ipaddress.ip_address(str(address[4][0]).split("%", 1)[0])
            if not ip.is_global:
                raise StorageError("image URL must resolve only to public addresses")

    def _decode_image(self, payload: bytes) -> Image.Image:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload), formats=_SUPPORTED_FORMATS) as probe:
                    probe.verify()
                with Image.open(io.BytesIO(payload), formats=_SUPPORTED_FORMATS) as opened:
                    if opened.width * opened.height > self.max_pixels:
                        raise StorageError("source image exceeds pixel limit")
                    image = ImageOps.exif_transpose(opened).copy()
        except StorageError:
            raise
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise StorageError("source is not a valid supported image") from exc
        if image.width < 1 or image.height < 1:
            raise StorageError("source image has invalid dimensions")
        if image.mode != "RGB":
            background = Image.new("RGB", image.size, "white")
            if "A" in image.getbands():
                background.paste(image.convert("RGBA"), mask=image.getchannel("A"))
            else:
                background.paste(image.convert("RGB"))
            image = background
        return image

    def _encode_jpeg(self, image: Image.Image) -> bytes:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95, optimize=True)
        payload = output.getvalue()
        if len(payload) > self.max_bytes:
            raise StorageError("normalized image exceeds byte limit")
        return payload

    def _managed_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if (
            not relative_path
            or candidate.is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or candidate.parts[0] not in _MANAGED_DIRECTORIES
            or candidate.suffix.lower() != ".jpg"
        ):
            raise StorageError("path must be a contained managed JPEG path")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root / candidate.parts[0])
        except ValueError as exc:
            raise StorageError("path escapes managed storage") from exc
        return resolved

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        self._private_directory(destination.parent)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".write-", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                if os.name == "posix":
                    os.fchmod(handle.fileno(), 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _private_directory(directory: Path) -> None:
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        if os.name == "posix":
            os.chmod(directory, 0o700)
