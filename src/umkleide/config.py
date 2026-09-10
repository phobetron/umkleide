"""Process configuration for the local server."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "umkleide"
DATABASE_FILENAME = "umkleide.sqlite3"
MEDIA_QUOTA_BYTES_ENV = "UMKLEIDE_MEDIA_QUOTA_BYTES"
DEFAULT_MEDIA_QUOTA_BYTES = 1024 * 1024 * 1024
DEFAULT_IMPORT_DIRECTORY = "umkleide-imports"

MANAGED_DIRECTORIES = ("person", "clothing", "generations")


def default_import_root() -> Path:
    """Return the per-user temporary directory used for local image imports."""
    return Path(tempfile.gettempdir()) / DEFAULT_IMPORT_DIRECTORY


def default_data_root(
    environ: Mapping[str, str] | None = None, *, platform: str | None = None
) -> Path:
    """Return the platform-appropriate data root without creating it."""
    environment = os.environ if environ is None else environ
    current_platform = sys.platform if platform is None else platform
    home = Path.home()
    if current_platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if current_platform.startswith("win"):
        app_data = environment.get("LOCALAPPDATA") or environment.get("APPDATA")
        if app_data:
            return Path(app_data) / APP_NAME
        return home / "AppData" / "Local" / APP_NAME

    xdg_data_home = environment.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / APP_NAME
    return home / ".local" / "share" / APP_NAME


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration owned by the local server process."""

    data_root: Path
    media_quota_bytes: int = DEFAULT_MEDIA_QUOTA_BYTES

    def __post_init__(self) -> None:
        if self.media_quota_bytes <= 0:
            raise ValueError("media_quota_bytes must be a positive integer")

    @property
    def database_path(self) -> Path:
        return self.data_root / DATABASE_FILENAME

    @property
    def import_root(self) -> Path:
        return default_import_root().absolute()

    def ensure_import_root(self) -> None:
        """Create the private temporary import directory."""
        import_root = self.import_root
        import_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        status = import_root.lstat()
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError("image import path is not a real directory")
        if os.name == "posix" and status.st_uid != os.getuid():
            raise RuntimeError("image import directory is owned by another user")
        _restrict_directory_permissions(import_root)

    def ensure_data_root(self) -> None:
        """Create the private application layout if it does not yet exist."""
        self.data_root.mkdir(parents=True, exist_ok=True)
        _restrict_directory_permissions(self.data_root)
        for directory in MANAGED_DIRECTORIES:
            managed_directory = self.data_root / directory
            managed_directory.mkdir(exist_ok=True)
            _restrict_directory_permissions(managed_directory)


def _restrict_directory_permissions(path: Path) -> None:
    if os.name == "posix":
        path.chmod(0o700)


def load_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    """Load non-secret process configuration."""
    environment = os.environ if environ is None else environ
    quota_value = environment.get(MEDIA_QUOTA_BYTES_ENV, str(DEFAULT_MEDIA_QUOTA_BYTES))
    try:
        quota = int(quota_value)
    except ValueError as exc:
        raise ValueError(f"{MEDIA_QUOTA_BYTES_ENV} must be a positive integer") from exc
    if quota <= 0:
        raise ValueError(f"{MEDIA_QUOTA_BYTES_ENV} must be a positive integer")
    return AppConfig(
        data_root=default_data_root(environment),
        media_quota_bytes=quota,
    )
