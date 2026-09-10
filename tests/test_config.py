import os
from pathlib import Path

import pytest

import umkleide.config as config_module
from umkleide.config import (
    APP_NAME,
    DATABASE_FILENAME,
    DEFAULT_IMPORT_DIRECTORY,
    DEFAULT_MEDIA_QUOTA_BYTES,
    MANAGED_DIRECTORIES,
    MEDIA_QUOTA_BYTES_ENV,
    AppConfig,
    load_config,
)


def test_config_creates_the_private_managed_layout(tmp_path: Path) -> None:
    config = AppConfig(data_root=tmp_path / APP_NAME)
    config.ensure_data_root()

    assert {path.name for path in config.data_root.iterdir()} == set(MANAGED_DIRECTORIES)
    assert config.database_path == config.data_root / DATABASE_FILENAME


def test_load_config_uses_a_positive_media_quota() -> None:
    assert load_config({}).media_quota_bytes == DEFAULT_MEDIA_QUOTA_BYTES
    assert load_config({MEDIA_QUOTA_BYTES_ENV: "123"}).media_quota_bytes == 123
    with pytest.raises(ValueError, match="positive integer"):
        load_config({MEDIA_QUOTA_BYTES_ENV: "0"})


def test_import_directory_uses_and_creates_a_private_temporary_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(config_module.tempfile, "gettempdir", lambda: str(tmp_path))
    config = load_config({})

    assert config.import_root == tmp_path / DEFAULT_IMPORT_DIRECTORY
    config.ensure_import_root()
    assert config.import_root.is_dir()
    if os.name == "posix":
        assert config.import_root.stat().st_mode & 0o777 == 0o700


def test_import_directory_rejects_a_preexisting_symlink(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config_module.tempfile, "gettempdir", lambda: str(tmp_path))
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / DEFAULT_IMPORT_DIRECTORY).symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="not a real directory"):
        load_config({}).ensure_import_root()
