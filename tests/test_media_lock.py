from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from umkleide.database import Database
from umkleide.repositories import Repository
from umkleide.storage import MediaStorage, PreparedImage
from umkleide.wardrobe import MediaQuotaError, WardrobeService


def _wardrobe(root: Path) -> WardrobeService:
    database = Database(root / "data.sqlite3")
    database.initialize()
    return WardrobeService(
        Repository(database), MediaStorage(root), quota_bytes=10, import_root=root / "imports"
    )


def _write_item(wardrobe: WardrobeService) -> None:
    wardrobe.add_clothing(
        person_id="me", name="shirt", category="top", prepared=PreparedImage(b"x" * 8, 1, 1)
    )


def test_in_process_lock_serializes_concurrent_quota_admission(tmp_path: Path) -> None:
    wardrobe = _wardrobe(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_write_item, wardrobe) for _ in range(2)]
    outcomes = [future.exception() for future in futures]

    assert sum(outcome is None for outcome in outcomes) == 1
    assert any(isinstance(outcome, MediaQuotaError) for outcome in outcomes)
    assert wardrobe.storage.managed_usage() == 8
    assert wardrobe.repository.database.record_counts()["clothing_items"] == 1


def test_separate_service_roots_do_not_share_a_lock(tmp_path: Path) -> None:
    first, second = _wardrobe(tmp_path / "first"), _wardrobe(tmp_path / "second")
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(_write_item, (first, second)))
    assert first.storage.managed_usage() == second.storage.managed_usage() == 8
