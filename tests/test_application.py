from pathlib import Path

import pytest

import umkleide.application as application_module
from umkleide.application import application
from umkleide.config import AppConfig
from umkleide.generations import Provider


class RecordingProvider:
    async def submit_flux_2_pro(self, request):  # pragma: no cover - protocol stub
        raise AssertionError("not used")

    async def poll(self, polling_url):  # pragma: no cover - protocol stub
        raise AssertionError("not used")

    async def download_result(self, result_url):  # pragma: no cover - protocol stub
        raise AssertionError("not used")


async def test_application_builds_complete_contexts_sharing_the_catalog(tmp_path: Path) -> None:
    config = AppConfig(tmp_path / "data")
    async with application(config, environ={}) as state:
        assert state.repository is state.wardrobe.repository
        assert state.storage is state.wardrobe.storage
        assert state.generation.repository is state.repository
        assert state.provider is None
        assert state.retriever is not None
        async with application(config, environ={}) as second:
            assert second.repository.get_person("me") == state.repository.get_person("me")


async def test_startup_failure_does_not_prevent_another_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(tmp_path / "data")
    monkeypatch.setattr(
        application_module.WardrobeService,
        "maintain",
        lambda _self: (_ for _ in ()).throw(RuntimeError("maintenance failed")),
    )
    with pytest.raises(RuntimeError, match="maintenance failed"):
        async with application(config, environ={}):
            pass
    monkeypatch.undo()
    async with application(config, environ={}) as state:
        assert state.repository.get_person("me").id == "me"


async def test_application_does_not_close_an_injected_provider(tmp_path: Path) -> None:
    provider: Provider = RecordingProvider()
    async with application(AppConfig(tmp_path / "data"), provider=provider, environ={}) as state:
        assert state.provider is provider
