"""Child runtime for real generation transport tests."""

from __future__ import annotations

import asyncio
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image

from umkleide.application import application
from umkleide.bfl import BFLJobState, BFLPollResult, BFLSubmission
from umkleide.config import AppConfig
from umkleide.server import create_server


class FakeProvider:
    async def submit_flux_2_pro(self, request: Any) -> BFLSubmission:
        del request
        return BFLSubmission("test-request", "https://test.invalid/poll")

    async def poll(self, polling_url: str) -> BFLPollResult:
        del polling_url
        return BFLPollResult(BFLJobState.READY, result_url="https://test.invalid/result")

    async def download_result(self, result_url: str) -> bytes:
        del result_url
        output = io.BytesIO()
        Image.new("RGB", (1088, 1920), "mediumseagreen").save(output, format="JPEG")
        return output.getvalue()


@dataclass(frozen=True, slots=True)
class TransportConfig(AppConfig):
    test_import_root: Path = Path()

    @property
    def import_root(self) -> Path:
        return self.test_import_root


async def _run(config: TransportConfig, transport: str, port: int | None) -> None:
    async with application(config, provider=FakeProvider(), environ={}) as app:
        server = create_server(app=app, config=config)
        if transport == "stdio":
            await server.run_stdio_async()
        else:
            assert port is not None
            server.settings.host, server.settings.port = "127.0.0.1", port
            await server.run_streamable_http_async()


def child_main() -> None:
    data_root, import_root = Path(sys.argv[1]), Path(sys.argv[2])
    transport = sys.argv[3]
    port = int(sys.argv[4]) if len(sys.argv) > 4 else None
    asyncio.run(_run(TransportConfig(data_root, test_import_root=import_root), transport, port))


if __name__ == "__main__":
    child_main()
