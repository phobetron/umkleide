"""Real stdio and HTTP regressions for background outfit-image retrieval."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import socket
import sys
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ElicitResult, ImageContent
from PIL import Image

from umkleide.database import Database
from umkleide.models import GenerationStatus
from umkleide.repositories import Repository


def _photo() -> dict[str, str]:
    output = io.BytesIO()
    Image.new("RGB", (3, 4), "steelblue").save(output, format="JPEG")
    return {
        "type": "data_uri",
        "data": "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(),
    }


def _result(response: CallToolResult) -> dict[str, Any]:
    assert not response.isError, response.content
    assert isinstance(response.structuredContent, Mapping)
    return dict(response.structuredContent)


async def _accept(context: Any, params: Any) -> ElicitResult:
    del context, params
    return ElicitResult(action="accept")


def _command(workspace: Path, transport: str, port: int | None = None) -> list[str]:
    arguments = [str(workspace / "data"), str(workspace / "imports"), transport]
    if port is not None:
        arguments.append(str(port))
    return [
        sys.executable,
        str(Path(__file__).parent / "helpers" / "generation_server.py"),
        *arguments,
    ]


async def _eventually(predicate: Callable[[], bool], timeout: float = 12) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), timeout)


@asynccontextmanager
async def _stdio(workspace: Path) -> AsyncIterator[ClientSession]:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=_command(workspace, "stdio")[1:],
        env=dict(os.environ, BFL_API_KEY=""),
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, elicitation_callback=_accept, read_timeout_seconds=timedelta(seconds=15)
        ) as client:
            await client.initialize()
            yield client


@asynccontextmanager
async def _http_server(workspace: Path) -> AsyncIterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = await asyncio.create_subprocess_exec(
        *_command(workspace, "streamable-http", port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        env=dict(os.environ, BFL_API_KEY=""),
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:

        async def started() -> None:
            while True:
                try:
                    async with streamable_http_client(url, terminate_on_close=False) as streams:
                        async with ClientSession(*streams[:2]) as client:
                            await client.initialize()
                    return
                except Exception:
                    if process.returncode is not None:
                        stderr = await process.stderr.read() if process.stderr else b""
                        raise RuntimeError(stderr.decode(errors="replace")) from None
                    await asyncio.sleep(0.05)

        await asyncio.wait_for(started(), 12)
        yield url
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


@asynccontextmanager
async def _http(url: str) -> AsyncIterator[ClientSession]:
    async with streamable_http_client(url, terminate_on_close=False) as streams:
        async with ClientSession(*streams[:2], elicitation_callback=_accept) as client:
            await client.initialize()
            yield client


async def _create(client: ClientSession, *, progress_callback=None) -> tuple[str, CallToolResult]:
    person = _result(
        await client.call_tool("add_person", {"names": ["Test person"], "photo": _photo()})
    )
    started = time.monotonic()
    response = await client.call_tool(
        "create_outfit_photo",
        {
            "prompt": "standing portrait",
            "person_id": str(person["person_id"]),
            "items": [
                {"type": "new", "name": "Test jacket", "category": "outerwear", "photo": _photo()}
            ],
            "wait_seconds": 1,
        },
        read_timeout_seconds=timedelta(seconds=15),
        progress_callback=progress_callback,
        meta={"progressToken": "test-progress"} if progress_callback else None,
    )
    assert time.monotonic() - started < 2
    return str(_result(response)["generation_id"]), response


def _ready(workspace: Path, generation_id: str) -> bool:
    root = workspace / "data"
    generation = Repository(Database(root / "umkleide.sqlite3")).get_generation(generation_id)
    return (
        generation.status is GenerationStatus.READY
        and generation.result_relative_path is not None
        and (root / generation.result_relative_path).is_file()
    )


async def _assert_ready_response(
    client: ClientSession, workspace: Path, generation_id: str
) -> None:
    response = await client.call_tool("get_generation", {"generation_id": generation_id})
    assert _result(response)["status"] == "ready"
    image = next(content for content in response.content if isinstance(content, ImageContent))
    stored = next((workspace / "data" / "generations").glob("*.jpg")).read_bytes()
    assert base64.b64decode(image.data) == stored


async def test_stdio_retains_generation_before_any_followup_call(tmp_path: Path) -> None:
    async with _stdio(tmp_path) as client:
        progress: list[object] = []

        async def observe(*event: object) -> None:
            progress.append(event)

        generation_id, created = await _create(client, progress_callback=observe)
        assert _result(created)["status"] == "processing"
        assert progress
        await _eventually(lambda: _ready(tmp_path, generation_id))
        await _assert_ready_response(client, tmp_path, generation_id)


async def test_http_disconnect_keeps_server_retrieving(tmp_path: Path) -> None:
    async with _http_server(tmp_path) as url:
        async with _http(url) as client:
            generation_id, created = await _create(client)
            assert _result(created)["status"] == "processing"
        await _eventually(lambda: _ready(tmp_path, generation_id))
        async with _http(url) as client:
            await _assert_ready_response(client, tmp_path, generation_id)
