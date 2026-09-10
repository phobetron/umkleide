"""Exercise shared-data access through two real CLI stdio processes."""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult
from PIL import Image

from umkleide.config import DATABASE_FILENAME
from umkleide.database import Database
from umkleide.storage import MediaStorage

_CLI_PROGRAM = """
from pathlib import Path
import sys
import umkleide.cli as cli
from umkleide.config import AppConfig

AppConfig.import_root = property(lambda self: self.data_root / 'imports')
cli.load_config = lambda: AppConfig(Path(sys.argv[1]), media_quota_bytes=int(sys.argv[2]))
raise SystemExit(cli.main(['--transport=stdio']))
"""


@asynccontextmanager
async def _client(root: Path, *, quota: int = 10_000_000) -> AsyncIterator[ClientSession]:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-c", _CLI_PROGRAM, str(root), str(quota)],
        env={"BFL_API_KEY": ""},
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=10)
        ) as client:
            await client.initialize()
            yield client


@pytest.fixture
def photo() -> dict[str, str]:
    output = io.BytesIO()
    Image.new("RGB", (2, 3), "steelblue").save(output, format="PNG")
    return {
        "type": "data_uri",
        "data": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(),
    }


def _result(response: CallToolResult):
    assert not response.isError, response.content
    assert response.structuredContent is not None
    return response.structuredContent


async def test_two_stdio_processes_share_catalog_and_survive_peer_shutdown(tmp_path: Path, photo):
    async with _client(tmp_path / "data") as first:
        async with _client(tmp_path / "data") as second:
            first_tools, second_tools = await asyncio.gather(
                first.list_tools(), second.list_tools()
            )
            names = {tool.name for tool in first_tools.tools}
            assert names == {tool.name for tool in second_tools.tools}
            assert {"get_diagnostics", "add_person", "get_person", "list_people"} <= names
            added = await asyncio.gather(
                first.call_tool("add_person", {"names": ["First"], "photo": photo}),
                second.call_tool("add_person", {"names": ["Second"], "photo": photo}),
            )
            person = _result(added[0])
            assert _result(added[1])["names"] == ["Second"]
            fetched = _result(
                await second.call_tool("get_person", {"person_id": person["person_id"]})
            )
            assert fetched["names"] == ["First"]
            people = _result(await first.call_tool("list_people", {}))["items"]
            assert {person["names"][0] for person in people} == {"me", "First", "Second"}
        diagnostics = _result(await first.call_tool("get_diagnostics", {}))
        assert diagnostics["records"]["people"] == 3


async def test_two_stdio_processes_admit_only_one_image_when_quota_allows_one(
    tmp_path: Path, photo
):
    root = tmp_path / "data"
    quota = MediaStorage(tmp_path / "sizing").prepare_user_image(photo["data"]).size_bytes
    async with _client(root, quota=quota) as first:
        async with _client(root, quota=quota) as second:
            results = await asyncio.gather(
                first.call_tool("add_person", {"names": ["First"], "photo": photo}),
                second.call_tool("add_person", {"names": ["Second"], "photo": photo}),
            )
            assert sum(not result.isError for result in results) == 1
            failure = next(result for result in results if result.isError)
            assert "quota" in failure.model_dump_json().lower()
            assert Database(root / DATABASE_FILENAME).record_counts()["people"] == 2
            images = list((root / "person").glob("*.jpg"))
            assert len(images) == 1
            assert images[0].stat().st_size == quota
