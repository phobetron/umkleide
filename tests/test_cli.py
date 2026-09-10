import getpass
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import umkleide.cli as cli
from umkleide.config import AppConfig
from umkleide.credentials import BFL_API_KEY_ENV, BflCredentialStore


class RecordingServer:
    def __init__(self) -> None:
        self.transports: list[str] = []

    async def run_stdio_async(self) -> None:
        self.transports.append("stdio")

    async def run_streamable_http_async(self) -> None:
        self.transports.append("streamable-http")


@pytest.mark.parametrize(
    ("argv", "expected_transport"),
    [
        ((), "streamable-http"),
        (("--transport=streamable-http",), "streamable-http"),
        (("--transport=stdio",), "stdio"),
        (("--transport", "stdio"), "stdio"),
    ],
)
def test_main_dispatches_the_selected_transport(
    argv: tuple[str, ...], expected_transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = RecordingServer()
    monkeypatch.setattr(cli, "load_config", lambda: AppConfig(Path("/tmp/umkleide-test")))

    class Context:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(cli, "application", lambda _config: Context())
    monkeypatch.setattr(cli, "create_server", lambda *, app, config: server)

    assert cli.main(list(argv)) == 0
    assert server.transports == [expected_transport]


def test_main_rejects_unknown_transport_before_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_run_server", lambda _transport: pytest.fail("started server"))
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--transport=websocket"])
    assert exc_info.value.code == 2


def test_configure_persists_environment_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(tmp_path / "data")
    monkeypatch.setattr(cli, "load_config", lambda _environment: config)
    assert cli._configure_credentials({BFL_API_KEY_ENV: "test-key"}, clear=False) == 0
    assert BflCredentialStore(config.data_root).read() == "test-key"


def test_cli_async_runner_returns_a_process_status(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def run(_transport: str) -> int:
        calls.append(_transport)
        return 7

    monkeypatch.setattr(cli, "_run_server", run)
    assert cli.main([]) == 7
    assert calls == ["streamable-http"]


def test_configure_saves_while_catalog_mutations_are_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = AppConfig(tmp_path / "data")
    config.data_root.mkdir()
    monkeypatch.setattr(cli, "load_config", lambda _environment: config)
    from filelock import FileLock

    lock = FileLock(config.data_root / ".media.lock", timeout=0)
    with lock:
        assert cli._configure_credentials({BFL_API_KEY_ENV: "test-key"}, clear=False) == 0
    assert BflCredentialStore(config.data_root).read() == "test-key"
    assert capsys.readouterr().err == ""


def test_module_cli_stdio_serves_mcp_without_non_protocol_stdout(tmp_path: Path) -> None:
    environment = os.environ | {
        "BFL_API_KEY": "",
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(tmp_path / "tmp"),
        "XDG_DATA_HOME": str(tmp_path / "data-home"),
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from umkleide.cli import main; raise SystemExit(main())",
            "--transport=stdio",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                }
            )
            + "\n"
        )
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        assert response["id"] == 1
        assert "result" in response
    finally:
        process.stdin.close()
        process.wait(timeout=10)
    assert process.returncode == 0


def test_configure_uses_environment_key_without_prompting_or_starting_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setattr(cli, "load_config", lambda _environment: AppConfig(data_root))
    monkeypatch.setenv(BFL_API_KEY_ENV, "environment-test-key")
    monkeypatch.setattr(
        cli.getpass, "getpass", lambda _prompt: pytest.fail("prompted despite env key")
    )
    monkeypatch.setattr(
        cli, "create_server", lambda: pytest.fail("constructed server while configuring")
    )

    assert cli.main(["configure"]) == 0

    captured = capsys.readouterr()
    assert BflCredentialStore(data_root).read() == "environment-test-key"
    assert "environment-test-key" not in captured.out + captured.err


def test_configure_prompts_for_a_missing_environment_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setattr(cli, "load_config", lambda _environment: AppConfig(data_root))
    monkeypatch.delenv(BFL_API_KEY_ENV, raising=False)
    prompts: list[str] = []
    monkeypatch.setattr(
        cli.getpass,
        "getpass",
        lambda prompt: prompts.append(prompt) or "prompt-test-key",
    )

    assert cli.main(["configure"]) == 0

    captured = capsys.readouterr()
    assert prompts == ["BFL API key: "]
    assert BflCredentialStore(data_root).read() == "prompt-test-key"
    assert "prompt-test-key" not in captured.out + captured.err


@pytest.mark.parametrize(
    "outcome", ["  ", EOFError(), getpass.GetPassWarning("hidden input failed")]
)
def test_configure_blank_or_non_tty_prompt_preserves_old_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str | EOFError | getpass.GetPassWarning,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    store = BflCredentialStore(data_root)
    store.write("old-test-key")
    monkeypatch.setattr(cli, "load_config", lambda _environment: AppConfig(data_root))
    monkeypatch.delenv(BFL_API_KEY_ENV, raising=False)

    def prompt(_prompt: str) -> str:
        if isinstance(outcome, EOFError):
            raise outcome
        if isinstance(outcome, getpass.GetPassWarning):
            warnings.warn(outcome, stacklevel=2)
            return "untrusted-prompt-key"
        return outcome

    monkeypatch.setattr(cli.getpass, "getpass", prompt)

    assert cli.main(["configure"]) == 1

    captured = capsys.readouterr()
    assert store.read() == "old-test-key"
    assert "old-test-key" not in captured.out + captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["configure", "--transport=stdio"],
        ["configure", "--transport", "stdio"],
        ["--transport=stdio", "configure"],
        ["--transport", "streamable-http", "configure"],
    ],
)
def test_configure_rejects_transport_without_prompting_mutating_or_starting_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setattr(cli, "load_config", lambda _environment: AppConfig(data_root))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: pytest.fail("configure prompted"))
    monkeypatch.setattr(cli, "create_server", lambda: pytest.fail("configure constructed a server"))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 2
    assert not BflCredentialStore(data_root).path.exists()


def test_configure_clear_ignores_environment_and_never_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    BflCredentialStore(data_root).write("stored-test-key")
    monkeypatch.setattr(cli, "load_config", lambda _environment: AppConfig(data_root))
    monkeypatch.setenv(BFL_API_KEY_ENV, "environment-test-key")
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: pytest.fail("clear prompted"))

    assert cli.main(["configure", "--clear"]) == 0

    captured = capsys.readouterr()
    assert BflCredentialStore(data_root).read() is None
    assert "stored-test-key" not in captured.out + captured.err
    assert "environment-test-key" not in captured.out + captured.err
