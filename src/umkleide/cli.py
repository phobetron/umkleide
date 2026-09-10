"""Command-line entry point for Umkleide."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
import warnings
from collections.abc import Mapping, Sequence

from .application import application
from .config import load_config
from .credentials import (
    BFL_API_KEY_ENV,
    BflCredentialStore,
    CredentialStoreError,
    normalize_credential_key,
)
from .server import create_server


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected MCP transport or configure the local BFL credential."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"))
    commands = parser.add_subparsers(dest="command")
    configure = commands.add_parser("configure")
    configure.add_argument("--clear", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "configure":
        if args.transport is not None:
            parser.error("--transport cannot be used with configure")
        return _configure_credentials(dict(os.environ), clear=args.clear)
    try:
        return asyncio.run(_run_server(args.transport or "streamable-http"))
    except KeyboardInterrupt:
        return 130


async def _run_server(transport: str) -> int:
    config = load_config()
    async with application(config) as app:
        server = create_server(app=app, config=config)
        if transport == "stdio":
            await server.run_stdio_async()
        else:
            await server.run_streamable_http_async()
    return 0


def _configure_credentials(environ: Mapping[str, str], *, clear: bool) -> int:
    """Configure the local BFL credential without starting the MCP server."""
    try:
        config = load_config(environ)
        config.ensure_data_root()
        store = BflCredentialStore(config.data_root)
        if clear:
            store.clear()
        else:
            key = normalize_credential_key(environ.get(BFL_API_KEY_ENV))
            if key is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    key = normalize_credential_key(getpass.getpass("BFL API key: "))
            if key is None:
                raise CredentialStoreError("credential value is invalid")
            store.write(key)
    except (
        CredentialStoreError,
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        getpass.GetPassWarning,
    ):
        print("Unable to configure BFL credentials.", file=sys.stderr)
        return 1
    if clear:
        print("BFL credentials cleared.")
        if environ.get(BFL_API_KEY_ENV, "").strip():
            print(
                "BFL_API_KEY remains set and will be saved again at the next server startup.",
                file=sys.stderr,
            )
    else:
        print("BFL credentials saved.")
    return 0
