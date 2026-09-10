"""Process-owned application runtime for the local MCP server."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .bfl import BFLClient
from .config import AppConfig
from .credentials import BflCredentialSource, BflCredentialStore, resolve_bfl_credential
from .database import Database
from .generations import GenerationService, Provider
from .locking import media_lock
from .repositories import Repository
from .retrieval import GenerationRetriever
from .storage import MediaStorage
from .wardrobe import WardrobeService
from .workers import run_blocking


@dataclass(frozen=True, slots=True)
class AppState:
    """Services constructed and owned by one local server process."""

    repository: Repository
    storage: MediaStorage
    wardrobe: WardrobeService
    generation: GenerationService
    retriever: GenerationRetriever
    provider: Provider | None
    bfl_api_key_source: BflCredentialSource


@asynccontextmanager
async def application(
    config: AppConfig,
    *,
    provider: Provider | None = None,
    environ: Mapping[str, str] | None = None,
) -> AsyncIterator[AppState]:
    """Build services for one connection host over the shared local data directory."""
    environment = dict(os.environ if environ is None else environ)
    config.data_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    owned_provider: BFLClient | None = None
    generation: GenerationService | None = None
    retriever: GenerationRetriever | None = None
    cancelled = False
    try:

        def initialize():
            with media_lock(config.data_root):
                config.ensure_data_root()
                config.ensure_import_root()
                credential = resolve_bfl_credential(
                    BflCredentialStore(config.data_root), environment
                )
                storage = MediaStorage(config.data_root)
                database = Database(config.database_path)
                database.initialize()
                return credential, storage, Repository(database)

        credential, storage, repository = await run_blocking(initialize)
        active_provider = provider
        if active_provider is None and credential.key is not None:
            owned_provider = BFLClient(credential.key)
            active_provider = owned_provider
        wardrobe = WardrobeService(
            repository,
            storage,
            quota_bytes=config.media_quota_bytes,
            import_root=config.import_root,
        )
        generation = GenerationService(repository, wardrobe, active_provider)
        retriever = GenerationRetriever(generation)
        state = AppState(
            repository=repository,
            storage=storage,
            wardrobe=wardrobe,
            generation=generation,
            retriever=retriever,
            provider=active_provider,
            bfl_api_key_source=credential.source,
        )
        await run_blocking(wardrobe.maintain)
        if active_provider is not None:
            retriever.start()
        yield state
    finally:
        if generation is not None:
            generation.close_intake()
        cancelled = await _cleanup(_shutdown(generation, retriever, owned_provider))
        if cancelled:
            raise asyncio.CancelledError


async def _cleanup(awaitable: Awaitable[object]) -> bool:
    """Drain shutdown work even when the enclosing server task is cancelled."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if task.done() and not task.cancelled():
        task.result()
    return cancelled


async def _shutdown(
    generation: GenerationService | None,
    retriever: GenerationRetriever | None,
    owned_provider: BFLClient | None,
) -> None:
    try:
        if generation is not None:
            try:
                await generation.drain_submissions(35.0)
            finally:
                if retriever is not None:
                    await retriever.aclose()
    finally:
        if owned_provider is not None:
            await owned_provider.aclose()
