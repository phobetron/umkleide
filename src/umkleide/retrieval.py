"""Periodic retrieval of durable generation jobs."""

from __future__ import annotations

import asyncio
import logging

from .generations import GenerationService
from .models import Generation, GenerationStatus
from .workers import run_blocking

_LOG = logging.getLogger(__name__)
_BATCH_SIZE = 4


class GenerationRetriever:
    """Advance every durable processing job from one process-owned task."""

    def __init__(
        self,
        service: GenerationService,
        *,
        interval_seconds: float = 5.0,
        wait_interval_seconds: float = 0.25,
    ) -> None:
        if interval_seconds <= 0 or wait_interval_seconds <= 0:
            raise ValueError("retrieval intervals must be positive")
        self.service = service
        self.interval_seconds = interval_seconds
        self.wait_interval_seconds = wait_interval_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None and self.service.provider is not None:
            self._task = asyncio.create_task(self._run(), name="generation-retriever")

    async def wait_for_result(self, generation_id: str, timeout: float) -> Generation:
        """Observe durable state without taking ownership of provider polling."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            generation = await self.service.get_generation_snapshot(generation_id)
            if _wait_complete(generation):
                return generation
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return generation
            await asyncio.sleep(min(remaining, self.wait_interval_seconds))

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while True:
            try:
                generation_ids = await run_blocking(
                    self.service.repository.processing_generation_ids
                )
                for start in range(0, len(generation_ids), _BATCH_SIZE):
                    results = await asyncio.gather(
                        *(
                            self.service.advance_generation(generation_id)
                            for generation_id in generation_ids[start : start + _BATCH_SIZE]
                        ),
                        return_exceptions=True,
                    )
                    if any(isinstance(result, BaseException) for result in results):
                        _LOG.warning("generation retrieval attempt failed")
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.warning("generation retrieval scan failed")
            await asyncio.sleep(self.interval_seconds)


def _wait_complete(generation: Generation) -> bool:
    return generation.status in {
        GenerationStatus.READY,
        GenerationStatus.FAILED,
        GenerationStatus.SUBMISSION_UNKNOWN,
    } or (generation.error is not None and generation.error.get("code") == "storage_unavailable")
