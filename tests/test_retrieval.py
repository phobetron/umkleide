from __future__ import annotations

import asyncio

import pytest

from umkleide.models import Generation, GenerationStatus
from umkleide.retrieval import GenerationRetriever


def _generation(
    identifier: str, *, status: GenerationStatus = GenerationStatus.PROCESSING
) -> Generation:
    return Generation(
        id=identifier,
        person_id="person",
        status=status,
        prompt="outfit",
        person_photo_id="photo",
        selected_clothing=(),
        polling_url="https://poll.example/job",
        created_at="2026-01-01T00:00:00.000000Z",
        updated_at="2026-01-01T00:00:00.000000Z",
    )


class _Repository:
    def __init__(self, generation_ids: tuple[str, ...]) -> None:
        self.generation_ids = generation_ids

    def processing_generation_ids(self) -> tuple[str, ...]:
        return self.generation_ids


class _Service:
    def __init__(
        self, generations: tuple[Generation, ...], *, provider: object | None = object()
    ) -> None:
        self.provider = provider
        self.repository = _Repository(tuple(generation.id for generation in generations))
        self.snapshots = {generation.id: generation for generation in generations}
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.active = self.maximum_active = 0
        self.release = asyncio.Event()
        self.cancel_cleanup: asyncio.Event | None = None

    async def get_generation_snapshot(self, generation_id: str) -> Generation:
        return self.snapshots[generation_id]

    async def advance_generation(self, generation_id: str) -> Generation:
        self.started.append(generation_id)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(generation_id)
            if self.cancel_cleanup is not None:
                await self.cancel_cleanup.wait()
            raise
        finally:
            self.active -= 1
        return self.snapshots[generation_id]


async def _eventually(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.005)
    assert predicate()


async def test_retriever_batches_all_snapshot_jobs_with_a_four_job_cap() -> None:
    service = _Service(tuple(_generation(str(index)) for index in range(5)))
    retriever = GenerationRetriever(service, interval_seconds=60)  # type: ignore[arg-type]
    retriever.start()
    await _eventually(lambda: service.started == ["0", "1", "2", "3"])
    assert service.maximum_active == 4

    service.release.set()
    await _eventually(lambda: service.started == ["0", "1", "2", "3", "4"])
    await retriever.aclose()


async def test_waiter_returns_terminal_storage_or_timeout_without_advancing() -> None:
    ready = _generation("ready", status=GenerationStatus.READY)
    stored = _generation("stored").model_copy(update={"error": {"code": "storage_unavailable"}})
    processing = _generation("processing")
    service = _Service((ready, stored, processing), provider=None)
    retriever = GenerationRetriever(service, wait_interval_seconds=0.01)  # type: ignore[arg-type]
    retriever.start()
    await asyncio.sleep(0)

    assert await retriever.wait_for_result("ready", 1) == ready
    assert await retriever.wait_for_result("stored", 1) == stored
    assert await retriever.wait_for_result("processing", 0) == processing
    task = asyncio.create_task(retriever.wait_for_result("processing", 1))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.started == []


async def test_close_cancels_the_single_inflight_batch() -> None:
    service = _Service((_generation("job"),))
    retriever = GenerationRetriever(service, interval_seconds=60)  # type: ignore[arg-type]
    retriever.start()
    await _eventually(lambda: service.started == ["job"])

    await asyncio.wait_for(retriever.aclose(), timeout=1)
    assert service.cancelled == ["job"]


async def test_close_waits_for_every_cancelled_batch_member_to_finish_cleanup() -> None:
    service = _Service((_generation("first"), _generation("second")))
    service.cancel_cleanup = asyncio.Event()
    retriever = GenerationRetriever(service, interval_seconds=60)  # type: ignore[arg-type]
    retriever.start()
    await _eventually(lambda: service.started == ["first", "second"])

    closing = asyncio.create_task(retriever.aclose())
    await _eventually(lambda: set(service.cancelled) == {"first", "second"})
    assert not closing.done()
    service.cancel_cleanup.set()
    await asyncio.wait_for(closing, timeout=1)
