import asyncio
import threading

import pytest

from umkleide.workers import run_blocking


async def test_cancellation_waits_for_running_write_to_finish():
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def write():
        started.set()
        assert release.wait(timeout=5)
        completed.set()

    task = asyncio.create_task(run_blocking(write))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed.is_set()


async def test_worker_failure_during_cancellation_preserves_cancellation():
    started = threading.Event()
    release = threading.Event()

    def write():
        started.set()
        assert release.wait(timeout=5)
        raise OSError("write failed")

    task = asyncio.create_task(run_blocking(write))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_anyio_cancellation_scope_waits_for_worker():
    import time

    import anyio

    completed = threading.Event()

    def write():
        time.sleep(0.05)
        completed.set()

    with anyio.move_on_after(0.01) as scope:
        await run_blocking(write)
    assert scope.cancel_called
    assert completed.is_set()
