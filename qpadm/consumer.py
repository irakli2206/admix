from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from qpadm.runner import run_qpadm_job

logger = logging.getLogger(__name__)

_queue: Optional[asyncio.Queue[str]] = None
_task: Optional[asyncio.Task] = None
_sem: Optional[asyncio.Semaphore] = None


def setup() -> None:
    global _queue, _sem
    n = max(1, int(os.environ.get("MAX_CONCURRENT_QPADM", "1")))
    _queue = asyncio.Queue()
    _sem = asyncio.Semaphore(n)


def enqueue(job_id: str) -> None:
    if _queue is None:
        raise RuntimeError("qpadm consumer not initialized")
    _queue.put_nowait(job_id)


async def _worker_loop() -> None:
    assert _queue is not None and _sem is not None
    while True:
        job_id = await _queue.get()
        try:
            async with _sem:
                await asyncio.to_thread(run_qpadm_job, job_id)
        except Exception:
            logger.exception("qpadm consumer error for job %s", job_id)
        finally:
            _queue.task_done()


def start_background_task() -> asyncio.Task:
    global _task
    if _task is not None:
        return _task
    setup()
    _task = asyncio.create_task(_worker_loop(), name="qpadm-consumer")
    return _task


async def stop_background_task() -> None:
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except asyncio.CancelledError:
        pass
    _task = None
