"""Background queue for ADMIXTURE jobs: bounded concurrency."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from admixture_jobs import store as admixture_store
from admixture_jobs.runner import run_admixture_job

logger = logging.getLogger(__name__)

_queue: Optional[asyncio.Queue[str]] = None
_task: Optional[asyncio.Task[None]] = None
_sem: Optional[asyncio.Semaphore] = None


def _max_concurrent() -> int:
    return max(1, int(os.environ.get("MAX_CONCURRENT_ADMIXTURE", "1")))


async def _worker_loop() -> None:
    assert _queue is not None and _sem is not None
    while True:
        job_id = await _queue.get()
        try:
            async with _sem:
                await asyncio.to_thread(run_admixture_job, job_id)
        except Exception:
            logger.exception("ADMIXTURE consumer error for job %s", job_id)
        finally:
            _queue.task_done()


def start_background_task() -> None:
    global _queue, _task, _sem
    if _task is not None and not _task.done():
        return
    _queue = asyncio.Queue()
    _sem = asyncio.Semaphore(_max_concurrent())
    for jid in admixture_store.queued_job_ids():
        _queue.put_nowait(jid)
        logger.info("ADMIXTURE re-queued job %s after startup", jid)
    _task = asyncio.create_task(_worker_loop(), name="admixture-consumer")
    logger.info("ADMIXTURE consumer started (max concurrent=%s)", _max_concurrent())


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


def enqueue(job_id: str) -> None:
    if _queue is None:
        raise RuntimeError("ADMIXTURE consumer not started")
    _queue.put_nowait(job_id)
