"""SQLite job store for qpAdm runs."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_lock = threading.Lock()

_REPO_ROOT = Path(__file__).resolve().parent.parent


def jobs_root() -> str:
    return os.environ.get(
        "QPADM_JOBS_ROOT",
        str(_REPO_ROOT / "qpadm_jobs_data"),
    )


def _db_path() -> str:
    root = jobs_root()
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "qpadm_jobs.db")


def init_db() -> None:
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    error TEXT,
                    result TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
    _recover_stale_running()


def _recover_stale_running() -> None:
    now = time.time()
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                """
                UPDATE jobs SET status = 'failed', error = ?, updated_at = ?
                WHERE status = 'running'
                """,
                ("Server restarted while job was running.", now),
            )
            conn.commit()


def new_job_id() -> str:
    return str(uuid.uuid4())


def queued_job_ids() -> list[str]:
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            cur = conn.execute(
                "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_at"
            )
            return [r[0] for r in cur.fetchall()]


def create_job(job_id: str) -> None:
    now = time.time()
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, error, result, created_at, updated_at)
                VALUES (?, 'queued', NULL, NULL, ?, ?)
                """,
                (job_id, now, now),
            )
            conn.commit()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("result"):
        try:
            d["result"] = json.loads(d["result"])
        except json.JSONDecodeError:
            d["result"] = None
    return d


def update_job(
    job_id: str,
    status: str,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    now = time.time()
    result_json = json.dumps(result) if result is not None else None
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            if result_json is not None:
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, error = ?, result = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (status, error, result_json, now, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, error = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (status, error, now, job_id),
                )
            conn.commit()
