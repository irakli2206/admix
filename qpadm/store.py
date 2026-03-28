from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional

_lock = threading.Lock()


def jobs_root() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("QPADM_JOBS_ROOT", os.path.join(base, "qpadm_jobs"))


def _db_path() -> str:
    root = jobs_root()
    state = os.path.join(root, "_state")
    os.makedirs(state, exist_ok=True)
    return os.path.join(state, "jobs.sqlite")


def init_db() -> None:
    with _lock:
        conn = sqlite3.connect(_db_path())
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    par_filename TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error TEXT,
                    result_json TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def create_job(job_id: str, par_filename: str) -> None:
    now = time.time()
    with _lock:
        conn = sqlite3.connect(_db_path())
        try:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, par_filename, created_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (job_id, par_filename, now, now),
            )
            conn.commit()
        finally:
            conn.close()


def update_job(
    job_id: str,
    status: str,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    now = time.time()
    rj = json.dumps(result) if result is not None else None
    with _lock:
        conn = sqlite3.connect(_db_path())
        try:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, error = ?, result_json = ?
                WHERE job_id = ?
                """,
                (status, now, error, rj, job_id),
            )
            conn.commit()
        finally:
            conn.close()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = sqlite3.connect(_db_path())
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            if d.get("result_json"):
                d["result"] = json.loads(d["result_json"])
            else:
                d["result"] = None
            del d["result_json"]
            return d
        finally:
            conn.close()
