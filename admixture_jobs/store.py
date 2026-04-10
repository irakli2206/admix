"""SQLite job store and on-disk job directories for ADMIXTURE runs."""

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
        "ADMIXTURE_JOBS_ROOT",
        str(_REPO_ROOT / "admixture_jobs_data"),
    )


def _db_path() -> str:
    root = jobs_root()
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "admixture_jobs.db")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(jobs)")
    cols = {row[1] for row in cur.fetchall()}
    if "job_kind" not in cols:
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN job_kind TEXT NOT NULL DEFAULT 'bundle'"
        )


def init_db() -> None:
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    plink_prefix TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    threads INTEGER NOT NULL,
                    cross_validation INTEGER NOT NULL,
                    error TEXT,
                    result TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            _migrate_schema(conn)
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


def create_job(
    job_id: str,
    plink_prefix: str,
    k: int,
    threads: int,
    cross_validation: bool,
    job_kind: str = "bundle",
) -> None:
    now = time.time()
    cv_int = 1 if cross_validation else 0
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, plink_prefix, k, threads, cross_validation,
                    error, result, created_at, updated_at, job_kind
                )
                VALUES (?, 'queued', ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (job_id, plink_prefix, k, threads, cv_int, now, now, job_kind),
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
    d["cross_validation"] = bool(d.get("cross_validation"))
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
