from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import zipfile
from typing import Any, Dict, List

from qpadm import store
from qpadm.materialize import DEFAULT_SOURCES_MANIFEST, materialize_pop_list_files

logger = logging.getLogger(__name__)

QPADM_BIN = os.environ.get("QPADM_BIN", "qpAdm")
QPADM_TIMEOUT_SEC = int(os.environ.get("QPADM_TIMEOUT_SEC", "3600"))
OUTPUT_READ_MAX = int(os.environ.get("QPADM_OUTPUT_READ_MAX", str(2 * 1024 * 1024)))
_sources_manifest = os.environ.get("QPADM_SOURCES_MANIFEST", DEFAULT_SOURCES_MANIFEST).strip()
QPADM_SOURCES_MANIFEST = (
    ""
    if _sources_manifest.lower() in ("-", "none")
    else _sources_manifest
)
QPADM_AUTO_POP_LISTS = os.environ.get("QPADM_AUTO_POP_LISTS", "true").lower() in (
    "1",
    "true",
    "yes",
)


def validate_par_filename(name: str) -> str:
    n = (name or "qpAdm.par").strip()
    n = os.path.basename(n.replace("\\", "/"))
    if not n or len(n) > 240:
        raise ValueError("Invalid par_filename")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", n):
        raise ValueError("Invalid par_filename")
    return n


def _safe_extract(zip_path: str, dest_dir: str) -> None:
    dest_abs = os.path.abspath(dest_dir)
    os.makedirs(dest_abs, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            rel = member.replace("\\", "/").lstrip("/")
            if ".." in rel.split("/"):
                raise ValueError(f"Unsafe zip entry: {member!r}")
            norm = os.path.normpath(rel)
            if norm.startswith(".."):
                raise ValueError(f"Unsafe zip entry: {member!r}")
            target = os.path.abspath(os.path.join(dest_abs, norm))
            if not target.startswith(dest_abs + os.sep) and target != dest_abs:
                raise ValueError(f"Unsafe zip entry: {member!r}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member, "r") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _read_text_file(path: str, limit: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _collect_output_files(work_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for root, _, files in os.walk(work_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith((".par", ".zip")):
                continue
            if any(
                low.endswith(suf)
                for suf in (
                    ".out",
                    ".log",
                    ".txt",
                    ".stderr",
                    ".stdout",
                )
            ) or "qpadm" in low:
                rel = os.path.relpath(os.path.join(root, fn), work_dir)
                content = _read_text_file(os.path.join(root, fn), OUTPUT_READ_MAX)
                if content:
                    out[rel.replace("\\", "/")] = content
    return out


def run_qpadm_job(job_id: str) -> None:
    root = store.jobs_root()
    job_dir = os.path.join(root, job_id)
    bundle = os.path.join(job_dir, "bundle.zip")
    work_dir = os.path.join(job_dir, "work")

    row = store.get_job(job_id)
    if not row:
        logger.error("qpadm job missing: %s", job_id)
        return
    if row["status"] not in ("queued",):
        return

    par_name = row["par_filename"]

    try:
        if not os.path.isfile(bundle):
            raise FileNotFoundError("bundle.zip missing on server")

        store.update_job(job_id, "running")

        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        _safe_extract(bundle, work_dir)

        par_path = os.path.join(work_dir, par_name)
        if not os.path.isfile(par_path):
            raise FileNotFoundError(
                f"Parameter file not found after extract: {par_name!r} (paths in .par must match zip layout)"
            )

        materialized = materialize_pop_list_files(
            work_dir,
            par_path,
            manifest_basename=QPADM_SOURCES_MANIFEST,
            auto_from_ind=QPADM_AUTO_POP_LISTS,
        )
        if materialized:
            logger.info("qpAdm job %s materialized: %s", job_id, "; ".join(materialized))

        env = os.environ.copy()
        # ADMIXTOOLS often expects PATH; optional QPADM_EXTRA_PATH prepended
        extra = os.environ.get("QPADM_EXTRA_PATH", "").strip()
        if extra:
            env["PATH"] = extra + os.pathsep + env.get("PATH", "")

        proc = subprocess.run(
            [QPADM_BIN, "-p", par_path],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=QPADM_TIMEOUT_SEC,
            errors="replace",
        )

        files_payload = _collect_output_files(work_dir)
        result: Dict[str, Any] = {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:OUTPUT_READ_MAX],
            "stderr": (proc.stderr or "")[:OUTPUT_READ_MAX],
            "output_files": files_payload,
            "materialized": materialized,
        }

        if proc.returncode != 0:
            store.update_job(
                job_id,
                "failed",
                error=f"qpAdm exited with code {proc.returncode}",
                result=result,
            )
        else:
            store.update_job(job_id, "done", result=result)
    except subprocess.TimeoutExpired:
        store.update_job(
            job_id,
            "failed",
            error=f"qpAdm timed out after {QPADM_TIMEOUT_SEC}s",
        )
    except FileNotFoundError as e:
        store.update_job(
            job_id,
            "failed",
            error=str(e),
        )
    except Exception as e:
        logger.exception("qpadm job %s failed", job_id)
        store.update_job(job_id, "failed", error=str(e))
