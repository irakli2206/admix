from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import zipfile
from typing import Any, Dict

from admixture_jobs import store

logger = logging.getLogger(__name__)

ADMIXTURE_BIN = os.environ.get("ADMIXTURE_BIN", "admixture")
ADMIXTURE_TIMEOUT_SEC = int(os.environ.get("ADMIXTURE_TIMEOUT_SEC", str(24 * 3600)))
OUTPUT_READ_MAX = int(
    os.environ.get("ADMIXTURE_OUTPUT_READ_MAX", str(2 * 1024 * 1024))
)


def validate_plink_prefix(name: str) -> str:
    """Basename only; must match .bed/.bim/.fam stem inside the zip."""
    n = (name or "").strip()
    n = os.path.basename(n.replace("\\", "/"))
    if not n or len(n) > 240:
        raise ValueError("Invalid plink_prefix")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", n):
        raise ValueError("Invalid plink_prefix")
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


def _collect_output_files(work_dir: str, prefix: str, k: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    candidates = [
        f"{prefix}.{k}.Q",
        f"{prefix}.{k}.P",
        f"{prefix}.{k}.log",
    ]
    for name in candidates:
        path = os.path.join(work_dir, name)
        if os.path.isfile(path):
            rel = name.replace("\\", "/")
            content = _read_text_file(path, OUTPUT_READ_MAX)
            if content:
                out[rel] = content
    return out


def run_admixture_job(job_id: str) -> None:
    root = store.jobs_root()
    job_dir = os.path.join(root, job_id)
    bundle = os.path.join(job_dir, "bundle.zip")
    work_dir = os.path.join(job_dir, "work")

    row = store.get_job(job_id)
    if not row:
        logger.error("ADMIXTURE job missing: %s", job_id)
        return
    if row["status"] not in ("queued",):
        return

    plink_prefix = row["plink_prefix"]
    k = int(row["k"])
    threads = max(1, int(row["threads"]))
    cv = bool(row.get("cross_validation"))

    try:
        if not os.path.isfile(bundle):
            raise FileNotFoundError("bundle.zip missing on server")

        store.update_job(job_id, "running")

        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        _safe_extract(bundle, work_dir)

        bed_path = os.path.join(work_dir, f"{plink_prefix}.bed")
        bim_path = os.path.join(work_dir, f"{plink_prefix}.bim")
        fam_path = os.path.join(work_dir, f"{plink_prefix}.fam")
        if not os.path.isfile(bed_path):
            raise FileNotFoundError(
                f"Missing {plink_prefix}.bed after extract (zip must contain "
                f"matching .bed/.bim/.fam with the same prefix)."
            )
        if not os.path.isfile(bim_path):
            raise FileNotFoundError(f"Missing {plink_prefix}.bim after extract")
        if not os.path.isfile(fam_path):
            raise FileNotFoundError(f"Missing {plink_prefix}.fam after extract")

        argv = [ADMIXTURE_BIN]
        if threads > 1:
            argv.append(f"-j{threads}")
        if cv:
            argv.append("--cv")
        argv.extend([bed_path, str(k)])

        env = os.environ.copy()
        extra = os.environ.get("ADMIXTURE_EXTRA_PATH", "").strip()
        if extra:
            env["PATH"] = extra + os.pathsep + env.get("PATH", "")

        proc = subprocess.run(
            argv,
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=ADMIXTURE_TIMEOUT_SEC,
            errors="replace",
        )

        files_payload = _collect_output_files(work_dir, plink_prefix, k)
        stdout_t = (proc.stdout or "")[:OUTPUT_READ_MAX]
        stderr_t = (proc.stderr or "")[:OUTPUT_READ_MAX]
        result: Dict[str, Any] = {
            "returncode": proc.returncode,
            "stdout": stdout_t,
            "stderr": stderr_t,
            "output_files": files_payload,
            "command": argv,
        }

        if proc.returncode != 0:
            err = f"admixture exited with code {proc.returncode}"
            store.update_job(job_id, "failed", error=err, result=result)
        else:
            store.update_job(job_id, "done", result=result)
    except subprocess.TimeoutExpired:
        store.update_job(
            job_id,
            "failed",
            error=f"admixture timed out after {ADMIXTURE_TIMEOUT_SEC}s",
        )
    except FileNotFoundError as e:
        store.update_job(job_id, "failed", error=str(e))
    except Exception as e:
        logger.exception("ADMIXTURE job %s failed", job_id)
        store.update_job(job_id, "failed", error=str(e))
