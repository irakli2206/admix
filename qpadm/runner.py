from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict

from qpadm import store
from qpadm.workdir import PAR_NAME

logger = logging.getLogger(__name__)

QPADM_BIN = os.environ.get("QPADM_BIN", "qpAdm")
QPADM_TIMEOUT_SEC = int(os.environ.get("QPADM_TIMEOUT_SEC", str(24 * 3600)))
OUTPUT_READ_MAX = int(os.environ.get("QPADM_OUTPUT_READ_MAX", str(2 * 1024 * 1024)))


def _exit_hint(code: int) -> str:
    if code < 0:
        sig = -code
        if sig == 6:
            return (
                " (SIGABRT — often qpAdm fatalx; check stderr for 'zero samples' / "
                "pop labels vs .ind col3)"
            )
        if sig == 9:
            return " (SIGKILL — often OOM or cgroup memory limit)"
        if sig == 11:
            return " (SIGSEGV — qpAdm crashed; try smaller runs or different build)"
    return ""


def _read_text_file(path: str, limit: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _cleanup_subset_files(work_dir: str) -> None:
    """Remove large subset EIGENSTRAT files written by ind_mode=custom after qpAdm exits."""
    for name in ("subset.geno", "subset.snp", "subset.ind"):
        path = os.path.join(work_dir, name)
        try:
            os.remove(path)
        except OSError:
            pass


def _collect_output_files(work_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    skip = {PAR_NAME.lower(), "request.json", "left_pops.txt", "right_pops.txt"}
    for root, _, files in os.walk(work_dir):
        for fn in files:
            low = fn.lower()
            if low in skip:
                continue
            if low.endswith((".par", ".zip")):
                continue
            if low == "gmon.out":
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
            ) or "qpadm" in low or low.endswith((".cov", ".f4")):
                rel = os.path.relpath(os.path.join(root, fn), work_dir)
                content = _read_text_file(os.path.join(root, fn), OUTPUT_READ_MAX)
                if content:
                    out[rel.replace("\\", "/")] = content
    return out


def run_qp_adm_job(job_id: str) -> None:
    root = store.jobs_root()
    work_dir = os.path.join(root, job_id, "work")
    par_path = os.path.join(work_dir, PAR_NAME)

    row = store.get_job(job_id)
    if not row:
        logger.error("qpAdm job missing: %s", job_id)
        return
    if row["status"] not in ("queued",):
        return

    try:
        if not os.path.isfile(par_path):
            raise FileNotFoundError(f"Missing {PAR_NAME} in job work dir")

        store.update_job(job_id, "running")

        env = os.environ.copy()
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

        _cleanup_subset_files(work_dir)
        files_payload = _collect_output_files(work_dir)
        stdout_t = (proc.stdout or "")[:OUTPUT_READ_MAX]
        stderr_t = (proc.stderr or "")[:OUTPUT_READ_MAX]
        hint = _exit_hint(proc.returncode)
        result: Dict[str, Any] = {
            "returncode": proc.returncode,
            "stdout": stdout_t,
            "stderr": stderr_t,
            "exit_hint": hint.strip() or None,
            "output_files": files_payload,
        }

        if proc.returncode != 0:
            err = f"qpAdm exited with code {proc.returncode}{hint}"
            if proc.returncode == 255 and "f4 stats all zero" in stdout_t:
                err += (
                    " — degenerate f4 matrix (Rank 0). Check right outgroups and "
                    "left populations vs .ind labels."
                )
            store.update_job(job_id, "failed", error=err, result=result)
        else:
            store.update_job(job_id, "done", result=result)
    except subprocess.TimeoutExpired:
        _cleanup_subset_files(work_dir)
        store.update_job(
            job_id,
            "failed",
            error=f"qpAdm timed out after {QPADM_TIMEOUT_SEC}s",
        )
    except FileNotFoundError as e:
        _cleanup_subset_files(work_dir)
        store.update_job(job_id, "failed", error=str(e))
    except Exception as e:
        _cleanup_subset_files(work_dir)
        logger.exception("qpAdm job %s failed", job_id)
        store.update_job(job_id, "failed", error=str(e))
