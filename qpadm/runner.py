"""Run ADMIXTOOLS 2 qpadm via Rscript and parse structured JSON output."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from qpadm import store
from qpadm.workdir import REQUEST_NAME

logger = logging.getLogger(__name__)

RSCRIPT_BIN = os.environ.get("QPADM_RSCRIPT", "Rscript")
QPADM_TIMEOUT_SEC = int(os.environ.get("QPADM_TIMEOUT_SEC", str(24 * 3600)))

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
RUN_QPADM_R = str(_SCRIPTS_DIR / "run_qpadm.R")


def run_qp_adm_job(job_id: str) -> None:
    root = store.jobs_root()
    work_dir = os.path.join(root, job_id, "work")
    request_path = os.path.join(work_dir, REQUEST_NAME)

    row = store.get_job(job_id)
    if not row:
        logger.error("qpAdm job missing: %s", job_id)
        return
    if row["status"] not in ("queued",):
        return

    try:
        store.update_job(job_id, "running")

        if not os.path.isfile(request_path):
            raise FileNotFoundError(f"Missing {REQUEST_NAME} in job work dir")

        proc = subprocess.run(
            [RSCRIPT_BIN, RUN_QPADM_R, request_path],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=QPADM_TIMEOUT_SEC,
            errors="replace",
        )

        stderr_t = (proc.stderr or "").strip()
        if stderr_t:
            logger.info("qpAdm R stderr (job %s):\n%s", job_id, stderr_t[:4096])

        if proc.returncode != 0:
            error_msg = f"qpAdm R script exited with code {proc.returncode}"
            result: Dict[str, Any] = {"returncode": proc.returncode, "stderr": stderr_t}
            try:
                parsed = json.loads(proc.stdout)
                if parsed.get("error"):
                    error_msg += f": {parsed['error']}"
                result["r_error"] = parsed.get("error")
            except (json.JSONDecodeError, TypeError):
                result["stdout"] = (proc.stdout or "")[:8192]
            store.update_job(job_id, "failed", error=error_msg, result=result)
            return

        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            store.update_job(
                job_id, "failed",
                error=f"Failed to parse R output as JSON: {e}",
                result={"stdout": (proc.stdout or "")[:8192], "stderr": stderr_t},
            )
            return

        store.update_job(job_id, "done", result=result)

    except subprocess.TimeoutExpired:
        store.update_job(
            job_id,
            "failed",
            error=f"qpAdm timed out after {QPADM_TIMEOUT_SEC}s",
        )
    except FileNotFoundError as e:
        store.update_job(job_id, "failed", error=str(e))
    except Exception as e:
        logger.exception("qpAdm job %s failed", job_id)
        store.update_job(job_id, "failed", error=str(e))
