import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import config
from app.deps import check_qpadm_bundle_upload_size, require_internal_api_key
from qpadm import consumer as qpadm_consumer
from qpadm import store as qpadm_store
from qpadm.runner import validate_par_filename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qpadm", tags=["qpAdm"])


@router.post("/jobs")
async def qpadm_create_job(
    bundle: UploadFile = File(
        ...,
        description="Zip: qpAdm .par (+ pop list files and optional qpadm_sources.json).",
    ),
    par_filename: str = Form(
        "qpAdm.par",
        description="Basename of the parameter file inside the zip.",
    ),
    _auth: None = Depends(require_internal_api_key),
    _: None = Depends(check_qpadm_bundle_upload_size),
):
    """
    Queue an ADMIXTOOLS qpAdm run. Poll GET /qpadm/jobs/{job_id} until done or failed.
    """
    if not config.QPADM_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="qpAdm is disabled (set QPADM_ENABLED=true).",
        )
    try:
        par_safe = validate_par_filename(par_filename)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid par_filename: use a basename like qpAdm.par",
        )

    job_id = qpadm_store.new_job_id()
    root = qpadm_store.jobs_root()
    job_dir = os.path.join(root, job_id)
    os.makedirs(job_dir, exist_ok=True)
    bundle_path = os.path.join(job_dir, "bundle.zip")

    max_b = config.QPADM_MAX_BUNDLE_BYTES
    total = 0
    try:
        with open(bundle_path, "wb") as out:
            while True:
                chunk = await bundle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_b:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Bundle too large (max {config.QPADM_MAX_BUNDLE_MB} MB).",
                    )
                out.write(chunk)
    except HTTPException:
        if os.path.isfile(bundle_path):
            os.remove(bundle_path)
        try:
            os.rmdir(job_dir)
        except OSError:
            pass
        raise

    qpadm_store.create_job(job_id, par_safe)
    qpadm_consumer.enqueue(job_id)
    logger.info("qpAdm job %s queued (par=%s, bytes=%s)", job_id, par_safe, total)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def qpadm_job_status(
    job_id: str,
    _auth: None = Depends(require_internal_api_key),
):
    if not config.QPADM_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="qpAdm is disabled (set QPADM_ENABLED=true).",
        )
    row = qpadm_store.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "par_filename": row["par_filename"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error": row.get("error"),
        "result": row.get("result"),
    }
