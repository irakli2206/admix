import os
import shutil
import uuid
import zipfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import config
from app.deps import check_qpadm_bundle_upload_size, require_internal_api_key
from qpadm import consumer as qpadm_consumer
from qpadm import store as qpadm_store
from qpadm.runner import validate_par_filename

router = APIRouter(prefix="/qpadm", tags=["qpAdm"])


@router.post("/jobs")
async def qpadm_create_job(
    bundle: UploadFile = File(
        ...,
        description=(
            "Zip: qpAdm .par (+ optional qpadm_sources.json). "
            "Missing pop-list files are created from the JSON or from indivname .ind (pop column)."
        ),
    ),
    par_filename: str = Form(
        "qpAdm.par",
        description="Basename of the parameter file inside the zip (no directories)",
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
    if not bundle.filename:
        raise HTTPException(status_code=400, detail="bundle file required")
    try:
        par_safe = validate_par_filename(par_filename)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid par_filename: use a basename like qpAdm.par",
        )

    job_id = str(uuid.uuid4())
    root = qpadm_store.jobs_root()
    job_dir = os.path.join(root, job_id)
    os.makedirs(job_dir, exist_ok=True)
    zip_path = os.path.join(job_dir, "bundle.zip")

    size = 0
    try:
        with open(zip_path, "wb") as out:
            while True:
                chunk = await bundle.read(256 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > config.QPADM_MAX_BUNDLE_BYTES:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Bundle too large (max {config.QPADM_MAX_BUNDLE_MB} MB).",
                    )
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    if size == 0 or not zipfile.is_zipfile(zip_path):
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="Upload must be a non-empty .zip file.",
        )

    qpadm_store.create_job(job_id, par_safe)
    qpadm_consumer.enqueue(job_id)
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
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "par_filename": row["par_filename"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error": row["error"],
        "result": row["result"],
    }
