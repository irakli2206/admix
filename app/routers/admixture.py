import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import config
from app.deps import check_admixture_bundle_upload_size, require_internal_api_key
from admixture_jobs import consumer as admixture_consumer
from admixture_jobs import store as admixture_store
from admixture_jobs.runner import resolve_host_plink_bed, validate_plink_prefix

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admixture", tags=["ADMIXTURE"])


def _wants_zip_upload(bundle: UploadFile | None) -> bool:
    """True if client attached a non-empty bundle filename (zip upload path)."""
    if bundle is None:
        return False
    fn = (bundle.filename or "").strip()
    return bool(fn)


@router.post("/jobs")
async def admixture_create_job(
    plink_prefix: str = Form(
        ...,
        description="Stem of mydata.bed / .bim / .fam (in zip or on server mount).",
    ),
    k: int = Form(
        ...,
        description="Number of ancestral populations (K).",
        ge=2,
        le=128,
    ),
    threads: int = Form(
        1,
        description="Thread count passed to admixture as -jN (if > 1).",
        ge=1,
        le=64,
    ),
    cross_validation: bool = Form(
        False,
        description="If true, run with --cv (slower; estimates CV error).",
    ),
    bundle: UploadFile | None = File(
        None,
        description=(
            "Optional zip with PLINK .bed/.bim/.fam at zip root. "
            "Omit this field to use files under ADMIXTURE_HOST_PLINK_ROOT (no large upload)."
        ),
    ),
    _auth: None = Depends(require_internal_api_key),
    _: None = Depends(check_admixture_bundle_upload_size),
):
    """
    Queue an ADMIXTURE run. Poll GET /admixture/jobs/{job_id} until done or failed.

    **Two modes (same URL):**

    - **Server PLINK mount (recommended):** multipart form with `plink_prefix`, `k`, etc. only —
      do **not** send `bundle`. Requires `ADMIXTURE_HOST_PLINK_ROOT` and a read-only mount.
    - **Zip upload:** include multipart file `bundle` (zip at root with matching `.bed/.bim/.fam`).
    """
    if not config.ADMIXTURE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ADMIXTURE is disabled (set ADMIXTURE_ENABLED=true).",
        )
    try:
        prefix_safe = validate_plink_prefix(plink_prefix)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid plink_prefix: use a basename like v62_HO_small",
        )

    use_zip = _wants_zip_upload(bundle)

    if use_zip:
        job_kind = "bundle"
    else:
        if not config.ADMIXTURE_HOST_PLINK_ROOT:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No bundle uploaded: configure ADMIXTURE_HOST_PLINK_ROOT and mount host "
                    "PLINK data, or attach a zip as form field 'bundle'."
                ),
            )
        try:
            resolve_host_plink_bed(prefix_safe)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        job_kind = "host_disk"

    job_id = admixture_store.new_job_id()
    root = admixture_store.jobs_root()
    job_dir = os.path.join(root, job_id)
    os.makedirs(job_dir, exist_ok=True)
    bundle_path = os.path.join(job_dir, "bundle.zip")

    total = 0
    if use_zip:
        assert bundle is not None
        max_b = config.ADMIXTURE_MAX_BUNDLE_BYTES
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
                            detail=f"Bundle too large (max {config.ADMIXTURE_MAX_BUNDLE_MB} MB).",
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
        if total == 0:
            if os.path.isfile(bundle_path):
                os.remove(bundle_path)
            try:
                os.rmdir(job_dir)
            except OSError:
                pass
            raise HTTPException(
                status_code=400,
                detail="bundle file is empty",
            )

    admixture_store.create_job(
        job_id,
        prefix_safe,
        k,
        threads,
        cross_validation,
        job_kind=job_kind,
    )
    admixture_consumer.enqueue(job_id)
    logger.info(
        "ADMIXTURE job %s queued (kind=%s, prefix=%s, k=%s, threads=%s, cv=%s, bytes=%s)",
        job_id,
        job_kind,
        prefix_safe,
        k,
        threads,
        cross_validation,
        total if use_zip else 0,
    )
    return {"job_id": job_id, "status": "queued", "job_kind": job_kind}


@router.get("/jobs/{job_id}")
def admixture_job_status(
    job_id: str,
    _auth: None = Depends(require_internal_api_key),
):
    if not config.ADMIXTURE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ADMIXTURE is disabled (set ADMIXTURE_ENABLED=true).",
        )
    row = admixture_store.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "job_kind": row.get("job_kind") or "bundle",
        "plink_prefix": row["plink_prefix"],
        "k": row["k"],
        "threads": row["threads"],
        "cross_validation": row["cross_validation"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error": row.get("error"),
        "result": row.get("result"),
    }
