import logging
import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from app import config
from app.deps import require_internal_api_key
from qpadm import consumer as qpadm_consumer
from qpadm import store as qpadm_store
from qpadm.workdir import save_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qpadm", tags=["qpAdm"])


class QpAdmJobCreate(BaseModel):
    """
    Build qpAdm.par under the job work dir. ``left_pops`` / ``right_pops`` are
    one population label per line (``.ind`` column 3), Reich-style: first left
    pop is the target, remaining left pops are sources; right pops are outgroups.
    """

    left_pops: list[str] = Field(..., min_length=1, max_length=64)
    right_pops: list[str] = Field(..., min_length=1, max_length=64)
    ind_mode: Literal["full", "custom"] = Field(
        "custom",
        description=(
            "custom (default): copy only individuals whose .ind column 3 is in "
            "left_pops or right_pops into the job work dir first (faster, lower memory). "
            "full: use the reference genotype matrix as-is."
        ),
    )
    genotypename: Optional[str] = None
    snpname: Optional[str] = None
    indivname: Optional[str] = None
    badsnpname: Optional[str] = None
    snplistname: Optional[str] = None
    allsnps: bool = False
    inbreed: bool = False
    details: bool = False

    @model_validator(mode="after")
    def eigenstrat_triplet(self) -> "QpAdmJobCreate":
        g, s, i = self.genotypename, self.snpname, self.indivname
        has_any = bool(g or s or i)
        has_all = bool(g and s and i)
        if has_any and not has_all:
            raise ValueError(
                "Provide all three of genotypename, snpname, indivname, "
                "or omit all to use server defaults."
            )
        return self


@router.post("/jobs")
def qp_adm_create_job(
    body: QpAdmJobCreate,
    _auth: None = Depends(require_internal_api_key),
):
    """
    Queue qpAdm from JSON.  Validates inputs and persists ``request.json``;
    the heavy subset + qpAdm run happen in the background worker.
    Poll GET /qpadm/jobs/{job_id}.
    """
    if not config.QPADM_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="qpAdm is disabled (set QPADM_ENABLED=true).",
        )
    job_id = qpadm_store.new_job_id()
    root = qpadm_store.jobs_root()
    work_dir = os.path.join(root, job_id, "work")

    try:
        snap = save_request(work_dir, body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    qpadm_store.create_job(job_id)
    qpadm_consumer.enqueue(job_id)
    logger.info(
        "qpAdm job %s queued (left=%s pops, right=%s pops)",
        job_id,
        len(body.left_pops),
        len(body.right_pops),
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "ind_mode": snap.get("ind_mode", "custom"),
    }


@router.get("/jobs/{job_id}")
def qp_adm_job_status(
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
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error": row.get("error"),
        "result": row.get("result"),
    }
