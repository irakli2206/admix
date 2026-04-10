"""Write request.json for ADMIXTOOLS 2 qpadm R script into a job work directory."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from qpadm.paths import resolve_under_allowed

REQUEST_NAME = "request.json"

_POP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-:]*$")


def validate_pop_token(s: str) -> str:
    t = (s or "").strip()
    if not t or len(t) > 256:
        raise ValueError(f"Invalid population label: {s!r}")
    if not _POP_RE.match(t):
        raise ValueError(f"Invalid population label (allowed: letters, digits, ._-:): {s!r}")
    return t


def _default_triplet() -> tuple[str, str, str]:
    geno = os.environ.get("QPADM_DEFAULT_GENO", "").strip()
    snp = os.environ.get("QPADM_DEFAULT_SNP", "").strip()
    ind = os.environ.get("QPADM_DEFAULT_IND", "").strip()
    if not (geno and snp and ind):
        raise ValueError(
            "Set QPADM_DEFAULT_GENO, QPADM_DEFAULT_SNP, QPADM_DEFAULT_IND on the server "
            "(or pass genotypename, snpname, indivname in the request)."
        )
    return (
        resolve_under_allowed(geno),
        resolve_under_allowed(snp),
        resolve_under_allowed(ind),
    )


def _geno_prefix(geno_path: str) -> str:
    """Strip the .geno extension to get the EIGENSTRAT prefix for ADMIXTOOLS 2."""
    for ext in (".geno",):
        if geno_path.endswith(ext):
            return geno_path[: -len(ext)]
    return geno_path


def f2_dir() -> str:
    return os.environ.get("QPADM_F2_DIR", "").strip()


def save_request(work_dir: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and persist the request JSON. No heavy I/O.

    Called by the POST handler so the API responds instantly.
    Returns a snap dict for the response.
    """
    os.makedirs(work_dir, exist_ok=True)

    left_pops: List[str] = [validate_pop_token(x) for x in body["left_pops"]]
    right_pops: List[str] = [validate_pop_token(x) for x in body["right_pops"]]
    if not left_pops or not right_pops:
        raise ValueError("left_pops and right_pops must be non-empty")

    if body.get("genotypename"):
        geno = resolve_under_allowed(str(body["genotypename"]))
        snp = resolve_under_allowed(str(body["snpname"]))
        ind = resolve_under_allowed(str(body["indivname"]))
    else:
        if body.get("snpname") or body.get("indivname"):
            raise ValueError("Provide all of genotypename, snpname, indivname or none for defaults")
        geno, snp, ind = _default_triplet()

    geno_prefix = _geno_prefix(geno)
    f2 = f2_dir()

    snap = {
        "left_pops": left_pops,
        "right_pops": right_pops,
        "geno_prefix": geno_prefix,
        "allsnps": bool(body.get("allsnps", False)),
        "inbreed": bool(body.get("inbreed", False)),
        "details": bool(body.get("details", False)),
    }

    if f2:
        snap["f2_dir"] = f2

    with open(os.path.join(work_dir, REQUEST_NAME), "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    return snap


def write_job_workdir(
    work_dir: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """One-shot: save request (used by tests)."""
    return save_request(work_dir, body)
