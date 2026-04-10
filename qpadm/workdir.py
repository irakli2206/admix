"""Write qpAdm.par and left/right pop list files into a job work directory."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from qpadm.eigenstrat_subset import subset_eigenstrat_by_pops
from qpadm.paths import resolve_under_allowed

PAR_NAME = "qpAdm.par"
LEFT_NAME = "left_pops.txt"
RIGHT_NAME = "right_pops.txt"
REQUEST_NAME = "request.json"

_POP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-:]*$")


def validate_pop_token(s: str) -> str:
    t = (s or "").strip()
    if not t or len(t) > 256:
        raise ValueError(f"Invalid population label: {s!r}")
    if not _POP_RE.match(t):
        raise ValueError(f"Invalid population label (allowed: letters, digits, ._-:): {s!r}")
    return t


def _yes_no(b: bool) -> str:
    return "YES" if b else "NO"


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


def write_job_workdir(
    work_dir: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create ``work_dir`` and write ``qpAdm.par``, pop lists, and request snapshot.
    Returns a dict of resolved paths for logging.
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

    ind_mode = str(body.get("ind_mode", "custom")).strip().lower()
    if ind_mode not in ("full", "custom"):
        raise ValueError('ind_mode must be "full" or "custom"')

    source_triplet: Optional[tuple[str, str, str]] = None
    if ind_mode == "custom":
        source_triplet = (geno, snp, ind)
        allowed = set(left_pops) | set(right_pops)
        geno, snp, ind = subset_eigenstrat_by_pops(geno, snp, ind, allowed, work_dir)

    optional_paths: Dict[str, str] = {}
    for key in ("badsnpname", "snplistname"):
        val = body.get(key)
        if val:
            optional_paths[key] = resolve_under_allowed(str(val))

    allsnps = bool(body.get("allsnps", False))
    inbreed = bool(body.get("inbreed", False))
    details = bool(body.get("details", False))

    with open(os.path.join(work_dir, LEFT_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(left_pops) + "\n")
    with open(os.path.join(work_dir, RIGHT_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(right_pops) + "\n")

    lines = [
        f"genotypename: {geno}",
        f"snpname: {snp}",
        f"indivname: {ind}",
        f"popleft: {LEFT_NAME}",
        f"popright: {RIGHT_NAME}",
        f"allsnps: {_yes_no(allsnps)}",
        f"inbreed: {_yes_no(inbreed)}",
    ]
    if details:
        lines.append("details: YES")
    for k, v in optional_paths.items():
        lines.append(f"{k}: {v}")

    with open(os.path.join(work_dir, PAR_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    snap = {
        "left_pops": left_pops,
        "right_pops": right_pops,
        "genotypename": geno,
        "snpname": snp,
        "indivname": ind,
        "ind_mode": ind_mode,
        "allsnps": allsnps,
        "inbreed": inbreed,
        "details": details,
        **{k: v for k, v in optional_paths.items()},
    }
    if source_triplet is not None:
        snap["subset_source"] = {
            "genotypename": source_triplet[0],
            "snpname": source_triplet[1],
            "indivname": source_triplet[2],
        }
    with open(os.path.join(work_dir, REQUEST_NAME), "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    return snap
