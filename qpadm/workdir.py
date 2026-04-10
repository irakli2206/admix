"""Write qpAdm.par and left/right pop list files into a job work directory."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from qpadm.eigenstrat_subset import subset_eigenstrat_by_pops
from qpadm.paths import resolve_under_allowed
from qpadm.subset_cache import get_cached, put_cached

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


def _try_symlink(src: str, dst: str) -> None:
    """Symlink src→dst, falling back to hard-link or no-op."""
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            pass


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

    ind_mode = str(body.get("ind_mode", "custom")).strip().lower()
    if ind_mode not in ("full", "custom"):
        raise ValueError('ind_mode must be "full" or "custom"')

    optional_paths: Dict[str, str] = {}
    for key in ("badsnpname", "snplistname"):
        val = body.get(key)
        if val:
            optional_paths[key] = resolve_under_allowed(str(val))

    snap = {
        "left_pops": left_pops,
        "right_pops": right_pops,
        "genotypename": geno,
        "snpname": snp,
        "indivname": ind,
        "ind_mode": ind_mode,
        "allsnps": bool(body.get("allsnps", False)),
        "inbreed": bool(body.get("inbreed", False)),
        "details": bool(body.get("details", False)),
        **{k: v for k, v in optional_paths.items()},
    }
    with open(os.path.join(work_dir, REQUEST_NAME), "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    return snap


def prepare_workdir(work_dir: str) -> Dict[str, Any]:
    """Build qpAdm.par from the saved request.json — called by the background worker.

    Performs the (potentially slow) subsetting here instead of in the POST handler.
    Uses the subset cache to skip repeated work.
    Returns the final snap dict.
    """
    req_path = os.path.join(work_dir, REQUEST_NAME)
    with open(req_path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    left_pops: List[str] = snap["left_pops"]
    right_pops: List[str] = snap["right_pops"]
    geno: str = snap["genotypename"]
    snp: str = snap["snpname"]
    ind: str = snap["indivname"]
    ind_mode: str = snap.get("ind_mode", "custom")

    source_triplet: Optional[tuple[str, str, str]] = None
    if ind_mode == "custom":
        allowed = set(left_pops) | set(right_pops)
        cached = get_cached(geno, allowed)
        if cached is not None:
            source_triplet = (geno, snp, ind)
            geno, snp, ind = cached
        else:
            new_geno, new_snp, new_ind = subset_eigenstrat_by_pops(
                geno, snp, ind, allowed, work_dir,
            )
            subsetted = os.path.realpath(new_geno) != os.path.realpath(geno)
            if subsetted:
                source_triplet = (geno, snp, ind)
                geno, snp, ind = put_cached(
                    snap["genotypename"], allowed, new_geno, new_snp, new_ind,
                )
            else:
                geno, snp, ind = new_geno, new_snp, new_ind

    with open(os.path.join(work_dir, LEFT_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(left_pops) + "\n")
    with open(os.path.join(work_dir, RIGHT_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(right_pops) + "\n")

    allsnps = snap.get("allsnps", False)
    inbreed = snap.get("inbreed", False)
    details = snap.get("details", False)

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
    for k in ("badsnpname", "snplistname"):
        v = snap.get(k)
        if v:
            lines.append(f"{k}: {v}")

    with open(os.path.join(work_dir, PAR_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    snap["genotypename"] = geno
    snap["snpname"] = snp
    snap["indivname"] = ind
    if source_triplet is not None:
        snap["subset_source"] = {
            "genotypename": source_triplet[0],
            "snpname": source_triplet[1],
            "indivname": source_triplet[2],
        }

    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    return snap


# Keep for backward compatibility with tests that call it directly
def write_job_workdir(
    work_dir: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """One-shot: save request + prepare workdir (used by tests)."""
    save_request(work_dir, body)
    return prepare_workdir(work_dir)
