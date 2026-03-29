"""Create qpAdm pop-list files from qpadm_sources.json and/or EIGENSTRAT .ind (col1=id, col3=pop)."""

from __future__ import annotations

import difflib
import json
import logging
import os
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

DEFAULT_SOURCES_MANIFEST = "qpadm_sources.json"


def _parse_par(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, rest = line.partition(":")
            out[key.strip().lower()] = rest.strip()
    return out


def _bad_token(t: str) -> bool:
    return not t or t in (".", "..") or ".." in t or "/" in t or "\\" in t


def _resolve_ind_path(work_dir: str, raw: str) -> str:
    p = raw.strip()
    if not p:
        raise ValueError("indivname in .par is empty")
    if os.path.isabs(p):
        return p
    # Windows: paths like /var/qpadm/... are not isabs() but are Unix VPS paths — do not join work_dir.
    if os.name == "nt" and p.startswith("/"):
        return p
    work_abs = os.path.abspath(work_dir)
    cand = os.path.abspath(os.path.normpath(os.path.join(work_abs, p)))
    if cand != work_abs and not cand.startswith(work_abs + os.sep):
        raise ValueError(f"indivname path escapes work directory: {p!r}")
    return cand


def _load_manifest(work_dir: str, basename: str) -> Dict[str, List[str]]:
    if not basename:
        return {}
    path = os.path.join(work_dir, basename)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{basename} must be a JSON object")
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or _bad_token(k):
            raise ValueError(f"Invalid manifest key: {k!r}")
        if isinstance(v, str):
            ids = [v.strip()] if v.strip() else []
        elif isinstance(v, list):
            ids = [str(x).strip() for x in v if str(x).strip()]
        else:
            raise ValueError(f"Manifest value for {k!r} must be list or string")
        out[k] = ids
    return out


def _pop_to_sample_ids(
    ind_path: str, *, skip_enhanced: bool = True
) -> Dict[str, List[str]]:
    """Map pop label -> sample ids. Skips ids containing '_enhanced' when skip_enhanced (AADR duplicate rows)."""
    m: Dict[str, List[str]] = {}
    with open(ind_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            sid = parts[0]
            if skip_enhanced and "_enhanced" in sid:
                continue
            m.setdefault(parts[2], []).append(sid)
    return m


def materialize_pop_list_files(
    work_dir: str,
    par_path: str,
    *,
    manifest_basename: str = DEFAULT_SOURCES_MANIFEST,
    auto_from_ind: bool = True,
) -> List[str]:
    """Missing list files for popleft/popright tokens: manifest → else .ind pop column. Skip if file exists."""
    par = _parse_par(par_path)
    tokens: Set[str] = set()
    for key in ("popleft", "popright"):
        if key in par:
            tokens.update(t for t in par[key].split() if t)
    if not tokens:
        return []

    manifest = _load_manifest(work_dir, manifest_basename)
    pop_map: Dict[str, List[str]] = {}
    ind_hint = ""
    if auto_from_ind and "indivname" in par:
        raw_ind = par["indivname"]
        try:
            ind_path = _resolve_ind_path(work_dir, raw_ind)
        except ValueError as e:
            logger.warning("qpAdm materialize: %s", e)
            ind_path = ""
            ind_hint = f" Could not resolve indivname: {e}"
        if ind_path and os.path.isfile(ind_path):
            _skip_enh = os.environ.get("QPADM_IND_SKIP_ENHANCED", "true").lower() in (
                "1",
                "true",
                "yes",
            )
            pop_map = _pop_to_sample_ids(ind_path, skip_enhanced=_skip_enh)
        elif ind_path:
            logger.warning("qpAdm materialize: indivname not found: %s", ind_path)
            ind_hint = (
                f" indivname file missing: {ind_path!r} "
                "(common on Windows if .par uses Linux paths like /var/qpadm/ref/...). "
                "Use a reachable path, or ship qpadm_sources.json / per-pop list files in the zip."
            )
    elif auto_from_ind:
        ind_hint = " No indivname: line in .par; cannot auto-expand from .ind."

    def _missing_msg(tok: str) -> str:
        base = (
            f"Missing pop list for {tok!r}. Add a key to {manifest_basename}, "
            "put a list file named like that token in the zip, or match .ind column 3 exactly."
        )
        if ind_hint:
            return base + ind_hint
        if pop_map:
            close = difflib.get_close_matches(tok, pop_map.keys(), n=6, cutoff=0.5)
            sub = [k for k in pop_map if tok.split("_")[0].lower() in k.lower() or tok.lower() in k.lower()]
            sub = [k for k in sub if k != tok][:8]
            extra = ""
            if close:
                extra += f" Close labels in .ind: {close}."
            elif sub:
                extra += f" Partial name matches in .ind: {sub}."
            return base + extra
        return base

    log: List[str] = []
    for token in sorted(tokens):
        if _bad_token(token):
            raise ValueError(f"Unsafe popleft/popright token: {token!r}")
        target = os.path.join(work_dir, token)
        if os.path.isfile(target):
            continue

        if token in manifest:
            ids, src = manifest[token], "manifest"
        elif token in pop_map:
            ids, src = pop_map[token], ".ind"
        else:
            raise FileNotFoundError(_missing_msg(token))

        ids = [i.strip() for i in ids if i.strip()]
        if not ids:
            raise ValueError(f"Pop list for {token!r} is empty")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(ids) + "\n")
        log.append(f"materialized {token!r} ({len(ids)} ids, from {src})")
    return log
