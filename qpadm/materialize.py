"""Create qpAdm pop-list files from qpadm_sources.json and/or EIGENSTRAT .ind (col1=id, col3=pop).

Reich qpAdm (see DReichLab qpAdm.c / CompPopGen workshop): each population list file must name
**population labels** matching **column 3** of the .ind — typically **one line per file**, the same
label as the filename (e.g. file ``French.DG`` contains the single line ``French.DG``). qpAdm then
includes **all** individuals with that pop label. Putting **individual IDs** (column 1) in those
files makes qpAdm treat each line as a population name → ``zero samples`` for IDs like ``S_French-2.DG``.

Precedence for each pop token:
1. Entry in qpadm_sources.json (if enabled) → write list (authoritative; use for explicit ID subsets).
2. Else expansion from indivname .ind (if enabled and file readable) → write pop label (default) or IDs
   (see QPADM_IND_LIST_STYLE).
3. Else keep an existing file from the zip (if present).

This avoids stale empty/broken list files shipped in the zip overriding good manifest/.ind data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

DEFAULT_SOURCES_MANIFEST = "qpadm_sources.json"

_POP_LINE = re.compile(r"^\s*(popleft|popright)\s*:\s*(.*)$", re.IGNORECASE)


def _par_tokens(par_path: str) -> Set[str]:
    tokens: Set[str] = set()
    try:
        with open(par_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _POP_LINE.match(line)
                if not m:
                    continue
                rest = m.group(2).strip()
                if not rest or rest.upper() == "YES" or rest.upper() == "NO":
                    continue
                for part in rest.split():
                    t = part.strip()
                    if t:
                        tokens.add(t)
    except OSError as e:
        logger.warning("qpAdm materialize: cannot read .par: %s", e)
    return tokens


def _read_manifest(work_dir: str, basename: str) -> Dict[str, List[str]]:
    path = os.path.join(work_dir, basename)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("qpAdm materialize: %s", e)
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, list):
            continue
        ids = [str(x).strip() for x in v if str(x).strip()]
        if ids:
            out[k] = ids
    return out


def _indivname_path(par_path: str) -> str | None:
    try:
        with open(par_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.lower().startswith("indivname:"):
                    p = s.split(":", 1)[1].strip()
                    return p or None
    except OSError:
        pass
    return None


def _parse_ind_for_pops(
    ind_path: str, skip_enhanced: bool
) -> Dict[str, List[str]]:
    """Map population label -> list of individual IDs (col1)."""
    by_pop: Dict[str, List[str]] = {}
    try:
        with open(ind_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                iid, _sex, pop = parts[0], parts[1], parts[2]
                if skip_enhanced and "_enhanced" in iid:
                    continue
                by_pop.setdefault(pop, []).append(iid)
    except OSError as e:
        logger.warning("qpAdm materialize: cannot read .ind: %s", e)
    return by_pop


def _resolve_ind_path(work_dir: str, raw: str) -> str | None:
    """Return readable .ind path: absolute as-is, or under work_dir."""
    raw = raw.strip()
    if not raw:
        return None
    if os.path.isabs(raw):
        return raw if os.path.isfile(raw) else None
    cand = os.path.normpath(os.path.join(work_dir, raw))
    return cand if os.path.isfile(cand) else None


def _write_list(path: str, ids: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for iid in ids:
            out.write(iid + "\n")


def materialize_pop_list_files(
    work_dir: str,
    par_path: str,
    *,
    manifest_basename: str,
    auto_from_ind: bool,
) -> List[str]:
    """Write missing pop list files; return human-readable log lines."""
    log: List[str] = []
    tokens = _par_tokens(par_path)
    if not tokens:
        return log

    manifest: Dict[str, List[str]] = {}
    if manifest_basename:
        manifest = _read_manifest(work_dir, manifest_basename)

    skip_enh = os.environ.get("QPADM_IND_SKIP_ENHANCED", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    by_pop: Dict[str, List[str]] = {}
    ind_resolved: str | None = None
    if auto_from_ind:
        raw_ind = _indivname_path(par_path)
        if raw_ind:
            ind_resolved = _resolve_ind_path(work_dir, raw_ind)
            if ind_resolved:
                by_pop = _parse_ind_for_pops(ind_resolved, skip_enh)
            else:
                logger.warning(
                    "qpAdm materialize: indivname not found: %s "
                    "(use a mounted server path or a relative path inside the zip).",
                    raw_ind,
                )

    for token in sorted(tokens):
        path = os.path.join(work_dir, token)
        if token in manifest:
            _write_list(path, manifest[token])
            log.append(
                f"materialized {token!r} ({len(manifest[token])} ids, from {manifest_basename})"
            )
            continue
        if auto_from_ind and ind_resolved and token in by_pop:
            ids = by_pop[token]
            use_individuals = os.environ.get(
                "QPADM_IND_LIST_STYLE", "poplabel"
            ).lower() in ("individuals", "ids", "samples")
            if use_individuals:
                _write_list(path, ids)
                log.append(
                    f"materialized {token!r} ({len(ids)} individual ids, from .ind)"
                )
            else:
                _write_list(path, [token])
                log.append(
                    f"materialized {token!r} (pop label; {len(ids)} samples in .ind)"
                )
            continue
        # Keep zip-provided file if present; otherwise qpAdm will fail clearly.
        if not os.path.isfile(path):
            log.append(
                f"warning: no list file for {token!r} (add to zip, {manifest_basename}, or readable .ind)"
            )

    return log
