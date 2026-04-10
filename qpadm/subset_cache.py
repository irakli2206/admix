"""Cache subset EIGENSTRAT files keyed by (source .geno path + pop set).

When the same set of populations is requested against the same reference panel,
reuse previously built subset files instead of reading the full binary again.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _cache_root() -> str:
    from qpadm.store import jobs_root
    root = os.path.join(jobs_root(), "_subset_cache")
    os.makedirs(root, exist_ok=True)
    return root


def _cache_key(geno_path: str, pops: Set[str]) -> str:
    real = os.path.realpath(geno_path)
    mtime = 0.0
    try:
        mtime = os.path.getmtime(real)
    except OSError:
        pass
    token = f"{real}\n{mtime}\n" + "\n".join(sorted(pops))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def get_cached(
    geno_path: str,
    pops: Set[str],
) -> Optional[Tuple[str, str, str]]:
    key = _cache_key(geno_path, pops)
    d = os.path.join(_cache_root(), key)
    geno = os.path.join(d, "subset.geno")
    snp = os.path.join(d, "subset.snp")
    ind = os.path.join(d, "subset.ind")
    with _lock:
        if os.path.isfile(geno) and os.path.isfile(snp) and os.path.isfile(ind):
            logger.info("subset cache HIT: %s", key)
            return (geno, snp, ind)
    return None


def put_cached(
    geno_path: str,
    pops: Set[str],
    out_geno: str,
    out_snp: str,
    out_ind: str,
) -> Tuple[str, str, str]:
    """Move job-local subset files into the shared cache dir.

    Returns the cache paths (caller should point .par at these).
    """
    key = _cache_key(geno_path, pops)
    d = os.path.join(_cache_root(), key)
    with _lock:
        os.makedirs(d, exist_ok=True)
        c_geno = os.path.join(d, "subset.geno")
        c_snp = os.path.join(d, "subset.snp")
        c_ind = os.path.join(d, "subset.ind")
        if not os.path.isfile(c_geno):
            os.replace(out_geno, c_geno)
        else:
            _try_remove(out_geno)
        if not os.path.isfile(c_snp):
            os.replace(out_snp, c_snp)
        else:
            _try_remove(out_snp)
        if not os.path.isfile(c_ind):
            os.replace(out_ind, c_ind)
        else:
            _try_remove(out_ind)
    logger.info("subset cache STORE: %s", key)
    return (c_geno, c_snp, c_ind)


def _try_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
