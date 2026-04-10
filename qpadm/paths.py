"""Restrict EIGENSTRAT and sidecar paths to server-managed roots."""

from __future__ import annotations

import os
from typing import Iterable


def _allowed_prefixes() -> list[str]:
    raw = os.environ.get("QPADM_ALLOWED_PATH_PREFIXES", "").strip()
    if raw:
        return [p.strip() for p in raw.split(":") if p.strip()]
    root = os.environ.get("QPADM_DEFAULT_REF_DIR", "/var/qpadm/ref").strip()
    return [root] if root else []


def resolve_under_allowed(path: str) -> str:
    """
    Return realpath of ``path`` if it lies under one of the allowed prefixes.
    Raises ValueError if not allowed or missing.
    """
    if not path or not path.strip():
        raise ValueError("Empty path")
    if ".." in path.split(os.sep):
        raise ValueError("Path must not contain '..'")
    real = os.path.realpath(os.path.expanduser(path.strip()))
    if not os.path.isfile(real):
        raise ValueError(f"Not a file or not found: {path}")
    prefixes = _allowed_prefixes()
    if not prefixes:
        raise ValueError("QPADM_ALLOWED_PATH_PREFIXES or QPADM_DEFAULT_REF_DIR not configured")
    for prefix in prefixes:
        preal = os.path.realpath(prefix)
        root_sep = preal if preal.endswith(os.sep) else preal + os.sep
        if real == preal or real.startswith(root_sep):
            return real
    raise ValueError(f"Path outside allowed qpAdm reference roots: {path}")
