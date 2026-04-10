"""Subset a packed EIGENSTRAT trio to individuals whose .ind column 3 is in ``pops``.

Supports three .geno layouts:

* **PACKEDANCESTRYMAP** — binary, 2 bits per genotype, "GENO" header.
  Detected by magic bytes + file-size check.  Output is text EIGENSTRAT so
  we don't need to recompute hashes for the header.
* **Text SNP-major** — one ASCII digit per individual per SNP row (standard
  EIGENSTRAT).
* **Text transposed** — one row per individual (rare).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TRANSPOSED_MAX_MATRIX_BYTES = int(
    os.environ.get("QPADM_TRANSPOSED_MAX_MATRIX_BYTES", str(512 * 1024 * 1024))
)


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _detect_packed(
    geno_path: str,
    n_ind: int,
) -> Tuple[bool, int, int]:
    """Return ``(True, rlen, n_snp)`` if *geno_path* is PACKEDANCESTRYMAP."""
    rlen = max(48, (n_ind + 3) // 4)
    file_size = os.path.getsize(geno_path)
    if file_size < rlen:
        return (False, 0, 0)

    with open(geno_path, "rb") as f:
        header = f.read(rlen)

    if len(header) < 4 or header[:4] != b"GENO":
        return (False, 0, 0)

    try:
        header_text = header.split(b"\x00")[0].decode("ascii", errors="replace")
        parts = header_text.split()
        h_nind = int(parts[1])
        h_nsnp = int(parts[2])
    except (ValueError, IndexError):
        return (False, 0, 0)

    if h_nind != n_ind:
        logger.warning(
            "packed .geno header nind=%d != .ind nind=%d; skipping packed detection",
            h_nind, n_ind,
        )
        return (False, 0, 0)

    expected = rlen * (1 + h_nsnp)
    if file_size != expected:
        return (False, 0, 0)

    return (True, rlen, h_nsnp)


def _count_newlines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def _detect_layout(
    geno_path: str,
    n_ind: int,
) -> Tuple[str, int, int]:
    """Return ``(layout, rlen, n_snp)``.

    *layout* is one of ``"packed"``, ``"snp_major"``, ``"transposed"``, or
    ``"unknown"``.  For ``"packed"`` *rlen* and *n_snp* are set; for text
    layouts they are 0.
    """
    is_packed, rlen, n_snp = _detect_packed(geno_path, n_ind)
    if is_packed:
        return ("packed", rlen, n_snp)

    with open(geno_path, "rb") as f:
        first = f.readline().rstrip(b"\r\n")
    w = len(first)

    if w == n_ind:
        return ("snp_major", 0, 0)

    n_lines = _count_newlines(geno_path)
    if n_lines == n_ind:
        return ("transposed", 0, 0)

    return ("unknown", 0, 0)


# ---------------------------------------------------------------------------
# Subset helpers
# ---------------------------------------------------------------------------

def _subset_packed_binary(
    geno_path: str,
    out_geno: str,
    n_ind: int,
    ki: np.ndarray,
    n_kept: int,
    rlen: int,
    n_snp: int,
) -> int:
    """Read PACKEDANCESTRYMAP binary, write **text EIGENSTRAT** subset.

    Bit packing (EIGENSOFT convention): individual *k* is stored at byte
    ``k // 4``, shift ``(3 - k % 4) * 2`` (high-order bits first).
    Values 0/1/2 = genotypes, 3 = missing (written as ``9`` in text output).
    """
    in_byte_idx = (ki // 4).astype(np.intp)
    in_bit_shift = ((3 - (ki % 4)) * 2).astype(np.uint8)

    chunk_budget = max(1, min(50_000, 100_000_000 // max(rlen, 1)))

    with open(geno_path, "rb") as fin, open(out_geno, "wb") as fout:
        hdr = fin.read(rlen)
        if len(hdr) != rlen:
            raise ValueError("packed .geno: header too short")

        remaining = n_snp
        while remaining > 0:
            chunk_n = min(chunk_budget, remaining)
            raw = fin.read(chunk_n * rlen)
            if len(raw) != chunk_n * rlen:
                raise ValueError(
                    f"packed .geno: expected {chunk_n * rlen} bytes, got {len(raw)}"
                )
            matrix = np.frombuffer(raw, dtype=np.uint8).reshape(chunk_n, rlen)
            geno_vals = (matrix[:, in_byte_idx] >> in_bit_shift) & 3
            text = np.where(geno_vals == 3, 9, geno_vals).astype(np.uint8) + ord("0")
            newlines = np.full((chunk_n, 1), ord("\n"), dtype=np.uint8)
            rows = np.hstack([text, newlines])
            fout.write(rows.tobytes())
            remaining -= chunk_n

    return n_snp


def _subset_snp_major(
    geno_path: str,
    out_geno: str,
    n_ind: int,
    ki: np.ndarray,
    n_kept: int,
) -> int:
    """Text SNP-major: each row = one SNP, one ASCII byte per individual."""
    n_snp = 0
    with open(geno_path, "rb") as fin, open(out_geno, "wb") as fout:
        for raw in fin:
            n_snp += 1
            line = raw.rstrip(b"\r\n")
            if len(line) != n_ind:
                raise ValueError(
                    f".geno line {n_snp}: length {len(line)} != "
                    f"n_individuals {n_ind} from .ind (SNP-major text layout)"
                )
            subset = np.frombuffer(line, dtype=np.uint8)[ki].tobytes()
            fout.write(subset + b"\n")
    return n_snp


def _subset_transposed(
    geno_path: str,
    out_geno: str,
    n_ind: int,
    ki: np.ndarray,
    n_kept: int,
) -> int:
    """Text transposed: each row = one individual; output SNP-major."""
    geno_rows: list[bytes] = []
    with open(geno_path, "rb") as fin:
        for i, raw in enumerate(fin):
            if i >= n_ind:
                raise ValueError(
                    f".geno has more than {n_ind} lines (transposed layout); "
                    "does not match .ind"
                )
            geno_rows.append(raw.rstrip(b"\r\n"))
    if len(geno_rows) != n_ind:
        raise ValueError(
            f".geno has {len(geno_rows)} lines but .ind has {n_ind} (transposed layout)"
        )
    n_snp = len(geno_rows[0])
    matrix_bytes = n_ind * n_snp
    if matrix_bytes > _TRANSPOSED_MAX_MATRIX_BYTES:
        raise ValueError(
            f"transposed .geno too large for in-memory subset ({matrix_bytes} bytes > "
            f"{_TRANSPOSED_MAX_MATRIX_BYTES}); use SNP-major reference files or raise "
            "QPADM_TRANSPOSED_MAX_MATRIX_BYTES"
        )
    row_len = n_snp
    for i, row in enumerate(geno_rows):
        if len(row) != row_len:
            raise ValueError(
                f".geno line {i + 1}: length {len(row)} != first row length {row_len}"
            )
    matrix = np.frombuffer(b"".join(geno_rows), dtype=np.uint8).reshape(n_ind, n_snp)
    kept = matrix[ki, :]
    snp_major = kept.T
    with open(out_geno, "wb") as fout:
        for row in snp_major:
            fout.write(row.tobytes() + b"\n")
    return n_snp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def subset_eigenstrat_by_pops(
    geno_path: str,
    snp_path: str,
    ind_path: str,
    pops: Iterable[str],
    out_dir: str,
    basename: str = "subset",
) -> tuple[str, str, str]:
    """
    Copy ``snp`` unchanged; write a new ``.ind`` with only rows whose population
    (3rd whitespace-separated field) is in ``pops``; write a new ``.geno`` with
    only the kept individuals (text EIGENSTRAT output, auto-detected by qpAdm).

    Supports PACKEDANCESTRYMAP binary, text SNP-major, and text transposed
    ``.geno`` layouts.
    """
    pop_set = {p for p in pops if p}
    if not pop_set:
        raise ValueError("custom ind_mode requires non-empty population labels")

    rows: list[str] = []
    with open(ind_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rows.append(line.rstrip("\r\n"))

    n_ind = len(rows)
    if n_ind == 0:
        raise ValueError(".ind file is empty")

    keep_indices: list[int] = []
    ind_out_lines: list[str] = []
    for i, line in enumerate(rows):
        if not line.strip():
            raise ValueError(f".ind line {i + 1} is empty")
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f".ind line {i + 1} has fewer than 3 fields")
        if parts[2] in pop_set:
            keep_indices.append(i)
            ind_out_lines.append(line)

    if not keep_indices:
        raise ValueError(
            "custom ind_mode: no individuals matched requested populations; "
            "check pop labels match .ind column 3 exactly"
        )

    out_geno = os.path.join(out_dir, f"{basename}.geno")
    out_ind = os.path.join(out_dir, f"{basename}.ind")
    out_snp = os.path.join(out_dir, f"{basename}.snp")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(snp_path, out_snp)

    with open(out_ind, "w", encoding="utf-8", newline="\n") as fout:
        fout.write("\n".join(ind_out_lines) + "\n")

    ki = np.asarray(keep_indices, dtype=np.intp)
    n_kept = len(ki)

    layout, rlen, packed_n_snp = _detect_layout(geno_path, n_ind)

    if layout == "unknown":
        logger.warning(
            "eigenstrat subset: .geno layout unrecognised — "
            "falling back to full reference files",
        )
        for p in (out_geno, out_ind, out_snp):
            try:
                os.remove(p)
            except OSError:
                pass
        return (
            os.path.abspath(geno_path),
            os.path.abspath(snp_path),
            os.path.abspath(ind_path),
        )

    logger.info(
        "eigenstrat subset layout=%s n_ind=%s n_kept=%s pops=%s",
        layout, n_ind, n_kept, len(pop_set),
    )

    if layout == "packed":
        n_snp = _subset_packed_binary(
            geno_path, out_geno, n_ind, ki, n_kept, rlen, packed_n_snp,
        )
    elif layout == "snp_major":
        n_snp = _subset_snp_major(geno_path, out_geno, n_ind, ki, n_kept)
    else:
        n_snp = _subset_transposed(geno_path, out_geno, n_ind, ki, n_kept)

    logger.info(
        "eigenstrat subset done: %s SNPs, %s -> %s individuals",
        n_snp, n_ind, n_kept,
    )

    return (
        os.path.abspath(out_geno),
        os.path.abspath(out_snp),
        os.path.abspath(out_ind),
    )
