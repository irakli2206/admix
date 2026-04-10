"""Subset a packed EIGENSTRAT trio to individuals whose .ind column 3 is in ``pops``."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Transposed layout is loaded fully into memory; cap to avoid OOM on mistaken huge matrices.
_TRANSPOSED_MAX_MATRIX_BYTES = int(
    os.environ.get("QPADM_TRANSPOSED_MAX_MATRIX_BYTES", str(512 * 1024 * 1024))
)


def _count_newlines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def _subset_snp_major(
    geno_path: str,
    out_geno: str,
    n_ind: int,
    ki: np.ndarray,
    n_kept: int,
) -> int:
    """Each row = one SNP, one byte per individual (qpAdm / Reich packed default)."""
    n_snp = 0
    with open(geno_path, "rb") as fin, open(out_geno, "wb") as fout:
        for raw in fin:
            n_snp += 1
            line = raw.rstrip(b"\r\n")
            if len(line) != n_ind:
                raise ValueError(
                    f".geno line {n_snp}: length {len(line)} != "
                    f"n_individuals {n_ind} from .ind (SNP-major layout)"
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
    """Each row = one individual, one byte per SNP; output SNP-major for qpAdm."""
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
    kept = matrix[ki, :]  # (n_kept, n_snp)
    snp_major = kept.T     # (n_snp, n_kept)
    with open(out_geno, "wb") as fout:
        for row in snp_major:
            fout.write(row.tobytes() + b"\n")
    return n_snp


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
    (3rd whitespace-separated field) is in ``pops``; write a new ``.geno`` with one
    packed genotype character per kept individual per SNP row.

    Supports **SNP-major** (default Reich / qpAdm: each .geno row = one SNP, width =
    n individuals) and **transposed** (each .geno row = one individual, width =
    n SNPs), detected from first row width vs n_ind.  Layout count only reads the
    full file when the first row width does not match n_ind (rare / non-standard).
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

    with open(geno_path, "rb") as f:
        first = f.readline().rstrip(b"\r\n")
    w = len(first)

    if w == n_ind:
        # Standard SNP-major: no need to count lines.
        layout = "snp_major"
    else:
        n_lines = _count_newlines(geno_path)
        if n_lines == n_ind:
            layout = "transposed"
        else:
            raise ValueError(
                f".geno does not match .ind: first row width {w}, "
                f"{n_lines} lines in .geno, {n_ind} lines in .ind. "
                "Expected SNP-major (each row length = n individuals, many rows) or "
                "transposed (each row = one individual, row count = n individuals). "
                "Check that genotypename/snpname/indivname are the same build."
            )

    logger.info(
        "eigenstrat subset layout=%s n_ind=%s n_kept=%s pops=%s",
        layout, n_ind, n_kept, len(pop_set),
    )

    if layout == "snp_major":
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
