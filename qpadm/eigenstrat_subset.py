"""Subset a packed EIGENSTRAT trio to individuals whose .ind column 3 is in ``pops``."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Iterable

logger = logging.getLogger(__name__)


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

    Individual order in the output matches the order of kept rows in the
    original ``.ind`` (same column order qpAdm expects for those samples).
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

    ki = tuple(keep_indices)
    max_ki = ki[-1]

    n_kept = len(ki)
    n_snp = 0
    with open(geno_path, "rb") as fin, open(out_geno, "wb") as fout:
        for raw in fin:
            n_snp += 1
            line = raw.rstrip(b"\r\n")
            if len(line) != n_ind:
                raise ValueError(
                    f".geno line {n_snp}: length {len(line)} != "
                    f"n_individuals {n_ind} from .ind"
                )
            if max_ki >= len(line):
                raise ValueError(f".geno line {n_snp}: inconsistent with .ind length")
            subset = bytes(line[j] for j in ki)
            if len(subset) != n_kept:
                raise ValueError("internal error building subset genotype row")
            fout.write(subset + b"\n")

    logger.info(
        "eigenstrat subset: %s SNPs, %s -> %s individuals (pops=%s)",
        n_snp,
        n_ind,
        n_kept,
        len(pop_set),
    )

    return (
        os.path.abspath(out_geno),
        os.path.abspath(out_snp),
        os.path.abspath(out_ind),
    )
