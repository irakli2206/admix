"""
Read raw DNA files (23andme, Ancestry, FTDNA, etc.) and read model data (SNPs + frequency matrix).
"""

import csv
import os
import admix_models


def _data_dir():
    """Directory containing data/ (same folder as this module)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def check_file(file_name):
    if not os.path.isfile(file_name):
        raise FileNotFoundError("Cannot find the file '%s'!" % file_name)


def _canonical_rsid(rsid):
    """Normalize rsID for lookup (lowercase, strip)."""
    return (rsid or "").strip().lower()


def twenty_three_and_me(data_file_name):
    check_file(data_file_name)
    processed_data = {}
    with open(data_file_name, "r", encoding="utf-8", errors="replace") as data:
        data = csv.reader(data, delimiter="\t")
        for row in data:
            if len(row) == 4 and row[-1][-1] in ["A", "T", "G", "C"]:
                processed_data[_canonical_rsid(row[0])] = row[-1]
    return processed_data


def ancestry(data_file_name):
    check_file(data_file_name)
    processed_data = {}
    with open(data_file_name, "r", encoding="utf-8", errors="replace") as data:
        data = csv.reader(data, delimiter="\t")
        for row in data:
            if len(row) == 5 and row[-1] in ["A", "T", "G", "C"]:
                processed_data[_canonical_rsid(row[0])] = "".join(row[-2:])
    return processed_data


def ftdna(data_file_name):
    check_file(data_file_name)
    processed_data = {}
    with open(data_file_name, "r", encoding="utf-8", errors="replace") as data:
        data = csv.reader(data)
        for row in data:
            if row[0] == "RSID":
                continue
            if row[0].startswith("#"):
                continue
            if len(row) == 4 and row[-1][-1] in ["A", "T", "G", "C"]:
                processed_data[_canonical_rsid(row[0])] = row[-1]
    return processed_data


def ftdna2(data_file_name):
    check_file(data_file_name)
    processed_data = {}
    with open(data_file_name, "r", encoding="utf-8", errors="replace") as data:
        data = csv.reader(data)
        for row in data:
            if row[0].startswith("#"):
                continue
            if len(row) == 5 and row[-1] in ["A", "T", "G", "C"]:
                processed_data[_canonical_rsid(row[0])] = "".join(row[-2:])
    return processed_data


def wegene(data_file_name):
    return twenty_three_and_me(data_file_name)


def myheritage(data_file_name):
    return ftdna(data_file_name)


def read_raw_data(data_format, data_file_name=None):
    if data_format == "23andme":
        if data_file_name is not None:
            return twenty_three_and_me(data_file_name)
        return twenty_three_and_me(
            os.path.join(_data_dir(), "demo_genome_23andme.txt")
        )
    if data_format == "ancestry":
        if data_file_name is None:
            raise ValueError("Data file not set!")
        return ancestry(data_file_name)
    if data_format == "ftdna":
        if data_file_name is None:
            raise ValueError("Data file not set!")
        return ftdna(data_file_name)
    if data_format == "ftdna2":
        if data_file_name is None:
            raise ValueError("Data file not set!")
        return ftdna2(data_file_name)
    if data_format == "wegene":
        if data_file_name is None:
            raise ValueError("Data file not set!")
        return wegene(data_file_name)
    if data_format == "myheritage":
        if data_file_name is None:
            raise ValueError("Data file not set!")
        return myheritage(data_file_name)
    raise ValueError("Data format does not exist: %s" % data_format)


def read_model(model):
    """
    Load SNP list and frequency matrix.
    K36.alleles format: rsID allele1 allele2 (space-separated).
    K36.36.F has one row per SNP: frequency of the SECOND allele (column 3) in each of 36 populations.
    We treat that allele as 'minor' and the first as 'major' so the likelihood matches the .F matrix.
    Returns: (snp, minor_alleles, major_alleles, frequency) as numpy arrays.
    frequency shape = (n_snps, n_populations).
    """
    snp_file_name = admix_models.snp_file_name(model)
    frequency_file_name = admix_models.frequency_file_name(model)
    base = _data_dir()

    snp = []
    minor_alleles = []
    major_alleles = []

    snp_path = os.path.join(base, snp_file_name)
    check_file(snp_path)
    with open(snp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                snp.append(parts[0])
                # .F matrix is frequency of second allele (column 3); treat as minor for likelihood
                major_alleles.append(parts[1])
                minor_alleles.append(parts[2])

    freq_path = os.path.join(base, frequency_file_name)
    check_file(freq_path)
    frequency = []
    with open(freq_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            row = [float(x) for x in line.split()]
            if row:
                frequency.append(row)

    import numpy as np
    # Align lengths: frequency matrix must have one row per SNP (same order as alleles).
    n_snp = len(snp)
    n_freq = len(frequency)
    n = min(n_snp, n_freq)
    if n_freq != n_snp:
        import logging
        logging.getLogger(__name__).warning(
            "K36 alleles file has %d SNPs but frequency file has %d rows; using first %d.",
            n_snp, n_freq, n,
        )
    return (
        np.array(snp[:n]),
        np.array(minor_alleles[:n]),
        np.array(major_alleles[:n]),
        np.array(frequency[:n]),
    )
