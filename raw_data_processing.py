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


# Cache loaded model data so we don't reload the big frequency matrix on every request (avoids OOM on Render).
_model_cache = {}


def read_model(model):
    """
    Load SNP list and frequency matrix. Reads frequency file first so we only load
    that many allele lines (saves memory when alleles file has more lines than frequency).
    Result is cached per model so repeated requests reuse the same arrays (critical on low-RAM).
    Returns: (snp, minor_alleles, major_alleles, frequency) as numpy arrays.
    """
    if model in _model_cache:
        return _model_cache[model]

    snp_file_name = admix_models.snp_file_name(model)
    frequency_file_name = admix_models.frequency_file_name(model)
    base = _data_dir()

    # Load frequency matrix first (smaller; one row per SNP).
    freq_path = os.path.join(base, frequency_file_name)
    check_file(freq_path)
    frequency = []
    with open(freq_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            row = [float(x) for x in line.split()]
            if row:
                frequency.append(row)
    n = len(frequency)

    # Load only the first n SNP lines to match frequency rows (avoids loading huge alleles file on low-RAM).
    snp = []
    minor_alleles = []
    major_alleles = []
    snp_path = os.path.join(base, snp_file_name)
    check_file(snp_path)
    with open(snp_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                snp.append(parts[0])
                major_alleles.append(parts[1])
                minor_alleles.append(parts[2])

    import numpy as np
    n_snp = len(snp)
    if n_snp < n:
        import logging
        logging.getLogger(__name__).warning(
            "K36 alleles has %d lines but frequency has %d rows; using %d.",
            n_snp, n, n_snp,
        )
        n = n_snp
        frequency = frequency[:n]
    result = (
        np.array(snp),
        np.array(minor_alleles),
        np.array(major_alleles),
        np.array(frequency),
    )
    _model_cache[model] = result
    return result
