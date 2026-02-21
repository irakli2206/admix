"""
Maximum-likelihood admixture estimation from genotype counts and population allele frequencies.
"""

import numpy as np
import scipy.optimize as optimize
from raw_data_processing import read_raw_data, read_model
import admix_models


def _canonical_rsid(rsid):
    """Must match raw_data_processing so lookup works (raw data stored with lowercase key)."""
    return (rsid if isinstance(rsid, str) else str(rsid)).strip().lower()


def genotype_matches(genome_data, snp, major_alleles, minor_alleles):
    """Count how many major and minor alleles the sample has at each SNP."""
    n = len(snp)
    g_major1 = np.zeros(n, dtype=float)
    g_major2 = np.zeros(n, dtype=float)
    g_minor1 = np.zeros(n, dtype=float)
    g_minor2 = np.zeros(n, dtype=float)
    for i in range(n):
        gt = genome_data.get(_canonical_rsid(snp[i]), "-")
        if len(gt) >= 1 and gt[0] in "ATGC":
            a1, a2 = gt[0], gt[-1] if len(gt) > 1 else gt[0]
            g_major1[i] = 1.0 if a1 == major_alleles[i] else 0.0
            g_major2[i] = 1.0 if a2 == major_alleles[i] else 0.0
            g_minor1[i] = 1.0 if a1 == minor_alleles[i] else 0.0
            g_minor2[i] = 1.0 if a2 == minor_alleles[i] else 0.0
    g_major = g_major1 + g_major2
    g_minor = g_minor1 + g_minor2
    return g_major, g_minor


# Small epsilon to avoid log(0) in likelihood
_EPS = 1e-10


def likelihood(g_major, g_minor, frequency, admixture_fraction):
    """Negative log-likelihood (minimize this)."""
    p_minor = np.dot(frequency, admixture_fraction)
    p_major = 1.0 - p_minor
    p_minor = np.clip(p_minor, _EPS, 1.0 - _EPS)
    p_major = np.clip(p_major, _EPS, 1.0 - _EPS)
    l1 = np.dot(g_major, np.log(p_major))
    l2 = np.dot(g_minor, np.log(p_minor))
    return -(l1 + l2)


def admix_fraction(model, raw_data_format, raw_data_file=None, tolerance=1e-3):
    """
    Compute admixture proportions via MLE.
    Returns 1D array of length n_populations (fractions sum to 1).
    """
    genome_data = read_raw_data(raw_data_format, raw_data_file)
    snp, minor_alleles, major_alleles, frequency = read_model(model)
    g_major, g_minor = genotype_matches(
        genome_data, snp, major_alleles, minor_alleles
    )

    # Use only SNPs where the sample has a call (otherwise no signal -> uniform result)
    has_call = (g_major + g_minor) > 0
    n_used = int(np.sum(has_call))
    if n_used == 0:
        raise ValueError(
            "No overlapping SNPs between your raw file and the K36 model. "
            "Check that the file is 23andme/Ancestry/FTDNA format and rsIDs match (e.g. rs12345)."
        )

    g_major_u = g_major[has_call]
    g_minor_u = g_minor[has_call]
    frequency_u = frequency[has_call]

    n_pop = admix_models.n_populations(model)
    initial_guess = np.ones(n_pop) / n_pop
    bounds = tuple((0.0, 1.0) for _ in range(n_pop))
    constraints = {"type": "eq", "fun": lambda af: np.sum(af) - 1}

    def objective(af):
        return likelihood(g_major_u, g_minor_u, frequency_u, af)

    result = optimize.minimize(
        objective,
        initial_guess,
        bounds=bounds,
        constraints=constraints,
        tol=float(tolerance),
    )
    return result.x
