"""
Admixture model definitions: SNP file, frequency matrix file, and population names.
K36 uses data/K36.alleles and data/K36.36.F (36 columns = 36 reference populations).
"""

import os

# K36 population names in the same order as columns in K36.36.F (standard Eurogenes K36 order).
# Must match the 36 columns of the frequency matrix.
K36_POPULATIONS = (
    "Amerindian",
    "Arabian",
    "Armenian",
    "Basque",
    "Central_African",
    "Central_Euro",
    "East_African",
    "East_Asian",
    "East_Balkan",
    "East_Central_Asian",
    "East_Central_Euro",
    "East_Med",
    "Eastern_Euro",
    "Fennoscandian",
    "French",
    "Iberian",
    "Indo_Chinese",
    "Italian",
    "Malayan",
    "Near_Eastern",
    "North_African",
    "North_Atlantic",
    "North_Caucasian",
    "North_Sea",
    "Northeast_African",
    "Oceanian",
    "Omotic",
    "Pygmy",
    "Siberian",
    "South_Asian",
    "South_Central_Asian",
    "South_Chinese",
    "Volga_Ural",
    "West_African",
    "West_Caucasian",
    "West_Med",
)

_MODEL_CONFIG = {
    "K36": {
        "snp_file": "K36.alleles",
        "frequency_file": "K36.36.F",
        "n_populations": 36,
        "populations": K36_POPULATIONS,
    },
}


def models():
    """Return list of available model names."""
    return list(_MODEL_CONFIG.keys())


def snp_file_name(model):
    """Return the SNP/alleles filename for the model (in data/)."""
    return _MODEL_CONFIG[model]["snp_file"]


def frequency_file_name(model):
    """Return the frequency matrix filename for the model (in data/)."""
    return _MODEL_CONFIG[model]["frequency_file"]


def n_populations(model):
    """Return the number of reference populations for the model."""
    return _MODEL_CONFIG[model]["n_populations"]


def populations(model):
    """Return list of (english_name, chinese_name) for display. We only use English."""
    names = _MODEL_CONFIG[model]["populations"]
    return [(n, n) for n in names]
