"""K36 MLE + K36→G25 regression (paths, matrices, sync pipeline for SSE)."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from progress_tracker import ConversionProgress

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_K36_ALLELES_PATH = PROJECT_ROOT / "data" / "K36.alleles"
_K36_FREQ_PATH = PROJECT_ROOT / "data" / "K36.36.F"
_K36_G25_MATRIX: Optional[pd.DataFrame] = None


def builtin_raw_to_k36_available() -> bool:
    return _K36_ALLELES_PATH.is_file() and _K36_FREQ_PATH.is_file()


def run_builtin_raw_to_k36(
    raw_path: str,
    vendor: str,
    progress: Optional[ConversionProgress] = None,
) -> Dict[str, float]:
    import admix_models
    from admix_fraction import admix_fraction

    frac = admix_fraction(
        "K36", vendor, raw_path, tolerance=1e-3, progress=progress
    )
    populations = admix_models.populations("K36")
    return {pop_en: round(100.0 * f, 2) for (pop_en, _), f in zip(populations, frac)}


def normalize_k36_key(key: str) -> str:
    key = key.strip().replace("-", "_").replace(" ", "_")
    variants = {
        "indo chinese": "Indo_Chinese",
        "central african": "Central_African",
        "east african": "East_African",
        "west african": "West_African",
        "north african": "North_African",
        "northeast african": "Northeast_African",
        "south asian": "South_Asian",
        "east asian": "East_Asian",
        "central euro": "Central_Euro",
        "eastern euro": "Eastern_Euro",
        "east central euro": "East_Central_Euro",
        "east balkan": "East_Balkan",
        "east med": "East_Med",
        "west med": "West_Med",
        "north sea": "North_Sea",
        "near eastern": "Near_Eastern",
        "north atlantic": "North_Atlantic",
        "north caucasian": "North_Caucasian",
        "west caucasian": "West_Caucasian",
        "east central asian": "East_Central_Asian",
        "south central asian": "South_Central_Asian",
        "south chinese": "South_Chinese",
        "volga ural": "Volga_Ural",
    }
    k = key.lower().replace("_", " ")
    return variants.get(k, key)


def get_k36_to_g25_matrix() -> pd.DataFrame:
    global _K36_G25_MATRIX
    if _K36_G25_MATRIX is None:
        weights_path = PROJECT_ROOT / "k36_to_g25_weights.csv"
        _K36_G25_MATRIX = pd.read_csv(weights_path, index_col=0)
    return _K36_G25_MATRIX


def k36_vector_from_dict(k36_results: Dict[str, float]) -> List[float]:
    matrix = get_k36_to_g25_matrix()
    components = [idx for idx in matrix.index if idx != "INTERCEPT"]
    normalized_results = {normalize_k36_key(k): v for k, v in k36_results.items()}
    vector: List[float] = []
    for name in components:
        norm_name = normalize_k36_key(name)
        vector.append(normalized_results.get(norm_name, 0.0))
    return vector


def k36_to_g25(user_k36_data: List[float]) -> List[float]:
    matrix = get_k36_to_g25_matrix()
    weights = matrix.drop("INTERCEPT").values
    intercept = matrix.loc["INTERCEPT"].values
    result = np.dot(user_k36_data, weights) + intercept
    return result.tolist()


def validated_g25_coords(raw_coords: List[float]) -> List[float]:
    coords = [float(c) for c in raw_coords]
    if len(coords) != 25:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected G25 dimension: expected 25, got {len(coords)}.",
        )
    if not all(np.isfinite(c) for c in coords):
        raise HTTPException(
            status_code=500,
            detail="Invalid G25 output: non-finite coordinate detected.",
        )
    return [round(c, 6) for c in coords]


def g25_response_dict(k36_results: Dict[str, float], sample_name: str) -> Dict:
    user_k36_vector = k36_vector_from_dict(k36_results)
    g25_coords = validated_g25_coords(k36_to_g25(user_k36_vector))
    g25_coords_csv = ",".join(str(c) for c in g25_coords)
    vahaduo_string = f"{sample_name},{g25_coords_csv}"
    return {
        "status": "success",
        "k36_results": k36_results,
        "g25_coordinates": g25_coords,
        "g25_coords_csv": g25_coords_csv,
        "vahaduo_format": vahaduo_string,
        "note": "SIMULATED G25 from K36 regression. `g25_coords_csv` is coords-only; `vahaduo_format` is sample label + coords.",
    }


def sync_raw_to_g25_with_progress(
    temp_path: str, vendor: str, sample_name: str, progress: ConversionProgress
) -> None:
    try:
        if not builtin_raw_to_k36_available():
            raise RuntimeError(
                "K36 data missing. Add data/K36.alleles and data/K36.36.F to the server."
            )
        progress.set(1.0, "starting")
        k36_results = run_builtin_raw_to_k36(temp_path, vendor, progress=progress)
        progress.bump(92.0, "g25_regression")
        payload = g25_response_dict(k36_results, sample_name)
        progress.complete(payload)
    except Exception as e:
        progress.fail(str(e))
        logger.exception("raw-to-g25 progress worker failed: %s", e)
