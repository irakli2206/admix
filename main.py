from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Literal
import asyncio
import logging
import os
import tempfile
import zlib

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_rss_mb() -> float:
    """Current process RSS in MB (Linux only; 0 on other platforms)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0.0


app = FastAPI()

VENDOR_CHOICES = Literal["23andme", "ancestry", "ftdna", "ftdna2", "wegene", "myheritage"]

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
# Max decompressed file size (env MAX_DECOMPRESSED_MB, default 50).
_max_decompressed_mb = int(os.environ.get("MAX_DECOMPRESSED_MB", "50"))
MAX_DECOMPRESSED_BYTES = _max_decompressed_mb * 1024 * 1024

MAX_CONCURRENT_CONVERSIONS = 1
_conversion_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONVERSIONS)

# Timeout for K36 conversion (seconds). Prevents hanging on low-memory (e.g. Render 512 MB).
K36_TIMEOUT = int(os.environ.get("K36_CONVERSION_TIMEOUT", "120"))

# Built-in K36: requires data/K36.alleles and data/K36.36.F
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_K36_ALLELES_PATH = os.path.join(_BASE_DIR, "data", "K36.alleles")
_K36_FREQ_PATH = os.path.join(_BASE_DIR, "data", "K36.36.F")


def _builtin_raw_to_k36_available() -> bool:
    return os.path.isfile(_K36_ALLELES_PATH) and os.path.isfile(_K36_FREQ_PATH)


def _run_builtin_raw_to_k36(raw_path: str, vendor: str) -> Dict[str, float]:
    """Run in-project K36 MLE; returns dict population_name -> percentage (0–100)."""
    import admix_models
    from admix_fraction import admix_fraction

    frac = admix_fraction("K36", vendor, raw_path, tolerance=1e-3)
    populations = admix_models.populations("K36")
    return {pop_en: round(100.0 * f, 2) for (pop_en, _), f in zip(populations, frac)}


_K36_G25_MATRIX = None


def get_k36_to_g25_matrix() -> pd.DataFrame:
    """
    Lazily load and cache the K36→G25 regression matrix from CSV.
    The CSV is expected at k36_to_g25_weights.csv in the same directory
    as this file.
    """
    global _K36_G25_MATRIX
    if _K36_G25_MATRIX is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "k36_to_g25_weights.csv")
        _K36_G25_MATRIX = pd.read_csv(weights_path, index_col=0)
    return _K36_G25_MATRIX


def k36_vector_from_dict(k36_results: Dict[str, float]) -> List[float]:
    """
    Build an ordered K36 vector that matches the row order of the
    K36→G25 matrix (excluding the INTERCEPT row).
    """
    matrix = get_k36_to_g25_matrix()
    components = [idx for idx in matrix.index if idx != "INTERCEPT"]

    # Normalise incoming keys once
    normalized_results = {normalize_k36_key(k): v for k, v in k36_results.items()}

    vector: List[float] = []
    for name in components:
        norm_name = normalize_k36_key(name)
        vector.append(normalized_results.get(norm_name, 0.0))

    return vector


def k36_to_g25(user_k36_data: List[float]) -> List[float]:
    """
    Core conversion, matching the provided reference implementation:

        result = np.dot(user_k36_data, weights) + intercept
    """
    matrix = get_k36_to_g25_matrix()

    weights = matrix.drop("INTERCEPT").values
    intercept = matrix.loc["INTERCEPT"].values

    result = np.dot(user_k36_data, weights) + intercept
    return result.tolist()


class K36Input(BaseModel):
    k36_results: Dict[str, float]
    sample_name: str = "Sample"

# Allow your Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your Vercel URL later for security
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Kvali Engine is running"}


@app.head("/")
def home_head():
    """Allow HEAD / for health checks (e.g. Render)."""
    return Response(status_code=200)


@app.get("/debug/memory")
def debug_memory():
    """Current process RSS in MB (Linux only; 0 elsewhere)."""
    return {"rss_mb": round(_get_rss_mb(), 2)}


def check_upload_size(request: Request) -> None:
    """Reject request body over MAX_UPLOAD_BYTES."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB per request.",
                )
        except ValueError:
            pass


async def write_upload_to_temp(
    file: UploadFile,
    temp_path: str,
    compressed: bool = False,
) -> float:
    """
    Stream uploaded file to temp_path. If compressed=True, stream-decompress gzip.
    Enforces MAX_DECOMPRESSED_BYTES on written size. Returns size written in MB.
    """
    size = 0
    try:
        if compressed:
            with open(temp_path, "wb") as out:
                d = zlib.decompressobj(zlib.MAX_WBITS + 32)  # gzip
                while True:
                    chunk = await file.read(256 * 1024)
                    if not chunk:
                        out.write(d.flush())
                        break
                    out.write(d.decompress(chunk))
                    size = out.tell()
                    if size > MAX_DECOMPRESSED_BYTES:
                        out.close()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File is too large to process on this server (max {MAX_DECOMPRESSED_BYTES // (1024*1024)} MB). Server has limited RAM; use a smaller export or upgrade the plan.",
                        )
                size = out.tell()
        else:
            with open(temp_path, "wb") as out:
                while True:
                    chunk = await file.read(512 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_DECOMPRESSED_BYTES:
                        out.close()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File is too large to process on this server (max {MAX_DECOMPRESSED_BYTES // (1024*1024)} MB). Server has limited RAM; use a smaller export or upgrade the plan.",
                        )
                    out.write(chunk)
        return size / (1024 * 1024)
    except HTTPException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


@app.post("/raw-to-k36")
async def process_dna(
    file: UploadFile = File(...),
    vendor: VENDOR_CHOICES = Form("23andme", description="Raw data format: 23andme, ancestry, ftdna, ftdna2, wegene, myheritage"),
    compressed: bool = Form(False, description="Set true if the uploaded file is gzip-compressed (.gz)"),
    _: None = Depends(check_upload_size),
):
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="No file selected. Choose a file to upload.")
    base_name = file.filename.rstrip(".gz") if file.filename.lower().endswith(".gz") else file.filename
    suffix = os.path.splitext(base_name)[1] or ".txt"
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="raw2k36_")
    os.close(fd)
    try:
        file_size_mb = await write_upload_to_temp(file, temp_path, compressed)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    try:
        async with _conversion_semaphore:
            rss_before = _get_rss_mb()
            logger.info(
                "Conversion started (raw-to-k36): rss_mb=%.2f file_size_mb=%.2f",
                rss_before,
                file_size_mb,
            )

            if not _builtin_raw_to_k36_available():
                raise HTTPException(
                    status_code=503,
                    detail="K36 data missing. Add data/K36.alleles and data/K36.36.F to the server.",
                )
            try:
                clean_results = await asyncio.wait_for(
                    asyncio.to_thread(_run_builtin_raw_to_k36, temp_path, vendor),
                    timeout=K36_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"K36 conversion timed out after {K36_TIMEOUT}s. Try a smaller file or set K36_CONVERSION_TIMEOUT.",
                )
            except Exception as e:
                logger.exception("K36 conversion failed: %s", e)
                raise HTTPException(
                    status_code=500,
                    detail="K36 conversion failed: " + str(e),
                )

            logger.info("K36 finished (raw-to-k36): rss_mb=%.2f", _get_rss_mb())
        return {"status": "success", "results": clean_results}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_k36_key(key: str) -> str:
    """Normalize K36 component names to match k36_to_g25_weights.csv index."""
    key = key.strip().replace("-", "_").replace(" ", "_")
    # Map display variants to matrix row names
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


@app.post("/k36-to-g25")
async def convert_k36_to_g25(data: K36Input):
    """
    Convert K36 admixture results to simulated G25 coordinates.
    Uses linear regression approximation based on reference population data.
    Note: These are SIMULATED coordinates, not official G25 from Davidski.
    """
    k36_results = data.k36_results

    # Validate that percentages sum to approximately 100
    total = sum(k36_results.values())
    if not (95 <= total <= 105):
        raise HTTPException(
            status_code=400,
            detail=f"K36 percentages should sum to ~100, got {total:.2f}"
        )

    # Build ordered K36 vector and run regression
    user_k36_vector = k36_vector_from_dict(k36_results)
    g25_coords = k36_to_g25(user_k36_vector)

    # Round to 6 decimal places for readability
    g25_coords = [round(c, 6) for c in g25_coords]

    # Format as Vahaduo-compatible string
    vahaduo_string = f"{data.sample_name}," + ",".join(str(c) for c in g25_coords)
    
    return {
        "status": "success",
        "sample_name": data.sample_name,
        "g25_coordinates": g25_coords,
        "vahaduo_format": vahaduo_string,
        "note": "These are SIMULATED G25 coordinates based on K36 regression. For official G25, use g25requests.app"
    }


@app.post("/raw-to-g25")
async def process_dna_to_g25(
    file: UploadFile = File(...),
    vendor: VENDOR_CHOICES = Form("23andme", description="Raw data format: 23andme, ancestry, ftdna, ftdna2, wegene, myheritage"),
    compressed: bool = Form(False, description="Set true if the uploaded file is gzip-compressed (.gz)"),
    _: None = Depends(check_upload_size),
):
    """
    Full pipeline: Raw DNA -> K36 -> Simulated G25 coordinates.
    Max request size 50 MB (raw or gzip). Set compressed=true for .gz uploads. Processing limit is MAX_DECOMPRESSED_MB (env, default 5).
    In Swagger: click "Try it out", choose a file, then Execute. Conversion can take 30–120 s for larger files.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="No file selected. Choose a file to upload.")
    base_name = file.filename.rstrip(".gz") if file.filename.lower().endswith(".gz") else file.filename
    suffix = os.path.splitext(base_name)[1] or ".txt"
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="raw2g25_")
    os.close(fd)
    try:
        file_size_mb = await write_upload_to_temp(file, temp_path, compressed)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    try:
        async with _conversion_semaphore:
            rss_before = _get_rss_mb()
            logger.info(
                "Conversion started (raw-to-g25): rss_mb=%.2f file_size_mb=%.2f",
                rss_before,
                file_size_mb,
            )
            if not _builtin_raw_to_k36_available():
                raise HTTPException(
                    status_code=503,
                    detail="K36 data missing. Add data/K36.alleles and data/K36.36.F to the server.",
                )
            try:
                k36_results = await asyncio.wait_for(
                    asyncio.to_thread(_run_builtin_raw_to_k36, temp_path, vendor),
                    timeout=K36_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"K36 conversion timed out after {K36_TIMEOUT}s. Try a smaller file or set K36_CONVERSION_TIMEOUT.",
                )
            except Exception as e:
                logger.exception("K36 conversion failed (raw-to-g25): %s", e)
                raise HTTPException(
                    status_code=500,
                    detail="K36 conversion failed: " + str(e),
                )

            logger.info("K36 finished (raw-to-g25): rss_mb=%.2f", _get_rss_mb())

        # K36 -> G25 regression
        user_k36_vector = k36_vector_from_dict(k36_results)
        g25_coords = k36_to_g25(user_k36_vector)

        g25_coords = [round(c, 6) for c in g25_coords]

        sample_name = file.filename.replace(".txt", "")
        vahaduo_string = f"{sample_name}," + ",".join(str(c) for c in g25_coords)

        return {
            "status": "success",
            "k36_results": k36_results,
            "g25_coordinates": g25_coords,
            "vahaduo_format": vahaduo_string,
            "note": "SIMULATED G25 from K36 regression. For official G25 use g25requests.app",
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
