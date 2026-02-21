from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional
import asyncio
import logging
import os
import subprocess
import tempfile
import zlib

try:
    import resource
except ImportError:
    resource = None  # Windows

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


def _get_memory_debug() -> Optional[Dict[str, float]]:
    """
    After an admix subprocess has just exited, return its max RSS + current process RSS
    so we can estimate peak RAM used. Linux only (resource module). Use when testing locally.
    """
    if not resource or not hasattr(resource, "RUSAGE_CHILDREN"):
        return None
    try:
        # ru_maxrss is in KB on Linux
        child_max_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        child_max_mb = child_max_kb / 1024.0
        python_mb = _get_rss_mb()
        return {
            "admix_child_max_mb": round(child_max_mb, 2),
            "python_rss_mb": round(python_mb, 2),
            "estimated_peak_mb": round(child_max_mb + python_mb, 2),
        }
    except (OSError, ValueError, AttributeError):
        return None

app = FastAPI()

# Vendor values supported by admix -v (see README Raw Data Format)
VENDOR_CHOICES = Literal["23andme", "ancestry", "ftdna", "ftdna2", "wegene", "myheritage"]

# Max request body size. We stream to disk.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
# Max file size we pass to admix. Default 50 MB so local/Swagger works; on Render set env MAX_DECOMPRESSED_MB=10 to avoid OOM.
_max_decompressed_mb = int(os.environ.get("MAX_DECOMPRESSED_MB", "50"))
MAX_DECOMPRESSED_BYTES = _max_decompressed_mb * 1024 * 1024

# Timeout for admix subprocess (seconds). Prevents hanging if admix is slow or stuck.
ADMIX_TIMEOUT = 120

# Max concurrent raw-to-K36 / raw-to-G25 conversions (each runs admix subprocess + memory).
# On 512 MB RAM use 1 so we don't run two admix processes at once.
MAX_CONCURRENT_CONVERSIONS = 1
_conversion_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONVERSIONS)


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
    """
    Return current process RSS (MB). Use to confirm deployment and baseline memory.
    Total RAM used = this process + admix subprocess during conversion.
    """
    rss = _get_rss_mb()
    return {
        "rss_mb": round(rss, 2),
        "note": "RSS is this process only. During conversion, admix subprocess adds more; check Render logs for 'Conversion started/finished' to see RSS growth.",
    }


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
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["admix", "-f", temp_path, "-v", vendor, "-m", "K36"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=ADMIX_TIMEOUT,
                )
            except FileNotFoundError:
                raise HTTPException(
                    status_code=503,
                    detail="admix CLI not found. Install with: pip install admix",
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=504,
                    detail=f"Conversion timed out after {ADMIX_TIMEOUT}s. Try a smaller file.",
                )
            rss_after = _get_rss_mb()
            logger.info("Admix finished (raw-to-k36): rss_mb=%.2f", rss_after)

        # Parse output into JSON
        clean_results = {}
        for line in result.stdout.split("\n"):
            if ":" in line and "%" in line:
                name, value = line.split(":")
                clean_results[name.strip()] = float(value.replace("%", "").strip())

        out: Dict = {"status": "success", "results": clean_results}
        mem = _get_memory_debug()
        if mem:
            out["memory_debug"] = mem
            logger.info(
                "Memory (raw-to-k36): admix_child_max_mb=%.2f estimated_peak_mb=%.2f",
                mem["admix_child_max_mb"],
                mem["estimated_peak_mb"],
            )
        return out

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": e.stderr or e.stdout or str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_k36_key(key: str) -> str:
    """Normalize K36 component names to match our matrix keys."""
    key = key.strip().replace("-", "_").replace(" ", "_")
    # Handle common variations
    replacements = {
        "Indo_Chinese": "Indo_Chinese",
        "IndoChinese": "Indo_Chinese",
        "Central African": "Central_African",
        "East African": "East_African",
        "West African": "West_African",
        "North African": "North_African",
        "Northeast African": "Northeast_African",
        "South Asian": "South_Asian",
        "East Asian": "East_Asian",
        "Central Euro": "Central_Euro",
        "Eastern Euro": "Eastern_Euro",
        "East Central Euro": "East_Central_Euro",
        "East Balkan": "East_Balkan",
        "East Med": "East_Med",
        "West Med": "West_Med",
        "North Sea": "North_Sea",
        "Near Eastern": "Near_Eastern",
        "North Atlantic": "North_Atlantic",
        "North Caucasian": "North_Caucasian",
        "West Caucasian": "West_Caucasian",
        "East Central Asian": "East_Central_Asian",
        "South Central Asian": "South_Central_Asian",
        "South Chinese": "South_Chinese",
        "Volga Ural": "Volga_Ural",
    }
    for old, new in replacements.items():
        if key.lower().replace("_", " ") == old.lower().replace("_", " "):
            return new
    return key


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
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["admix", "-f", temp_path, "-v", vendor, "-m", "K36"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=ADMIX_TIMEOUT,
                )
            except FileNotFoundError:
                raise HTTPException(
                    status_code=503,
                    detail="admix CLI not found. Install with: pip install admix",
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=504,
                    detail=f"Conversion timed out after {ADMIX_TIMEOUT}s. Try a smaller file.",
                )
            rss_after = _get_rss_mb()
            logger.info("Admix finished (raw-to-g25): rss_mb=%.2f", rss_after)

        mem = _get_memory_debug()
        if mem:
            logger.info(
                "Memory (raw-to-g25): admix_child_max_mb=%.2f estimated_peak_mb=%.2f",
                mem["admix_child_max_mb"],
                mem["estimated_peak_mb"],
            )

        # Parse K36 output
        k36_results = {}
        for line in result.stdout.split("\n"):
            if ":" in line and "%" in line:
                name, value = line.split(":")
                k36_results[name.strip()] = float(value.replace("%", "").strip())

        # Step 2: Convert K36 to G25 using the regression matrix
        user_k36_vector = k36_vector_from_dict(k36_results)
        g25_coords = k36_to_g25(user_k36_vector)

        g25_coords = [round(c, 6) for c in g25_coords]

        sample_name = file.filename.replace(".txt", "")
        vahaduo_string = f"{sample_name}," + ",".join(str(c) for c in g25_coords)

        out = {
            "status": "success",
            "k36_results": k36_results,
            "g25_coordinates": g25_coords,
            "vahaduo_format": vahaduo_string,
            "note": "These are SIMULATED G25 coordinates based on K36 regression. For official G25, use g25requests.app",
        }
        if mem:
            out["memory_debug"] = mem
        return out

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": e.stderr or e.stdout or str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
