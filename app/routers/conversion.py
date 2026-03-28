import asyncio
import json
import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from progress_tracker import ConversionProgress

from app import config
from app.deps import check_upload_size, require_internal_api_key
from app import k36_g25
from app.schemas import K36Input, VENDOR_CHOICES
from app.upload import write_upload_to_temp
from app.memory_util import rss_mb

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversion"])


@router.post("/raw-to-k36")
async def process_dna(
    file: UploadFile = File(...),
    vendor: VENDOR_CHOICES = Form(
        "23andme",
        description="Raw data format: 23andme, ancestry, ftdna, ftdna2, wegene, myheritage",
    ),
    compressed: bool = Form(
        False, description="Set true if the uploaded file is gzip-compressed (.gz)"
    ),
    _auth: None = Depends(require_internal_api_key),
    _: None = Depends(check_upload_size),
):
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="No file selected. Choose a file to upload.")
    base_name = (
        file.filename.rstrip(".gz")
        if file.filename.lower().endswith(".gz")
        else file.filename
    )
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
        async with config.conversion_semaphore:
            logger.info(
                "Conversion started (raw-to-k36): rss_mb=%.2f file_size_mb=%.2f",
                rss_mb(),
                file_size_mb,
            )
            if not k36_g25.builtin_raw_to_k36_available():
                raise HTTPException(
                    status_code=503,
                    detail="K36 data missing. Add data/K36.alleles and data/K36.36.F to the server.",
                )
            try:
                clean_results = await asyncio.wait_for(
                    asyncio.to_thread(
                        k36_g25.run_builtin_raw_to_k36, temp_path, vendor
                    ),
                    timeout=config.K36_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"K36 conversion timed out after {config.K36_TIMEOUT}s. Try a smaller file or set K36_CONVERSION_TIMEOUT.",
                )
            except Exception as e:
                logger.exception("K36 conversion failed: %s", e)
                raise HTTPException(
                    status_code=500,
                    detail="K36 conversion failed: " + str(e),
                )
            logger.info("K36 finished (raw-to-k36): rss_mb=%.2f", rss_mb())
        return {"status": "success", "results": clean_results}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/k36-to-g25")
async def convert_k36_to_g25(
    data: K36Input,
    _auth: None = Depends(require_internal_api_key),
):
    """
    Convert K36 admixture results to simulated G25 coordinates.
    Note: These are SIMULATED coordinates, not official G25 from Davidski.
    """
    k36_results = data.k36_results
    total = sum(k36_results.values())
    if not (95 <= total <= 105):
        raise HTTPException(
            status_code=400,
            detail=f"K36 percentages should sum to ~100, got {total:.2f}",
        )
    user_k36_vector = k36_g25.k36_vector_from_dict(k36_results)
    g25_coords = k36_g25.validated_g25_coords(k36_g25.k36_to_g25(user_k36_vector))
    g25_coords_csv = ",".join(str(c) for c in g25_coords)
    vahaduo_string = f"{data.sample_name},{g25_coords_csv}"
    return {
        "status": "success",
        "sample_name": data.sample_name,
        "g25_coordinates": g25_coords,
        "g25_coords_csv": g25_coords_csv,
        "vahaduo_format": vahaduo_string,
        "note": "These are SIMULATED G25 coordinates based on K36 regression. `g25_coords_csv` is coords-only; `vahaduo_format` is sample label + coords.",
    }


@router.post("/raw-to-g25")
async def process_dna_to_g25(
    file: UploadFile = File(...),
    vendor: VENDOR_CHOICES = Form(
        "23andme",
        description="Raw data format: 23andme, ancestry, ftdna, ftdna2, wegene, myheritage",
    ),
    compressed: bool = Form(
        False, description="Set true if the uploaded file is gzip-compressed (.gz)"
    ),
    _auth: None = Depends(require_internal_api_key),
    _: None = Depends(check_upload_size),
):
    """
    Full pipeline: Raw DNA -> K36 -> Simulated G25 coordinates.
    Max request size 50 MB (raw or gzip). Conversion can take 30–120 s for larger files.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="No file selected. Choose a file to upload.")
    base_name = (
        file.filename.rstrip(".gz")
        if file.filename.lower().endswith(".gz")
        else file.filename
    )
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
        async with config.conversion_semaphore:
            logger.info(
                "Conversion started (raw-to-g25): rss_mb=%.2f file_size_mb=%.2f",
                rss_mb(),
                file_size_mb,
            )
            if not k36_g25.builtin_raw_to_k36_available():
                raise HTTPException(
                    status_code=503,
                    detail="K36 data missing. Add data/K36.alleles and data/K36.36.F to the server.",
                )
            try:
                k36_results = await asyncio.wait_for(
                    asyncio.to_thread(
                        k36_g25.run_builtin_raw_to_k36, temp_path, vendor
                    ),
                    timeout=config.K36_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"K36 conversion timed out after {config.K36_TIMEOUT}s. Try a smaller file or set K36_CONVERSION_TIMEOUT.",
                )
            except Exception as e:
                logger.exception("K36 conversion failed (raw-to-g25): %s", e)
                raise HTTPException(
                    status_code=500,
                    detail="K36 conversion failed: " + str(e),
                )
            logger.info("K36 finished (raw-to-g25): rss_mb=%.2f", rss_mb())
        sample_name = file.filename.replace(".txt", "")
        return k36_g25.g25_response_dict(k36_results, sample_name)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/raw-to-g25/stream")
async def raw_to_g25_stream(
    file: UploadFile = File(...),
    vendor: VENDOR_CHOICES = Form(
        "23andme",
        description="Raw data format: 23andme, ancestry, ftdna, ftdna2, wegene, myheritage",
    ),
    compressed: bool = Form(
        False, description="Set true if the uploaded file is gzip-compressed (.gz)"
    ),
    _auth: None = Depends(require_internal_api_key),
    _: None = Depends(check_upload_size),
):
    """
    Same pipeline as `/raw-to-g25`, but the response is Server-Sent Events (SSE).
    Use fetch() + ReadableStream; EventSource cannot POST multipart.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=400, detail="No file selected. Choose a file to upload."
        )
    base_name = (
        file.filename.rstrip(".gz")
        if file.filename.lower().endswith(".gz")
        else file.filename
    )
    suffix = os.path.splitext(base_name)[1] or ".txt"
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="raw2g25s_")
    os.close(fd)
    try:
        file_size_mb = await write_upload_to_temp(file, temp_path, compressed)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    sample_name = file.filename.replace(".txt", "")

    async def sse_gen():
        try:
            async with config.conversion_semaphore:
                logger.info(
                    "raw-to-g25/stream: rss_mb=%.2f file_size_mb=%.2f",
                    rss_mb(),
                    file_size_mb,
                )
                progress = ConversionProgress()
                loop = asyncio.get_running_loop()
                fut = loop.run_in_executor(
                    None,
                    lambda: k36_g25.sync_raw_to_g25_with_progress(
                        temp_path, vendor, sample_name, progress
                    ),
                )
                deadline = loop.time() + config.K36_TIMEOUT + 90
                while True:
                    await asyncio.sleep(0.15)
                    snap = progress.snapshot()
                    yield f"data: {json.dumps(snap)}\n\n"
                    if snap.get("done"):
                        break
                    if loop.time() > deadline:
                        yield f"data: {json.dumps({'done': True, 'error': 'stream_timeout', 'progress': snap.get('progress'), 'stage': snap.get('stage')})}\n\n"
                        break
                try:
                    await asyncio.wait_for(fut, timeout=120)
                except asyncio.TimeoutError:
                    logger.warning("raw-to-g25/stream executor wait timed out")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
