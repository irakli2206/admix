"""Stream multipart uploads to disk with size limits."""

from __future__ import annotations

import os
import zlib

from fastapi import HTTPException, UploadFile

from app import config


async def write_upload_to_temp(
    file: UploadFile,
    temp_path: str,
    compressed: bool = False,
) -> float:
    size = 0
    try:
        if compressed:
            with open(temp_path, "wb") as out:
                d = zlib.decompressobj(zlib.MAX_WBITS + 32)
                while True:
                    chunk = await file.read(256 * 1024)
                    if not chunk:
                        out.write(d.flush())
                        break
                    out.write(d.decompress(chunk))
                    size = out.tell()
                    if size > config.MAX_DECOMPRESSED_BYTES:
                        out.close()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File is too large to process on this server (max {config.MAX_DECOMPRESSED_BYTES // (1024*1024)} MB). Server has limited RAM; use a smaller export or upgrade the plan.",
                        )
                size = out.tell()
        else:
            with open(temp_path, "wb") as out:
                while True:
                    chunk = await file.read(512 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > config.MAX_DECOMPRESSED_BYTES:
                        out.close()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File is too large to process on this server (max {config.MAX_DECOMPRESSED_BYTES // (1024*1024)} MB). Server has limited RAM; use a smaller export or upgrade the plan.",
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
