"""FastAPI dependencies: auth, upload size checks."""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app import config

logger = logging.getLogger(__name__)

# Declared for OpenAPI / Swagger "Authorize" (sends X-Internal-Api-Key on Try it out).
_internal_api_key_header = APIKeyHeader(
    name=config.INTERNAL_API_KEY_HEADER,
    auto_error=False,
    description="Same secret as server env INTERNAL_API_KEY",
)


def check_upload_size(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > config.MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large. Max {config.MAX_UPLOAD_BYTES // (1024*1024)} MB per request.",
                )
        except ValueError:
            pass


def check_admixture_bundle_upload_size(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > config.ADMIXTURE_MAX_BUNDLE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Request too large for ADMIXTURE bundle. "
                        f"Max {config.ADMIXTURE_MAX_BUNDLE_MB} MB."
                    ),
                )
        except ValueError:
            pass


def require_internal_api_key(
    api_key: Optional[str] = Security(_internal_api_key_header),
) -> None:
    if not config.INTERNAL_API_KEY:
        logger.error("INTERNAL_API_KEY is not configured on the server")
        raise HTTPException(
            status_code=503,
            detail="Server auth is not configured.",
        )
    provided = api_key or ""
    if not provided or not secrets.compare_digest(provided, config.INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")
