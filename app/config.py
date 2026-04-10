"""Environment-backed settings and shared async primitives."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "").strip()
INTERNAL_API_KEY_HEADER = "X-Internal-Api-Key"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_max_decompressed_mb = int(os.environ.get("MAX_DECOMPRESSED_MB", "50"))
MAX_DECOMPRESSED_BYTES = _max_decompressed_mb * 1024 * 1024

MAX_CONCURRENT_CONVERSIONS = int(os.environ.get("MAX_CONCURRENT_CONVERSIONS", "5"))
conversion_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONVERSIONS)

K36_TIMEOUT = int(os.environ.get("K36_CONVERSION_TIMEOUT", "120"))

ADMIXTURE_ENABLED = os.environ.get("ADMIXTURE_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
ADMIXTURE_MAX_BUNDLE_MB = int(os.environ.get("ADMIXTURE_MAX_BUNDLE_MB", "6144"))
ADMIXTURE_MAX_BUNDLE_BYTES = ADMIXTURE_MAX_BUNDLE_MB * 1024 * 1024

CORS_ORIGINS = [
    "https://kvali.app",
    "https://www.kvali.app",
    "https://staging.kvali.app",
    "http://localhost:3000",
]
