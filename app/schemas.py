"""Pydantic models and typing aliases for API routes."""

from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel

VENDOR_CHOICES = Literal[
    "23andme", "ancestry", "ftdna", "ftdna2", "wegene", "myheritage"
]


class K36Input(BaseModel):
    k36_results: Dict[str, float]
    sample_name: str = "Sample"
