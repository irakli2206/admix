"""
Thread-safe progress for long-running conversions (updated from worker threads, read from async SSE).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class ConversionProgress:
    """Monotonic-ish progress 0–100 and stage label for UI / progress bars."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.percent: float = 0.0
        self.stage: str = "init"
        self.done: bool = False
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None

    def set(self, percent: float, stage: str) -> None:
        with self._lock:
            self.percent = min(100.0, max(0.0, percent))
            self.stage = stage

    def bump(self, percent: float, stage: str) -> None:
        """Only increase percent (smoother bar when multiple sub-steps report)."""
        with self._lock:
            self.percent = min(100.0, max(self.percent, percent))
            self.stage = stage

    def complete(self, result: Dict[str, Any]) -> None:
        with self._lock:
            self.percent = 100.0
            self.stage = "complete"
            self.done = True
            self.result = result

    def fail(self, message: str) -> None:
        with self._lock:
            self.stage = "error"
            self.error = message
            self.done = True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {
                "progress": round(self.percent, 1),
                "stage": self.stage,
                "done": self.done,
            }
            if self.error is not None:
                out["error"] = self.error
            if self.done and self.result is not None:
                out["result"] = self.result
            return out
