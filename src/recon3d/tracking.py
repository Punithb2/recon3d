"""Experiment-tracking interface.

Training code only talks to this small interface. Today it prints to the console; Block C adds an
MLflow tracker with the same methods, and train.py does not change.
"""

from __future__ import annotations

from typing import Any, Protocol


class Tracker(Protocol):
    def start_run(self, run_name: str, params: dict[str, Any], tags: dict[str, str] | None = None) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def end_run(self, status: str = "FINISHED") -> None: ...


class NullTracker:
    """Does nothing. Default in tests."""

    def start_run(self, run_name, params, tags=None):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path):
        pass

    def end_run(self, status="FINISHED"):
        pass


class ConsoleTracker(NullTracker):
    def start_run(self, run_name, params, tags=None):
        print(f"[tracker] start {run_name} {tags or ''}")

    def log_metrics(self, metrics, step=None):
        parts = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[tracker] step={step} {parts}")

    def end_run(self, status="FINISHED"):
        print(f"[tracker] end ({status})")
