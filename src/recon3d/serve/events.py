"""Request event log: the monitoring data source.

One JSON line per request with metadata only (never the uploaded image): time, latency, input statistics,
OOD score, outcome, model version. Enough to answer "is the input distribution drifting?", "how often do
users upload unfamiliar objects?" and "is latency OK?", without storing anything personal.

On the Space the folder is synced to a private Hugging Face dataset every few minutes, because the
Space's own disk is wiped on every restart.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Protocol

from .settings import Settings


class EventLog(Protocol):
    def write(self, event: dict) -> None: ...
    def close(self) -> None: ...


class NullEventLog:
    def write(self, event: dict) -> None:
        pass

    def close(self) -> None:
        pass


class JsonlEventLog:
    """Appends to <dir>/events-<date>-<process id>.jsonl. One file per process avoids write conflicts."""

    def __init__(self, folder: str | Path, lock: threading.Lock | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = lock or threading.Lock()
        self.proc = uuid.uuid4().hex[:8]
        self.scheduler = None

    def path(self) -> Path:
        return self.folder / f"events-{time.strftime('%Y-%m-%d', time.gmtime())}-{self.proc}.jsonl"

    def write(self, event: dict) -> None:
        line = json.dumps(event, separators=(",", ":"), default=str)
        with self.lock, open(self.path(), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def close(self) -> None:
        if self.scheduler is not None:
            with contextlib.suppress(Exception):
                self.scheduler.trigger().result(timeout=30)     # push the last events before shutdown
            self.scheduler.stop()


def make_event_log(settings: Settings) -> EventLog:
    if not settings.event_log_dir:
        return NullEventLog()
    if settings.event_log_dataset and settings.hf_token:
        from huggingface_hub import CommitScheduler

        folder = Path(settings.event_log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        scheduler = CommitScheduler(repo_id=settings.event_log_dataset, repo_type="dataset", folder_path=folder,
                                    path_in_repo="events", every=settings.event_log_every_min, private=True,
                                    token=settings.hf_token, allow_patterns=["*.jsonl"])
        log = JsonlEventLog(folder, lock=scheduler.lock)       # never upload a half-written line
        log.scheduler = scheduler
        return log
    return JsonlEventLog(settings.event_log_dir)
