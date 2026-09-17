"""All runtime settings come from environment variables (12-factor style), with safe defaults.
On Hugging Face Spaces they are set under Settings -> Variables and secrets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path


def _env(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    model_repo: str = "Punith25/recon3d-resnet18"   # RECON3D_MODEL_REPO
    model_revision: str = "v1.0.1"                  # RECON3D_MODEL_REVISION: always a pinned tag
    model_dir: str = ""                             # RECON3D_MODEL_DIR: local bundle instead of the Hub (dev/tests)
    torch_threads: int = 4                          # RECON3D_TORCH_THREADS: os.cpu_count() lies on shared hosts
    max_concurrent: int = 2                         # RECON3D_MAX_CONCURRENT: inferences at once
    max_upload_mb: float = 5.0                      # RECON3D_MAX_UPLOAD_MB
    max_pixels: int = 4096 * 4096                   # RECON3D_MAX_PIXELS: decompression-bomb guard
    frame_fill: float = 0.6                         # RECON3D_FRAME_FILL: object size in frame (see framing.json)
    cache_size: int = 256                           # RECON3D_CACHE_SIZE: results kept in memory (0 = off)
    rate_limit_per_min: int = 30                    # RECON3D_RATE_LIMIT_PER_MIN per client (0 = off)
    event_log_dir: str = "event_logs"               # RECON3D_EVENT_LOG_DIR: request metadata as JSON lines ("" = off)
    event_log_dataset: str = ""                     # RECON3D_EVENT_LOG_DATASET: HF dataset to sync logs to
    event_log_every_min: int = 10                   # RECON3D_EVENT_LOG_EVERY_MIN
    hf_token: str = ""                              # HF_TOKEN (secret): only needed for private repos / log upload

    @classmethod
    def from_env(cls, framing_file: str | Path | None = None, **overrides) -> Settings:
        """framing_file: framing.json from `recon3d framing`; its measured fill is used unless
        RECON3D_FRAME_FILL is set explicitly."""
        values = {}
        for f in fields(cls):
            env_name = "HF_TOKEN" if f.name == "hf_token" else f"RECON3D_{f.name.upper()}"
            values[f.name] = _env(env_name, f.default, type(f.default))
        if framing_file and Path(framing_file).exists() and "RECON3D_FRAME_FILL" not in os.environ:
            values["frame_fill"] = float(json.loads(Path(framing_file).read_text())["recommended_fill"])
        values.update(overrides)
        return cls(**values)
