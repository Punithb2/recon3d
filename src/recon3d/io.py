"""Small file helpers: atomic checkpoint writes and point-cloud export."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def save_ckpt(path: str | Path, state: dict) -> None:
    """Write to a temp file, then rename. A crash mid-write never leaves a half-written checkpoint."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_ply(points: np.ndarray, path: str | Path | None = None) -> bytes:
    """ASCII PLY (opens in MeshLab, Blender, and most 3D viewers). Returns the bytes; writes them if path is given."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected [N, 3] points, got {pts.shape}")
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    body = "\n".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in pts)
    data = (header + body + "\n").encode()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def read_ply(data: bytes | str | Path) -> np.ndarray:
    """Reads the ASCII PLY written by write_ply."""
    if not isinstance(data, bytes):
        data = Path(data).read_bytes()
    text = data.decode()
    header, _, body = text.partition("end_header\n")
    n = int(next(line.split()[-1] for line in header.splitlines() if line.startswith("element vertex")))
    pts = np.array([[float(v) for v in line.split()] for line in body.strip().splitlines()], dtype=np.float32)
    if pts.shape != (n, 3):
        raise ValueError(f"PLY header says {n} vertices, found {pts.shape}")
    return pts
