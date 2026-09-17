"""Dataset access and image preprocessing.

Dataset folder layout (produced by the Colab preprocessing notebook):
    meta.json                 categories, views, counts
    index_{split}.csv         idx, mesh_id, category, label, ...
    sil_{split}.npy           uint8  [N, V, H, W]   white object on black background
    points_{split}.npy        float32 [N, 8192, 3]  surface samples, bbox centre 0, diagonal 1
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SPLIT_NAMES = ("train", "val", "test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetMeta:
    root: Path
    categories: list[str]
    views: list[tuple[int, int]]          # (azimuth, elevation), index = view id
    counts: dict[str, int]

    @classmethod
    def load(cls, root: str | Path) -> DatasetMeta:
        root = Path(root)
        m = json.loads((root / "meta.json").read_text())
        return cls(root=root, categories=list(m["categories"]),
                   views=[(int(v["azimuth"]), int(v["elevation"])) for v in m["views"]],
                   counts={k: int(v) for k, v in m["counts"].items()})

    @property
    def run_tag(self) -> str:
        """n1000 = 1000 training meshes per category. Keeps run names unique across dataset versions."""
        return f"n{self.counts['train'] // len(self.categories)}"

    def view_index(self, az_el) -> int:
        return self.views.index(tuple(int(a) for a in az_el))


def find_dataset(search_root: str | Path = "/kaggle/input") -> Path:
    hits = [p.parent for p in Path(search_root).rglob("meta.json") if (p.parent / "sil_train.npy").exists()]
    if not hits:
        raise FileNotFoundError(f"no dataset (meta.json + sil_train.npy) under {search_root}")
    if len(hits) > 1:
        print(f"WARNING: {len(hits)} datasets found, using {hits[0]}")
    return hits[0]


class Split:
    """One split. Silhouettes stay in CPU RAM (uint8); points go to `device`."""

    def __init__(self, root: str | Path, name: str, device: torch.device | str = "cpu", in_ram: bool = True):
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        root, self.name, self.device = Path(root), name, torch.device(device)
        t0 = time.time()
        self.index = pd.read_csv(root / f"index_{name}.csv", dtype={"mesh_id": str})
        self.sil = np.load(root / f"sil_{name}.npy", mmap_mode=None if in_ram else "r")
        self.points = torch.from_numpy(np.load(root / f"points_{name}.npy")).to(self.device)
        self.labels = self.index["label"].to_numpy()
        self.N, self.V = self.sil.shape[:2]
        if len(self.index) != self.N or self.points.shape[0] != self.N:
            raise ValueError(f"{name}: index ({len(self.index)}), silhouettes ({self.N}) and points "
                             f"({self.points.shape[0]}) disagree on the number of meshes")
        print(f"{name}: {self.N} meshes x {self.V} views ({self.sil.nbytes / 1e9:.2f} GB) "
              f"loaded in {time.time() - t0:.0f}s")

    def images(self, mesh_idx, view_idx) -> torch.Tensor:
        """uint8 [B, H, W] on the split's device."""
        x = torch.from_numpy(np.ascontiguousarray(self.sil[np.asarray(mesh_idx), np.asarray(view_idx)]))
        return x.to(self.device, non_blocking=True)

    def gt(self, mesh_idx, n: int = 2048, random_subset: bool = False) -> torch.Tensor:
        """[B, n, 3]. Training: a fresh random subset of the cached points. Evaluation: the first n (fixed)."""
        p = self.points[torch.as_tensor(np.asarray(mesh_idx), device=self.device)]
        if not random_subset:
            return p[:, :n]
        sel = torch.rand(p.shape[0], p.shape[1], device=self.device).argsort(dim=1)[:, :n]
        return torch.gather(p, 1, sel[..., None].expand(-1, -1, 3))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random scale (0.9-1.1) + shift (+-8%). Safe because targets are centred and size-normalised:
    moving or scaling the silhouette in the frame does not change the 3D answer."""
    B = x.shape[0]
    s = torch.empty(B, device=x.device).uniform_(0.9, 1.1)
    t = torch.empty(B, 2, device=x.device).uniform_(-0.08, 0.08)
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1 / s
    theta[:, 1, 1] = 1 / s
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def preprocess(x_uint8: torch.Tensor, img_res: int = 224, aug: bool = False) -> torch.Tensor:
    """uint8 silhouettes [B, H, W] -> normalised float [B, 3, img_res, img_res].
    Training, evaluation and the web service all use this one function (no train/serve skew)."""
    if x_uint8.dtype != torch.uint8 or x_uint8.ndim != 3:
        raise ValueError(f"expected uint8 [B, H, W], got {x_uint8.dtype} {tuple(x_uint8.shape)}")
    x = x_uint8.float().div_(255).unsqueeze(1)
    if aug:
        x = augment(x)
    if x.shape[-1] != img_res or x.shape[-2] != img_res:
        x = F.interpolate(x, size=(img_res, img_res), mode="bilinear", align_corners=False)
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    return x
