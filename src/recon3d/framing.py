"""How big and where is the object in a training silhouette? And a few example images for the demo.

The web app crops each upload to its object and rescales it so the object fills the frame the way
training images do. Guessing that fill ratio would silently shift uploads away from the training
distribution, so it is measured here from the real validation silhouettes.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data import DatasetMeta


def bbox_stats(sil: np.ndarray, thr: int = 127) -> dict | None:
    """sil: uint8 [H, W]. Returns fill (longest bbox side / image side), centre offset and foreground share."""
    m = sil > thr
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    h, w = sil.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return dict(fill=max(y1 - y0, x1 - x0) / max(h, w),
                cx=(x0 + x1) / 2 / w - 0.5, cy=(y0 + y1) / 2 / h - 0.5,
                fg=float(m.mean()))


def framing_stats(root: str | Path, split: str = "val", max_meshes: int | None = None) -> dict:
    root = Path(root)
    meta = DatasetMeta.load(root)
    sil = np.load(root / f"sil_{split}.npy", mmap_mode="r")          # never loads the whole array
    index = pd.read_csv(root / f"index_{split}.csv", dtype={"mesh_id": str})
    n = len(index) if max_meshes is None else min(max_meshes, len(index))
    rows = []
    for m in range(n):
        block = np.asarray(sil[m])
        for v in range(block.shape[0]):
            s = bbox_stats(block[v])
            if s:
                rows.append(dict(category=index.category[m], elevation=meta.views[v][1], **s))
    df = pd.DataFrame(rows)

    def q(x):
        return {f"p{p}": round(float(np.quantile(x, p / 100)), 4) for p in (5, 25, 50, 75, 95)}

    return dict(
        split=split, images=len(df), image_size=int(sil.shape[-1]),
        fill=q(df.fill), center_x=q(df.cx), center_y=q(df.cy), foreground=q(df.fg),
        fill_by_category={c: round(float(g.fill.median()), 4) for c, g in df.groupby("category")},
        fill_by_elevation={int(e): round(float(g.fill.median()), 4) for e, g in df.groupby("elevation")},
        recommended_fill=round(float(df.fill.median()), 3),
    )


def export_examples(root: str | Path, out_dir: str | Path, split: str = "test", per_category: int = 2,
                    view: tuple[int, int] = (45, 30), prefix: str = "") -> list[str]:
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = DatasetMeta.load(root)
    v = meta.view_index(view)
    sil = np.load(root / f"sil_{split}.npy", mmap_mode="r")
    index = pd.read_csv(root / f"index_{split}.csv", dtype={"mesh_id": str})
    names = []
    for cat, g in index.groupby("category"):
        for i, r in enumerate(g.head(per_category).itertuples()):
            name = f"{prefix}{cat}_{i + 1}.png"
            Image.fromarray(np.asarray(sil[r.Index, v])).save(out / name)
            names.append(name)
    return names


def run(data: str | Path, out: str | Path, unseen: str | Path | None = None, per_category: int = 2,
        max_meshes: int | None = None) -> Path:
    out = Path(out)
    ex_dir = out / "examples"
    stats = framing_stats(data, "val", max_meshes)
    stats["examples_seen"] = export_examples(data, ex_dir, "test", per_category)
    if unseen:
        stats["examples_unseen"] = export_examples(unseen, ex_dir, "test", 1, prefix="unseen_")
    (out / "framing.json").write_text(json.dumps(stats, indent=1))
    zpath = out / "framing_bundle.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(out / "framing.json", "framing.json")
        for f in sorted(ex_dir.glob("*.png")):
            z.write(f, f"examples/{f.name}")
    return zpath
