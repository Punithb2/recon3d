"""Shared fixtures: a tiny synthetic dataset in the exact on-disk format of the real one."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from recon3d.config import TrainConfig

CATS = ["airplane", "chair"]
UNSEEN_CATS = ["lamp", "sofa"]
KIND = {"airplane": 0, "chair": 1, "lamp": 2, "sofa": 3}      # which synthetic shape each category gets
VIEWS = [(az, el) for el in (-30, 0, 30) for az in range(0, 360, 45)]
SIL_RES, N_POINTS = 32, 512
COUNTS = {"train": 6, "val": 2, "test": 2}          # per category


def _shape(rng, kind):
    """Box (0), sphere (1), cylinder (2) or flat plate (3) surface with random extents."""
    if kind == 2:
        t = rng.uniform(0, 2 * np.pi, N_POINTS)
        p = np.stack([np.cos(t), rng.uniform(-1, 1, N_POINTS), np.sin(t)], 1)
    elif kind == 3:
        p = rng.uniform(-1, 1, (N_POINTS, 3))
        p[:, 1] *= 0.05
    elif kind == 0:
        p = rng.uniform(-1, 1, (N_POINTS, 3))
        ax = rng.integers(0, 3, N_POINTS)
        p[np.arange(N_POINTS), ax] = np.sign(p[np.arange(N_POINTS), ax])
    else:
        p = rng.normal(size=(N_POINTS, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
    ext = rng.uniform(0.15, 0.45, 3)
    return (p * ext).astype(np.float32), ext


def _silhouettes(rng, ext, kind):
    """24 views of a filled ellipse/rectangle whose size follows the shape's extents."""
    yy, xx = np.mgrid[-1:1:SIL_RES * 1j, -1:1:SIL_RES * 1j]
    out = np.zeros((len(VIEWS), SIL_RES, SIL_RES), np.uint8)
    for v, (az, el) in enumerate(VIEWS):
        a = np.deg2rad(az)
        w = abs(np.cos(a)) * ext[0] + abs(np.sin(a)) * ext[2]
        h = ext[1] + 0.2 * abs(np.sin(np.deg2rad(el))) * ext[0]
        if kind == 0:
            m = (np.abs(xx) < 2 * w) & (np.abs(yy) < 2 * h)
        elif kind == 1:
            m = (xx / (2 * w)) ** 2 + (yy / (2 * h)) ** 2 < 1
        elif kind == 2:                                   # cross
            m = ((np.abs(xx) < 0.3 * w) & (np.abs(yy) < 2 * h)) | ((np.abs(yy) < 0.3 * h) & (np.abs(xx) < 2 * w))
        else:                                             # thin bar
            m = (np.abs(xx) < 2 * w) & (np.abs(yy) < 0.15)
        out[v] = m * 255
    return out


def make_dataset(root, seed=0, cats=CATS, counts=COUNTS, extra_meta=None):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for split, n in counts.items():
        rows, sils, pts = [], [], []
        for label, cat in enumerate(cats):
            for i in range(n):
                p, ext = _shape(rng, KIND[cat])
                pts.append(p)
                sils.append(_silhouettes(rng, ext, KIND[cat]))
                rows.append(dict(idx=len(rows), label=label, category=cat, mesh_id=f"{cat}_{split}_{i:03d}"))
        pd.DataFrame(rows).to_csv(root / f"index_{split}.csv", index=False)
        np.save(root / f"sil_{split}.npy", np.stack(sils))
        np.save(root / f"points_{split}.npy", np.stack(pts))
    meta = dict(categories=list(cats),
                counts={s: n * len(cats) for s, n in counts.items()},
                views=[dict(index=i, azimuth=a, elevation=e) for i, (a, e) in enumerate(VIEWS)],
                **(extra_meta or {}))
    (root / "meta.json").write_text(json.dumps(meta))
    return root


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def unseen_dataset(tmp_path_factory):
    """Same format as the real unseen set: only test + val, categories the model never trained on."""
    return make_dataset(tmp_path_factory.mktemp("unseen"), seed=1, cats=UNSEEN_CATS,
                        counts={"test": 3, "val": 2}, extra_meta=dict(role="unseen_categories", seen_categories=CATS))


@pytest.fixture
def tiny_cfg():
    """Small and fast: random-init encoder (no download), 64 px input, 128 points."""
    return TrainConfig(name="frozen", pretrained=False, img_res=64, n_pred=128, n_gt=128, epochs=2,
                       batch_size=16, lr=1e-3, views_per_mesh=4, warmup_epochs=1, amp=False)


@pytest.fixture
def session(dataset, tmp_path):
    from recon3d.train import Session
    torch.manual_seed(0)
    return Session(data_root=dataset, runs_dir=tmp_path / "runs", device="cpu")
