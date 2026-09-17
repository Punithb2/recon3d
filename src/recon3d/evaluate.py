"""Evaluation: model vs nearest-neighbour retrieval baseline, per (mesh, view).

Output files match the notebook (eval_{split}_summary.json, eval_{split}_per_sample.csv.gz), so the
error-analysis and scaling notebooks keep working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import TrainConfig
from .data import Split, preprocess
from .metrics import chamfer, fscore
from .models import load_checkpoint
from .plots import plot_qualitative
from .train import Session, encode_split

_TRAIN_FEATS: dict[str, torch.Tensor] = {}


@dataclass
class TrainBank:
    """L2-normalised global features of every training image, for retrieval and similarity."""

    feats: torch.Tensor          # [N_train * V, D], unit length
    mesh: torch.Tensor           # [N_train * V] training mesh index of each row


def train_bank(model, cfg: TrainConfig, run_dir: Path, session: Session) -> TrainBank:
    key = str(run_dir)
    tr = session.split("train")
    if key not in _TRAIN_FEATS:                       # encoding 144k images takes a while: cache per run
        _TRAIN_FEATS[key] = F.normalize(encode_split(model, tr, cfg.img_res, session).flatten(0, 1).float(), dim=1)
    mesh = torch.arange(tr.N, device=session.device).repeat_interleave(tr.V)
    return TrainBank(_TRAIN_FEATS[key], mesh)


@dataclass
class SplitScores:
    long: pd.DataFrame        # one row per (method, mesh, view): cd_x1000, f@tau
    extra: pd.DataFrame       # one row per (mesh, view): retrieved training mesh, its cosine similarity
    keep: dict                # {mesh: (pred, retrieval)} point clouds at keep_view, for figures
    feats: torch.Tensor       # [N * V, D] unit-length image features (CPU, float16), row order = extra


@torch.no_grad()
def score_split(model, cfg: TrainConfig, sp: Split, tr: Split, bank: TrainBank, session: Session,
                keep_view: int | None = None, chunk: int = 512) -> SplitScores:
    """Predict every (mesh, view) of `sp` and score model + retrieval against the ground truth."""
    rows, extra, keep, feats = [], [], {}, []
    mesh_idx = np.repeat(np.arange(sp.N), sp.V)
    view_idx = np.tile(np.arange(sp.V), sp.N)
    for s in range(0, len(mesh_idx), chunk):
        mi, vi = mesh_idx[s:s + chunk], view_idx[s:s + chunk]
        gt = sp.gt(mi, cfg.n_gt)
        with session.autocast():
            x = preprocess(sp.images(mi, vi), cfg.img_res)
            f = model.embed(x)
            pred = model.decode(f) if hasattr(model, "decode") else model(x)
        fn = F.normalize(f.float(), dim=1)
        feats.append(fn.half().cpu())
        sim, j = (fn @ bank.feats.T).max(1)                                  # closest training image
        nn_train = bank.mesh[j]
        retr = tr.gt(nn_train.cpu().numpy(), cfg.n_gt)
        for method, P in (("model", pred), ("retrieval", retr)):
            cd, d_pg, d_gp = chamfer(P, gt)
            out = {"cd_x1000": (1000 * cd).cpu().numpy()}
            for t in cfg.fscore_taus:
                out[f"f@{t}"] = fscore(d_pg, d_gp, t).cpu().numpy()
            for k in range(len(mi)):
                rows.append(dict(method=method, mesh=int(mi[k]), view=int(vi[k]),
                                 **{name: float(v[k]) for name, v in out.items()}))
        extra.append(pd.DataFrame(dict(mesh=mi, view=vi, nn_train_mesh=nn_train.cpu().numpy(),
                                       max_sim=sim.cpu().numpy())))
        if keep_view is not None:
            for k, m in enumerate(mi):
                if vi[k] == keep_view:
                    keep[int(m)] = (pred[k].float().cpu().numpy(), retr[k].cpu().numpy())
    return SplitScores(pd.DataFrame(rows), pd.concat(extra, ignore_index=True), keep, torch.cat(feats))


@torch.no_grad()
def evaluate(run_dir: str | Path, split_name: str, session: Session, n_vis: int = 2) -> dict:
    run_dir = Path(run_dir)
    model, cfg, ck = load_checkpoint(run_dir / "best.pt", session.device)
    sp, tr = session.split(split_name), session.split("train")
    views = session.meta.views
    vis_view = session.meta.view_index(cfg.val_views[0])

    bank = train_bank(model, cfg, run_dir, session)
    res = score_split(model, cfg, sp, tr, bank, session, keep_view=vis_view)
    df, keep = res.long, res.keep
    df["category"] = sp.index["category"].to_numpy()[df.mesh]
    df["elevation"] = [views[v][1] for v in df.view]
    metrics = ["cd_x1000"] + [f"f@{t}" for t in cfg.fscore_taus]
    overall = df.groupby("method")[metrics].mean()
    per_cat = df.pivot_table(index="category", columns="method", values=metrics, aggfunc="mean")
    df.to_csv(run_dir / f"eval_{split_name}_per_sample.csv.gz", index=False)
    summary = dict(run=run_dir.name, split=split_name, best_epoch=int(ck["epoch"]),
                   n_predictions=int(len(df) // 2),
                   overall=overall.to_dict(orient="index"),
                   per_category={c: per_cat.loc[c].unstack().to_dict() for c in per_cat.index})
    (run_dir / f"eval_{split_name}_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\n=== {run_dir.name} | {split_name} | {len(df) // 2} predictions | best epoch {ck['epoch'] + 1} ===")
    print(overall.round(4).to_string())

    picks = sp.index.groupby("category").head(n_vis).index.to_numpy()
    plot_qualitative([(sp.index.mesh_id[m], sp.sil[m, vis_view], sp.points[m, :cfg.n_gt].cpu().numpy(), *keep[int(m)])
                      for m in picks if int(m) in keep], run_dir / f"qualitative_{split_name}.png")
    return summary


def results_table(runs_dir: str | Path) -> pd.DataFrame:
    """One row per (run, split, method) from every eval summary under runs_dir."""
    rows = []
    for p in sorted(Path(runs_dir).glob("*/eval_*_summary.json")):
        s = json.loads(p.read_text())
        for method, m in s["overall"].items():
            rows.append(dict(run=s["run"], split=s["split"], method=method, best_epoch=s["best_epoch"] + 1, **m))
    df = pd.DataFrame(rows)
    return df.sort_values(["split", "run", "method"]).reset_index(drop=True) if len(df) else df
