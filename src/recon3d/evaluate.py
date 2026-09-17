"""Evaluation: model vs nearest-neighbour retrieval baseline, per (mesh, view).

Output files match the notebook (eval_{split}_summary.json, eval_{split}_per_sample.csv.gz), so the
error-analysis and scaling notebooks keep working.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import preprocess
from .metrics import chamfer, fscore
from .models import load_checkpoint
from .plots import plot_qualitative
from .train import Session, encode_split

_TRAIN_FEATS: dict[str, torch.Tensor] = {}


@torch.no_grad()
def evaluate(run_dir: str | Path, split_name: str, session: Session, n_vis: int = 2) -> dict:
    run_dir = Path(run_dir)
    model, cfg, ck = load_checkpoint(run_dir / "best.pt", session.device)
    sp, tr = session.split(split_name), session.split("train")
    views = session.meta.views
    vis_view = session.meta.view_index(cfg.val_views[0])

    # 1) global features of every (mesh, view); train features feed the retrieval baseline
    f_eval = encode_split(model, sp, cfg.img_res, session).flatten(0, 1).float()
    key = str(run_dir)
    if key not in _TRAIN_FEATS:
        _TRAIN_FEATS[key] = encode_split(model, tr, cfg.img_res, session).flatten(0, 1).float()
    f_train = _TRAIN_FEATS[key]
    mesh_of_train = torch.arange(tr.N, device=session.device).repeat_interleave(tr.V)
    fe_n, ft_n = F.normalize(f_eval, dim=1), F.normalize(f_train, dim=1)

    rows, keep = [], {}
    mesh_idx = np.repeat(np.arange(sp.N), sp.V)
    view_idx = np.tile(np.arange(sp.V), sp.N)
    for s in range(0, len(f_eval), 512):
        sl = slice(s, s + 512)
        mi, vi = mesh_idx[sl], view_idx[sl]
        gt = sp.gt(mi, cfg.n_gt)
        with session.autocast():
            pred = model(preprocess(sp.images(mi, vi), cfg.img_res))
        nn_train = mesh_of_train[(fe_n[sl] @ ft_n.T).argmax(1)]          # closest training image
        retr = tr.gt(nn_train.cpu().numpy(), cfg.n_gt)
        for method, P in (("model", pred), ("retrieval", retr)):
            cd, d_pg, d_gp = chamfer(P, gt)
            out = {"cd_x1000": (1000 * cd).cpu().numpy()}
            for t in cfg.fscore_taus:
                out[f"f@{t}"] = fscore(d_pg, d_gp, t).cpu().numpy()
            for j in range(len(mi)):
                rows.append(dict(method=method, mesh=int(mi[j]), view=int(vi[j]),
                                 **{k: float(v[j]) for k, v in out.items()}))
        for j, m in enumerate(mi):
            if vi[j] == vis_view:
                keep[int(m)] = (pred[j].float().cpu().numpy(), retr[j].cpu().numpy())

    df = pd.DataFrame(rows)
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
