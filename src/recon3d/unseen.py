"""Generalisation to categories the model never saw, and the OOD detector the web service uses.

Questions answered (numbers land in runs/<run>/unseen/):
  1. How much worse is the model on unseen categories than on the trained ones?
  2. What does it produce instead? (which training category do unseen images resemble)
  3. Can a cheap feature-similarity score tell unseen inputs apart from seen ones? (AUROC)
  4. Does that score predict bad reconstructions? (rank correlation with Chamfer)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import DatasetMeta, Split
from .evaluate import score_split, train_bank
from .models import load_checkpoint
from .ood import OODDetector, auroc, file_sha256, ood_scores, spearman, threshold_at
from .plots import plot_category_bars, plot_ood_hist, plot_qualitative, plot_score_vs_error
from .train import Session, encode_split


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                 chunk: int = 200) -> tuple[float, float]:
    """95% CI of the mean by resampling. Pass one value per MESH: the 24 views of a mesh are not
    independent samples, so resampling views would give falsely narrow intervals.

    Resampling happens in chunks: one (n_boot, n) index array would be n_boot * n * 8 bytes
    (hundreds of MB for a large sample), which is a needless memory spike."""
    v = np.asarray(values, np.float64)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        means[s:s + k] = v[rng.integers(0, len(v), (k, len(v)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _splits_with_data(meta: DatasetMeta) -> list[str]:
    return [s for s in ("test", "val", "train") if meta.counts.get(s, 0) > 0]


@torch.no_grad()
def evaluate_unseen(run_dir: str | Path, session: Session, unseen_root: str | Path, keep: float = 0.95,
                    bank_views: list[int] | None = None, n_vis: int = 2) -> dict:
    run_dir, unseen_root = Path(run_dir), Path(unseen_root)
    out = run_dir / "unseen"
    out.mkdir(parents=True, exist_ok=True)
    dev = session.device
    model, cfg, ck = load_checkpoint(run_dir / "best.pt", dev)
    umeta = DatasetMeta.load(unseen_root)
    overlap = set(umeta.categories) & set(session.meta.categories)
    if overlap:
        raise ValueError(f"'unseen' dataset contains training categories: {sorted(overlap)}")
    if umeta.views != session.meta.views:
        raise ValueError("unseen dataset uses a different view grid than the training data")
    metrics = ["cd_x1000"] + [f"f@{t}" for t in cfg.fscore_taus]
    vis_view = session.meta.view_index(cfg.val_views[0])

    # --- feature banks: full (every training image) and compact (what the web service ships)
    tr = session.split("train")
    bank = train_bank(model, cfg, run_dir, session)
    bank_views = bank_views or list(range(0, tr.V, 3))                 # 8 of 24 views, spread over az and el
    rows = (torch.arange(tr.N, device=dev)[:, None] * tr.V + torch.tensor(bank_views, device=dev)[None]).flatten()
    compact = bank.feats[rows]

    # --- threshold from SEEN validation images only
    val = session.split("val")
    f_val = F.normalize(encode_split(model, val, cfg.img_res, session).flatten(0, 1).float(), dim=1)
    val_scores = {"full": ood_scores(f_val, bank.feats).cpu().numpy(),
                  "compact": ood_scores(f_val, compact).cpu().numpy()}
    thr = {k: threshold_at(v, keep) for k, v in val_scores.items()}

    # --- score seen test + every unseen split
    parts = [("seen", "test", session.split("test"))]
    parts += [("unseen", s, Split(unseen_root, s, dev)) for s in _splits_with_data(umeta)]
    frames, vis_rows = [], []
    for source, split_name, sp in parts:
        res = score_split(model, cfg, sp, tr, bank, session, keep_view=vis_view)
        wide = res.long.pivot_table(index=["mesh", "view"], columns="method", values=metrics)
        wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
        d = res.extra.copy()
        d["score_full"] = 1 - d.max_sim
        d["score"] = ood_scores(res.feats.to(dev), compact).cpu().numpy()
        d = d.merge(wide.reset_index(), on=["mesh", "view"])
        d["source"], d["split"] = source, split_name
        d["category"] = sp.index["category"].to_numpy()[d.mesh]
        d["mesh_id"] = sp.index["mesh_id"].to_numpy()[d.mesh]
        d["retrieved_category"] = tr.index["category"].to_numpy()[d.nn_train_mesh]
        d["elevation"] = [session.meta.views[v][1] for v in d.view]
        frames.append(d)
        if source == "unseen":
            for m in sp.index.groupby("category").head(n_vis).index.to_numpy():
                if int(m) in res.keep:
                    vis_rows.append((f"{sp.index.category[m]} / {sp.index.mesh_id[m]}", sp.sil[m, vis_view],
                                     sp.points[m, :cfg.n_gt].cpu().numpy(), *res.keep[int(m)]))
    df = pd.concat(frames, ignore_index=True)
    df["flagged"] = df.score > thr["compact"]
    df.to_csv(out / "per_sample.csv.gz", index=False)

    # --- 1) seen vs unseen quality (mesh-level bootstrap)
    per_mesh = df.groupby(["source", "category", "split", "mesh_id"])[
        [f"model_{m}" for m in metrics] + ["retrieval_cd_x1000", "score", "flagged"]].mean().reset_index()
    overall = {}
    for source, g in per_mesh.groupby("source"):
        o = {"meshes": int(len(g))}
        for col in [f"model_{m}" for m in metrics] + ["retrieval_cd_x1000"]:
            lo, hi = bootstrap_ci(g[col])
            o[col] = {"mean": float(g[col].mean()), "ci95": [lo, hi]}
        o["flagged_share"] = float(df[df.source == source].flagged.mean())
        overall[source] = o

    # --- 2) per-category table
    cat_rows = []
    for (source, cat), g in per_mesh.groupby(["source", "category"]):
        s = df[(df.source == source) & (df.category == cat)]
        top = s.retrieved_category.value_counts(normalize=True)
        lo, hi = bootstrap_ci(g.model_cd_x1000)
        cat_rows.append(dict(category=cat, source=source, meshes=len(g),
                             model_cd_x1000=g.model_cd_x1000.mean(), model_cd_x1000_lo=lo, model_cd_x1000_hi=hi,
                             **{f"model_{m}": g[f"model_{m}"].mean() for m in metrics[1:]},
                             retrieval_cd_x1000=g.retrieval_cd_x1000.mean(),
                             flagged_share=s.flagged.mean(), score_mean=s.score.mean(),
                             looks_like=top.index[0], looks_like_share=top.iloc[0]))
    cats = pd.DataFrame(cat_rows)
    cats.to_csv(out / "per_category.csv", index=False)

    # --- 3) OOD separation
    seen, unseen = df[df.source == "seen"], df[df.source == "unseen"]
    ood = {
        "threshold": thr["compact"], "threshold_full_bank": thr["full"], "keep_seen_val": keep,
        "auroc": auroc(seen.score, unseen.score),
        "auroc_full_bank": auroc(seen.score_full, unseen.score_full),
        "auroc_per_category": {c: auroc(seen.score, g.score) for c, g in unseen.groupby("category")},
        "false_alarm_rate_seen_test": float(seen.flagged.mean()),
        "detection_rate_unseen": float(unseen.flagged.mean()),
        "bank": {"views": bank_views, "rows": int(len(compact)), "dim": int(compact.shape[1]),
                 "size_mb_float16": round(compact.numel() * 2 / 1e6, 1), "full_rows": int(len(bank.feats))},
    }

    # --- 4) does the score predict error?
    link = {
        "spearman_score_vs_cd_all": spearman(df.score, df.model_cd_x1000),
        "spearman_score_vs_cd_seen": spearman(seen.score, seen.model_cd_x1000),
        "mean_cd_accepted": float(df[~df.flagged].model_cd_x1000.mean()),
        "mean_cd_flagged": float(df[df.flagged].model_cd_x1000.mean()) if df.flagged.any() else None,
    }

    sha = file_sha256(run_dir / "best.pt")
    detector = OODDetector(bank=compact.half().cpu().numpy(), threshold=thr["compact"], model_sha256=sha,
                           info=dict(run=run_dir.name, bank_views=bank_views, keep_seen_val=keep,
                                     auroc_unseen=ood["auroc"], train_meshes=int(tr.N),
                                     created=time.strftime("%Y-%m-%d %H:%M:%S")))
    detector.save(out / "ood_detector.npz")

    summary = dict(run=run_dir.name, checkpoint_sha256=sha, best_epoch=int(ck["epoch"]),
                   seen_categories=session.meta.categories, unseen_categories=umeta.categories,
                   overall=overall, ood=ood, score_vs_error=link,
                   per_category=json.loads(cats.to_json(orient="records")))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "findings.md").write_text(findings_markdown(summary, cats))

    plot_category_bars(cats, out / "category_cd.png")
    plot_ood_hist({"seen val": val_scores["compact"], "seen test": seen.score, "unseen": unseen.score},
                  thr["compact"], out / "ood_hist.png")
    plot_score_vs_error(df.score, df.model_cd_x1000, df.source, thr["compact"], out / "score_vs_cd.png")
    plot_qualitative(vis_rows, out / "qualitative_unseen.png")

    print(findings_markdown(summary, cats))
    return summary


def _fmt_ci(d) -> str:
    return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def findings_markdown(s: dict, cats: pd.DataFrame) -> str:
    seen, unseen, ood, link = s["overall"]["seen"], s["overall"]["unseen"], s["ood"], s["score_vs_error"]
    ratio = unseen["model_cd_x1000"]["mean"] / seen["model_cd_x1000"]["mean"]
    beats = unseen["model_cd_x1000"]["mean"] < unseen["retrieval_cd_x1000"]["mean"]
    lines = [
        f"# Unseen-category evaluation: {s['run']}",
        "",
        f"Trained on: {', '.join(s['seen_categories'])}  ",
        f"Never seen: {', '.join(s['unseen_categories'])}",
        "",
        "## 1. Quality drop (mean over meshes, 95% bootstrap CI)",
        "",
        "| | meshes | model CDx1000 | model F@1% | retrieval CDx1000 |",
        "|---|---|---|---|---|",
    ]
    for name, o in (("seen (test)", seen), ("unseen", unseen)):
        f1 = next(k for k in o if k.startswith("model_f@"))
        lines.append(f"| {name} | {o['meshes']} | {_fmt_ci(o['model_cd_x1000'])} | {_fmt_ci(o[f1])} | "
                     f"{_fmt_ci(o['retrieval_cd_x1000'])} |")
    lines += [
        "",
        f"Unseen Chamfer is **{ratio:.1f}x** the seen Chamfer. On unseen categories the model is "
        f"{'better' if beats else 'not better'} than copying the nearest training shape.",
        "",
        "## 2. Per category",
        "",
        "| category | source | meshes | model CDx1000 | retrieval CDx1000 | flagged | looks like |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in cats.sort_values(["source", "model_cd_x1000"]).itertuples():
        lines.append(f"| {r.category} | {r.source} | {r.meshes} | {r.model_cd_x1000:.3f} | "
                     f"{r.retrieval_cd_x1000:.3f} | {r.flagged_share:.0%} | "
                     f"{r.looks_like} ({r.looks_like_share:.0%}) |")
    lines += [
        "",
        "'looks like' = the category of the nearest training image in feature space: what the model "
        "most likely falls back to.",
        "",
        "## 3. Detecting unfamiliar inputs",
        "",
        f"- Score = 1 - max cosine similarity to {ood['bank']['rows']:,} training images "
        f"({ood['bank']['size_mb_float16']} MB). Threshold {ood['threshold']:.4f} lets "
        f"{ood['keep_seen_val']:.0%} of seen validation images through.",
        f"- AUROC seen-test vs unseen: **{ood['auroc']:.3f}** (full bank of {ood['bank']['full_rows']:,} "
        f"images: {ood['auroc_full_bank']:.3f}).",
        f"- At the threshold: {ood['detection_rate_unseen']:.0%} of unseen images flagged, "
        f"{ood['false_alarm_rate_seen_test']:.0%} of seen test images flagged (false alarms).",
        "- AUROC per unseen category: " + ", ".join(f"{c} {v:.2f}" for c, v in
                                                    sorted(ood["auroc_per_category"].items(), key=lambda x: x[1])),
        "",
        "## 4. Does the score predict bad reconstructions?",
        "",
        f"- Spearman(score, Chamfer): {link['spearman_score_vs_cd_all']:.2f} over all images, "
        f"{link['spearman_score_vs_cd_seen']:.2f} within seen categories.",
        f"- Mean Chamfer x1000: accepted {link['mean_cd_accepted']:.3f}"
        + (f", flagged {link['mean_cd_flagged']:.3f}." if link["mean_cd_flagged"] is not None else "."),
        "",
        "Categories with low AUROC resemble a training category: the detector cannot tell them apart, "
        "and neither can the model. A flag is a warning, not a guarantee.",
    ]
    return "\n".join(lines) + "\n"
