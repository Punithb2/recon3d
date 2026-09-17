"""Figures. matplotlib is optional (pip install recon3d[plots]); without it, plots are skipped."""

from __future__ import annotations

from pathlib import Path


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")                 # no display needed (servers, CI, Kaggle background runs)
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed: skipping figure")
        return None


def plot_curves(history: list[dict], path: str | Path, title: str = "") -> None:
    plt = _plt()
    if plt is None or not history:
        return
    ep = [h["epoch"] + 1 for h in history]
    fig = plt.figure(figsize=(6, 3.5))
    plt.plot(ep, [h["train_cd"] for h in history], label="train")
    plt.plot(ep, [h["val_cd"] for h in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("Chamfer x1000")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_qualitative(rows: list[tuple[str, object, object, object, object]], path: str | Path) -> None:
    """rows: (title, silhouette uint8 [H, W], gt [N,3], pred [N,3], retrieval [N,3]); side view (z, y)."""
    plt = _plt()
    if plt is None or not rows:
        return
    fig, axes = plt.subplots(len(rows), 4, figsize=(9, 2.2 * len(rows)), squeeze=False)
    for ax_row, (title, sil, gt, pred, retr) in zip(axes, rows, strict=True):
        ax_row[0].imshow(sil, cmap="gray")
        ax_row[0].set_title(title, fontsize=8)
        for ax, P, t in zip(ax_row[1:], (gt, pred, retr), ("ground truth", "model", "retrieval"), strict=True):
            ax.scatter(P[:, 2], P[:, 1], s=0.2, c="k")
            ax.set_xlim(-.5, .5)
            ax.set_ylim(-.5, .5)
            ax.set_aspect("equal")
            ax.set_title(t, fontsize=8)
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_category_bars(table, path: str | Path, metric: str = "model_cd_x1000", title: str = "") -> None:
    """table: DataFrame with columns category, source ('seen'/'unseen'), metric, and optional metric_lo/_hi."""
    plt = _plt()
    if plt is None or table is None or len(table) == 0:
        return
    t = table.sort_values(["source", metric])
    colors = ["#4C78A8" if s == "seen" else "#E45756" for s in t.source]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(t)), t[metric], color=colors)
    if f"{metric}_lo" in t:
        ax.errorbar(range(len(t)), t[metric], yerr=[t[metric] - t[f"{metric}_lo"], t[f"{metric}_hi"] - t[metric]],
                    fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t.category, rotation=40, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or "blue = trained categories, red = never seen")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_ood_hist(scores: dict[str, object], threshold: float, path: str | Path) -> None:
    """scores: {label: 1-D array}. Draws overlaid histograms and the threshold line."""
    plt = _plt()
    if plt is None:
        return
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    allv = np.concatenate([np.asarray(v) for v in scores.values()])
    bins = np.linspace(allv.min(), allv.max(), 50)
    for label, v in scores.items():
        ax.hist(np.asarray(v), bins=bins, alpha=0.5, density=True, label=label)
    ax.axvline(threshold, color="k", ls="--", lw=1, label=f"threshold {threshold:.4f}")
    ax.set_xlabel("OOD score = 1 - max cosine similarity to training images")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_score_vs_error(score, error, source, threshold: float, path: str | Path) -> None:
    plt = _plt()
    if plt is None:
        return
    import numpy as np
    score, error, source = np.asarray(score), np.asarray(error), np.asarray(source)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for s, c in (("seen", "#4C78A8"), ("unseen", "#E45756")):
        m = source == s
        ax.scatter(score[m], error[m], s=2, alpha=0.3, c=c, label=s)
    ax.axvline(threshold, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("OOD score")
    ax.set_ylabel("model Chamfer x1000 (log)")
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
