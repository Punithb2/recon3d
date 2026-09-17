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
