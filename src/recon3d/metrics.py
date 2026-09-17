"""Point-cloud metrics. All point clouds live in the normalised frame (bbox diagonal = 1)."""

from __future__ import annotations

import torch


def chamfer(pred: torch.Tensor, gt: torch.Tensor, chunk: int = 32):
    """Squared-L2 Chamfer distance, averaged in both directions.

    pred: [B, Np, 3], gt: [B, Ng, 3]
    Returns (cd [B], d_pg [B, Np], d_gp [B, Ng]).

    Nearest neighbours are searched without autograd (cheap on memory), then the distances to
    those neighbours are recomputed with autograd. The gradient is identical to differentiating
    through min(), because min() only passes gradient to the selected neighbour anyway.
    """
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[-1] != 3 or gt.shape[-1] != 3:
        raise ValueError(f"expected [B, N, 3] tensors, got {tuple(pred.shape)} and {tuple(gt.shape)}")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError("pred and gt must have the same batch size")
    pred, gt = pred.float(), gt.float()
    with torch.no_grad():
        i_pg, i_gp = [], []
        for s in range(0, pred.shape[0], chunk):
            p, g = pred[s:s + chunk], gt[s:s + chunk]
            d = (p * p).sum(-1, keepdim=True) - 2 * p @ g.transpose(1, 2) + (g * g).sum(-1).unsqueeze(1)
            i_pg.append(d.argmin(2))
            i_gp.append(d.argmin(1))
        i_pg, i_gp = torch.cat(i_pg), torch.cat(i_gp)
    nn_g = torch.gather(gt, 1, i_pg[..., None].expand(-1, -1, 3))
    nn_p = torch.gather(pred, 1, i_gp[..., None].expand(-1, -1, 3))
    d_pg = (pred - nn_g).pow(2).sum(-1)
    d_gp = (gt - nn_p).pow(2).sum(-1)
    return d_pg.mean(1) + d_gp.mean(1), d_pg, d_gp


def precision_recall(d_pg: torch.Tensor, d_gp: torch.Tensor, tau: float):
    """precision: share of predicted points within tau of the surface.
    recall: share of surface points within tau of a prediction. (d_* are squared distances.)"""
    return (d_pg < tau * tau).float().mean(1), (d_gp < tau * tau).float().mean(1)


def fscore(d_pg: torch.Tensor, d_gp: torch.Tensor, tau: float) -> torch.Tensor:
    prec, rec = precision_recall(d_pg, d_gp, tau)
    return 2 * prec * rec / (prec + rec).clamp_min(1e-8)


def spread_ratio(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean distance from the centroid, pred / gt. < 1 means the prediction is shrunk ('blurry')."""
    def spread(x):
        return (x - x.mean(1, keepdim=True)).norm(dim=-1).mean(1)
    return spread(pred.float()) / spread(gt.float()).clamp_min(1e-8)
