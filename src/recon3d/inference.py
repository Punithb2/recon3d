"""Minimal single-image prediction. The web service (Block E) builds on this and adds input
validation, framing normalisation, the out-of-distribution check, caching and logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import preprocess
from .models import load_checkpoint


def load_silhouette(path_or_image, size: int = 224) -> np.ndarray:
    """Image -> uint8 [size, size], white object on black background (the training format).
    If the corners are bright (dark object on a white page), the image is inverted."""
    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    if img.mode in ("RGBA", "LA"):                      # transparent PNG: alpha channel is the silhouette
        arr = np.asarray(img.getchannel("A"))
    else:
        arr = np.asarray(img.convert("L"))
        corners = np.concatenate([arr[:4, :4].ravel(), arr[:4, -4:].ravel(),
                                  arr[-4:, :4].ravel(), arr[-4:, -4:].ravel()])
        if corners.mean() > 127:
            arr = 255 - arr
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.uint8)


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.cfg, self.meta = load_checkpoint(checkpoint, self.device)

    @torch.inference_mode()
    def predict(self, silhouettes: np.ndarray) -> np.ndarray:
        """uint8 [H, W] or [B, H, W] -> float32 [B, n_pred, 3]."""
        x = torch.from_numpy(np.array(silhouettes, dtype=np.uint8, copy=True))
        if x.ndim == 2:
            x = x[None]
        return self.model(preprocess(x.to(self.device), self.cfg.img_res)).float().cpu().numpy()

    @torch.inference_mode()
    def embed(self, silhouettes: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.array(silhouettes, dtype=np.uint8, copy=True))
        if x.ndim == 2:
            x = x[None]
        return self.model.embed(preprocess(x.to(self.device), self.cfg.img_res)).float().cpu().numpy()
