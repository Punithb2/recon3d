"""Check that a checkpoint trained in the Kaggle notebook loads and predicts with the package.

usage:
    python scripts/check_checkpoint.py path/to/best.pt [path/to/silhouette.png]

Prints the stored config, parameter count and a prediction summary. Exit code 0 = all checks passed.
"""

import sys
import time

import numpy as np

from recon3d.inference import Predictor, load_silhouette
from recon3d.io import write_ply


def main(ckpt: str, image: str | None = None) -> int:
    t = time.perf_counter()
    p = Predictor(ckpt)
    n = sum(v.numel() for v in p.model.parameters())
    print(f"loaded in {time.perf_counter() - t:.1f}s | decoder={p.cfg.decoder} | {n / 1e6:.1f}M params | "
          f"best epoch {p.meta['epoch'] + 1} | val CDx1000 {p.meta['val_cd']:.3f}")

    if image:
        sil = load_silhouette(image, p.cfg.img_res)
    else:                                   # synthetic chair-like silhouette: seat, back, legs
        sil = np.zeros((p.cfg.img_res, p.cfg.img_res), np.uint8)
        sil[40:120, 70:90] = 255
        sil[110:130, 70:160] = 255
        sil[130:190, 75:82] = 255
        sil[130:190, 150:157] = 255
    p.predict(sil)                          # warm-up
    t = time.perf_counter()
    pts = p.predict(sil)[0]
    ms = (time.perf_counter() - t) * 1000

    checks = {
        f"shape is ({p.cfg.n_pred}, 3)": pts.shape == (p.cfg.n_pred, 3),
        "all values finite": bool(np.isfinite(pts).all()),
        "points inside the unit frame (|x| < 0.75)": bool((np.abs(pts) < 0.75).mean() > 0.99),
        "not collapsed to a point (spread > 0.05)": bool(np.linalg.norm(pts - pts.mean(0), axis=1).mean() > 0.05),
    }
    print(f"CPU prediction: {ms:.0f} ms | bbox min {pts.min(0).round(3)} max {pts.max(0).round(3)}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    write_ply(pts, "check_prediction.ply")
    print("wrote check_prediction.ply (open it in MeshLab or https://3dviewer.net)")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)                       # usage: path to best.pt, optional silhouette image
        sys.exit(2)
    sys.exit(main(*sys.argv[1:3]))
