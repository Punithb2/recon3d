"""Check a published model version the way the web app will use it: fresh download, load, predict.

    python scripts/verify_hub.py                          # Punith25/recon3d-resnet18 @ v1.0
    python scripts/verify_hub.py --revision v1.0.1

Downloads into a temporary folder (not your normal cache), so it proves the Hub copy works on its own.
"""

import argparse
import re
import tempfile
import time

import numpy as np
import torch

from recon3d.data import preprocess
from recon3d.hub import BUNDLE_FILES, load_from_hub
from recon3d.ood import file_sha256


def chair_like(size: int = 224) -> np.ndarray:
    s = np.zeros((size, size), np.uint8)
    s[40:120, 70:90] = 255        # backrest
    s[110:130, 70:160] = 255      # seat
    s[130:190, 75:82] = 255       # legs
    s[130:190, 150:157] = 255
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Punith25/recon3d-resnet18")
    ap.add_argument("--revision", default="v1.0")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads, like a small Space")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    checks = {}
    with tempfile.TemporaryDirectory() as cache:
        t = time.perf_counter()
        b = load_from_hub(args.repo, revision=args.revision, cache_dir=cache)
        print(f"downloaded + loaded {args.repo}@{args.revision} in {time.perf_counter() - t:.1f}s")
        checks["all 5 bundle files present"] = all((b.path / f).exists() for f in BUNDLE_FILES)
        checks["weights hash matches config.json"] = \
            file_sha256(b.path / "model.safetensors") == b.config["weights_sha256"]
        card = (b.path / "README.md").read_text()
        ids = set(re.findall(r'load_from_hub\("([^"]+)"', card))
        checks[f"model card points to {args.repo}"] = ids == {args.repo}

        x = preprocess(torch.from_numpy(chair_like())[None])
        with torch.inference_mode():
            b.model(x)                                            # warm-up
            t = time.perf_counter()
            f = b.model.embed(x)
            pts = b.model.decode(f)[0].numpy()
            ms = (time.perf_counter() - t) * 1000
        score = float(b.detector.scores(f.numpy())[0])
        checks["2,048 finite points"] = pts.shape == (2048, 3) and bool(np.isfinite(pts).all())
        checks["points inside the unit frame"] = bool((np.abs(pts) < 0.75).mean() > 0.99)

    print(f"prediction: {ms:.0f} ms on {args.threads} CPU threads | OOD score {score:.4f} "
          f"(threshold {b.detector.threshold:.4f}) | epoch {b.config['best_epoch']}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
