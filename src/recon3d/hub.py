"""Model bundle for the Hugging Face Hub: the one artifact the web app downloads.

A bundle is a folder with:
    model.safetensors   weights (safetensors: plain tensors, no pickle, so loading can't run code)
    config.json         TrainConfig + provenance (source checkpoint hash, epoch, package version)
    ood_detector.npz    feature bank + threshold, bound to the hash of model.safetensors
    metrics.json        seen-test, unseen and OOD numbers the card was generated from
    README.md           model card

Versioning: every publish is one Hub commit plus an immutable git tag (v1.0, v1.1, ...). The app pins a
tag, so a new upload never changes what is deployed until the app is told to use it. That is the
"model registry" of this project, without running a registry server.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import TrainConfig
from .models import build_model, load_checkpoint
from .ood import OODDetector, file_sha256

BUNDLE_FILES = ("model.safetensors", "config.json", "ood_detector.npz", "metrics.json", "README.md")


@dataclass
class Bundle:
    model: torch.nn.Module
    cfg: TrainConfig
    detector: OODDetector
    config: dict
    metrics: dict
    path: Path


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def export_bundle(run_dir: str | Path, out_dir: str | Path, repo_id: str = "Punith25/recon3d-resnet18") -> Path:
    """Turn a finished run folder into a Hub bundle. Expects in run_dir:
    best.pt, unseen/ood_detector.npz, unseen/summary.json, and (optional) eval_test_summary.json."""
    from safetensors.torch import save_file

    run_dir, out = Path(run_dir), Path(out_dir)
    ckpt = run_dir / "best.pt"
    det_path = run_dir / "unseen" / "ood_detector.npz"
    for p in (ckpt, det_path, run_dir / "unseen" / "summary.json"):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new folder so old files can't leak into the upload")
    out.mkdir(parents=True, exist_ok=True)

    model, cfg, meta = load_checkpoint(ckpt)
    detector = OODDetector.load(det_path, checkpoint=ckpt)          # refuses a detector built for other weights
    src_sha = detector.model_sha256

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt", "recon3d": __version__})
    weights_sha = file_sha256(out / "model.safetensors")

    # same bank and threshold, now bound to the exported weights file
    detector.model_sha256 = weights_sha
    detector.info = {**detector.info, "source_checkpoint_sha256": src_sha}
    detector.save(out / "ood_detector.npz")

    config = dict(recon3d_version=__version__, cfg=cfg.to_dict(), best_epoch=int(meta["epoch"]) + 1,
                  val_cd_x1000=float(meta["val_cd"]), source_checkpoint_sha256=src_sha,
                  weights_sha256=weights_sha, run=run_dir.name,
                  exported=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "config.json").write_text(json.dumps(config, indent=1))

    unseen = _read_json(run_dir / "unseen" / "summary.json")
    seen = _read_json(run_dir / "eval_test_summary.json")
    metrics = dict(seen_test=seen, unseen={k: unseen[k] for k in ("overall", "per_category", "unseen_categories",
                                                                   "seen_categories")},
                   ood=unseen["ood"], score_vs_error=unseen["score_vs_error"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "README.md").write_text(model_card(config, metrics, repo_id))

    # prove the bundle reproduces the checkpoint before anything is uploaded
    b = load_bundle(out)
    x = torch.randn(2, 3, cfg.img_res, cfg.img_res)
    with torch.no_grad():
        if not torch.equal(model(x), b.model(x)):
            raise RuntimeError("exported weights do not reproduce the checkpoint's predictions")
    return out


def load_bundle(path: str | Path, device: str | torch.device = "cpu") -> Bundle:
    from safetensors.torch import load_file

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    cfg = TrainConfig.from_dict(config["cfg"])
    model = build_model(cfg, device=device, pretrained=False)
    model.load_state_dict(load_file(str(path / "model.safetensors"), device=str(device)), strict=True)
    model.eval()
    detector = OODDetector.load(path / "ood_detector.npz", checkpoint=path / "model.safetensors")
    metrics = _read_json(path / "metrics.json") or {}
    return Bundle(model, cfg, detector, config, metrics, path)


def load_from_hub(repo_id: str, revision: str, device: str | torch.device = "cpu",
                  token: str | None = None, cache_dir: str | None = None) -> Bundle:
    """Download (cached after the first time) and load a pinned version, e.g. revision='v1.0'."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id, revision=revision, allow_patterns=list(BUNDLE_FILES),
                              token=token, cache_dir=cache_dir)
    b = load_bundle(local, device)
    b.config["hub"] = {"repo_id": repo_id, "revision": revision}
    return b


def publish_bundle(bundle_dir: str | Path, repo_id: str, tag: str, private: bool = False,
                   token: str | None = None) -> str:
    """Upload the bundle as one commit and tag it. Tags are never overwritten."""
    from huggingface_hub import HfApi

    bundle_dir = Path(bundle_dir)
    missing = [f for f in BUNDLE_FILES if not (bundle_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"bundle incomplete, missing {missing}")
    load_bundle(bundle_dir)                                   # never upload something that doesn't load
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    existing = {t.name for t in api.list_repo_refs(repo_id).tags}
    if tag in existing:
        raise ValueError(f"tag {tag} already exists on {repo_id}; published versions are immutable, pick a new tag")
    info = api.upload_folder(repo_id=repo_id, folder_path=str(bundle_dir), allow_patterns=list(BUNDLE_FILES),
                             commit_message=f"recon3d model {tag}")
    api.create_tag(repo_id, tag=tag, tag_message=f"recon3d {tag}", revision=info.oid)
    return f"https://huggingface.co/{repo_id}/tree/{tag}"


# ---------------------------------------------------------------- model card
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def model_card(config: dict, metrics: dict, repo_id: str) -> str:
    cfg = config["cfg"]
    seen = metrics.get("seen_test")
    un = metrics["unseen"]
    ood = metrics["ood"]
    link = metrics["score_vs_error"]
    f1 = f"f@{cfg['fscore_taus'][0]}"
    f2 = f"f@{cfg['fscore_taus'][1]}"
    m = re.search(r"_n(\d+)$", config["run"])
    per_cat = f"{int(m.group(1)):,} training meshes per category" if m else "training meshes"

    lines = [
        "---",
        "license: other",
        "library_name: pytorch",
        "pipeline_tag: image-to-3d",
        "tags: [3d-reconstruction, point-cloud, single-view-reconstruction, shapenet, resnet18]",
        "---",
        "",
        "# recon3d: single-view silhouette to 3D point cloud",
        "",
        f"A ResNet-18 encoder and an MLP decoder that turn one {cfg['img_res']}x{cfg['img_res']} silhouette into "
        f"{cfg['n_pred']:,} (x, y, z) points. Trained with Chamfer distance on six ShapeNet categories: "
        f"{', '.join(un['seen_categories'])}.",
        "",
        "Code, training pipeline and tests: https://github.com/Punithb2/recon3d",
        "",
        "## Intended use",
        "",
        "Research and demonstration. Input: a silhouette (white object on black, or a dark object on a light "
        "background) of an object from one of the six training categories, roughly centred. Output: a point "
        "cloud in a normalised frame (bounding-box centre at the origin, diagonal = 1, Y up). "
        "Not for measurement, safety-relevant or commercial use.",
        "",
        "## How to use",
        "",
        "```python",
        "from recon3d.hub import load_from_hub",
        "from recon3d.data import preprocess",
        "import torch",
        "",
        f'b = load_from_hub("{repo_id}", revision="<tag>")',
        "sil = torch.zeros(1, 224, 224, dtype=torch.uint8)      # your silhouette, uint8 [1, H, W]",
        "with torch.no_grad():",
        "    feats = b.model.embed(preprocess(sil))",
        "    points = b.model.decode(feats)[0]                # [2048, 3]",
        "familiar = not b.detector.is_ood(feats.numpy())[0]",
        "```",
        "",
        "## Training",
        "",
        f"- Data: {', '.join(un['seen_categories'])}; {per_cat}, 24 rendered views "
        "each (azimuth every 45 deg, elevation -30/0/30 deg). Targets: 8,192 area-weighted surface samples per mesh.",
        f"- Recipe: linear probe (frozen ImageNet encoder), then fine-tuning of layer3 + layer4 "
        f"(decoder lr {cfg['lr']}, encoder lr {cfg['enc_lr']}, batch {cfg['batch_size']}, {cfg['epochs']} epochs, "
        "cosine schedule, mixed precision, scale/shift augmentation) on a free Kaggle T4.",
        f"- Checkpoint: epoch {config['best_epoch']} (lowest validation CD x1000: {config['val_cd_x1000']:.3f}).",
        "",
        "## Evaluation",
        "",
        "CD = squared-L2 Chamfer distance x1000 (2,048 vs 2,048 points, both directions). F@1% / F@2% = F-score "
        "at 1% / 2% of the bounding-box diagonal. Every test mesh is evaluated from all 24 views. "
        "Retrieval = copying the training shape whose image features are closest (a strong baseline).",
        "",
    ]
    if seen:
        lines += ["### Trained categories (test split)", "",
                  "| category | CD (model) | CD (retrieval) | F@1% | F@2% |", "|---|---|---|---|---|"]
        for c, v in sorted(seen["per_category"].items(), key=lambda kv: kv[1]["model"]["cd_x1000"]):
            lines.append(f"| {c} | {_f(v['model']['cd_x1000'])} | {_f(v['retrieval']['cd_x1000'])} | "
                         f"{_f(v['model'][f1])} | {_f(v['model'][f2])} |")
        o = seen["overall"]
        lines += [f"| **all** | **{_f(o['model']['cd_x1000'])}** | {_f(o['retrieval']['cd_x1000'])} | "
                  f"**{_f(o['model'][f1])}** | **{_f(o['model'][f2])}** |", ""]

    us, uu = un["overall"]["seen"], un["overall"]["unseen"]
    ratio = uu["model_cd_x1000"]["mean"] / us["model_cd_x1000"]["mean"]
    lines += [
        "### Categories never seen in training",
        "",
        f"Evaluated on {uu['meshes']:,} meshes from {', '.join(un['unseen_categories'])}.",
        "",
        "| category | CD (model) | CD (retrieval) | flagged as unfamiliar |",
        "|---|---|---|---|",
    ]
    for r in sorted((r for r in un["per_category"] if r["source"] == "unseen"), key=lambda r: r["model_cd_x1000"]):
        lines.append(f"| {r['category']} | {_f(r['model_cd_x1000'])} | {_f(r['retrieval_cd_x1000'])} | "
                     f"{r['flagged_share']:.0%} |")
    lines += [
        f"| **all unseen** | **{_f(uu['model_cd_x1000']['mean'])}** | {_f(uu['retrieval_cd_x1000']['mean'])} | "
        f"{uu['flagged_share']:.0%} |",
        "",
        f"Error on unseen categories is about **{ratio:.1f}x** higher than on trained ones "
        f"(F@1% {_f(uu[f'model_{f1}']['mean'])} vs {_f(us[f'model_{f1}']['mean'])}). The model does not "
        "generalise to new categories; it produces shapes that resemble its training categories.",
        "",
        "### Unfamiliar-input warning",
        "",
        f"`ood_detector.npz` holds image features of {ood['bank']['rows']:,} training images. An input is flagged "
        f"when 1 - (highest cosine similarity) exceeds {ood['threshold']:.4f}, a threshold that passes "
        f"{ood['keep_seen_val']:.0%} of trained-category validation images.",
        "",
        f"- AUROC trained-test vs unseen: {ood['auroc']:.3f}; flags {ood['detection_rate_unseen']:.0%} of unseen "
        f"images and {ood['false_alarm_rate_seen_test']:.0%} of trained-category test images.",
        "- Per category AUROC: " + ", ".join(f"{c} {v:.2f}" for c, v in
                                             sorted(ood["auroc_per_category"].items(), key=lambda kv: kv[1])) + ".",
        f"- The score correlates with reconstruction error (Spearman {link['spearman_score_vs_cd_all']:.2f}), "
        "but it measures how unfamiliar the *image* looks, not how hard the 3D shape is. "
        "Objects whose silhouettes resemble training objects can pass unflagged and still be reconstructed badly.",
        "",
        "## Limitations",
        "",
        "- Six categories only; anything else is reconstructed poorly (see above).",
        "- Silhouettes only: no colour or texture, so front/back ambiguities remain.",
        "- Chamfer training favours averaged, slightly blurred shapes; thin parts are the first to suffer.",
        "- Trained on clean synthetic renders; photos need a good object mask first.",
        "",
        "## Data and licence",
        "",
        "Trained on renders of ShapeNet models. ShapeNet is available for non-commercial research under its terms "
        "of use, so these weights are shared for research and demonstration only. The training data is not "
        "redistributed here.",
        "",
        "## Provenance",
        "",
        f"- recon3d {config['recon3d_version']}, run `{config['run']}`, exported {config['exported']}",
        f"- weights sha256 `{config['weights_sha256']}`",
        f"- source checkpoint sha256 `{config['source_checkpoint_sha256']}`",
    ]
    return "\n".join(lines) + "\n"

