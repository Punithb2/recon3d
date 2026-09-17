"""Command line interface.

    recon3d train    --data DIR --runs DIR configs/overfit.yaml configs/frozen.yaml configs/finetune.yaml
    recon3d evaluate --data DIR --runs DIR --run finetune_resnet18_n1000 --splits val,test
    recon3d unseen   --data DIR --unseen DIR --runs DIR --run finetune_resnet18_n1000
    recon3d export   --run-dir runs/finetune_resnet18_n1000 --out bundle_v1.0
    recon3d publish  --bundle bundle_v1.0 --repo Punithb2/recon3d-resnet18 --tag v1.0
    recon3d predict  --checkpoint best.pt --image chair.png --out chair.ply
    recon3d table    --runs DIR
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _check_gpu(dev: torch.device) -> None:
    if dev.type != "cuda":
        print("WARNING: running on CPU (fine for tests, far too slow for real training)")
        return
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        raise SystemExit(f"{name} (sm_{cap[0]}{cap[1]}) is not supported by torch {torch.__version__}. "
                         "On Kaggle choose GPU T4 x2.")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {name} | torch {torch.__version__}")


def _session(args):
    from .data import find_dataset
    from .train import Session

    data = Path(args.data) if args.data else find_dataset()
    dev = _device(args.device)
    _check_gpu(dev)
    return Session(data_root=data, runs_dir=Path(args.runs), device=dev,
                   time_limit_h=getattr(args, "time_limit_h", 8.0))


def cmd_train(args) -> int:
    from .config import load_config
    from .evaluate import evaluate
    from .train import is_finished, run_overfit, train

    cfgs = [load_config(p, args.set) for p in args.configs]      # fail fast on a bad config
    s = _session(args)
    print(f"data {s.data_root} | {s.meta.counts} | tag {s.meta.run_tag} | "
          f"experiments {[c.name for c in cfgs]}")
    for cfg in cfgs:
        print(f"\n################ {cfg.name} ################ ({s.time_left_h():.2f} h left)")
        if cfg.kind == "overfit":
            run_overfit(cfg, s)
            continue
        run_dir = train(cfg, s)
        if not is_finished(run_dir):
            print(f"[{cfg.name}] not finished: skipping evaluation and later experiments. Re-run to resume.")
            return 3
        if not args.no_eval:
            for split in args.eval_splits.split(","):
                evaluate(run_dir, split, s)
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate

    s = _session(args)
    for split in args.splits.split(","):
        evaluate(s.runs_dir / args.run, split, s)
    return 0


def cmd_unseen(args) -> int:
    from .unseen import evaluate_unseen

    s = _session(args)
    evaluate_unseen(s.runs_dir / args.run, s, args.unseen, keep=args.keep)
    return 0


def cmd_export(args) -> int:
    from .hub import export_bundle, load_bundle

    out = export_bundle(args.run_dir, args.out, repo_id=args.repo)
    b = load_bundle(out)
    print(f"bundle written to {out} (verified: reloads and reproduces the checkpoint)")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<20} {f.stat().st_size / 1e6:8.1f} MB")
    print(f"weights sha256 {b.config['weights_sha256'][:16]}... | OOD threshold {b.detector.threshold:.4f}")
    print(f"review {out / 'README.md'} (the model card), then run: recon3d publish --bundle {out} --tag v1.0")
    return 0


def cmd_publish(args) -> int:
    from .hub import publish_bundle

    url = publish_bundle(args.bundle, args.repo, args.tag, private=args.private)
    print(f"published {args.tag}: {url}")
    return 0


def cmd_predict(args) -> int:
    from .inference import Predictor, load_silhouette
    from .io import write_ply

    p = Predictor(args.checkpoint, device=args.device if args.device != "auto" else "cpu")
    sil = load_silhouette(args.image, p.cfg.img_res)
    t = time.perf_counter()
    pts = p.predict(sil)[0]
    ms = (time.perf_counter() - t) * 1000
    out = Path(args.out or Path(args.image).with_suffix(".ply"))
    write_ply(pts, out)
    print(f"{len(pts)} points -> {out} ({ms:.0f} ms)")
    return 0


def cmd_table(args) -> int:
    from .evaluate import results_table

    df = results_table(args.runs)
    if df.empty:
        print("no eval summaries found")
        return 1
    out = Path(args.runs).parent / "results_table.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nsaved {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recon3d", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp):
        sp.add_argument("--data", help="dataset folder (default: search /kaggle/input)")
        sp.add_argument("--runs", default="runs", help="where run folders are written")
        sp.add_argument("--device", default="auto", help="auto | cpu | cuda")

    t = sub.add_parser("train", help="run one or more experiments in order")
    data_args(t)
    t.add_argument("configs", nargs="+", help="YAML config files, run in the given order")
    t.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)")
    t.add_argument("--time-limit-h", type=float, default=8.0, help="stop cleanly before this many hours")
    t.add_argument("--eval-splits", default="val,test", help="comma-separated, e.g. val,test")
    t.add_argument("--no-eval", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a finished run")
    data_args(e)
    e.add_argument("--run", required=True, help="run folder name, e.g. finetune_resnet18_n1000")
    e.add_argument("--splits", default="val,test", help="comma-separated")
    e.set_defaults(func=cmd_evaluate)

    u = sub.add_parser("unseen", help="evaluate on never-seen categories and build the OOD detector")
    data_args(u)
    u.add_argument("--unseen", required=True, help="dataset folder with categories the model never trained on")
    u.add_argument("--run", required=True, help="run folder name, e.g. finetune_resnet18_n1000")
    u.add_argument("--keep", type=float, default=0.95, help="share of seen val images the threshold lets through")
    u.set_defaults(func=cmd_unseen)

    ex = sub.add_parser("export", help="package a finished run as a Hugging Face Hub bundle")
    ex.add_argument("--run-dir", required=True, help="folder with best.pt and unseen/ood_detector.npz")
    ex.add_argument("--out", required=True, help="new, empty folder for the bundle")
    ex.add_argument("--repo", default="Punithb2/recon3d-resnet18", help="Hub repo id (used in the model card)")
    ex.set_defaults(func=cmd_export)

    pb = sub.add_parser("publish", help="upload a bundle to the Hub and tag the version")
    pb.add_argument("--bundle", required=True)
    pb.add_argument("--repo", default="Punithb2/recon3d-resnet18")
    pb.add_argument("--tag", required=True, help="version tag, e.g. v1.0 (never reused)")
    pb.add_argument("--private", action="store_true", help="create the repo as private")
    pb.set_defaults(func=cmd_publish)

    pr = sub.add_parser("predict", help="point cloud for one image")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--image", required=True)
    pr.add_argument("--out", help="output .ply (default: next to the image)")
    pr.add_argument("--device", default="cpu")
    pr.set_defaults(func=cmd_predict)

    tb = sub.add_parser("table", help="collect all eval summaries into results_table.csv")
    tb.add_argument("--runs", default="runs")
    tb.set_defaults(func=cmd_table)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
