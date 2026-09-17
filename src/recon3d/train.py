"""Training: the overfit sanity check and the main training loop (same logic as the Kaggle notebook)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import TrainConfig
from .data import DatasetMeta, Split, preprocess
from .io import save_ckpt
from .metrics import chamfer
from .models import build_model, configure_encoder, count_params, decoder_params, set_modes
from .tracking import NullTracker, Tracker


class OverfitError(RuntimeError):
    """The pipeline could not fit 16 samples: something is broken, do not start real training."""


@dataclass
class Session:
    """Everything shared by the experiments of one session (one Kaggle run)."""

    data_root: Path
    runs_dir: Path
    device: torch.device
    time_limit_h: float = 8.0
    tracker: Tracker = field(default_factory=NullTracker)
    t0: float = field(default_factory=time.time)
    splits: dict[str, Split] = field(default_factory=dict)

    def __post_init__(self):
        self.data_root, self.runs_dir = Path(self.data_root), Path(self.runs_dir)
        self.device = torch.device(self.device)
        self.meta = DatasetMeta.load(self.data_root)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def split(self, name: str) -> Split:
        if name not in self.splits:                     # load lazily, once
            self.splits[name] = Split(self.data_root, name, self.device)
        return self.splits[name]

    def time_left_h(self) -> float:
        return self.time_limit_h - (time.time() - self.t0) / 3600

    def run_dir(self, cfg: TrainConfig, name: str | None = None) -> Path:
        return self.runs_dir / f"{name or cfg.name}_{cfg.backbone}_{self.meta.run_tag}"

    def autocast(self, enabled: bool = True):
        on = enabled and self.device.type == "cuda"
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=on)


# ---------------------------------------------------------------- utilities
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device: torch.device) -> dict:
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def set_rng_state(s: dict, device: torch.device) -> None:
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def cosine_schedule(total_steps: int, warmup_steps: int):
    """Linear warm-up, then cosine decay down to 2% of the base learning rate."""
    warmup_steps = max(1, warmup_steps)

    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


@torch.no_grad()
def encode_split(model, split: Split, img_res: int, session: Session, meshes_per_chunk: int = 16) -> torch.Tensor:
    """Encoder features for every (mesh, view): [N, V, D] float16."""
    model.eval()
    feats = torch.empty(split.N, split.V, model.feat_dim, dtype=torch.float16, device=session.device)
    for m in range(0, split.N, meshes_per_chunk):
        idx = np.arange(m, min(m + meshes_per_chunk, split.N))
        x = torch.from_numpy(np.ascontiguousarray(split.sil[idx])).to(session.device)
        with session.autocast():
            f = model.embed(preprocess(x.flatten(0, 1), img_res))
        feats[idx] = f.view(len(idx), split.V, -1).half()
    return feats


@torch.no_grad()
def validate(model, cfg: TrainConfig, session: Session, val_feats=None) -> float:
    """Mean Chamfer x1000 over every val mesh at the fixed validation views."""
    model.eval()
    sp = session.split("val")
    total, n = 0.0, 0
    for v in [session.meta.view_index(av) for av in cfg.val_views]:
        for s in range(0, sp.N, 256):
            mi = np.arange(s, min(s + 256, sp.N))
            vi = np.full(len(mi), v)
            with session.autocast(cfg.amp):
                pred = model.decode(val_feats[mi, vi]) if val_feats is not None \
                    else model(preprocess(sp.images(mi, vi), cfg.img_res))
            cd, _, _ = chamfer(pred, sp.gt(mi, cfg.n_gt))
            total += cd.sum().item()
            n += len(mi)
    return 1000 * total / n


# ---------------------------------------------------------------- experiments
def run_overfit(cfg: TrainConfig, session: Session) -> list[float]:
    """16 fixed (mesh, view) pairs, fixed targets, frozen encoder. Chamfer must fall below 10% of its
    starting value; otherwise raise OverfitError before any GPU hours are spent."""
    seed_all(cfg.seed)
    model = build_model(cfg, session.device)
    configure_encoder(model, False)
    tr = session.split("train")
    rng = np.random.default_rng(cfg.seed)
    bs = min(cfg.batch_size, tr.N)
    mi = rng.choice(tr.N, bs, replace=False)
    vi = rng.integers(0, tr.V, bs)
    with torch.no_grad():
        model.eval()
        x = preprocess(tr.images(mi, vi), cfg.img_res)
        if cfg.decoder == "attn":
            mem, g = (t.float() for t in model.tokens(x))

            def decode():
                return model.decode_tokens(mem, g)
        else:
            f = model.embed(x).float()

            def decode():
                return model.decode(f)
    model.train()
    gt = tr.gt(mi, cfg.n_gt)
    opt = torch.optim.Adam(decoder_params(model), lr=cfg.lr)
    hist = []
    for step in range(cfg.steps):
        loss = chamfer(decode(), gt)[0].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(1000 * loss.item())
        if step % 100 == 0 or step == cfg.steps - 1:
            print(f"  overfit[{cfg.decoder}] step {step:4d}  CDx1000 {hist[-1]:.3f}")
    ok = hist[-1] < 0.1 * hist[0]
    print(f"OVERFIT[{cfg.decoder}] {'PASSED' if ok else 'FAILED'}: {hist[0]:.2f} -> {hist[-1]:.3f}")
    if not ok:
        raise OverfitError("the pipeline cannot fit 16 samples; stop and debug")
    return hist


def train(cfg: TrainConfig, session: Session) -> Path:
    """Train one experiment. Resumable: re-running continues from last.pt; a finished run is skipped.
    Writes best.pt (lowest val Chamfer), last.pt (full state), history.csv."""
    cfg.validate()
    run_dir = session.run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_p, best_p = run_dir / "last.pt", run_dir / "best.pt"
    dev = session.device

    seed_all(cfg.seed)
    model = build_model(cfg, dev)
    enc_params = configure_encoder(model, cfg.train_encoder, cfg.unfreeze)
    groups = [{"params": decoder_params(model), "lr": cfg.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    trainable = [p for g in groups for p in g["params"]]

    tr = session.split("train")
    steps_per_epoch = max(1, tr.N * cfg.views_per_mesh // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_schedule(total_steps, steps_per_epoch * cfg.warmup_epochs))
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda" and cfg.amp)

    start_epoch, best, history = 0, float("inf"), []
    if last_p.exists():
        ck = torch.load(last_p, map_location=dev, weights_only=False)   # our own file: contains RNG state
        if ck.get("finished"):
            print(f"[{cfg.name}] already finished (best val CDx1000 {ck['best']:.3f}), skipping")
            return run_dir
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        set_rng_state(ck["rng"], dev)
        print(f"[{cfg.name}] resumed from epoch {start_epoch}")
    elif cfg.init_decoder_from:
        src = session.run_dir(cfg, cfg.init_decoder_from) / "best.pt"
        if src.exists():
            sd = torch.load(src, map_location=dev, weights_only=True)["model"]
            model.decoder.load_state_dict({k[len("decoder."):]: v for k, v in sd.items()
                                           if k.startswith("decoder.")})
            print(f"[{cfg.name}] decoder initialised from {src}")
        else:
            print(f"[{cfg.name}] WARNING: {src} not found, decoder starts from scratch")

    feats = {}
    if not cfg.train_encoder:          # frozen encoder: compute features once, train only the decoder
        t0 = time.time()
        feats = {s: encode_split(model, session.split(s), cfg.img_res, session) for s in ("train", "val")}
        print(f"[{cfg.name}] cached encoder features in {time.time() - t0:.0f}s")

    session.tracker.start_run(run_dir.name, cfg.to_dict(), tags={"dataset": session.meta.run_tag})
    print(f"[{cfg.name}] decoder={cfg.decoder} ({count_params(decoder_params(model)) / 1e6:.2f}M params) | "
          f"{cfg.epochs} epochs x {steps_per_epoch} steps (batch {cfg.batch_size})")
    epoch_times: list[float] = []
    status = "FINISHED"
    try:
        for epoch in range(start_epoch, cfg.epochs):
            if epoch_times and session.time_left_h() * 3600 < 1.5 * np.mean(epoch_times) + 300:
                print(f"[{cfg.name}] stopping early: session time limit close. Re-run to resume.")
                status = "KILLED"
                break
            t0 = time.time()
            enc_frozen = epoch < cfg.freeze_encoder_epochs
            for p in enc_params:
                p.requires_grad_(not enc_frozen)
            set_modes(model, cfg, encoder_frozen=enc_frozen)
            mesh_ids = np.repeat(np.arange(tr.N), cfg.views_per_mesh)
            view_ids = np.concatenate([np.random.permutation(tr.V)[:cfg.views_per_mesh] for _ in range(tr.N)])
            order = np.random.permutation(len(mesh_ids))
            run_loss = 0.0
            for step in range(steps_per_epoch):
                b = order[step * cfg.batch_size:(step + 1) * cfg.batch_size]
                mi, vi = mesh_ids[b], view_ids[b]
                gt = tr.gt(mi, cfg.n_gt, random_subset=True)
                with session.autocast(cfg.amp):
                    pred = model.decode(feats["train"][mi, vi]) if feats \
                        else model(preprocess(tr.images(mi, vi), cfg.img_res, aug=cfg.aug))
                loss = chamfer(pred, gt)[0].mean()          # always float32
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += loss.item()
                if not math.isfinite(run_loss):
                    raise FloatingPointError(f"[{cfg.name}] loss became {run_loss} at epoch {epoch} step {step}")
            val_cd = validate(model, cfg, session, feats.get("val"))
            dt = time.time() - t0
            epoch_times.append(dt)
            row = dict(epoch=epoch, train_cd=1000 * run_loss / steps_per_epoch, val_cd=val_cd,
                       lr=sched.get_last_lr()[0], sec=dt)
            history.append(row)
            session.tracker.log_metrics({k: row[k] for k in ("train_cd", "val_cd", "lr")}, step=epoch)
            improved = val_cd < best
            if improved:
                best = val_cd
                save_ckpt(best_p, dict(model=model.state_dict(), epoch=epoch, val_cd=val_cd, cfg=cfg.to_dict()))
            save_ckpt(last_p, dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                   scaler=scaler.state_dict(), epoch=epoch, best=best, history=history,
                                   rng=rng_state(dev), cfg=cfg.to_dict(), finished=epoch == cfg.epochs - 1))
            print(f"[{cfg.name}] epoch {epoch + 1:3d}/{cfg.epochs}  train {row['train_cd']:.3f}  "
                  f"val {val_cd:.3f}{' *' if improved else ''}  ({dt:.0f}s, {session.time_left_h():.2f} h left)")
    except BaseException:
        status = "FAILED"
        raise
    finally:
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        session.tracker.end_run(status)

    if history:
        from .plots import plot_curves
        plot_curves(history, run_dir / "curves.png", title=cfg.name)
    return run_dir


def is_finished(run_dir: Path) -> bool:
    p = Path(run_dir) / "last.pt"
    return p.exists() and bool(torch.load(p, map_location="cpu", weights_only=False).get("finished"))
