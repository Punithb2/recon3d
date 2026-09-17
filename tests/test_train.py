"""End-to-end training on the tiny synthetic dataset. Slow-ish (seconds each) but catches real bugs:
shape mismatches, resume logic, checkpoint contents, evaluation outputs."""

import json

import numpy as np
import pytest
import torch

from recon3d.evaluate import evaluate, results_table
from recon3d.io import save_ckpt
from recon3d.tracking import NullTracker
from recon3d.train import OverfitError, cosine_schedule, is_finished, run_overfit, train

pytestmark = pytest.mark.slow


class RecordingTracker(NullTracker):
    def __init__(self):
        self.events = []

    def start_run(self, run_name, params, tags=None):
        self.events.append(("start", run_name))

    def log_metrics(self, metrics, step=None):
        self.events.append(("metrics", step, metrics))

    def end_run(self, status="FINISHED"):
        self.events.append(("end", status))


def test_cosine_schedule_shape():
    f = cosine_schedule(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(100) == pytest.approx(0.02)
    assert all(f(s) >= f(s + 1) for s in range(10, 100))


def test_overfit_check_passes(session, tiny_cfg):
    hist = run_overfit(tiny_cfg.replace(kind="overfit", steps=150, batch_size=8), session)
    assert hist[-1] < 0.1 * hist[0]


def test_overfit_check_fails_loudly_when_nothing_can_be_learned(session, tiny_cfg):
    with pytest.raises(OverfitError):
        run_overfit(tiny_cfg.replace(kind="overfit", steps=3, lr=1e-9), session)


def test_train_frozen_writes_checkpoints_and_history(session, tiny_cfg):
    tracker = RecordingTracker()
    session.tracker = tracker
    run_dir = train(tiny_cfg, session)
    assert run_dir.name == "frozen_resnet18_n6"
    for f in ("best.pt", "last.pt", "history.csv"):
        assert (run_dir / f).exists()
    assert is_finished(run_dir)
    best = torch.load(run_dir / "best.pt", weights_only=True)
    assert set(best) == {"model", "epoch", "val_cd", "cfg"}
    assert best["cfg"]["name"] == "frozen"
    assert [e[0] for e in tracker.events] == ["start", "metrics", "metrics", "end"]
    assert tracker.events[-1] == ("end", "FINISHED")

    # a finished run is skipped, not retrained
    mtime = (run_dir / "last.pt").stat().st_mtime
    train(tiny_cfg, session)
    assert (run_dir / "last.pt").stat().st_mtime == mtime


def test_train_resumes_from_last_checkpoint(session, tiny_cfg, capsys):
    run_dir = train(tiny_cfg, session)
    ck = torch.load(run_dir / "last.pt", weights_only=False)
    ck.update(finished=False, epoch=0, history=ck["history"][:1])     # pretend we stopped after epoch 1
    save_ckpt(run_dir / "last.pt", ck)
    train(tiny_cfg, session)
    assert "resumed from epoch 1" in capsys.readouterr().out
    assert is_finished(run_dir)
    assert len(torch.load(run_dir / "last.pt", weights_only=False)["history"]) == 2


def test_finetune_initialises_decoder_from_frozen_run(session, tiny_cfg, capsys):
    train(tiny_cfg, session)
    ft = tiny_cfg.replace(name="finetune", train_encoder=True, unfreeze=["layer4"], aug=True,
                          init_decoder_from="frozen", epochs=1, batch_size=8, lr=3e-4, enc_lr=1e-4)
    run_dir = train(ft, session)
    assert "decoder initialised from" in capsys.readouterr().out
    assert is_finished(run_dir)


def test_attn_decoder_trains(session, tiny_cfg):
    cfg = tiny_cfg.replace(name="attn", decoder="attn", n_queries=16, pts_per_query=8, n_pred=128,
                           d_model=32, n_layers=1, n_heads=4, train_encoder=True, unfreeze=["layer4"],
                           freeze_encoder_epochs=1, epochs=2, batch_size=8)
    assert is_finished(train(cfg, session))


def test_evaluate_writes_notebook_compatible_outputs(session, tiny_cfg):
    run_dir = train(tiny_cfg, session)
    summary = evaluate(run_dir, "test", session)
    assert set(summary["overall"]) == {"model", "retrieval"}
    assert set(summary["overall"]["model"]) == {"cd_x1000", "f@0.01", "f@0.02"}
    assert summary["n_predictions"] == 4 * 24                    # 4 test meshes x 24 views
    assert set(summary["per_category"]) == {"airplane", "chair"}
    saved = json.loads((run_dir / "eval_test_summary.json").read_text())
    assert saved["run"] == run_dir.name
    import pandas as pd
    df = pd.read_csv(run_dir / "eval_test_per_sample.csv.gz")
    assert len(df) == 2 * 96 and np.isfinite(df.cd_x1000).all()
    table = results_table(session.runs_dir)
    assert list(table.method) == ["model", "retrieval"]
