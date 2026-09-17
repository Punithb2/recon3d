import numpy as np
import pytest
from PIL import Image

from recon3d.cli import main
from recon3d.inference import Predictor, load_silhouette
from recon3d.io import read_ply

TINY = ["--set", "pretrained=false", "--set", "img_res=64", "--set", "n_pred=128", "--set", "n_gt=128",
        "--set", "epochs=1", "--set", "batch_size=16", "--set", "views_per_mesh=2", "--set", "amp=false"]


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("train", "evaluate", "predict", "table"):
        assert cmd in out


def test_bad_config_fails_before_loading_data(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("epoch: 3\n")
    with pytest.raises(ValueError):
        main(["train", "--data", str(tmp_path / "missing"), str(bad)])


@pytest.mark.slow
def test_train_evaluate_predict_table(dataset, tmp_path, capsys):
    runs = tmp_path / "runs"
    cfg = tmp_path / "frozen.yaml"
    cfg.write_text("lr: 1.0e-3\n")
    assert main(["train", "--data", str(dataset), "--runs", str(runs), "--device", "cpu",
                 "--eval-splits", "test", str(cfg), *TINY]) == 0
    run_dir = runs / "frozen_resnet18_n6"
    assert (run_dir / "eval_test_summary.json").exists()

    assert main(["evaluate", "--data", str(dataset), "--runs", str(runs), "--device", "cpu",
                 "--run", run_dir.name, "--splits", "val"]) == 0
    assert (run_dir / "eval_val_summary.json").exists()

    img = tmp_path / "shape.png"
    arr = np.full((100, 100), 255, np.uint8)            # dark object on white paper
    arr[30:70, 20:80] = 0
    Image.fromarray(arr).save(img)
    assert main(["predict", "--checkpoint", str(run_dir / "best.pt"), "--image", str(img)]) == 0
    pts = read_ply(img.with_suffix(".ply"))
    assert pts.shape == (128, 3) and np.isfinite(pts).all()

    assert main(["table", "--runs", str(runs)]) == 0
    assert (tmp_path / "results_table.csv").exists()


def test_load_silhouette_normalises_polarity_and_alpha(tmp_path):
    white_page = np.full((50, 50), 255, np.uint8)
    white_page[10:40, 10:40] = 0
    s = load_silhouette(Image.fromarray(white_page), size=64)
    assert s.shape == (64, 64) and s.dtype == np.uint8
    assert s[32, 32] == 255 and s[0, 0] == 0                 # inverted: white object, black background

    rgba = np.zeros((50, 50, 4), np.uint8)
    rgba[10:40, 10:40, 3] = 255                              # transparent PNG: alpha is the mask
    s = load_silhouette(Image.fromarray(rgba, "RGBA"), size=64)
    assert s[32, 32] == 255 and s[0, 0] == 0


@pytest.mark.slow
def test_predictor_batch_and_embedding(dataset, tmp_path):
    from recon3d.config import TrainConfig
    from recon3d.io import save_ckpt
    from recon3d.models import build_model
    cfg = TrainConfig(pretrained=False, img_res=64, n_pred=128)
    save_ckpt(tmp_path / "best.pt", dict(model=build_model(cfg).state_dict(), epoch=0, val_cd=0.0,
                                         cfg=cfg.to_dict()))
    p = Predictor(tmp_path / "best.pt")
    sil = np.load(dataset / "sil_test.npy")[:3, 0]
    assert p.predict(sil).shape == (3, 128, 3)
    assert p.predict(sil[0]).shape == (1, 128, 3)
    assert p.embed(sil).shape == (3, 512)
