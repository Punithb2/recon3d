import json
import zipfile

import numpy as np
import pytest

from recon3d.framing import bbox_stats, framing_stats, run


def test_bbox_stats():
    s = np.zeros((100, 100), np.uint8)
    s[10:30, 40:90] = 255
    st = bbox_stats(s)
    assert st["fill"] == pytest.approx(0.5)
    assert st["cx"] == pytest.approx(0.15) and st["cy"] == pytest.approx(-0.3)
    assert bbox_stats(np.zeros((5, 5), np.uint8)) is None


def test_framing_run_writes_stats_and_examples(dataset, unseen_dataset, tmp_path):
    z = run(dataset, tmp_path / "out", unseen=unseen_dataset, per_category=2)
    stats = json.loads((tmp_path / "out" / "framing.json").read_text())
    f = stats["fill"]
    assert 0 < f["p5"] <= f["p50"] <= f["p95"] <= 1
    assert stats["images"] == 4 * 24
    assert set(stats["fill_by_category"]) == {"airplane", "chair"}
    assert stats["examples_seen"] == ["airplane_1.png", "airplane_2.png", "chair_1.png", "chair_2.png"]
    assert stats["examples_unseen"] == ["unseen_lamp_1.png", "unseen_sofa_1.png"]
    names = zipfile.ZipFile(z).namelist()
    assert "framing.json" in names and "examples/unseen_lamp_1.png" in names and len(names) == 7


def test_framing_cli(dataset, tmp_path, capsys):
    from recon3d.cli import main
    assert main(["framing", "--data", str(dataset), "--out", str(tmp_path / "f"), "--max-meshes", "2"]) == 0
    out = capsys.readouterr().out
    assert "recommended RECON3D_FRAME_FILL" in out and "framing_bundle.zip" in out


def test_framing_stats_limit(dataset):
    assert framing_stats(dataset, "val", max_meshes=1)["images"] == 24
