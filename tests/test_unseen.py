import json

import numpy as np
import pandas as pd
import pytest

from recon3d.ood import OODDetector
from recon3d.train import train
from recon3d.unseen import bootstrap_ci, evaluate_unseen


def test_bootstrap_ci_contains_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, large = rng.normal(5, 1, 20), rng.normal(5, 1, 2000)
    lo, hi = bootstrap_ci(small)
    assert lo < small.mean() < hi
    lo2, hi2 = bootstrap_ci(large)
    assert (hi2 - lo2) < (hi - lo)
    assert np.isnan(bootstrap_ci([1.0])[0])


@pytest.mark.slow
def test_evaluate_unseen_end_to_end(session, tiny_cfg, unseen_dataset):
    run_dir = train(tiny_cfg, session)
    s = evaluate_unseen(run_dir, session, unseen_dataset)
    out = run_dir / "unseen"
    for f in ("per_sample.csv.gz", "per_category.csv", "summary.json", "findings.md", "ood_detector.npz"):
        assert (out / f).exists(), f

    assert set(s["overall"]) == {"seen", "unseen"}
    assert s["overall"]["unseen"]["meshes"] == 2 * (3 + 2)         # 2 categories x (test + val)
    assert s["overall"]["seen"]["meshes"] == 4
    assert 0.0 <= s["ood"]["auroc"] <= 1.0
    assert set(s["ood"]["auroc_per_category"]) == {"lamp", "sofa"}
    assert s["ood"]["bank"]["rows"] == 12 * 8                        # 12 train meshes x 8 bank views

    df = pd.read_csv(out / "per_sample.csv.gz")
    assert len(df) == (4 + 10) * 24
    assert set(df.retrieved_category) <= {"airplane", "chair"}     # retrieval can only return training shapes
    assert (df.score >= -1e-4).all() and (df.score <= 2 + 1e-4).all()
    assert ((df.score > s["ood"]["threshold"]) == df.flagged).all()

    det = OODDetector.load(out / "ood_detector.npz", checkpoint=run_dir / "best.pt")
    assert det.bank.shape == (96, 512)
    assert json.loads((out / "summary.json").read_text())["checkpoint_sha256"] == det.model_sha256
    assert "Unseen Chamfer is" in (out / "findings.md").read_text()


@pytest.mark.slow
def test_unseen_rejects_overlapping_categories(session, tiny_cfg, dataset):
    run_dir = train(tiny_cfg, session)
    with pytest.raises(ValueError, match="training categories"):
        evaluate_unseen(run_dir, session, dataset)                  # the seen set is not 'unseen'
