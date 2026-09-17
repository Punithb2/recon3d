import numpy as np
import pytest
import torch

from recon3d.data import IMAGENET_MEAN, IMAGENET_STD, DatasetMeta, Split, augment, find_dataset, preprocess


def test_meta_and_run_tag(dataset):
    meta = DatasetMeta.load(dataset)
    assert meta.categories == ["airplane", "chair"]
    assert len(meta.views) == 24
    assert meta.run_tag == "n6"
    assert meta.view_index([45, 30]) == meta.views.index((45, 30))


def test_find_dataset(dataset):
    assert find_dataset(dataset.parent) == dataset
    with pytest.raises(FileNotFoundError):
        find_dataset(dataset / "nothing_here")


def test_split_loading_and_ground_truth(dataset):
    sp = Split(dataset, "train")
    assert (sp.N, sp.V) == (12, 24)
    assert sp.images([0, 1], [0, 5]).shape == (2, 32, 32)
    fixed = sp.gt([0, 1], n=100)
    assert torch.equal(fixed, sp.points[[0, 1], :100])           # evaluation: always the same points
    rnd = sp.gt([0], n=100, random_subset=True)
    assert rnd.shape == (1, 100, 3)
    all_pts = {tuple(p) for p in sp.points[0].tolist()}
    assert all(tuple(p) in all_pts for p in rnd[0].tolist())      # subset of the cached samples
    assert not torch.equal(rnd, fixed[:1])


def test_split_rejects_unknown_name(dataset):
    with pytest.raises(ValueError):
        Split(dataset, "holdout")


def test_split_detects_inconsistent_files(dataset, tmp_path):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(dataset, bad)
    np.save(bad / "points_val.npy", np.load(bad / "points_val.npy")[:-1])
    with pytest.raises(ValueError, match="disagree"):
        Split(bad, "val")


def test_preprocess_shape_and_normalisation():
    x = torch.zeros(2, 32, 32, dtype=torch.uint8)
    x[1] = 255
    y = preprocess(x, img_res=64)
    assert y.shape == (2, 3, 64, 64) and y.dtype == torch.float32
    for c in range(3):
        assert y[0, c].mean().item() == pytest.approx(-IMAGENET_MEAN[c] / IMAGENET_STD[c], abs=1e-5)
        assert y[1, c].mean().item() == pytest.approx((1 - IMAGENET_MEAN[c]) / IMAGENET_STD[c], abs=1e-5)


def test_preprocess_rejects_wrong_input():
    with pytest.raises(ValueError):
        preprocess(torch.zeros(1, 32, 32))                  # float instead of uint8
    with pytest.raises(ValueError):
        preprocess(torch.zeros(1, 1, 32, 32, dtype=torch.uint8))


def test_augment_keeps_shape_and_roughly_keeps_area():
    torch.manual_seed(0)
    x = torch.zeros(8, 1, 64, 64)
    x[:, :, 24:40, 24:40] = 1
    y = augment(x)
    assert y.shape == x.shape
    ratio = y.sum((1, 2, 3)) / x.sum((1, 2, 3))
    assert ((ratio > 0.7) & (ratio < 1.3)).all()             # scale 0.9-1.1 -> area 0.81-1.21
    assert not torch.equal(y, x)
