import numpy as np
import pytest
import torch

from recon3d.io import read_ply, save_ckpt, write_ply


def test_ply_round_trip(tmp_path):
    pts = np.random.default_rng(0).uniform(-0.5, 0.5, (2048, 3)).astype(np.float32)
    data = write_ply(pts, tmp_path / "x.ply")
    assert data.startswith(b"ply\n")
    np.testing.assert_allclose(read_ply(tmp_path / "x.ply"), pts, atol=1e-5)
    np.testing.assert_allclose(read_ply(data), pts, atol=1e-5)


def test_ply_rejects_bad_shape():
    with pytest.raises(ValueError):
        write_ply(np.zeros((10, 2)))


def test_save_ckpt_is_atomic(tmp_path):
    p = tmp_path / "last.pt"
    save_ckpt(p, {"a": torch.ones(2)})
    save_ckpt(p, {"a": torch.zeros(2)})
    assert torch.equal(torch.load(p)["a"], torch.zeros(2))
    assert [f.name for f in tmp_path.iterdir()] == ["last.pt"]      # no leftover .tmp file
