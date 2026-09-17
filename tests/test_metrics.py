import pytest
import torch

from recon3d.metrics import chamfer, fscore, precision_recall, spread_ratio


def brute_force(a, b):
    d = torch.cdist(a, b).pow(2)
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)


@pytest.mark.parametrize("chunk", [1, 2, 32])
def test_chamfer_matches_brute_force_value_and_gradient(chunk):
    torch.manual_seed(0)
    a = torch.randn(3, 50, 3, requires_grad=True)
    b = torch.randn(3, 60, 3)
    cd, d_pg, d_gp = chamfer(a, b, chunk=chunk)
    ref = brute_force(a, b)
    assert torch.allclose(cd, ref, atol=1e-5)
    assert d_pg.shape == (3, 50) and d_gp.shape == (3, 60)
    g1 = torch.autograd.grad(cd.sum(), a)[0]
    g2 = torch.autograd.grad(ref.sum(), a)[0]
    assert torch.allclose(g1, g2, atol=1e-5)


def test_chamfer_is_zero_for_identical_and_symmetric():
    torch.manual_seed(1)
    a, b = torch.randn(2, 40, 3), torch.randn(2, 40, 3)
    assert torch.allclose(chamfer(a, a)[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(chamfer(a, b)[0], chamfer(b, a)[0], atol=1e-5)


def test_chamfer_handles_half_precision_input():
    a = torch.randn(1, 20, 3).half()
    cd = chamfer(a, a.float())[0]
    assert cd.dtype == torch.float32


def test_chamfer_rejects_bad_shapes():
    with pytest.raises(ValueError):
        chamfer(torch.randn(2, 10, 2), torch.randn(2, 10, 3))
    with pytest.raises(ValueError):
        chamfer(torch.randn(2, 10, 3), torch.randn(3, 10, 3))


def test_fscore_extremes():
    a = torch.rand(2, 100, 3)
    _, d_pg, d_gp = chamfer(a, a)
    assert torch.allclose(fscore(d_pg, d_gp, 0.01), torch.ones(2))
    _, d_pg, d_gp = chamfer(a, a + 5)
    assert torch.allclose(fscore(d_pg, d_gp, 0.01), torch.zeros(2))


def test_precision_recall_separate_failure_modes():
    """Prediction covers only half the object: precision stays 1, recall drops to ~0.5."""
    gt = torch.cat([torch.rand(1, 500, 3) * 0.1, torch.rand(1, 500, 3) * 0.1 + 1], dim=1)
    pred = gt[:, :500]
    _, d_pg, d_gp = chamfer(pred, gt)
    prec, rec = precision_recall(d_pg, d_gp, 0.05)
    assert prec.item() == 1.0
    assert rec.item() == pytest.approx(0.5, abs=0.01)


def test_spread_ratio_detects_shrinking():
    gt = torch.randn(1, 200, 3)
    assert spread_ratio(gt * 0.5, gt).item() == pytest.approx(0.5, rel=1e-4)
