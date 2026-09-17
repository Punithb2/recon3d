import numpy as np
import pytest
import torch

from recon3d.ood import OODDetector, auroc, file_sha256, ood_scores, spearman, threshold_at


def test_ood_score_is_zero_for_bank_members_and_grows_with_distance():
    bank = torch.eye(4)
    q = torch.tensor([[1.0, 0, 0, 0], [1.0, 1.0, 0, 0], [-1.0, 0, 0, 0]])
    s = ood_scores(q, bank)
    assert s[0] == pytest.approx(0.0, abs=1e-6)
    assert s[1] == pytest.approx(1 - 1 / np.sqrt(2), abs=1e-6)
    assert s[2] == pytest.approx(1.0, abs=1e-6)          # opposite of e0, orthogonal to the rest
    assert ood_scores(q * 10, bank * 3).allclose(s)      # scale does not matter (cosine)


def test_ood_scores_chunking_matches():
    torch.manual_seed(0)
    q, b = torch.randn(50, 8), torch.randn(30, 8)
    assert torch.allclose(ood_scores(q, b, chunk=7), ood_scores(q, b, chunk=1000))


def test_auroc_extremes_and_ties():
    assert auroc([0, 1, 2], [3, 4]) == 1.0
    assert auroc([3, 4], [0, 1, 2]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([0, 2], [1, 3]) == 0.75


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 40), rng.integers(1, 7, 30)          # many ties on purpose
    pairwise = np.mean([(y > x) + 0.5 * (y == x) for x in a for y in b])
    assert auroc(a, b) == pytest.approx(pairwise)
    with pytest.raises(ValueError):
        auroc([], [1])


def test_threshold_keeps_the_requested_share():
    s = np.arange(1000) / 1000
    t = threshold_at(s, 0.95)
    assert (s <= t).mean() == pytest.approx(0.95, abs=0.002)


def test_spearman():
    x = np.arange(20)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_detector_round_trip_and_checkpoint_binding(tmp_path):
    ck = tmp_path / "best.pt"
    ck.write_bytes(b"weights v1")
    bank = np.eye(8, dtype=np.float32)[:4]               # training features span only the first 4 axes
    det = OODDetector(bank=bank.astype(np.float16), threshold=0.3, model_sha256=file_sha256(ck), info={"a": 1})
    det.save(tmp_path / "ood.npz")
    loaded = OODDetector.load(tmp_path / "ood.npz", checkpoint=ck)
    assert loaded.threshold == 0.3 and loaded.info == {"a": 1} and loaded.bank.dtype == np.float16
    flags = loaded.is_ood(np.stack([bank[0], np.eye(8, dtype=np.float32)[6]]))   # familiar, unfamiliar
    assert flags.tolist() == [False, True]

    ck.write_bytes(b"weights v2")                        # different model -> refuse the stale bank
    with pytest.raises(ValueError, match="different checkpoint"):
        OODDetector.load(tmp_path / "ood.npz", checkpoint=ck)
