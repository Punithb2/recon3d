import hashlib
import pickle

import pytest
import torch

from recon3d.config import TrainConfig
from recon3d.io import save_ckpt
from recon3d.models import (
    Recon,
    ReconAttn,
    build_model,
    configure_encoder,
    decoder_params,
    load_checkpoint,
    set_modes,
)

# sha256 of "name:shape" lines of the checkpoints written by the Kaggle notebook (resnet18, 2,048 points).
# If a refactor renames or reshapes a parameter, these change and the trained weights stop loading.
NOTEBOOK_FINGERPRINT = {
    "mlp": ("4d1cc339cbbc0db2b3cd8ae76e2a1efd259e9c59f7129e0fc32b0189d5bdaf54", 126),
    "attn": ("e145eb4c6e93097fc09491471a3aa493e60d49cad0fd04b33049a8f790240b43", 207),
}


def fingerprint(sd):
    s = "\n".join(f"{k}:{tuple(v.shape)}" for k, v in sorted(sd.items()))
    return hashlib.sha256(s.encode()).hexdigest(), len(sd)


@pytest.mark.parametrize("decoder", ["mlp", "attn"])
def test_parameter_names_match_notebook_checkpoints(decoder):
    cfg = TrainConfig(decoder=decoder, pretrained=False)
    assert fingerprint(build_model(cfg).state_dict()) == NOTEBOOK_FINGERPRINT[decoder]


def test_recon_output_shape_and_embedding():
    m = Recon(pretrained=False, n_pred=128).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert m(x).shape == (2, 128, 3)
        assert m.embed(x).shape == (2, 512)


def test_attn_output_shape():
    m = ReconAttn(pretrained=False, img_res=64, n_queries=16, pts_per_query=4, d_model=32,
                  n_layers=1, n_heads=4).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 64, 64)).shape == (2, 64, 3)


def test_configure_encoder_freezes_everything_except_named_stages():
    m = Recon(pretrained=False, n_pred=16)
    params = configure_encoder(m, True, ["layer4"])
    layer4 = set(map(id, m.encoder.layer4.parameters()))
    assert {id(p) for p in params} == layer4
    for name, p in m.encoder.named_parameters():
        assert p.requires_grad == name.startswith("layer4.")
    assert all(p.requires_grad for p in decoder_params(m))
    assert configure_encoder(m, False, ["layer4"]) == []


def test_frozen_stages_keep_batchnorm_in_eval_mode():
    m = Recon(pretrained=False, n_pred=16)
    cfg = TrainConfig(train_encoder=True, unfreeze=["layer4"])
    set_modes(m, cfg)
    assert m.decoder.training and m.encoder.layer4.training
    assert not m.encoder.layer1.training and not m.encoder.bn1.training
    set_modes(m, cfg, encoder_frozen=True)
    assert not m.encoder.layer4.training


def test_frozen_batchnorm_statistics_do_not_change():
    m = Recon(pretrained=False, n_pred=16)
    cfg = TrainConfig(train_encoder=True, unfreeze=["layer4"])
    set_modes(m, cfg)
    before = m.encoder.layer1[0].bn1.running_mean.clone()
    m(torch.randn(4, 3, 64, 64))
    assert torch.equal(before, m.encoder.layer1[0].bn1.running_mean)


def test_checkpoint_round_trip_gives_identical_predictions(tmp_path):
    cfg = TrainConfig(pretrained=False, n_pred=64, img_res=64)
    m = build_model(cfg).eval()
    path = tmp_path / "best.pt"
    save_ckpt(path, dict(model=m.state_dict(), epoch=3, val_cd=1.5, cfg=cfg.to_dict()))
    m2, cfg2, meta = load_checkpoint(path)
    assert cfg2 == cfg and meta["epoch"] == 3 and not m2.training
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))


def test_notebook_style_checkpoint_loads(tmp_path):
    """Notebook checkpoints store cfg with tuples and without newer keys."""
    m = Recon(pretrained=False, n_pred=2048)
    old_cfg = {"n_pred": 2048, "n_gt": 2048, "unfreeze": ("layer3", "layer4"), "train_encoder": True,
               "backbone": "resnet18", "img_res": 224, "epochs": 40}
    torch.save(dict(model=m.state_dict(), epoch=1, val_cd=2.0, cfg=old_cfg), tmp_path / "best.pt")
    m2, cfg, _ = load_checkpoint(tmp_path / "best.pt")
    assert isinstance(m2, Recon) and cfg.unfreeze == ["layer3", "layer4"]


class Evil:
    """Stands in for a malicious object hidden inside a checkpoint."""


def test_load_checkpoint_refuses_pickled_code(tmp_path):
    """weights_only=True: a checkpoint containing arbitrary Python objects must not load."""
    torch.save(dict(model={}, cfg={}, evil=Evil()), tmp_path / "bad.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(tmp_path / "bad.pt")
