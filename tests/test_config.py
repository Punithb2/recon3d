from pathlib import Path

import pytest

from recon3d.config import TrainConfig, load_config, parse_overrides

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


def test_configs_exist():
    names = {p.stem for p in CONFIGS}
    assert {"overfit", "frozen", "finetune", "attn", "overfit_attn"} <= names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_is_valid(path):
    cfg = load_config(path)
    assert cfg.name == path.stem


def test_finetune_matches_the_notebook_run():
    cfg = load_config(CONFIGS[[p.stem for p in CONFIGS].index("finetune")])
    assert (cfg.epochs, cfg.batch_size, cfg.lr, cfg.enc_lr) == (40, 64, 3e-4, 1e-4)
    assert cfg.unfreeze == ["layer3", "layer4"] and cfg.aug and cfg.init_decoder_from == "frozen"


def test_overrides_are_typed():
    assert parse_overrides(["epochs=3", "amp=false", "unfreeze=[layer4]", "lr=1e-3"]) == {
        "epochs": 3, "amp": False, "unfreeze": ["layer4"], "lr": 1e-3}
    with pytest.raises(ValueError):
        parse_overrides(["epochs"])


def test_numbers_written_as_text_are_coerced(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("lr: 1e-3\nepochs: '5'\n")
    cfg = load_config(p)
    assert cfg.lr == 1e-3 and cfg.epochs == 5 and isinstance(cfg.epochs, int)
    p.write_text("lr: fast\n")
    with pytest.raises(ValueError, match="lr must be a number"):
        load_config(p)


def test_override_applies(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("epochs: 10\n")
    assert load_config(p, ["epochs=1"]).epochs == 1


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("epoch: 10\n")                 # typo must not be silently ignored
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


def test_old_notebook_checkpoint_config_loads():
    old = {"weight_decay": 1e-4, "warmup_epochs": 2, "grad_clip": 1.0, "amp": True, "n_pred": 2048,
           "n_gt": 2048, "epochs": 40, "batch_size": 64, "lr": 3e-4, "enc_lr": 1e-4, "views_per_mesh": 8,
           "train_encoder": True, "unfreeze": ("layer3", "layer4"), "aug": True,
           "init_decoder_from": "frozen", "backbone": "resnet18", "img_res": 224}
    cfg = TrainConfig.from_dict(old)
    assert cfg.decoder == "mlp" and cfg.unfreeze == ["layer3", "layer4"]
    cfg.validate()


@pytest.mark.parametrize("bad", [
    dict(decoder="conv"),
    dict(kind="eval"),
    dict(decoder="attn", n_pred=100),
    dict(train_encoder=True, unfreeze=[]),
    dict(batch_size=0),
])
def test_validation_catches_bad_values(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad).validate()


def test_round_trip():
    cfg = TrainConfig(name="x", unfreeze=["layer4"], train_encoder=True)
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
