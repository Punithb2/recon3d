"""Experiment configuration.

One dataclass holds every knob. YAML files in configs/ pick values for each experiment, and
`--set key=value` on the command line overrides them. The resolved config is saved inside every
checkpoint, so a checkpoint always knows how to rebuild its own model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ATTN_KEYS = ("n_queries", "pts_per_query", "d_model", "n_layers", "n_heads", "patch_radius")


@dataclass
class TrainConfig:
    name: str = "finetune"                 # experiment name, used in the run folder name
    kind: str = "train"                    # "train" or "overfit" (sanity check)

    # model
    backbone: str = "resnet18"
    pretrained: bool = True                # ImageNet weights for the encoder
    decoder: str = "mlp"                   # "mlp" (global vector) or "attn" (spatial tokens)
    img_res: int = 224
    n_pred: int = 2048                     # points the model outputs
    n_gt: int = 2048                       # ground-truth points compared per step
    n_queries: int = 256                   # attn decoder only
    pts_per_query: int = 8
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    patch_radius: float = 0.1

    # optimisation
    epochs: int = 40
    steps: int = 600                       # overfit check only
    batch_size: int = 64
    lr: float = 3e-4                       # decoder learning rate
    enc_lr: float = 1e-4                   # encoder learning rate (when trained)
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    amp: bool = True                       # mixed precision (GPU only)
    views_per_mesh: int = 8                # views sampled per mesh per epoch
    train_encoder: bool = False
    unfreeze: list[str] = field(default_factory=list)   # ResNet stages to train, e.g. [layer3, layer4]
    freeze_encoder_epochs: int = 0         # keep the encoder frozen for the first N epochs
    aug: bool = False                      # random scale/shift of the silhouette
    init_decoder_from: str | None = None   # experiment name whose best decoder initialises this one

    # evaluation
    val_views: list[list[int]] = field(default_factory=lambda: [[45, 30], [135, 0]])   # (azimuth, elevation)
    fscore_taus: list[float] = field(default_factory=lambda: [0.01, 0.02])
    seed: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any], **extra: Any) -> TrainConfig:
        """Build from a dict, ignoring unknown keys. Old notebook checkpoints lack some keys
        (name, decoder, ...), so missing keys fall back to the defaults above."""
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in {**d, **extra}.items() if k in known}
        for k in ("unfreeze",):                     # tuples in old checkpoints -> lists
            if k in data and data[k] is not None:
                data[k] = list(data[k])
        defaults = cls()
        for k, v in data.items():                   # YAML reads "1e-3" as a string: coerce numbers
            d = getattr(defaults, k)
            if isinstance(v, str) and isinstance(d, (int, float)) and not isinstance(d, bool):
                try:
                    data[k] = type(d)(float(v)) if isinstance(d, int) and float(v).is_integer() else float(v)
                except ValueError as e:
                    raise ValueError(f"{k} must be a number, got {v!r}") from e
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> TrainConfig:
        return dataclasses.replace(self, **changes)

    def validate(self) -> TrainConfig:
        if self.kind not in ("train", "overfit"):
            raise ValueError(f"kind must be 'train' or 'overfit', got {self.kind!r}")
        if self.decoder not in ("mlp", "attn"):
            raise ValueError(f"decoder must be 'mlp' or 'attn', got {self.decoder!r}")
        if self.decoder == "attn" and self.n_pred != self.n_queries * self.pts_per_query:
            raise ValueError("attn decoder: n_pred must equal n_queries * pts_per_query")
        if self.decoder == "attn" and self.img_res % 32:
            raise ValueError("attn decoder: img_res must be divisible by 32")
        if self.train_encoder and not self.unfreeze:
            raise ValueError("train_encoder=True needs at least one stage in unfreeze")
        if self.batch_size <= 0 or self.epochs < 0:
            raise ValueError("batch_size must be > 0 and epochs >= 0")
        return self


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """["epochs=1", "unfreeze=[layer4]"] -> {"epochs": 1, "unfreeze": ["layer4"]} (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override must look like key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):                 # YAML 1.1 treats "1e-3" as text
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        out[key.strip()] = parsed
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.setdefault("name", Path(path).stem)
    raw.update(parse_overrides(overrides))
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    return TrainConfig.from_dict(raw).validate()
