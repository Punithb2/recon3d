"""Models. Parameter names match the Kaggle notebook exactly, so its checkpoints load unchanged."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from .config import ATTN_KEYS, TrainConfig


def _resnet(backbone: str, pretrained: bool) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    return getattr(torchvision.models, backbone)(weights=weights)


class Recon(nn.Module):
    """Global-vector model: ResNet -> 512-d pooled vector -> MLP -> n_pred x 3 points."""

    def __init__(self, backbone: str = "resnet18", n_pred: int = 2048, pretrained: bool = True):
        super().__init__()
        enc = _resnet(backbone, pretrained)
        self.feat_dim = enc.fc.in_features
        enc.fc = nn.Identity()
        self.encoder = enc
        self.n_pred = n_pred
        self.decoder = nn.Sequential(
            nn.Linear(self.feat_dim, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, n_pred * 3),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Global image feature [B, 512]. Also used by the retrieval baseline and the OOD check."""
        return self.encoder(x)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f.float()).view(-1, self.n_pred, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.embed(x))


class ReconAttn(nn.Module):
    """Spatial-token decoder (experiment v2, kept for reproducibility; it did not beat Recon).
    layer3 (14x14) + layer4 (7x7) feature maps become 245 tokens; 256 learned queries cross-attend
    to them and each emits a centre plus pts_per_query points within patch_radius of it."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, img_res: int = 224,
                 n_queries: int = 256, pts_per_query: int = 8, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 8, patch_radius: float = 0.1):
        super().__init__()
        enc = _resnet(backbone, pretrained)
        self.feat_dim = enc.fc.in_features
        enc.fc = nn.Identity()
        self.encoder = enc
        c3 = [m for m in enc.layer3[-1].modules() if isinstance(m, nn.BatchNorm2d)][-1].num_features
        s3, s4 = img_res // 16, img_res // 32
        self.proj3 = nn.Conv2d(c3, d_model, 1)
        self.proj4 = nn.Conv2d(self.feat_dim, d_model, 1)
        self.pos3 = nn.Parameter(torch.randn(1, s3 * s3, d_model) * 0.02)
        self.pos4 = nn.Parameter(torch.randn(1, s4 * s4, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.glob = nn.Linear(self.feat_dim, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.point_decoder = nn.TransformerDecoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.center = nn.Linear(d_model, 3)
        self.offset = nn.Linear(d_model, 3 * pts_per_query)
        self.k, self.r = pts_per_query, patch_radius
        self.n_pred = n_queries * pts_per_query
        nn.init.normal_(self.center.weight, std=0.01)
        nn.init.zeros_(self.center.bias)

    def trunk(self, x):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        f3 = e.layer3(e.layer2(e.layer1(x)))
        return f3, e.layer4(f3)

    def embed(self, x):
        return self.trunk(x)[1].mean((2, 3))

    def tokens(self, x):
        f3, f4 = self.trunk(x)
        mem = torch.cat([self.proj3(f3).flatten(2).transpose(1, 2) + self.pos3,
                         self.proj4(f4).flatten(2).transpose(1, 2) + self.pos4], dim=1)
        return mem, f4.mean((2, 3))

    def decode_tokens(self, mem, g):
        q = self.queries.expand(mem.shape[0], -1, -1) + self.glob(g).unsqueeze(1)
        h = self.norm(self.point_decoder(q, mem))
        c = self.center(h).unsqueeze(2)
        o = self.r * torch.tanh(self.offset(h)).view(h.shape[0], h.shape[1], self.k, 3)
        return (c + o).flatten(1, 2).float()

    def forward(self, x):
        return self.decode_tokens(*self.tokens(x))


def build_model(cfg: TrainConfig, device: torch.device | str = "cpu", pretrained: bool | None = None) -> nn.Module:
    """pretrained=None -> use cfg.pretrained. Pass False when a checkpoint will be loaded anyway
    (skips the ImageNet download)."""
    pretrained = cfg.pretrained if pretrained is None else pretrained
    if cfg.decoder == "attn":
        m = ReconAttn(backbone=cfg.backbone, pretrained=pretrained, img_res=cfg.img_res,
                      **{k: getattr(cfg, k) for k in ATTN_KEYS})
    else:
        m = Recon(backbone=cfg.backbone, n_pred=cfg.n_pred, pretrained=pretrained)
    m = m.to(device)
    if torch.device(device).type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    return m


def decoder_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if not n.startswith("encoder.")]


def configure_encoder(model: nn.Module, train_encoder: bool, unfreeze=()) -> list[nn.Parameter]:
    """Freeze every ResNet stage except those named in `unfreeze`. Returns the trainable encoder params."""
    params = []
    for name, module in model.encoder.named_children():
        trainable = train_encoder and name in unfreeze
        for p in module.parameters():
            p.requires_grad_(trainable)
            if trainable:
                params.append(p)
    return params


def set_modes(model: nn.Module, cfg: TrainConfig, encoder_frozen: bool = False) -> None:
    """Frozen stages stay in eval mode so their BatchNorm running statistics don't drift."""
    model.train()
    for name, module in model.encoder.named_children():
        if encoder_frozen or not (cfg.train_encoder and name in cfg.unfreeze):
            module.eval()


def count_params(params) -> int:
    return sum(p.numel() for p in params)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Load a best.pt for inference/evaluation.

    weights_only=True: torch.load otherwise unpickles arbitrary Python objects, which can run code
    from a malicious file. best.pt only contains tensors and plain types, so the safe mode works.
    Returns (model in eval mode, TrainConfig, checkpoint dict without the weights).
    """
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = TrainConfig.from_dict(ck["cfg"])
    model = build_model(cfg, device=device, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    meta = {k: v for k, v in ck.items() if k != "model"}
    return model, cfg, meta
