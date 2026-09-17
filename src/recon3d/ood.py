"""Out-of-distribution (OOD) check: "does this image look like anything the model was trained on?"

Method: nearest-neighbour similarity in the encoder's feature space.
    score(x) = 1 - max_i cos(f(x), f(train_i))        higher = less familiar
An input is flagged when its score is above a threshold chosen so that 95% of *seen-category
validation* images pass. The threshold never looks at unseen categories: at serving time we
don't know which unfamiliar objects people will upload.

Why this method: no extra training, one matrix multiply per request, and the same features the
retrieval baseline already uses. Alternatives (Mahalanobis distance, a trained classifier's
confidence, ensembles) are discussed in docs/block_ood.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def ood_scores(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """1 - max cosine similarity of each query to the bank. Both inputs may be un-normalised."""
    q = F.normalize(queries.float(), dim=1)
    b = F.normalize(bank.float(), dim=1)
    out = [1 - (q[s:s + chunk] @ b.T).max(1).values for s in range(0, len(q), chunk)]
    return torch.cat(out) if out else torch.empty(0, device=q.device)


def auroc(scores_in, scores_out) -> float:
    """Probability that a random OOD sample scores higher than a random in-distribution sample
    (Mann-Whitney U / area under the ROC curve). 0.5 = useless, 1.0 = perfect separation. Ties count half."""
    s_in, s_out = np.asarray(scores_in, np.float64), np.asarray(scores_out, np.float64)
    if len(s_in) == 0 or len(s_out) == 0:
        raise ValueError("auroc needs at least one score on each side")
    allv = np.concatenate([s_in, s_out])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):                         # average ranks over ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_out = ranks[len(s_in):].sum()
    u = r_out - len(s_out) * (len(s_out) + 1) / 2
    return float(u / (len(s_in) * len(s_out)))


def threshold_at(scores_in, keep: float = 0.95) -> float:
    """Score below which `keep` of the in-distribution samples fall."""
    return float(np.quantile(np.asarray(scores_in, np.float64), keep))


def spearman(a, b) -> float:
    """Rank correlation (no scipy needed)."""
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class OODDetector:
    """What the web service loads: a compact feature bank + threshold, tied to one checkpoint."""

    bank: np.ndarray              # float16 [M, D], unit length
    threshold: float
    model_sha256: str             # the bank is only valid for the checkpoint that produced it
    info: dict

    def scores(self, feats: np.ndarray | torch.Tensor) -> np.ndarray:
        q = torch.as_tensor(np.asarray(feats, np.float32))
        return ood_scores(q, torch.from_numpy(self.bank.astype(np.float32))).numpy()

    def is_ood(self, feats) -> np.ndarray:
        return self.scores(feats) > self.threshold

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, bank=self.bank.astype(np.float16), threshold=np.float64(self.threshold),
                            model_sha256=np.array(self.model_sha256), info=np.array(json.dumps(self.info)))

    @classmethod
    def load(cls, path: str | Path, checkpoint: str | Path | None = None) -> OODDetector:
        with np.load(path, allow_pickle=False) as z:
            det = cls(bank=z["bank"], threshold=float(z["threshold"]), model_sha256=str(z["model_sha256"]),
                      info=json.loads(str(z["info"])))
        if checkpoint is not None and file_sha256(checkpoint) != det.model_sha256:
            raise ValueError(f"{path} was built for a different checkpoint than {checkpoint}")
        return det
