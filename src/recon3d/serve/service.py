"""InferenceService: everything the app does, with no web framework in sight.

FastAPI routes and the Gradio UI are thin adapters around this class, so the logic is written and
tested once. Swapping FastAPI for another framework would not touch this file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from ..data import preprocess
from .preprocess import InputError, prepare
from .settings import Settings


class ServiceBusy(RuntimeError):
    """All inference slots are taken; the API answers 503 and asks the client to retry."""


class NotReady(RuntimeError):
    """The model is still loading."""


@dataclass(frozen=True)
class Reconstruction:
    points: np.ndarray            # float32 [N, 3], normalised frame, Y up
    ood_score: float              # 1 - max cosine similarity to training images
    flagged: bool                 # ood_score above the calibrated threshold
    familiarity: float            # display scale 0..1 (1 = looks like training data); not a probability
    silhouette: np.ndarray        # uint8 [S, S]: exactly what the model saw
    input_info: dict
    timings_ms: dict
    cached: bool
    key: str                      # content hash of (silhouette, model); also the cache key


class LRUCache:
    """Small thread-safe in-memory cache. With several app replicas this would move to Redis, so that
    all replicas share hits; one Space replica doesn't need that."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: OrderedDict[str, Reconstruction] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> Reconstruction | None:
        if self.capacity <= 0:
            return None
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Reconstruction) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._data)


def familiarity(score: float, threshold: float) -> float:
    """Maps the OOD score to 0..1 for display: 1 at score 0, 0.5 at the threshold, 0 at twice the threshold."""
    return float(np.clip(1 - score / (2 * threshold), 0.0, 1.0))


class InferenceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bundle = None
        self.cache = LRUCache(settings.cache_size)
        self._slots = threading.BoundedSemaphore(settings.max_concurrent)
        self._load_lock = threading.Lock()
        self.load_seconds: float | None = None

    # ------------------------------------------------------------- lifecycle
    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def load(self) -> InferenceService:
        with self._load_lock:
            if self.ready:
                return self
            from ..hub import load_bundle, load_from_hub

            torch.set_num_threads(self.settings.torch_threads)
            t = time.perf_counter()
            s = self.settings
            if s.model_dir:
                bundle = load_bundle(s.model_dir, device="cpu")
                bundle.config["hub"] = {"repo_id": "local", "revision": s.model_dir}
            else:
                bundle = load_from_hub(s.model_repo, s.model_revision, device="cpu", token=s.hf_token or None)
            size = bundle.cfg.img_res
            with torch.inference_mode():                        # warm-up: first call allocates buffers
                bundle.model(torch.zeros(1, 3, size, size))
            self.bundle = bundle
            self.load_seconds = round(time.perf_counter() - t, 2)
            return self

    # ------------------------------------------------------------- info
    def model_info(self) -> dict:
        if not self.ready:
            raise NotReady("model is loading")
        b = self.bundle
        m = b.metrics or {}
        seen = (m.get("seen_test") or {}).get("overall", {}).get("model", {})
        un = (m.get("unseen") or {}).get("overall", {}).get("unseen", {})
        return {
            "repo_id": b.config["hub"]["repo_id"],
            "revision": b.config["hub"]["revision"],
            "weights_sha256": b.config["weights_sha256"],
            "run": b.config.get("run"),
            "best_epoch": b.config.get("best_epoch"),
            "architecture": {"backbone": b.cfg.backbone, "decoder": b.cfg.decoder, "input_size": b.cfg.img_res,
                             "points": b.cfg.n_pred},
            "trained_categories": (m.get("unseen") or {}).get("seen_categories"),
            "metrics": {
                "test_chamfer_x1000": seen.get("cd_x1000"),
                "test_fscore_1pct": seen.get(f"f@{b.cfg.fscore_taus[0]}"),
                "unseen_chamfer_x1000": (un.get("model_cd_x1000") or {}).get("mean"),
                "ood_auroc": (m.get("ood") or {}).get("auroc"),
            },
            "ood_threshold": b.detector.threshold,
            "runtime": {"torch_threads": torch.get_num_threads(), "max_concurrent": self.settings.max_concurrent,
                        "frame_fill": self.settings.frame_fill, "load_seconds": self.load_seconds,
                        "cache_entries": len(self.cache), "cache_hits": self.cache.hits,
                        "cache_misses": self.cache.misses},
        }

    # ------------------------------------------------------------- inference
    def reconstruct(self, data, *, fill_outlines: bool = False, timeout_s: float = 30.0) -> Reconstruction:
        if not self.ready:
            raise NotReady("model is loading")
        b, s = self.bundle, self.settings
        t0 = time.perf_counter()
        prep = prepare(data, b.cfg.img_res, s.frame_fill, fill_outlines=fill_outlines,
                       max_bytes=int(s.max_upload_mb * 1e6), max_pixels=s.max_pixels)
        t1 = time.perf_counter()
        key = hashlib.sha256(prep.silhouette.tobytes() + b.config["weights_sha256"].encode()).hexdigest()

        hit = self.cache.get(key)
        if hit is not None:
            return dataclasses.replace(hit, cached=True, input_info=prep.info,
                                       timings_ms={"preprocess": _ms(t0, t1), "inference": 0.0,
                                                   "total": _ms(t0, time.perf_counter())})

        if not self._slots.acquire(timeout=timeout_s):
            raise ServiceBusy("server is busy, retry in a few seconds")
        try:
            t2 = time.perf_counter()
            with torch.inference_mode():
                x = preprocess(torch.from_numpy(prep.silhouette.copy())[None], b.cfg.img_res)
                feats = b.model.embed(x)
                points = b.model.decode(feats)[0] if hasattr(b.model, "decode") else b.model(x)[0]
            t3 = time.perf_counter()
        finally:
            self._slots.release()
        score = float(b.detector.scores(feats.float().numpy())[0])
        t4 = time.perf_counter()
        result = Reconstruction(
            points=points.float().numpy(), ood_score=score, flagged=score > b.detector.threshold,
            familiarity=familiarity(score, b.detector.threshold), silhouette=prep.silhouette,
            input_info=prep.info, cached=False, key=key,
            timings_ms={"preprocess": _ms(t0, t1), "wait": _ms(t1, t2), "inference": _ms(t2, t3),
                        "ood": _ms(t3, t4), "total": _ms(t0, t4)})
        self.cache.put(key, result)
        return result


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000, 1)


__all__ = ["InferenceService", "Reconstruction", "InputError", "ServiceBusy", "NotReady", "LRUCache", "familiarity"]
