"""FastAPI application: the public HTTP API, with the Gradio UI mounted at /ui.

    GET  /healthz          liveness: the process is up
    GET  /readyz           readiness: the model is loaded (503 until then)
    GET  /v1/model         which model version is serving, its metrics and runtime settings
    POST /v1/reconstruct   image -> 2,048 points + OOD warning (JSON, or ?format=ply for a file)
    GET  /metrics          Prometheus metrics
    GET  /docs             interactive API documentation (OpenAPI)
    GET  /                 Gradio demo (mounted last, so the routes above take priority)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .. import __version__
from ..io import write_ply
from .events import make_event_log
from .preprocess import InputError
from .service import InferenceService, NotReady, Reconstruction, ServiceBusy
from .settings import Settings

log = logging.getLogger("recon3d.api")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# ------------------------------------------------------------------ response schemas (appear in /docs)
class OODResult(BaseModel):
    score: float = Field(description="1 - highest cosine similarity to a training image; higher = less familiar")
    threshold: float = Field(description="scores above this are flagged (95% of training-category images pass)")
    flagged: bool
    familiarity: float = Field(description="display scale 0..1; 0.5 = at the threshold. Not a probability.")
    message: str


class ModelRef(BaseModel):
    repo_id: str
    revision: str
    weights_sha256: str


class ReconstructResponse(BaseModel):
    request_id: str
    model: ModelRef
    num_points: int
    points: list[list[float]] = Field(description="[x, y, z] per point; bbox centre at origin, diagonal 1, Y up")
    ood: OODResult
    input: dict
    timings_ms: dict
    cached: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str


# ------------------------------------------------------------------ helpers
class RateLimiter:
    """Token bucket per client: `per_min` requests per minute, bursts up to the same number."""

    def __init__(self, per_min: int):
        self.capacity = float(per_min)
        self.rate = per_min / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        if self.capacity <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                ok, wait = True, 0.0
            else:
                self._buckets[key] = (tokens, now)
                ok, wait = False, (1 - tokens) / self.rate
            if len(self._buckets) > 10_000:                     # forget idle clients
                cutoff = now - 3600
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
        return ok, wait


def client_key(request: Request) -> str:
    # Behind the Hugging Face proxy the real client is the first X-Forwarded-For entry. A client can
    # forge this header when calling the app directly, which is acceptable for a demo rate limit.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def ood_message(r: Reconstruction, categories: list[str] | None) -> str:
    cats = ", ".join(categories or [])
    if r.flagged:
        return (f"This input looks unlike the training objects ({cats}). "
                "The reconstruction is probably unreliable.")
    return (f"This input looks similar to the training objects ({cats}). "
            "Similar-looking objects from other categories can still be reconstructed poorly.")


class Telemetry:
    """Prometheus metrics + event log, shared by the API routes and the Gradio UI."""

    def __init__(self, settings: Settings, service: InferenceService):
        self.service = service
        self.events = make_event_log(settings)
        self.registry = CollectorRegistry()
        self.requests = Counter("recon3d_requests_total", "Reconstruction requests", ["source", "status"],
                                registry=self.registry)
        self.latency = Histogram("recon3d_request_seconds", "End-to-end reconstruction time", ["source"],
                                 buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5), registry=self.registry)
        self.ood_score = Histogram("recon3d_ood_score", "OOD score of accepted inputs",
                                   buckets=(0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3),
                                   registry=self.registry)
        self.flagged = Counter("recon3d_ood_flagged_total", "Inputs flagged as unfamiliar", ["source"],
                               registry=self.registry)
        self.cache_hits = Counter("recon3d_cache_hits_total", "Results served from cache", registry=self.registry)
        self.model_info = Gauge("recon3d_model_info", "Serving model version", ["revision", "weights_sha256"],
                                registry=self.registry)

    def record(self, source: str, request_id: str, started: float, result: Reconstruction | None = None,
               status: int = 200, error: str | None = None) -> None:
        elapsed = time.perf_counter() - started
        self.requests.labels(source, str(status)).inc()
        event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request_id": request_id,
                 "source": source, "status": status, "latency_ms": round(elapsed * 1000, 1)}
        b = self.service.bundle
        if b is not None:
            event["model_revision"] = b.config["hub"]["revision"]
        if result is not None:
            self.latency.labels(source).observe(elapsed)
            self.ood_score.observe(result.ood_score)
            if result.flagged:
                self.flagged.labels(source).inc()
            if result.cached:
                self.cache_hits.inc()
            info = result.input_info
            event.update(cached=result.cached, ood_score=round(result.ood_score, 5), flagged=result.flagged,
                         timings_ms=result.timings_ms,
                         input={k: info.get(k) for k in ("width", "height", "source", "inverted",
                                                          "object_share", "filled_outlines")})
        if error:
            event["error"] = error[:200]
        try:
            self.events.write(event)
        except Exception:                                   # monitoring must never break a request
            log.exception("event log write failed")


# ------------------------------------------------------------------ app factory
def create_app(settings: Settings | None = None, service: InferenceService | None = None, *,
               with_ui: bool = True, examples_dir: str | Path | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or InferenceService(settings)
    telemetry = Telemetry(settings, service)
    limiter = RateLimiter(settings.rate_limit_per_min)
    max_body = int(settings.max_upload_mb * 1e6) + 64_000          # multipart overhead

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not service.ready:
            await run_in_threadpool(service.load)
        b = service.bundle
        telemetry.model_info.labels(b.config["hub"]["revision"], b.config["weights_sha256"][:12]).set(1)
        log.info(json.dumps({"event": "model_loaded", "revision": b.config["hub"]["revision"],
                             "seconds": service.load_seconds}))
        yield
        telemetry.events.close()

    app = FastAPI(title="recon3d API", version=__version__, lifespan=lifespan,
                  description="Single-view 3D reconstruction: silhouette image -> 2,048-point cloud, "
                              "with a warning when the input looks unlike the training data.")
    app.state.service, app.state.telemetry, app.state.settings = service, telemetry, settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if REQUEST_ID_RE.match(rid) else uuid.uuid4().hex[:16]
        request.state.request_id = rid
        started = time.perf_counter()
        length = request.headers.get("content-length")
        if request.url.path.startswith("/v1/") and length and length.isdigit() and int(length) > max_body:
            response = _error(413, "payload_too_large", f"upload limit is {settings.max_upload_mb:g} MB", rid)
        else:
            response = await call_next(request)
        ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-Ms"] = str(ms)
        if request.url.path.startswith(("/v1/", "/healthz", "/readyz")):
            log.info(json.dumps({"event": "http", "method": request.method, "path": request.url.path,
                                 "status": response.status_code, "ms": ms, "request_id": rid}))
        return response

    @app.exception_handler(NotReady)
    async def not_ready(request: Request, exc: NotReady):
        return _error(503, "not_ready", str(exc), request.state.request_id, retry_after=10)

    responses = {c: {"model": ErrorResponse} for c in (400, 413, 415, 422, 429, 503)}

    @app.get("/healthz", tags=["ops"])
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"], responses={503: {"model": ErrorResponse}})
    def readyz(request: Request):
        if not service.ready:
            return _error(503, "not_ready", "model is loading", request.state.request_id, retry_after=10)
        return {"status": "ready", "revision": service.bundle.config["hub"]["revision"]}

    @app.get("/v1/model", tags=["model"])
    def model():
        return service.model_info()

    @app.post("/v1/reconstruct", tags=["model"], response_model=ReconstructResponse, responses={
        **responses, 200: {"content": {"application/json": {}, "application/octet-stream": {}},
                           "description": "JSON by default; a PLY file with ?format=ply"}})
    async def reconstruct(
        request: Request,
        file: UploadFile = File(..., description="PNG/JPEG/WEBP/BMP: a silhouette, or one object on a plain "
                                                 "background"),
        format: Literal["json", "ply"] = Query("json", description="json or ply"),
        fill_outlines: bool = Query(False, description="treat closed outlines (drawings) as solid shapes"),
    ):
        rid, started = request.state.request_id, time.perf_counter()
        ok, wait = limiter.allow(client_key(request))
        if not ok:
            telemetry.record("api", rid, started, status=429, error="rate_limited")
            return _error(429, "rate_limited", "too many requests", rid, retry_after=int(wait) + 1)
        data = await file.read(max_body + 1)
        try:
            result = await run_in_threadpool(service.reconstruct, data, fill_outlines=fill_outlines)
        except InputError as e:
            telemetry.record("api", rid, started, status=e.status, error=str(e))
            return _error(e.status, "invalid_input", str(e), rid)
        except ServiceBusy as e:
            telemetry.record("api", rid, started, status=503, error="busy")
            return _error(503, "busy", str(e), rid, retry_after=5)
        telemetry.record("api", rid, started, result)
        b = service.bundle
        if format == "ply":
            return Response(write_ply(result.points), media_type="application/octet-stream",
                            headers={"Content-Disposition": f'attachment; filename="recon3d_{result.key[:8]}.ply"',
                                     "X-OOD-Score": f"{result.ood_score:.5f}",
                                     "X-OOD-Flagged": str(result.flagged).lower()})
        cats = (b.metrics.get("unseen") or {}).get("seen_categories")
        return ReconstructResponse(
            request_id=rid,
            model=ModelRef(repo_id=b.config["hub"]["repo_id"], revision=b.config["hub"]["revision"],
                           weights_sha256=b.config["weights_sha256"]),
            num_points=len(result.points),
            points=result.points.round(4).tolist(),
            ood=OODResult(score=round(result.ood_score, 5), threshold=round(b.detector.threshold, 5),
                          flagged=result.flagged, familiarity=round(result.familiarity, 3),
                          message=ood_message(result, cats)),
            input=result.input_info, timings_ms=result.timings_ms, cached=result.cached)

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    def metrics():
        return Response(generate_latest(telemetry.registry), media_type=CONTENT_TYPE_LATEST)

    if with_ui:
        import gradio as gr

        from .ui import build_ui

        @app.get("/ui", include_in_schema=False)
        def old_ui_path():
            return RedirectResponse("/")

        demo = build_ui(service, telemetry, examples_dir=examples_dir)
        # Mounted at the root: with a sub-path (/ui) Gradio 6 builds some file URLs without the prefix.
        app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False,
                                  max_file_size=f"{settings.max_upload_mb:g}mb")
    else:
        @app.get("/", include_in_schema=False)
        def root_docs():
            return RedirectResponse("/docs")

    return app


def _error(status: int, code: str, detail: str, request_id: str, retry_after: int | None = None) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse({"error": code, "detail": detail, "request_id": request_id}, status_code=status,
                        headers=headers)
