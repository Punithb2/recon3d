"""Service, HTTP API and UI, tested against a real (tiny) bundle. No network access needed."""

import io
import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from recon3d.io import read_ply
from recon3d.serve.api import RateLimiter, create_app
from recon3d.serve.preprocess import InputError
from recon3d.serve.service import InferenceService, LRUCache, NotReady, ServiceBusy, familiarity
from recon3d.serve.settings import Settings

pytestmark = pytest.mark.slow


def chair_png(offset=0, fmt="PNG") -> bytes:
    img = Image.new("L", (256, 256), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((70 + offset, 40, 90 + offset, 130), fill=0)
    d.rectangle((70 + offset, 120, 170 + offset, 138), fill=0)
    d.rectangle((75 + offset, 138, 82 + offset, 210), fill=0)
    d.rectangle((160 + offset, 138, 167 + offset, 210), fill=0)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def shared_service(bundle_dir):
    """One loaded model for the whole file: a ResNet-18 per test would use hundreds of MB."""
    return InferenceService(Settings(model_dir=str(bundle_dir), torch_threads=1, event_log_dir="")).load()


@pytest.fixture(autouse=True)
def _fresh_cache(shared_service):
    """The shared service is reused, so each test starts with an empty result cache."""
    shared_service.cache.clear()


@pytest.fixture
def settings(bundle_dir, tmp_path):
    return Settings(model_dir=str(bundle_dir), torch_threads=1, event_log_dir=str(tmp_path / "events"),
                    rate_limit_per_min=0)


@pytest.fixture
def client(settings, shared_service):
    """API-only app (no Gradio UI) sharing the loaded model."""
    with TestClient(create_app(settings, shared_service, with_ui=False)) as c:
        yield c


@pytest.fixture
def ui_client(settings, shared_service):
    with TestClient(create_app(settings, shared_service)) as c:
        yield c


# ------------------------------------------------------------------ service
def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RECON3D_TORCH_THREADS", "3")
    monkeypatch.setenv("RECON3D_FRAME_FILL", "0.55")
    monkeypatch.setenv("RECON3D_MODEL_REVISION", "v9")
    monkeypatch.setenv("HF_TOKEN", "secret")
    s = Settings.from_env(cache_size=7)
    assert (s.torch_threads, s.frame_fill, s.model_revision, s.hf_token, s.cache_size) == (3, 0.55, "v9", "secret", 7)
    assert Settings.from_env().model_repo == "Punith25/recon3d-resnet18"


def test_framing_file_sets_fill_unless_env_overrides(tmp_path, monkeypatch):
    f = tmp_path / "framing.json"
    f.write_text(json.dumps({"recommended_fill": 0.42}))
    monkeypatch.delenv("RECON3D_FRAME_FILL", raising=False)
    assert Settings.from_env(framing_file=f).frame_fill == 0.42
    assert Settings.from_env(framing_file=tmp_path / "missing.json").frame_fill == 0.6
    monkeypatch.setenv("RECON3D_FRAME_FILL", "0.7")
    assert Settings.from_env(framing_file=f).frame_fill == 0.7


def test_service_lifecycle_and_cache(settings):
    svc = InferenceService(settings)
    with pytest.raises(NotReady):
        svc.reconstruct(chair_png())
    svc.load()
    assert svc.ready and svc.load_seconds is not None
    r1 = svc.reconstruct(chair_png())
    assert r1.points.shape == (128, 3) and r1.points.dtype == np.float32
    assert r1.silhouette.shape == (64, 64) and not r1.cached
    assert r1.ood_score >= -1e-4 and 0 <= r1.familiarity <= 1
    assert r1.flagged == (r1.ood_score > svc.bundle.detector.threshold)
    r2 = svc.reconstruct(chair_png(offset=10))           # same shape moved: framing makes it identical
    assert r2.cached and r2.key == r1.key and np.array_equal(r2.points, r1.points)
    info = svc.model_info()
    assert info["runtime"]["cache_hits"] == 1 and info["architecture"]["points"] == 128
    assert info["revision"] == settings.model_dir


def test_busy_service_refuses(shared_service):
    for _ in range(shared_service.settings.max_concurrent):
        shared_service._slots.acquire()                    # simulate long-running requests
    try:
        with pytest.raises(ServiceBusy):
            shared_service.reconstruct(chair_png(offset=3), timeout_s=0.05)
    finally:
        for _ in range(shared_service.settings.max_concurrent):
            shared_service._slots.release()


def test_concurrent_requests_are_safe(shared_service):
    svc = shared_service
    results, errors = [], []

    def work(i):
        try:
            results.append(svc.reconstruct(chair_png(offset=i)).points)
        except Exception as e:                              # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]      # same shape, different offsets
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(results) == 6
    assert all(np.allclose(r, results[0], atol=1e-5) for r in results)   # same framed input -> same output


def test_lru_cache_evicts_oldest():
    c = LRUCache(2)
    c.put("x", 0)
    c.clear()
    assert len(c) == 0 and c.hits == 0
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)                                         # evicts b (least recently used)
    assert c.get("b") is None and c.get("a") == 1 and len(c) == 2
    off = LRUCache(0)
    off.put("a", 1)
    assert off.get("a") is None


def test_familiarity_scale():
    assert familiarity(0.0, 0.1) == 1.0
    assert familiarity(0.1, 0.1) == pytest.approx(0.5)
    assert familiarity(0.5, 0.1) == 0.0


def test_rate_limiter():
    rl = RateLimiter(2)
    assert rl.allow("x")[0] and rl.allow("x")[0]
    ok, wait = rl.allow("x")
    assert not ok and 0 < wait <= 30
    assert rl.allow("y")[0]                               # separate bucket per client
    assert RateLimiter(0).allow("x") == (True, 0.0)


# ------------------------------------------------------------------ HTTP API
def test_health_ready_model(client):
    assert client.get("/healthz").json()["status"] == "ok"
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    m = client.get("/v1/model").json()
    assert m["architecture"]["decoder"] == "mlp" and m["ood_threshold"] > 0
    assert m["trained_categories"] == ["airplane", "chair"]


def test_reconstruct_json(client):
    r = client.post("/v1/reconstruct", files={"file": ("chair.png", chair_png(), "image/png")},
                    headers={"X-Request-ID": "test-123"})
    assert r.status_code == 200, r.text
    assert r.headers["x-request-id"] == "test-123" and float(r.headers["x-process-time-ms"]) > 0
    body = r.json()
    assert body["request_id"] == "test-123" and body["num_points"] == 128 and len(body["points"]) == 128
    assert all(len(p) == 3 for p in body["points"])
    assert set(body["ood"]) == {"score", "threshold", "flagged", "familiarity", "message"}
    assert "airplane, chair" in body["ood"]["message"]
    assert body["input"]["inverted"] is True and body["cached"] is False
    assert body["model"]["weights_sha256"]


def test_reconstruct_ply(client):
    r = client.post("/v1/reconstruct?format=ply", files={"file": ("c.jpg", chair_png(fmt="JPEG"), "image/jpeg")})
    assert r.status_code == 200
    assert r.headers["content-disposition"].startswith('attachment; filename="recon3d_')
    assert r.headers["x-ood-flagged"] in ("true", "false")
    assert read_ply(r.content).shape == (128, 3)


def test_invalid_request_id_is_replaced(client):
    r = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces <script>"})
    assert r.headers["x-request-id"] != "bad id with spaces <script>" and len(r.headers["x-request-id"]) == 16


@pytest.mark.parametrize("payload,status,code", [
    (b"", 400, "invalid_input"),
    (b"GIF89a not really", 415, "invalid_input"),
    (None, 422, "invalid_input"),                         # blank white image
])
def test_reconstruct_errors(client, payload, status, code):
    if payload is None:
        buf = io.BytesIO()
        Image.new("L", (100, 100), 255).save(buf, "PNG")
        payload = buf.getvalue()
    r = client.post("/v1/reconstruct", files={"file": ("x.png", payload, "image/png")})
    assert r.status_code == status, r.text
    body = r.json()
    assert body["error"] == code and body["request_id"] == r.headers["x-request-id"]


def test_missing_file_is_422(client):
    assert client.post("/v1/reconstruct").status_code == 422


def test_oversized_upload_rejected_before_reading(settings, shared_service):
    app = create_app(Settings(**{**settings.__dict__, "max_upload_mb": 0.01}), shared_service, with_ui=False)
    with TestClient(app) as c:
        r = c.post("/v1/reconstruct", files={"file": ("big.png", b"0" * 200_000, "image/png")})
    assert r.status_code == 413 and r.json()["error"] == "payload_too_large"


def test_rate_limit_returns_429(settings, shared_service):
    app = create_app(Settings(**{**settings.__dict__, "rate_limit_per_min": 2}), shared_service, with_ui=False)
    with TestClient(app) as c:
        codes = [c.post("/v1/reconstruct", files={"file": ("c.png", chair_png(), "image/png")},
                        headers={"X-Forwarded-For": "1.2.3.4"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = c.post("/v1/reconstruct", files={"file": ("c.png", chair_png(), "image/png")},
                   headers={"X-Forwarded-For": "1.2.3.4"})
        assert int(r.headers["retry-after"]) >= 1
        other = c.post("/v1/reconstruct", files={"file": ("c.png", chair_png(), "image/png")},
                       headers={"X-Forwarded-For": "5.6.7.8"})
        assert other.status_code == 200


def test_metrics_and_event_log(client, settings):
    client.post("/v1/reconstruct", files={"file": ("c.png", chair_png(), "image/png")})
    client.post("/v1/reconstruct", files={"file": ("c.png", chair_png(), "image/png")})    # cache hit
    client.post("/v1/reconstruct", files={"file": ("x.png", b"junk", "image/png")})
    text = client.get("/metrics").text
    assert 'recon3d_requests_total{source="api",status="200"} 2.0' in text
    assert 'recon3d_requests_total{source="api",status="415"} 1.0' in text
    assert "recon3d_cache_hits_total 1.0" in text and "recon3d_model_info{" in text
    assert "recon3d_request_seconds_bucket" in text

    lines = [json.loads(line) for f in sorted(__import__("pathlib").Path(settings.event_log_dir).glob("*.jsonl"))
             for line in f.read_text().splitlines()]
    assert [e["status"] for e in lines] == [200, 200, 415]
    ok = lines[0]
    assert {"ts", "request_id", "latency_ms", "ood_score", "flagged", "input", "model_revision"} <= set(ok)
    assert "points" not in ok and "silhouette" not in ok        # no image data is stored
    assert lines[1]["cached"] is True and "error" in lines[2]


def test_not_ready_before_startup(settings):
    c = TestClient(create_app(settings, with_ui=False))    # no `with`: startup (model load) never runs

    assert c.get("/readyz").status_code == 503
    r = c.get("/v1/model")
    assert r.status_code == 503 and r.headers["retry-after"] == "10"


def test_docs_and_ui_are_served(ui_client):
    client = ui_client
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/v1/reconstruct" in schema["paths"] and "ReconstructResponse" in schema["components"]["schemas"]
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/ui", follow_redirects=False).headers["location"] == "/"
    assert client.get("/healthz").json()["status"] == "ok"          # API routes still win over the UI mount


def test_without_ui_root_goes_to_docs(settings, shared_service):
    with TestClient(create_app(settings, shared_service, with_ui=False)) as c:
        assert c.get("/", follow_redirects=False).headers["location"] == "/docs"
        assert c.get("/ui").status_code == 404


# ------------------------------------------------------------------ Gradio handlers
def _ui_fn(app, name):
    for route in app.routes:
        blocks = getattr(getattr(route, "app", None), "blocks", None)
        if blocks is not None:
            for f in blocks.fns.values():
                if f.fn is not None and f.fn.__name__ == name:
                    return f.fn
    raise LookupError(name)


def test_ui_handlers(settings, shared_service, tmp_path):
    ex = tmp_path / "ex"
    ex.mkdir()
    Image.open(io.BytesIO(chair_png())).save(ex / "chair_1.png")
    Image.open(io.BytesIO(chair_png())).save(ex / "unseen_bus_1.png")
    app = create_app(settings, shared_service, examples_dir=ex)
    with TestClient(app):
        upload = _ui_fn(app, "run_upload")
        fig, status, ply, sil = upload(Image.open(io.BytesIO(chair_png())))
        assert fig.data[0].x.shape == (128,) and sil.shape == (64, 64)
        assert "familiarity" in status and read_ply(ply).shape == (128, 3)

        drawing = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(drawing).rectangle((50, 50, 150, 150), outline=(0, 0, 0, 255), width=4)
        _, _, _, sil = _ui_fn(app, "run_draw")({"background": None, "layers": [], "composite": drawing})
        assert sil[32, 32] == 255                             # outline was filled

        import gradio as gr
        with pytest.raises(gr.Error):
            upload(None)
        with pytest.raises(gr.Error, match="no visible object"):
            upload(Image.new("L", (100, 100), 255))
        about = _ui_fn(app, "<lambda>")()
        assert "Test Chamfer" in about


def test_input_error_carries_status():
    assert InputError("x", 413).status == 413 and InputError("y").status == 422
