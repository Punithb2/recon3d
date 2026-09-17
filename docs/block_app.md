# Block: the web app (FastAPI + Gradio on a ZeroGPU Space)

## Architecture

```
Hugging Face Space (Gradio SDK, ZeroGPU hardware, free)
└── space/app.py ── uvicorn ── FastAPI app  (recon3d.serve.api)
                                ├── /v1/reconstruct, /v1/model   public JSON API
                                ├── /healthz, /readyz, /metrics  operations
                                ├── /docs                        OpenAPI docs
                                └── /  Gradio UI  (recon3d.serve.ui, mounted last)
                                         │
             both call ─────────► InferenceService (recon3d.serve.service)
                                   ├── prepare(): decode, mask, frame   (serve.preprocess)
                                   ├── model + OOD detector from the Hub bundle, pinned tag
                                   ├── LRU cache (content hash)
                                   └── concurrency limit (semaphore)
                                         │
                                  Telemetry: Prometheus counters + JSONL event log
                                             (synced to a private HF dataset)
```

**One service, two front ends.** The service knows nothing about HTTP or Gradio. The API and the UI are
thin adapters, so validation, caching and monitoring are written once and tested once (`tests/test_serve.py`).

## Request flow (`POST /v1/reconstruct`)

1. **Middleware** assigns a request ID (or keeps a valid `X-Request-ID`) and rejects bodies over the upload limit
   *before* reading them (413).
2. **Rate limit**: token bucket per client IP (from `X-Forwarded-For` behind the HF proxy); 429 + `Retry-After`.
3. **Decode safely**: allowed formats only (415), byte and pixel limits against decompression bombs (413),
   `verify()` before decoding, EXIF rotation applied.
4. **Make a silhouette** like the training data: alpha channel if the image is transparent, otherwise Otsu
   threshold with automatic inversion (dark object on light paper); optional filling of closed outlines
   (drawings); crop to the object, centre it, scale it to the **measured** training fill ratio, downsample with
   anti-aliasing. Blank or full images are rejected with a clear message (422).
5. **Cache lookup**: key = sha256(framed silhouette + weights hash). The same object moved or resized in the
   upload frames to the same silhouette, so it's a hit.
6. **Inference** in a thread pool (FastAPI doesn't block its event loop), at most `max_concurrent` at once;
   extra requests wait up to 30 s, then get 503 + `Retry-After`.
7. **OOD score** against the 48,000-vector bank; flag + familiarity + honest message.
8. **Telemetry**: metrics + one event line (metadata only, never the image).
9. **Response**: JSON (points rounded to 4 decimals) or `?format=ply`.

## Decisions and trade-offs

| Decision | Why | Cost / alternative |
|---|---|---|
| FastAPI as the server, Gradio mounted inside | A real, documented API (OpenAPI, typed schemas, status codes) plus a UI, in one process | Needed two ZeroGPU workarounds (see `serve/zerogpu.py`); plain `demo.launch()` would need none |
| CPU inference on a GPU Space | ~40 ms per image on 2 threads; ZeroGPU only grants GPUs through Gradio's queue and has a daily quota | Would need the GPU for a much larger model |
| `torch_threads` set explicitly (4) | `os.cpu_count()` reports the whole shared host (192 cores); using all of them oversubscribes | Tune from `/v1/model` → `runtime` |
| Model pinned to a Hub tag (`v1.0.1`) | Deploys are reproducible; rollback = change one variable | New models need an explicit version bump |
| Package pinned to a git commit in the Space | The Space runs exactly the tested code | Each deploy writes a new `requirements.txt` |
| In-memory LRU cache | One replica; zero infrastructure | With several replicas → Redis, so hits are shared |
| In-memory token-bucket rate limit | Enough for a demo | Per replica only; `X-Forwarded-For` can be forged when calling the app directly |
| JSONL events synced to an HF dataset | Space disks are wiped on restart; free; no images stored | Minutes of delay; for real traffic → a log pipeline / database |
| Prometheus text at `/metrics` | Industry standard; scrapeable | No Prometheus server is running on the free tier; it documents what would be monitored |
| Measured frame fill (`framing.json`) | Uploads look like training images in size and position | Needs one small Kaggle run |
| UI at `/`, API under `/v1` | Space link opens the demo directly; Gradio 6 builds some file URLs without a sub-path prefix | `/ui` kept as a redirect |

## Settings (environment variables)

All in `serve/settings.py`, prefixed `RECON3D_`: `MODEL_REPO`, `MODEL_REVISION`, `MODEL_DIR`, `TORCH_THREADS`,
`MAX_CONCURRENT`, `MAX_UPLOAD_MB`, `MAX_PIXELS`, `FRAME_FILL`, `CACHE_SIZE`, `RATE_LIMIT_PER_MIN`,
`EVENT_LOG_DIR`, `EVENT_LOG_DATASET`, `EVENT_LOG_EVERY_MIN`, plus the secret `HF_TOKEN`.
On the Space: Settings → Variables and secrets.

## Using the API

```powershell
curl.exe -F "file=@chair.png" https://punith25-recon3d.hf.space/v1/reconstruct
curl.exe -F "file=@chair.png" "https://punith25-recon3d.hf.space/v1/reconstruct?format=ply" -o chair.ply
curl.exe https://punith25-recon3d.hf.space/v1/model
```

```python
import requests
r = requests.post("https://punith25-recon3d.hf.space/v1/reconstruct", files={"file": open("chair.png", "rb")})
body = r.json(); print(body["ood"], len(body["points"]))
```

## Tests (`tests/test_preprocess.py`, `tests/test_serve.py`, `tests/test_framing.py`, `tests/test_zerogpu.py`)

Framing and polarity, alpha handling, outline filling, every error status, cache hits for a shifted
object, thread-safety, busy/not-ready states, rate limiting, request IDs, metrics and event-log content
(including "no image data is stored"), OpenAPI schema, UI handlers, the ZeroGPU hook. Two real bugs were
caught while writing them: flood fill silently doing nothing on a NumPy-backed image, and integer JSON keys
breaking `recon3d framing`. A browser test (Playwright) found a third: example images 404'd when the UI was
mounted under `/ui`.

## Running the tests on a laptop

`pytest` peaks at roughly 1.4 GB of RAM: PyTorch itself takes about 1 GB, and the tests add models and
services on top. Three things keep it there instead of growing with the number of tests:

- one loaded model shared by all API tests (`shared_service`), instead of a ResNet-18 per test;
- `gc.collect()` after every test (an autouse fixture), so finished models are released immediately;
- bootstrap resampling in chunks, instead of one big `(2000, n)` index array.

If a machine is still short of memory, `pytest -m "not slow"` runs the fast checks in about 10 seconds,
and `pytest tests/test_serve.py` etc. runs one file at a time. CI runs the full suite anyway.

## Interview questions

- **"Why FastAPI if Gradio already serves a UI?"** A typed, versioned API that other programs can call, with
  proper status codes, limits and docs; the UI is just one client of the same service.
- **"How do you avoid training/serving skew?"** The same `preprocess()` as training, plus upload framing
  calibrated on the training silhouettes, and the UI shows exactly what the model saw.
- **"What happens under load?"** Bounded concurrency, queueing with a timeout, 503 + `Retry-After`,
  per-client rate limiting, a result cache; next steps would be horizontal replicas + Redis.
- **"How do you monitor it without storing user data?"** Metadata-only event logs (score, latency, input
  size/area, outcome) and Prometheus counters; the OOD score distribution over time is the drift signal.
- **"How do you roll back?"** Model: set `RECON3D_MODEL_REVISION` to the previous tag. Code: redeploy an
  older commit with `scripts/deploy_space.py --ref <sha>`.
