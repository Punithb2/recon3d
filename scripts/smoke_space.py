"""Smoke-test a deployed app: wait until it is ready, then send a real image through it.

    python scripts/smoke_space.py                                   # the live Space
    python scripts/smoke_space.py --base-url http://127.0.0.1:8000  # a local run
    python scripts/smoke_space.py --expect-revision v1.0.1 --timeout 900

Used by the deploy workflow: a green deploy means the Space actually answers with a reconstruction,
not just that the files uploaded.
"""

import argparse
import io
import sys
import time

import httpx  # already a dependency (used by the test client); `requests` is not
from PIL import Image, ImageDraw


def chair_png() -> bytes:
    """A chair-like silhouette, drawn here so the check needs no files from the repo."""
    img = Image.new("L", (256, 256), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((70, 40, 90, 130), fill=0)
    d.rectangle((70, 120, 170, 138), fill=0)
    d.rectangle((75, 138, 82, 210), fill=0)
    d.rectangle((160, 138, 167, 210), fill=0)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def wait_ready(base: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/readyz", timeout=20)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:120]}"
        except httpx.HTTPError as e:
            last = str(e)[:120]
        print(f"  waiting for {base}/readyz ... {last}", flush=True)
        time.sleep(15)
    raise TimeoutError(f"not ready after {timeout:.0f}s: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="Punith25/recon3d")
    ap.add_argument("--base-url", help="override (default: the Space URL built from --space)")
    ap.add_argument("--expect-revision", help="fail unless the app serves this model revision")
    ap.add_argument("--timeout", type=float, default=900, help="seconds to wait for the app to come up")
    args = ap.parse_args()
    base = (args.base_url or f"https://{args.space.replace('/', '-').lower()}.hf.space").rstrip("/")

    ready = wait_ready(base, args.timeout)
    print(f"ready: {ready}")
    model = httpx.get(f"{base}/v1/model", timeout=30).json()
    t = time.perf_counter()
    r = httpx.post(f"{base}/v1/reconstruct", files={"file": ("chair.png", chair_png(), "image/png")}, timeout=60)
    ms = (time.perf_counter() - t) * 1000
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}

    checks = {
        "reconstruct returns 200": r.status_code == 200,
        f"{model['architecture']['points']} points returned": body.get("num_points") == model["architecture"]["points"],
        "ood block present": {"score", "threshold", "flagged"} <= set(body.get("ood", {})),
        "request id echoed in header": bool(r.headers.get("X-Request-ID")),
        "metrics endpoint served": httpx.get(f"{base}/metrics", timeout=30).text.startswith("# HELP"),
        "openapi served": "/v1/reconstruct" in httpx.get(f"{base}/openapi.json", timeout=30).json()["paths"],
    }
    if args.expect_revision:
        checks[f"serving revision {args.expect_revision}"] = model["revision"] == args.expect_revision

    print(f"model {model['repo_id']}@{model['revision']} | round trip {ms:.0f} ms | "
          f"threads {model['runtime']['torch_threads']} | model time {body.get('timings_ms', {}).get('inference')} ms")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print(r.text[:500])
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
