"""Gradio demo. Calls InferenceService directly (same process), not the HTTP API: no extra network hop,
and both paths share the same validation, cache, metrics and event log."""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr

from ..io import write_ply
from .preprocess import InputError
from .service import InferenceService, NotReady, ServiceBusy
from .viz import point_cloud_figure

PLY_DIR = Path(tempfile.gettempdir()) / "recon3d_ply"

INTRO = """
# recon3d: one silhouette in, a 3D point cloud out

A ResNet-18 encoder and an MLP decoder predict **2,048 3D points** from a single silhouette.
Trained on six ShapeNet categories: **airplane, bench, cabinet, chair, rifle, table**.
Anything else is reconstructed poorly, and the app tells you when an input looks unfamiliar.

**Try it:** pick an example, upload a dark shape on a light background (or a transparent PNG), or draw one.
[Code](https://github.com/Punithb2/recon3d) · [Model card](https://huggingface.co/Punith25/recon3d-resnet18) ·
[API docs](/docs) · [Metrics](/metrics)
"""


def _about(service: InferenceService) -> str:
    try:
        info = service.model_info()
    except NotReady:
        return "Model is loading..."
    m = info["metrics"]

    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    return f"""
| | |
|---|---|
| Model | `{info['repo_id']}` @ `{info['revision']}` (epoch {info['best_epoch']}) |
| Test Chamfer x1000 (trained categories) | {f(m['test_chamfer_x1000'])} |
| Test F-score @1% | {f(m['test_fscore_1pct'])} |
| Chamfer x1000 on 6 unseen categories | {f(m['unseen_chamfer_x1000'])} |
| Unfamiliar-input detector AUROC | {f(m['ood_auroc'])} |
| Weights sha256 | `{info['weights_sha256'][:16]}...` |

**How the warning works:** the image's encoder features are compared with 48,000 training images.
If nothing is similar enough, the result is flagged. It detects unfamiliar-*looking* inputs; an object
that looks like a training category (a bus seen from the side looks like a cabinet) can pass and still
be reconstructed badly.

**Limitations:** six categories only; silhouettes carry no depth or texture, so the model guesses the
hidden side from what it learned; results are blurrier on thin parts.
"""


def _status(r, threshold: float, categories: str) -> str:
    pct = int(round(r.familiarity * 100))
    if r.flagged:
        head = (f"### ⚠️ Unfamiliar input (familiarity {pct}%)\nThis doesn't look like the training objects "
                f"({categories}). Treat the result as unreliable.")
    else:
        head = (f"### ✅ Familiar input (familiarity {pct}%)\nLooks similar to the training objects "
                f"({categories}).")
    t = r.timings_ms
    speed = "served from cache" if r.cached else f"model {t.get('inference', 0):.0f} ms"
    return f"{head}\n\nOOD score {r.ood_score:.3f} (flag above {threshold:.3f}) · {speed} · total {t['total']:.0f} ms"


def build_ui(service: InferenceService, telemetry, examples_dir: str | Path | None = None) -> gr.Blocks:
    PLY_DIR.mkdir(parents=True, exist_ok=True)

    def run(image, fill_outlines: bool):
        rid, started = "ui-" + uuid.uuid4().hex[:12], time.perf_counter()
        if isinstance(image, dict):                     # Sketchpad value: {"background", "layers", "composite"}
            image = image.get("composite")
        if image is None:
            raise gr.Error("Nothing to reconstruct yet: upload an image, or pick the brush tool (left of the "
                           "canvas) and draw.")
        try:
            r = service.reconstruct(image, fill_outlines=fill_outlines)
        except InputError as e:
            telemetry.record("ui", rid, started, status=e.status, error=str(e))
            raise gr.Error(str(e)) from e
        except (ServiceBusy, NotReady) as e:
            telemetry.record("ui", rid, started, status=503, error=str(e))
            raise gr.Error(str(e)) from e
        telemetry.record("ui", rid, started, r)
        b = service.bundle
        cats = ", ".join((b.metrics.get("unseen") or {}).get("seen_categories") or [])
        ply = PLY_DIR / f"recon3d_{r.key[:12]}.ply"
        if not ply.exists():
            write_ply(r.points, ply)
        return point_cloud_figure(r.points), _status(r, b.detector.threshold, cats), str(ply), r.silhouette

    def run_upload(image):
        return run(image, False)

    def run_draw(sketch):
        return run(sketch, True)

    examples = sorted(str(p) for p in Path(examples_dir).glob("*.png")) if examples_dir else []
    seen_ex = [e for e in examples if not Path(e).name.startswith("unseen_")]
    unseen_ex = [e for e in examples if Path(e).name.startswith("unseen_")]

    with gr.Blocks(title="recon3d: silhouette to 3D") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Upload"):
                        image = gr.Image(type="pil", image_mode=None, label="Silhouette or object on a plain "
                                         "background", sources=["upload", "clipboard"], height=300)
                        upload_btn = gr.Button("Reconstruct", variant="primary")
                        if seen_ex:
                            gr.Examples(seen_ex, inputs=image, label="Trained categories", examples_per_page=12)
                        if unseen_ex:
                            gr.Examples(unseen_ex, inputs=image, label="Never-seen categories (expect a warning)")
                    with gr.Tab("Draw"):
                        gr.Markdown("1. Click the **brush** icon at the left of the canvas.  \n"
                                    "2. Draw a side view of a chair, table or bench. Closed outlines are filled in.  \n"
                                    "3. Press **Reconstruct drawing**.")
                        sketch = gr.Sketchpad(type="pil", image_mode="RGBA", canvas_size=(512, 512),
                                              brush=gr.Brush(colors=["#000000"], default_size=6))
                        draw_btn = gr.Button("Reconstruct drawing", variant="primary")
                seen_by_model = gr.Image(label="What the model sees (after cropping and framing)", height=200,
                                         interactive=False)
            with gr.Column(scale=1):
                plot = gr.Plot(label="3D point cloud (drag to rotate)")
                status = gr.Markdown()
                ply_file = gr.File(label="Download point cloud (.ply)")
        with gr.Accordion("About the model", open=False):
            about = gr.Markdown()

        outputs = [plot, status, ply_file, seen_by_model]
        upload_btn.click(run_upload, image, outputs, api_name=False, concurrency_limit=4)
        draw_btn.click(run_draw, sketch, outputs, api_name=False, concurrency_limit=4)
        demo.load(lambda: _about(service), None, about, api_name=False)
    return demo

