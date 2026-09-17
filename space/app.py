"""Hugging Face Space entry point (Gradio SDK, ZeroGPU hardware).

The app itself lives in the recon3d package (recon3d.serve); this file only starts it.
FastAPI is the server; the Gradio UI is mounted inside it at /ui.
"""

import spaces  # noqa: F401  (must be imported before torch on ZeroGPU)

import logging
import os
from pathlib import Path

import uvicorn

from recon3d.serve.api import create_app
from recon3d.serve.settings import Settings
from recon3d.serve.zerogpu import report_startup

logging.basicConfig(level=logging.INFO, format="%(message)s")


@spaces.GPU(duration=5)
def _gpu_placeholder():
    """ZeroGPU only starts apps that register at least one GPU function. The model runs on CPU
    (about 40 ms per image), so this is never called."""
    return None


HERE = Path(__file__).parent
app = create_app(Settings.from_env(framing_file=HERE / "framing.json"), examples_dir=HERE / "examples")

if __name__ == "__main__":
    report_startup()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)),
                proxy_headers=True, forwarded_allow_ips="*")
