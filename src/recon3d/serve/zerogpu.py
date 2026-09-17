"""Hugging Face ZeroGPU compatibility (free GPU Spaces).

ZeroGPU Spaces only start if the app reports its @spaces.GPU functions to the platform. The `spaces`
package sends that report from inside Gradio's demo.launch(). This app starts uvicorn itself (FastAPI
is the server, Gradio is mounted inside it), so launch() never runs and the report must be sent here.

`spaces.zero.startup` is internal to the `spaces` package, so this is isolated in one function and a
test fails loudly if a new `spaces` version removes it. The model itself runs on CPU.
"""

from __future__ import annotations


def is_zerogpu() -> bool:
    try:
        from spaces.config import Config
    except ImportError:
        return False
    return bool(Config.zero_gpu)


def report_startup() -> bool:
    """Call after every @spaces.GPU function is defined and before the server starts. Returns True if sent."""
    if not is_zerogpu():
        return False
    import spaces.zero

    startup = getattr(spaces.zero, "startup", None)
    if startup is None:
        raise RuntimeError("this `spaces` version has no spaces.zero.startup(); pin spaces==0.51.3 or update "
                           "recon3d.serve.zerogpu")
    startup()
    return True
