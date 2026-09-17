"""The ops scripts are part of the deploy path, so at least their entry points are checked.

This also pins their dependencies: the deploy job installs only huggingface_hub, httpx and pillow, so a
script that imports something else (e.g. `requests`) must fail here rather than in CI.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


@pytest.mark.parametrize("name", ["deploy_space.py", "smoke_space.py", "verify_hub.py", "check_checkpoint.py"])
def test_scripts_start_and_document_themselves(name):
    r = subprocess.run([sys.executable, str(SCRIPTS / name), "--help"], capture_output=True, text=True)
    assert r.returncode in (0, 2), r.stderr
    assert "usage" in (r.stdout + r.stderr).lower()


def test_smoke_reports_an_unreachable_app():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("smoke_space", SCRIPTS / "smoke_space.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.chair_png()) > 100
    with pytest.raises(TimeoutError):
        mod.wait_ready("http://127.0.0.1:9", timeout=0.1)
