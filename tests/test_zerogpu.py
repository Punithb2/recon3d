import types

import pytest

from recon3d.serve import zerogpu

spaces = pytest.importorskip("spaces")


def test_not_on_zerogpu_does_nothing():
    assert zerogpu.is_zerogpu() is False
    assert zerogpu.report_startup() is False


def test_reports_startup_on_zerogpu(monkeypatch):
    import spaces.config
    monkeypatch.setattr(spaces.config.Config, "zero_gpu", True)
    calls = []
    fake = types.SimpleNamespace(startup=lambda: calls.append(1))
    monkeypatch.setitem(__import__("sys").modules, "spaces.zero", fake)
    monkeypatch.setattr(spaces, "zero", fake, raising=False)
    assert zerogpu.report_startup() is True and calls == [1]


def test_missing_internal_hook_fails_loudly(monkeypatch):
    import spaces.config
    monkeypatch.setattr(spaces.config.Config, "zero_gpu", True)
    fake = types.SimpleNamespace()
    monkeypatch.setitem(__import__("sys").modules, "spaces.zero", fake)
    monkeypatch.setattr(spaces, "zero", fake, raising=False)
    with pytest.raises(RuntimeError, match="spaces.zero.startup"):
        zerogpu.report_startup()


def test_installed_spaces_still_has_the_hook():
    """Guards against a `spaces` upgrade silently removing the function we rely on."""
    import inspect

    import spaces.zero as z
    src = inspect.getsource(z)
    assert "def startup" in src
