import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recon3d import hub
from recon3d.evaluate import evaluate
from recon3d.hub import BUNDLE_FILES, export_bundle, load_bundle, load_from_hub, publish_bundle
from recon3d.ood import OODDetector
from recon3d.train import train
from recon3d.unseen import evaluate_unseen

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, dataset, unseen_dataset):
    from recon3d.config import TrainConfig
    from recon3d.train import Session
    s = Session(data_root=dataset, runs_dir=tmp_path_factory.mktemp("hubruns"), device="cpu")
    cfg = TrainConfig(name="finetune", pretrained=False, img_res=64, n_pred=128, n_gt=128, epochs=1,
                      batch_size=16, views_per_mesh=2, warmup_epochs=1, amp=False)
    rd = train(cfg, s)
    evaluate(rd, "test", s)
    evaluate_unseen(rd, s, unseen_dataset)
    return rd


def test_export_round_trip(run_dir, tmp_path):
    out = export_bundle(run_dir, tmp_path / "bundle")
    assert sorted(p.name for p in out.iterdir()) == sorted(BUNDLE_FILES)
    b = load_bundle(out)
    from recon3d.models import load_checkpoint
    m, _, _ = load_checkpoint(run_dir / "best.pt")
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(m(x), b.model(x))

    # detector: same bank and threshold as the run's, but bound to the exported weights
    orig = OODDetector.load(run_dir / "unseen" / "ood_detector.npz", checkpoint=run_dir / "best.pt")
    assert np.array_equal(orig.bank, b.detector.bank) and orig.threshold == b.detector.threshold
    assert b.detector.info["source_checkpoint_sha256"] == orig.model_sha256
    assert b.config["weights_sha256"] == b.detector.model_sha256

    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: other")
    assert "Categories never seen in training" in card and "lamp" in card and "| **all** |" in card
    assert "6 training meshes per category" in card
    assert json.loads((out / "metrics.json").read_text())["ood"]["threshold"] == orig.threshold


def test_export_refuses_non_empty_folder(run_dir, tmp_path):
    (tmp_path / "old.txt").write_text("stale")
    with pytest.raises(FileExistsError):
        export_bundle(run_dir, tmp_path)


def test_tampered_weights_are_refused(run_dir, tmp_path):
    out = export_bundle(run_dir, tmp_path / "b")
    from safetensors.torch import load_file, save_file
    sd = load_file(str(out / "model.safetensors"))
    sd["decoder.4.bias"] = sd["decoder.4.bias"] + 1                  # e.g. someone swapped in other weights
    save_file(sd, str(out / "model.safetensors"))
    with pytest.raises(ValueError, match="different checkpoint"):
        load_bundle(out)


class FakeApi:
    """Stands in for huggingface_hub.HfApi so tests never touch the network."""
    tags: set = set()
    uploads: list = []

    def __init__(self, token=None):
        pass

    def create_repo(self, repo_id, private=None, exist_ok=False):
        self.created = (repo_id, private)

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in FakeApi.tags])

    def upload_folder(self, repo_id, folder_path, allow_patterns, commit_message):
        FakeApi.uploads.append((repo_id, sorted(allow_patterns), commit_message))
        return SimpleNamespace(oid="abc123")

    def create_tag(self, repo_id, tag, tag_message, revision):
        assert revision == "abc123"
        FakeApi.tags.add(tag)


def test_publish_tags_once(run_dir, tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    FakeApi.tags, FakeApi.uploads = set(), []
    out = export_bundle(run_dir, tmp_path / "b")
    url = publish_bundle(out, "user/repo", "v1.0")
    assert url.endswith("/tree/v1.0") and FakeApi.tags == {"v1.0"}
    assert FakeApi.uploads[0][1] == sorted(BUNDLE_FILES)
    with pytest.raises(ValueError, match="immutable"):
        publish_bundle(out, "user/repo", "v1.0")
    (out / "metrics.json").unlink()
    with pytest.raises(FileNotFoundError):
        publish_bundle(out, "user/repo", "v1.1")


def test_load_from_hub_uses_pinned_revision(run_dir, tmp_path, monkeypatch):
    out = export_bundle(run_dir, tmp_path / "b")
    calls = {}

    def fake_download(repo_id, revision, allow_patterns, token, cache_dir):
        calls.update(repo_id=repo_id, revision=revision)
        return str(out)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    b = load_from_hub("user/repo", revision="v1.0")
    assert calls == {"repo_id": "user/repo", "revision": "v1.0"}
    assert b.config["hub"] == {"repo_id": "user/repo", "revision": "v1.0"}


def test_hub_module_imports_without_hub_libraries():
    assert hub.BUNDLE_FILES[0] == "model.safetensors"    # heavy imports happen inside the functions
