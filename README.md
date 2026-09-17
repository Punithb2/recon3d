# recon3d: single-view 3D reconstruction

[![CI](https://github.com/Punithb2/recon3d/actions/workflows/ci.yml/badge.svg)](https://github.com/Punithb2/recon3d/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/Punith25/recon3d)
[![Model](https://img.shields.io/badge/model-Punith25%2Frecon3d--resnet18-yellow)](https://huggingface.co/Punith25/recon3d-resnet18)

A silhouette image goes in; a 2,048-point 3D point cloud comes out.

```
silhouette (224x224) -> ResNet-18 encoder -> 512-d vector -> MLP decoder -> 2,048 (x, y, z) points
```

Trained with Chamfer Distance on 6 ShapeNet categories (airplane, rifle, bench, cabinet, chair, table),
1,000 training meshes per category x 24 views, on a free Kaggle T4.

## Project layout

```
configs/            one YAML file per experiment (overfit, frozen, finetune, attn, overfit_attn)
src/recon3d/
  config.py         TrainConfig dataclass, YAML loading, --set overrides, validation
  data.py           dataset files, Split, preprocess(), augment()
  models.py         Recon (MLP decoder), ReconAttn (transformer decoder), checkpoint loading
  metrics.py        Chamfer distance, F-score, precision/recall, spread ratio
  train.py          overfit sanity check, resumable training loop
  evaluate.py       model vs nearest-neighbour retrieval baseline
  unseen.py         evaluation on never-seen categories
  ood.py            out-of-distribution detector (feature similarity + threshold)
  inference.py      Predictor for single images
  tracking.py       experiment-tracking interface (MLflow plugs in here)
  io.py, plots.py   checkpoints, PLY export, figures
  hub.py            export/publish/load model bundles on the Hugging Face Hub
  framing.py        object framing statistics of training silhouettes
  serve/            web app: settings, preprocessing, InferenceService, FastAPI API, Gradio UI, telemetry
  cli.py            `recon3d train | evaluate | unseen | export | publish | serve | framing | predict | table`
tests/              pytest suite on a tiny synthetic dataset (runs on CPU in about a minute)
notebooks/          thin Kaggle launchers that pip-install this package and call the CLI
space/              Hugging Face Space entry point (deployed by scripts/deploy_space.py)
scripts/            one-off helpers (check_checkpoint.py)
docs/               design notes for each block
```

## Setup (CPU laptop)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest                    # 122 tests, about 1 min, peaks near 1.4 GB RAM
pytest -m "not slow"      # 10 s smoke run (skips the end-to-end training tests)
ruff check .
```

## Usage

```bash
# training (on a GPU; the dataset folder holds meta.json, sil_*.npy, points_*.npy, index_*.csv)
recon3d train --data /kaggle/input/recon3d-shapenet6-v2 --runs runs \
    configs/overfit.yaml configs/frozen.yaml configs/finetune.yaml

# evaluation of a finished run
recon3d evaluate --data DATA --runs runs --run finetune_resnet18_n1000 --splits val,test

# never-seen categories + OOD detector for the web app
recon3d unseen --data DATA --unseen UNSEEN_DATA --runs runs --run finetune_resnet18_n1000

# web app locally (UI at http://127.0.0.1:8000, API docs at /docs)
recon3d serve --model-dir bundle_v1.0.1

# one image -> .ply
recon3d predict --checkpoint models/best.pt --image chair.png --out chair.ply
```

Any config value can be overridden: `--set epochs=5 --set lr=1e-3`.

## Continuous integration

`.github/workflows/ci.yml` runs ruff and the test suite on Python 3.11 and 3.12 for every push and pull
request. Pushes to `main` that pass also deploy the Space (pinned to the commit) and smoke-test the
running app. Needs one repository secret: `HF_TOKEN` with write access.

## Data

The dataset is built from ShapeNet, whose terms do not allow redistribution, so it is not in this
repository. Model weights are published separately with a model card.
