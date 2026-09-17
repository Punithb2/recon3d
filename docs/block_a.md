# Block A: from notebook to a tested Python package

## What changed and why

The Kaggle notebook works, but nothing else can reuse it. The web service can't import a notebook cell,
GitHub Actions can't test one, and every copy-paste between notebooks is a chance for training and
serving to drift apart. Block A moves the same logic into a package, `recon3d`, that:

- **Training (Kaggle), tests (GitHub) and the web service (Hugging Face) all install.** One source of truth.
- **Has tests** that run on a laptop CPU in about 30 seconds, using a tiny fake dataset.
- **Loads the checkpoints you already trained**, unchanged.

Nothing about the model or the training recipe changed. On notebook-trained checkpoints, the package's
evaluation matches the notebook's stored test metrics to within 0.0003 on Chamfer x1000
(floating-point noise).

## File by file

| File | What it does | Decision worth defending |
|---|---|---|
| `pyproject.toml` | Package metadata, dependencies, the `recon3d` command, pytest + ruff settings | `torch` is **not** pinned: Kaggle, Spaces and your laptop each have their own CUDA/CPU build, and forcing one would reinstall 2 GB of torch everywhere. |
| `src/` layout | Code lives in `src/recon3d`, not at the repo root | Tests import the *installed* package, not whatever file happens to be in the current folder. That catches packaging mistakes (e.g. a file missing from the package). |
| `config.py` | `TrainConfig` dataclass + YAML files + `--set key=value` | Replaces the `PRESETS` dict and environment variables. Unknown keys are **rejected** (a typo like `epoch: 10` would otherwise silently train for the default 40). The resolved config is saved in every checkpoint. |
| `configs/*.yaml` | One file per experiment | DVC (Block B) and MLflow (Block C) read parameters from these files, so each run is fully described by a file in git. |
| `models.py` | `Recon`, `ReconAttn`, `build_model`, freezing helpers, `load_checkpoint` | Parameter names are identical to the notebook's, and a test proves it with a fingerprint of every name and shape. `load_checkpoint` uses `torch.load(weights_only=True)`: a normal `.pt` file is a pickle, and a pickle can run arbitrary code when loaded. |
| `data.py` | `DatasetMeta`, `Split`, `preprocess`, `augment` | `preprocess` is the **only** image-to-tensor function. Training, evaluation and the web API all call it, which prevents *training/serving skew* (the model seeing differently prepared images in production). |
| `metrics.py` | Chamfer, F-score, precision/recall, spread ratio | Same no-grad nearest-neighbour trick as before. A test checks the value **and the gradient** against brute force. |
| `train.py` | `Session`, `run_overfit`, `train` | No global variables: device, paths and time limit live in a `Session` object, so tests can create one pointing at a temp folder. Behaviour is unchanged: resumable, time-limited, best/last checkpoints, LP-FT decoder initialisation. |
| `tracking.py` | A 4-method `Tracker` interface; currently a no-op | Training calls `tracker.log_metrics(...)` without knowing who listens. Block C adds an MLflow tracker without touching `train.py` (the *dependency inversion* idea). |
| `evaluate.py` | Model vs retrieval baseline, per (mesh, view) | Writes the same files as the notebook, so your error-analysis and scaling notebooks still work. |
| `inference.py` | `Predictor` and `load_silhouette` | Handles dark-on-white images (inverts them) and transparent PNGs (uses the alpha channel). Cropping and resizing uploads to match training framing comes in Block E. |
| `cli.py` | `recon3d train / evaluate / predict / table` | Kaggle will run `recon3d train ...` instead of 8 notebook cells, so the notebook becomes a 3-line launcher (Block F). |
| `io.py` | Atomic checkpoint saves, PLY export | Write to `.tmp`, then rename: a crash mid-save never leaves a corrupt `last.pt`. |
| `scripts/check_checkpoint.py` | Loads a real `best.pt`, predicts, runs sanity checks | Your check on the real weights (step 4 below). |

## What the tests protect (61 tests)

| Test file | Protects against |
|---|---|
| `test_config.py` | Invalid or misspelled configs; `1e-3` being read as text (a real YAML trap, found while writing these tests); old notebook configs failing to load |
| `test_metrics.py` | A wrong Chamfer value or gradient; F-score/precision/recall mistakes; half-precision leaking into the loss |
| `test_models.py` | **Renamed parameters breaking your trained weights** (fingerprint test); freezing the wrong layers; BatchNorm statistics drifting in frozen layers; malicious checkpoints loading |
| `test_data.py` | Wrong normalisation; mismatched dataset files; the random GT subset not being a subset; augmentation destroying the object |
| `test_io.py` | Broken PLY files; half-written checkpoints |
| `test_train.py` | Overfit check not catching a broken pipeline; resume logic; skipping finished runs; LP-FT initialisation; attention decoder; evaluation output format |
| `test_cli.py` | The full `train -> evaluate -> predict -> table` path from the command line; image polarity/alpha handling |

`tests/conftest.py` builds a synthetic dataset (boxes and spheres, 32x32 silhouettes) in exactly the
real on-disk format. Tests use a randomly initialised encoder, 64-px inputs and 128 points, so no
download and no GPU is needed.

Two real bugs showed up while writing the tests, which is the point of having them:

1. `--eval-splits val test configs/frozen.yaml` swallowed the config file as a split name. Splits are now comma-separated (`--eval-splits val,test`).
2. `--set lr=1e-3` produced the *string* `"1e-3"`, because YAML 1.1 needs a dot (`1.0e-3`). Numbers given as text are now converted, and non-numbers are rejected with a clear error.

## Steps for you

### 1. Put the code on your computer
Unzip `recon3d_block_a.zip` into your projects folder.

### 2. Create the environment and run the tests (Windows PowerShell)
```powershell
cd recon3d
py -3.11 -m venv .venv            # any Python 3.10-3.12 works
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest                            # expect: 61 passed
ruff check .                      # expect: All checks passed!
```
`-e` means *editable*: Python uses your source folder directly, so edits apply without reinstalling.
If PowerShell refuses to run `activate`, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Put it on GitHub
Create an empty **public** repository named `recon3d` on github.com (no README, no .gitignore;
public repos get free GitHub Actions minutes). Then:
```powershell
git init
git add .
git commit -m "Package the training pipeline as recon3d with tests"
git branch -M main
git remote add origin https://github.com/<your-username>/recon3d.git
git push -u origin main
```
`.gitignore` already keeps data, checkpoints and `.npy` files out of git.

### 4. Check the real checkpoint
In your Kaggle notebook's latest version: **Output** -> `runs/finetune_resnet18_n1000/best.pt` -> download.
Save it as `models/finetune_n1000.pt` inside the repo (the `models/` folder is git-ignored). Then:
```powershell
python scripts/check_checkpoint.py models/finetune_n1000.pt
recon3d predict --checkpoint models/finetune_n1000.pt --image <any silhouette png> --out test.ply
```
All four checks should say PASS. Open `check_prediction.ply` at https://3dviewer.net to see the cloud.

## Interview questions this block prepares you for

- **"Why not just use notebooks?"** A notebook can't be imported, tested or versioned cleanly.
  The package gives one tested code path shared by training, CI and serving.
- **"How do you know the refactor didn't change the model?"** Parameter-name fingerprints match the
  notebook checkpoints, and re-evaluating notebook checkpoints with the package reproduces the stored
  metrics.
- **"How do you test ML code without a GPU or the real data?"** A synthetic dataset in the real file
  format, a tiny model configuration, and an overfit test that proves the pipeline can learn.
- **"What is training/serving skew?"** The model getting differently prepared inputs in production.
  Here one `preprocess()` function is shared.
- **"Why `weights_only=True`?"** `torch.load` unpickles, and unpickling can execute code. Safe mode only
  allows tensors and plain types.

## Next: Block B (data and model versioning)
DVC will record exactly which dataset version (v1 = 500/category, v2 = 1000/category) and which
config produced each model, without putting the data in git.
