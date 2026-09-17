# Block: CI/CD with GitHub Actions

One workflow, `.github/workflows/ci.yml`, with two jobs.

## Job 1: tests (every push and pull request)

- Python **3.11 and 3.12** in parallel, so the package isn't accidentally tied to one version.
- Installs the **CPU-only** PyTorch wheel (about 10x smaller than the CUDA build, and the runners have no GPU).
- Runs `ruff check .` then `pytest -q`.
- `concurrency` cancels the previous run when you push again, so queued runs don't pile up.
- Free for public repositories.

## Job 2: deploy to the Space (pushes to `main` only)

- `needs: test` — a commit whose tests failed is never deployed.
- `concurrency: deploy-space` — only one deploy at a time.
- `scripts/deploy_space.py --ref $GITHUB_SHA` uploads `space/` and writes a `requirements.txt` that pins
  the package to a commit.
- **The pin is the last commit that touched `src/` or `pyproject.toml`**, not necessarily `HEAD`.
  A docs-only commit therefore leaves `requirements.txt` unchanged, and the Space restarts in seconds
  instead of reinstalling PyTorch for several minutes.
- `scripts/smoke_space.py` then waits for `/readyz` and sends a real silhouette through
  `/v1/reconstruct`, checking the point count, the OOD block, the request-ID header, `/metrics` and
  `/openapi.json`. **A green deploy means the app answered, not just that files were uploaded.**
- A manual run (**Actions → CI → Run workflow**) accepts a model tag; it sets the Space variable
  `RECON3D_MODEL_REVISION` and then asserts the app really serves that version. That is the
  promote/rollback button for models.

## Secrets

| Where | Name | Why |
|---|---|---|
| GitHub → Settings → Secrets and variables → Actions | `HF_TOKEN` | lets the workflow upload to the Space |
| Space → Settings → Variables and secrets | `RECON3D_*` (optional), `HF_TOKEN` (only for a private model or log upload) | runtime configuration |

The token is never in the repository, and `deploy_space.py` reads it from the environment.

## What this project deliberately does *not* automate

- **Retraining.** There is no stream of new data, so a scheduled retrain would only burn the free GPU
  quota. Training is triggered by hand on Kaggle, and the notebook is thin: install the package, run the CLI.
- **Promotion by metric threshold.** The comparison that matters (bootstrap CIs, paired tests, the
  unseen-category evaluation) is a deliberate reading, not a single number to gate on. The registry step
  is a person choosing a tag.

Saying *why* something is not automated is a stronger answer in an interview than automating it badly.

## Failure modes and where they show up

| Problem | Where you see it |
|---|---|
| A change breaks the package | red `tests` job, deploy never runs |
| Formatting/lint drift | red `ruff` step |
| The Space starts but the model can't load | smoke test fails on `/readyz` after 15 minutes |
| The Space serves an old model | smoke test's revision check (manual run) |
| A deploy needs undoing | re-run the workflow on an older commit, or `python scripts/deploy_space.py --ref <sha>` |
