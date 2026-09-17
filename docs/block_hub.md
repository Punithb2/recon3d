# Block: the model on the Hugging Face Hub

## Why a separate model repo

The web app should not contain the weights, and git (GitHub) is the wrong place for a 76 MB binary.
The Hugging Face Hub is free, built for model files, and gives every upload a commit hash.
So the project has three homes, each for one kind of thing:

| What | Where | Versioned by |
|---|---|---|
| Code, configs, tests | GitHub `Punithb2/recon3d` | git commits |
| Data | private Kaggle datasets (v1, v2, unseen6-v1) | Kaggle dataset versions |
| Trained model + detector + card | HF Hub `Punithb2/recon3d-resnet18` | git tags `v1.0`, `v1.1`, ... |

## The bundle

`recon3d export` turns a finished run folder into five files:

| File | Why |
|---|---|
| `model.safetensors` | Weights in a format that stores only tensors. Loading a `.pt` file unpickles Python objects, which can execute code; safetensors can't. It also loads faster (memory-mapped). |
| `config.json` | Everything needed to rebuild the model (the `TrainConfig`), plus provenance: which checkpoint it came from (sha256), which epoch, which package version. |
| `ood_detector.npz` | The feature bank and threshold from the unseen-category evaluation. Re-bound to the hash of `model.safetensors`, so a bundle whose weights were swapped refuses to load. |
| `metrics.json` | The numbers the card was generated from, machine-readable (the app's `/v1/model` endpoint will serve them). |
| `README.md` | The model card: intended use, training, results including the unseen-category failure, limitations, licence note. Generated from the metrics, so it can't drift from them. |

Export checks itself before returning: it reloads the bundle and confirms the exported model gives
**bit-identical** predictions to `best.pt`.

## Versions without a registry server

`recon3d publish` uploads the bundle as one commit and adds a **tag** (`v1.0`). Tags are never reused:
publishing `v1.0` twice is an error. The app loads `revision="v1.0"`, so:

- uploading a new model never changes the live app by accident;
- rolling back means changing one setting from `v1.1` back to `v1.0`;
- anyone can see exactly which files a version contained.

That covers what a model registry is for (versions, lineage, promotion, rollback) with a free service.
MLflow's registry does the same with more features, but needs a server.

## Public or private?

The weights are trained on ShapeNet, whose terms allow non-commercial research use. Many published
ShapeNet models share weights for research, and the card states this restriction. Public is best for a
portfolio (anyone can open the card). If you'd rather be careful, use `--private`: private model repos are
free, and the Space reads it with an access token stored as a Space secret.

## Interview questions

- **"How do you version models?"** Immutable Hub tags; the deployment pins a tag; provenance hashes link
  weights to the training checkpoint and the OOD bank to the weights.
- **"Why safetensors?"** No arbitrary code execution on load, and fast memory-mapped loading.
- **"How do you know the exported model is the trained model?"** Export reloads it and compares predictions
  bit for bit before anything is uploaded.
- **"How would you roll back?"** Point the app at the previous tag and restart.
