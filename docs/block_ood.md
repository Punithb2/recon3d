# Block: unseen categories and the OOD detector

## The question

The model was trained on 6 categories. Real users will upload anything: a car, a lamp, a mug.
Two things need honest answers before the demo goes public:

1. **How badly does the model do on objects it never saw?** Measure it; don't assume it.
2. **Can the app notice?** If an upload looks unlike anything in training, the app should say
   "this result is probably unreliable" instead of silently returning a confident-looking point cloud.

The second part is called **out-of-distribution (OOD) detection**.

## Data: 6 unseen categories

| Category | Why it's in the set |
|---|---|
| loudspeaker | boxy, like cabinet: a *near* OOD case the detector should find hard |
| bathtub | furniture-like, partly similar to bench/cabinet |
| bus | a vehicle; nothing like it in training |
| guitar | thin and elongated, a bit like rifle |
| telephone | small, flat, rounded |
| laptop | two thin planes at an angle |

Picking a mix of near and far categories matters. A detector tested only on very different objects
looks better than it really is.

The Drive `test/` and `val/` folders are both used (up to 100 ShapeNet meshes each). For a category
the model never trained on, *every* mesh is unseen, so both folders are simply one evaluation pool.
The ShapeNet-vs-ModelNet rule filters them exactly as before. Some categories (laptop, guitar) may
end up with fewer than 100 per folder because many of their meshes are ModelNet.

## How the detector works

The encoder already turns every image into a 512-number feature vector. Images of similar objects
get similar vectors (that is also why the retrieval baseline works).

```
score(upload) = 1 - (highest cosine similarity between the upload and any training image)
```

- Close to 0: the upload looks like something in training.
- Larger: nothing in training looks like it.

**Threshold.** An upload is flagged when its score is above a threshold. The threshold is set so
that **95% of seen-category validation images pass**. Unseen categories are *not* used to choose it:
in production we don't know what strange things people will upload, so tuning on six specific
unseen categories would overstate how well it works on the seventh.

**Compact bank.** The full bank (6,000 meshes x 24 views = 144,000 vectors) is about 150 MB, too much
to ship to the web app. The shipped bank keeps 8 of the 24 views (48,000 vectors, about 25 MB
as float16). The evaluation reports AUROC for both, so you can show the compression costs little
(or see if it doesn't).

**Checkpoint binding.** The bank only makes sense for the exact weights that produced it.
`ood_detector.npz` stores the SHA-256 hash of `best.pt`, and `OODDetector.load(..., checkpoint=...)`
refuses a mismatch. That prevents a classic deployment bug: new model, stale detector.

### Why this method and not another

| Option | Pros | Cons |
|---|---|---|
| **Nearest-neighbour cosine (chosen)** | No training; same features the retrieval baseline uses; easy to explain; one matrix multiply | Bank must ship with the app; cost grows with bank size |
| Mahalanobis distance (fit a Gaussian to features) | Tiny (a mean and a covariance) | Assumes one blob-shaped cluster; our 6 categories form several clusters |
| Classifier confidence (train a 6-way category head) | Standard baseline | Needs extra training; networks are often confidently wrong on OOD inputs |
| Deep ensembles / MC dropout | Strong uncertainty estimates | Several models or passes per request; heavy for a free CPU Space |

## Metrics you'll see

- **AUROC**: pick one seen and one unseen image at random. AUROC is the probability the unseen one
  gets the higher score. 0.5 = coin flip, 1.0 = perfect. It doesn't depend on the threshold.
- **Detection rate / false-alarm rate**: at the chosen threshold, the share of unseen images flagged,
  and the share of *seen* test images wrongly flagged (expected to be about 5% by construction).
- **Spearman correlation (score vs Chamfer)**: does a higher score go with a worse reconstruction?
  This decides whether the flag is useful to users, not just statistically separable.
- **Bootstrap CI over meshes**: views of the same mesh are not independent, so confidence intervals
  resample whole meshes (same reasoning as the earlier error analysis).
- **"looks like"**: the category of the nearest training image. It hints at what the model falls back
  to, e.g. loudspeakers turning into cabinets.

## Code map

| File | Role |
|---|---|
| `src/recon3d/ood.py` | `ood_scores`, `auroc`, `threshold_at`, `spearman`, `OODDetector` (save/load, checkpoint hash) |
| `src/recon3d/unseen.py` | `evaluate_unseen`: runs everything above, writes `runs/<run>/unseen/` |
| `src/recon3d/evaluate.py` | now shares `score_split` with `unseen.py` (one prediction+scoring loop, not two) |
| `src/recon3d/cli.py` | new `recon3d unseen` command |
| `tests/test_ood.py`, `tests/test_unseen.py` | hand-checkable AUROC cases, detector round trip, full run on synthetic data |
| `notebooks/unseen_eval_kaggle.ipynb` | Kaggle launcher: installs the package from GitHub, runs a parity check, then `recon3d unseen` |

## Outputs (`runs/finetune_resnet18_n1000/unseen/`)

`findings.md` (the write-up with all numbers), `summary.json`, `per_category.csv`,
`per_sample.csv.gz`, `ood_detector.npz` (goes to the web app), `category_cd.png`, `ood_hist.png`,
`score_vs_cd.png`, `qualitative_unseen.png`.

## Interview questions this prepares you for

- **"Does your model generalise?"** "Not to new categories, and here's by how much, with confidence
  intervals, on six held-out categories, compared to a retrieval baseline."
- **"How do you handle inputs unlike your training data?"** Nearest-neighbour feature similarity, a
  threshold set on in-distribution validation data only, AUROC reported per category, and the warning
  shown in the UI.
- **"Why not tune the threshold on the unseen set?"** Because production OOD inputs are unknown;
  tuning on the test OOD set leaks information and overstates performance.
- **"What if you retrain the model?"** The detector is bound to the checkpoint hash and must be rebuilt;
  the loader refuses a stale one.
