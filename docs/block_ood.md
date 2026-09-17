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

**Compact bank.** The full bank (6,000 meshes x 24 views = 144,000 vectors) is about 150 MB as float16,
too much to ship to the web app. The shipped bank keeps 8 of the 24 views (48,000 x 512 x 2 bytes
= 49 MB, 43 MB compressed). The evaluation reports AUROC for both, so you can show the compression costs little
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

## Results (model `finetune_resnet18_n1000`, Kaggle run on the real data)

**Parity.** Re-evaluating the notebook's checkpoint with the package reproduced the notebook's test
metrics to within 0.01% (CD x1000 1.2268 vs 1.2269).

| | meshes | model CD x1000 | model F@1% | retrieval CD x1000 |
|---|---|---|---|---|
| trained categories (test) | 600 | 1.23 [1.12, 1.36] | 0.277 | 2.46 |
| unseen categories | 1,140 | 15.43 [14.53, 16.36] | 0.041 | 31.96 |

What the numbers say:

1. **No generalisation to new categories.** Error is 12.6x higher, and F@1% drops from 0.28 to 0.04.
2. **"Beats retrieval" is not "works".** On unseen objects the model still has half the retrieval error,
   but its F-score is near zero. Chamfer punishes a confident wrong shape (retrieval copies a real
   training object) more than a vague average shape (what the model tends to produce), so the lower
   CD most likely reflects hedging, not understanding.
3. **Distance from training matters.** Box-like categories close to cabinet/table (loudspeaker 5.0,
   bathtub 5.7, telephone 6.6) degrade 3-5x; guitar (18.7, resembles rifle) and bus (44.9) fail badly.
4. **The detector is useful but weak.** AUROC 0.77 (0.79 with the full bank, so the 3x smaller bank
   costs 0.02). With 4% false alarms on trained categories it flags only 26% of unseen images. It is good
   on guitar (0.93) and laptop (0.90) and near chance on loudspeaker (0.59).
5. **Its biggest miss is the worst category.** Buses have the highest error but only 14% are flagged:
   a bus silhouette looks like a long box, close to cabinets in feature space. The score measures how
   unfamiliar the *image* looks, not how hard the *3D shape* is. The app's wording must reflect that.
6. **The score is still a meaningful confidence signal.** Spearman 0.58 with error overall and 0.57
   *within* trained categories, so the app shows it as a continuous familiarity value, not only a flag.
