# Images used by the top-level README

Copy these four files here (they are produced by the pipeline, not by hand):

| File | Where it comes from |
|---|---|
| `demo.gif` | screen recording of the live Space: pick an example -> Reconstruct -> rotate the cloud -> try an "unseen" example and show the warning. 10-15 s, under 5 MB. Windows: ScreenToGif. |
| `category_cd.png` | `runs/finetune_resnet18_n1000/unseen/category_cd.png` (the `recon3d unseen` Kaggle run) |
| `ood_hist.png` | `runs/finetune_resnet18_n1000/unseen/ood_hist.png` (same run) |
| `qualitative_test.png` | `runs/finetune_resnet18_n1000/qualitative_test.png` (the training/evaluation run) — optional, add it to the README where you like |

PNGs from the pipeline are a few hundred KB each, which is fine for git. Keep the GIF small: it is the
first thing anyone sees.
