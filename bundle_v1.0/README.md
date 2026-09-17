---
license: other
library_name: pytorch
pipeline_tag: image-to-3d
tags: [3d-reconstruction, point-cloud, single-view-reconstruction, shapenet, resnet18]
---

# recon3d: single-view silhouette to 3D point cloud

A ResNet-18 encoder and an MLP decoder that turn one 224x224 silhouette into 2,048 (x, y, z) points. Trained with Chamfer distance on six ShapeNet categories: airplane, rifle, bench, cabinet, chair, table.

Code, training pipeline and tests: https://github.com/Punithb2/recon3d

## Intended use

Research and demonstration. Input: a silhouette (white object on black, or a dark object on a light background) of an object from one of the six training categories, roughly centred. Output: a point cloud in a normalised frame (bounding-box centre at the origin, diagonal = 1, Y up). Not for measurement, safety-relevant or commercial use.

## How to use

```python
from recon3d.hub import load_from_hub
from recon3d.data import preprocess
import torch

b = load_from_hub("Punith25/recon3d-resnet18", revision="<tag>")
sil = torch.zeros(1, 224, 224, dtype=torch.uint8)      # your silhouette, uint8 [1, H, W]
with torch.no_grad():
    feats = b.model.embed(preprocess(sil))
    points = b.model.decode(feats)[0]                # [2048, 3]
familiar = not b.detector.is_ood(feats.numpy())[0]
```

## Training

- Data: airplane, rifle, bench, cabinet, chair, table; 1,000 training meshes per category, 24 rendered views each (azimuth every 45 deg, elevation -30/0/30 deg). Targets: 8,192 area-weighted surface samples per mesh.
- Recipe: linear probe (frozen ImageNet encoder), then fine-tuning of layer3 + layer4 (decoder lr 0.0003, encoder lr 0.0001, batch 64, 40 epochs, cosine schedule, mixed precision, scale/shift augmentation) on a free Kaggle T4.
- Checkpoint: epoch 36 (lowest validation CD x1000: 1.429).

## Evaluation

CD = squared-L2 Chamfer distance x1000 (2,048 vs 2,048 points, both directions). F@1% / F@2% = F-score at 1% / 2% of the bounding-box diagonal. Every test mesh is evaluated from all 24 views. Retrieval = copying the training shape whose image features are closest (a strong baseline).

### Trained categories (test split)

| category | CD (model) | CD (retrieval) | F@1% | F@2% |
|---|---|---|---|---|
| rifle | 0.623 | 1.373 | 0.455 | 0.845 |
| airplane | 0.635 | 1.507 | 0.512 | 0.845 |
| bench | 1.164 | 2.044 | 0.279 | 0.705 |
| cabinet | 1.454 | 2.610 | 0.115 | 0.516 |
| table | 1.530 | 3.631 | 0.200 | 0.609 |
| chair | 1.955 | 3.580 | 0.103 | 0.464 |
| **all** | **1.227** | 2.458 | **0.277** | **0.664** |

### Categories never seen in training

Evaluated on 1,140 meshes from loudspeaker, bathtub, bus, guitar, telephone, laptop.

| category | CD (model) | CD (retrieval) | flagged as unfamiliar |
|---|---|---|---|
| loudspeaker | 5.013 | 9.960 | 13% |
| bathtub | 5.704 | 9.098 | 13% |
| telephone | 6.635 | 17.449 | 16% |
| laptop | 9.961 | 18.755 | 41% |
| guitar | 18.748 | 63.080 | 62% |
| bus | 44.903 | 69.436 | 14% |
| **all unseen** | **15.434** | 31.956 | 26% |

Error on unseen categories is about **12.6x** higher than on trained ones (F@1% 0.041 vs 0.277). The model does not generalise to new categories; it produces shapes that resemble its training categories.

### Unfamiliar-input warning

`ood_detector.npz` holds image features of 48,000 training images. An input is flagged when 1 - (highest cosine similarity) exceeds 0.1045, a threshold that passes 95% of trained-category validation images.

- AUROC trained-test vs unseen: 0.769; flags 26% of unseen images and 4% of trained-category test images.
- Per category AUROC: loudspeaker 0.59, bathtub 0.70, telephone 0.75, bus 0.79, laptop 0.90, guitar 0.93.
- The score correlates with reconstruction error (Spearman 0.58), but it measures how unfamiliar the *image* looks, not how hard the 3D shape is. Objects whose silhouettes resemble training objects can pass unflagged and still be reconstructed badly.

## Limitations

- Six categories only; anything else is reconstructed poorly (see above).
- Silhouettes only: no colour or texture, so front/back ambiguities remain.
- Chamfer training favours averaged, slightly blurred shapes; thin parts are the first to suffer.
- Trained on clean synthetic renders; photos need a good object mask first.

## Data and licence

Trained on renders of ShapeNet models. ShapeNet is available for non-commercial research under its terms of use, so these weights are shared for research and demonstration only. The training data is not redistributed here.

## Provenance

- recon3d 0.1.0, run `finetune_resnet18_n1000`, exported 2026-09-17 20:24:30
- weights sha256 `b65e5367349464d2f2472ac7e51deac5a44d14eb3385ee797ea84bd535802596`
- source checkpoint sha256 `f82a1f33f8934f216c85d9d7d03c19646a4e13826eae2e386fe1c337f6a088ab`
