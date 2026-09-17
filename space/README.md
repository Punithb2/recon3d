---
title: recon3d
emoji: 🪑
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
short_description: One silhouette in, a 3D point cloud out
models:
  - Punith25/recon3d-resnet18
---

# recon3d demo

Single-view 3D reconstruction: a ResNet-18 + MLP model turns one silhouette into 2,048 3D points,
and warns when the input looks unlike its training data.

- UI: `/` · API docs: `/docs` · Metrics: `/metrics`
- Model: [Punith25/recon3d-resnet18](https://huggingface.co/Punith25/recon3d-resnet18)
- Code: [github.com/Punithb2/recon3d](https://github.com/Punithb2/recon3d)

This folder is deployed by `scripts/deploy_space.py`, which pins the package to an exact git commit.
