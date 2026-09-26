---
title: HaHaScore Cascade v8
emoji: 🎭
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# HaHaScore Cascade v8 — Humor Strength Predictor

Multimodal humor strength predictor (text + audio fusion) using a **Cascade Gate** architecture.

- **Pretrained** on 30K Reddit jokes (continuous upvote scores)
- **Fine-tuned** on 639 standup comedy files
- **Val AUC: 0.802**
- **Inference: <5ms** (ONNX INT8, 2.9 MB model)

For full architecture details, see [Hayasuki/hahascore-cascade](https://huggingface.co/Hayasuki/hahascore-cascade).
