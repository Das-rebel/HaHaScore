---
language:
  - en
license: apache-2.0
library_name: pytorch
pipeline_tag: other
tags:
  - humor-detection
  - comedy
  - multimodal
  - fusion-model
---

# HaHaScore Fusion v3 — Superseded

**Hugging Face Model ID:** `Hayasuki/hahascore-fusion-v3`

⚠️ **DEPRECATED. Please use [Hayasuki/hahascore-fusion-v5](https://huggingface.co/Hayasuki/hahascore-fusion-v5) instead.**

This is an older fusion model superseded by v5 (5-Fold CV AUC 0.632).

## Why v3 was replaced

- **Architecture**: Bare MLP with no text/audio projections — different from v5's bilinear fusion
- **Performance**: Lower AUC than v5
- **v5 has**: Per-segment prosody, bilinear fusion with hadamard product, much better calibrated scores

## Upgrade to v5

```python
# Old (v3) — no longer recommended
# model_path = hf_hub_download("Hayasuki/hahascore-fusion-v3", "pytorch_model.bin")

# New (v5) — current best
model_path = hf_hub_download("Hayasuki/hahascore-fusion-v5", "pytorch_model.bin")
```

See full documentation at **[Hayasuki/hahascore-fusion-v5](https://huggingface.co/Hayasuki/hahascore-fusion-v5)**.

## License

Apache 2.0

*Updated 2026-09-16*
