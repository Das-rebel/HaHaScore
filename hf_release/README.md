---
license: mit
tags:
- humor
- multimodal
- text-audio
- cascade-gate
- pytorch
- onnx
datasets:
- SocialGrep/one-million-reddit-jokes
language: en
---

# HaHaScore Cascade v8 — Sentence-Level Humor Strength Predictor

A multimodal **sentence-level humor strength** predictor (0–100 continuous score) that fuses text + audio through a **Cascade Gate** architecture, with **Reddit pretraining** for humor-style priors.

## 🎯 Quick Start

```python
# Option 1: Use ONNX model (fast, no PyTorch)
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("v8_cascade_int8.onnx")
text = np.random.randn(1, 20, 768).astype(np.float32)    # text features
audio = np.random.randn(1, 20, 791).astype(np.float32)  # audio features
cross = np.zeros((1, 20, 16), dtype=np.float32)         # optional cross-modal

scores, conf, gate = session.run(None, {
    'text': text, 'audio': audio, 'cross': cross
})
print(f"Avg score: {scores.mean():.3f}, Max: {scores.max():.3f}")
```

```python
# Option 2: PyTorch
import torch
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(repo_id="Hayasuki/hahascore-cascade",
                             filename="v8_finetuned.pt")
ckpt = torch.load(ckpt_path, map_location='cpu')
```

## 📊 Performance

| Model | Reddit Pretrain | Val AUC | CPU Inference |
|-------|----------------|---------|---------------|
| v7 baseline | None | 0.860 (5-fold CV) | ~10ms |
| v8 v1 (smoke Reddit) | 2K samples | 0.807 | ~13ms |
| **v8 v1 (full Reddit)** | **30K samples** | **0.802** | **~9ms** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Reddit DistilBERT (66M params)         │  ← Continuous funniness pretraining
│  Pretrained on 30K Reddit upvotes       │
└─────────────┬───────────────────────────┘
              │ 768-dim
              ▼
┌─────────────────────────────────────────┐
│  Cascade Gate Fusion v8                 │
│  • text_proj: 768→128                   │
│  • audio_proj: 791→128 (WavLM/Prosody)  │
│  • Multi-scale Attention                │
│  • Cross-Modal Attention (4 heads)      │
│  • Dynamic Gating (text-confidence)     │
│  • BiGRU (2 layers, hidden=128)         │
│  • Sigmoid Score Head                   │
└─────────────┬───────────────────────────┘
              │ (20, 1) per segment
              ▼
       Humor Strength Score (0-1)
```

## 📦 Training Pipeline

### Stage 1: Data Acquisition (Google Drive, no local overflow)
- **1M Reddit jokes** (CC-BY-4.0) — continuous funniness via upvotes
- **ColBERT 200K** (CC-BY-2.0) — text humor detection pretraining
- **639 standup files** with per-segment pseudo-labels

### Stage 2: Reddit Pretrain (30K samples)
- DistilBERT (66M params)
- 1 epoch, batch=8, LR=2e-5, max_len=128
- Streaming dataset (never materialized locally)
- **Final loss: 0.0177** (started at 0.0673, 74% reduction)

### Stage 3: v8 Fine-tune (639 standup files)
- Cascade Gate initialized with Reddit DistilBERT
- 5 epochs, batch=4, LR=1e-4
- Text encoder frozen, audio + gate trained
- **Val AUC: 0.802**

### Stage 4: ONNX Export
- FP32: 0.3 MB, ~3.3ms CPU
- **INT8: 2.9 MB, ~3.3ms CPU** (recommended)

## 🔬 Inputs/Outputs

### Inputs
- **text**: float32, shape (batch, 20, 768) — pre-extracted text features
- **audio**: float32, shape (batch, 20, 791) — WavLM + prosody features
- **cross**: float32, shape (batch, 20, 16) — optional cross-modal features

### Outputs
- **scores**: float32, shape (batch, 20, 1) — humor strength per segment [0-1]
- **text_confidence**: float32, shape (batch, 20, 1)
- **gate_weight**: float32, shape (batch, 20, 1)

## 📁 Files in this Repository

| File | Description |
|------|-------------|
| `v8_finetuned.pt` | PyTorch model (270 MB) |
| `v8_cascade_fp32.onnx` | ONNX FP32 (0.3 MB) |
| `v8_cascade_int8.onnx` | ONNX INT8 quantized (2.9 MB) |
| `v8_finetuned.json` | Training metrics |
| `v8_cascade_metadata.json` | Architecture spec |
| `reddit_distilbert.json` | Pretrain metrics |

## 📚 Citation

```bibtex
@misc{hahascore-cascade-v8,
  author = {Das-rebel},
  title = {HaHaScore Cascade v8: Multimodal Humor Strength Predictor with Reddit Pretraining},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Das-rebel/HaHaScore}},
}
```

## 📜 License

MIT License. Reddit data: CC-BY-4.0 (credit r/Jokes).
