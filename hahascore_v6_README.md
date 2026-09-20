---
license: apache-2.0
tags:
- humor-detection
- multimodal
- audio
- speech
- nlp
- comedy
- cascade-gate
- cross-attention
- gru
- pytorch
datasets:
- StandUp4AI
metrics:
- AUC
model_index:
- name: Bridge 7 Cascade Gate
  type: multimodal_humor_detector
  accuracy: 0.860
  literature:
    arxiv: "https://github.com/Das-rebel/HaHaScore"
---

# HaHaScore: Sentence-Level Humor Strength Prediction

**Sentence-level humor strength prediction (0-1) via multimodal fusion of text and audio.**

## Models in this repo (Best to Baseline)

| File | Model | AUC | Description |
|------|-------|-----|-------------|
| `bridge7_cascade.pt` / `bridge7_inference.py` | **Bridge 7** | **0.860** | **✓ Current best** |
| `v6_trimodal.pt` / `v6_inference.py` | v6 | 0.858 | TriModal Cross-Attention |
| `pytorch_model.bin` / `bridge4_arc_tracker.py` | Bridge 4 | 0.842 | Audio-only BiGRU |
| `fusion_v5_model.bin` / `fusion_v5_inference.py` | Fusion v5 | 0.632 | Bilinear fusion (per-segment) |

## Bridge 7: Cascade Gate (Best Model)

**5-fold CV AUC: 0.860 ± 0.018** — current best.

```
Text (RoBERTa, 768d) → text_proj → text_confidence (scalar) → sigmoid
Audio (WavLM+prosody, 791d) → audio_proj (128d)
gated_audio = audio_proj * text_confidence
fused = concat(text_proj, gated_audio, cross_attn_output)
BiGRU(128d×2) → MLP → Score
```

**Architecture**:
- Text: RoBERTa-base CLS embedding (768d) → Linear(768→128) + LayerNorm + ReLU + Dropout(0.3)
- Text confidence: Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
- Audio: Linear(791→128) + LayerNorm + ReLU + Dropout(0.3)
- Cross-attention: 4-head attention (audio → text)
- Gated audio: `audio_proj * text_confidence`
- BiGRU(128d, 2 layers, bidirectional)
- MLP(256→128→1) + Sigmoid
- Trainable params: ~1.0M

**Results** (5-fold CV):
| Fold | AUC |
|------|-----|
| 1 | 0.856 |
| 2 | 0.873 |
| 3 | 0.869 |
| 4 | 0.877 |
| 5 | 0.827 |
| **Mean** | **0.860 ± 0.018** |

## v6: TriModal Cross-Attention

AUC: 0.858 ± 0.015

Same architecture but with bidirectional cross-attention (text↔audio both directions).

## Key Findings

1. **Text alone is random** (AUC 0.50) — words carry no humor signal
2. **Audio alone achieves AUC 0.842** — delivery is the dominant signal
3. **Cascade gate achieves AUC 0.860** — gating mechanism is better than pure cross-attention
4. **Self-training hurts** — iterative confidence filtering creates distributional shift
5. **Humor ≠ laughter** — pseudo-labels (funniness) ≠ gold labels (laughter)

## Usage

```python
from bridge7_inference import score_segments
import numpy as np

# Load features (20 segments × feature_dim)
text_features = np.load("text_features.npy")   # (20, 768)
audio_features = np.load("audio_features.npy")  # (20, 791)

# Score
scores, confidences = score_segments(text_features, audio_features)
# scores: (20,) humor strength 0-1
# confidences: (20,) text confidence 0-1
print(f"Mean humor strength: {scores.mean():.3f}")
```

## Bridge 4 (Audio-Only Baseline)

AUC: 0.842 ± 0.027

- BiGRU(128d, 2 layers, bidirectional) over WavLM+prosody segments
- Input: 768d WavLM + 23d prosody + 4d position = 795d
- Trainable params: ~790K

## GitHub

https://github.com/Das-rebel/HaHaScore

## Citation

```bibtex
@misc{das2026hahascore,
  title={HaHaScore: Sentence-Level Humor Strength Prediction},
  author={Subhajit Das},
  year={2026},
  url={https://github.com/Das-rebel/HaHaScore}
}
```
