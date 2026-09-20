---
license: apache-2.0
tags:
- humor-detection
- multimodal
- audio
- speech
- nlp
- comedy
- cross-attention
- gru
- pytorch
datasets:
- StandUp4AI
metrics:
- AUC
model_index:
- name: v6 TriModal
  type: multimodal_humor_detector
  accuracy: 0.858
  literature:
    arxiv: "https://github.com/Das-rebel/HaHaScore"
---

# HaHaScore: Sentence-Level Humor Strength Prediction

**Sentence-level humor strength prediction (0-1) via multimodal fusion of text and audio.**

## Models in this repo

| File | Model | AUC | Description |
|------|-------|-----|-------------|
| `pytorch_model.bin` / `bridge4_arc_tracker.py` | Bridge 4 | 0.842 | BiGRU over WavLM+prosody, audio-only |
| `fusion_v5_model.bin` / `fusion_v5_inference.py` | Fusion v5 | 0.632 | Bilinear fusion (DeBERTa + WavLM), per-segment |
| `v6_trimodal.pt` / `v6_inference.py` | **v6** | **0.858** | **Best — TriModal Cross-Attention (RoBERTa + WavLM)** |

## v6: TriModal Cross-Attention (Best Model)

**5-fold CV AUC: 0.858 ± 0.015** — exceeds our 0.85 target.

```
Text (RoBERTa, 768d) ──→ Cross-Attention ──┐
                                                 ├──→ BiGRU(128d×2) → MLP → Score
Audio (WavLM+prosody, 791d) ──→ BiGRU ────────┘
```

### Architecture
- Text: RoBERTa-base CLS embedding per segment (768d)
- Audio: WavLM-base-plus + 23d prosody per segment (791d)
- Cross-attention: 4-head bidirectional (text ↔ audio)
- BiGRU(128d, 2 layers, bidirectional)
- MLP(256→128→1) + Sigmoid
- Trainable params: ~1.16M

### Training
- Whisper-base transcription of 639 StandUp4AI files (66,851 words, 0 errors)
- RoBERTa-base text features aligned to 20 equal-duration segments per file
- 5-fold CV on pseudo-labels from Bridge 1

### v6 Results (5-fold CV)
| Fold | AUC |
|------|-----|
| 1 | 0.860 |
| 2 | 0.879 |
| 3 | 0.856 |
| 4 | 0.864 |
| 5 | 0.833 |
| **Mean** | **0.858 ± 0.015** |

## Key Findings

1. **Text alone is random** (AUC 0.50) — words carry no humor signal
2. **Audio alone achieves AUC 0.842** — delivery is the dominant signal
3. **Text+audio achieves AUC 0.858** (+0.016) — text provides semantic context
4. **Self-training hurts** — iterative confidence filtering creates distributional shift
5. **Humor ≠ laughter** — pseudo-labels (funniness) ≠ gold labels (laughter)

## Usage

```python
from v6_inference import score_segments
import numpy as np

# Load features
text_features = np.load("text_features.npy")   # (20, 768)
audio_features = np.load("audio_features.npy")   # (20, 791)

# Score
scores = score_segments(text_features, audio_features)  # (20,)
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
