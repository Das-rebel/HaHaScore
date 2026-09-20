# HaHaScore 😄

**Sentence-level humor strength prediction (0–100) via multimodal fusion of text and audio.**

> **v6 trained.** TriModal Cross-Attention fusion achieves AUC **0.858** (+0.016 over audio-only Bridge 4). Target AUC ≥ 0.85 achieved!

## Performance

| Model | AUC | Notes |
|-------|-----|-------|
| Text-only (RoBERTa) | 0.499 | Random guessing |
| Audio-only (WavLM LR) | 0.538 | Weak signal |
| Fusion v5 (bilinear) | 0.632 ± 0.007 | Per-segment text+audio |
| Bridge 4 (BiGRU arcs) | 0.842 ± 0.027 | Audio-only, sequential |
| **v6 (TriModal CrossAttn)** | **0.858 ± 0.015** | **✓ Target achieved!** |

## Bridge Status

| Bridge | AUC | Status |
|--------|-----|--------|
| Bridge 0 (Text features) | 0.50 | ✅ Random |
| Bridge 1 (Pseudo-labels) | — | ✅ 639 files done |
| Bridge 2 (Cascade gate) | — | 🔄 Needs text features |
| Bridge 3 (Incongruity) | — | ✅ Validated |
| Bridge 4 (BiGRU arcs) | 0.842 | ✅ Audio-only best |
| Bridge 5 (Self-training) | 0.840 | ✅ Confirmed hurts |
| **v6 (TriModal)** | **0.858** | ✅ **Cross-attention fusion** |

## v6: TriModal Cross-Attention Fusion

```
Text (RoBERTa, 768d) ──→ Cross-Attention ──┐
                                                 ├──→ BiGRU(128d×2) → MLP → Score
Audio (WavLM+prosody, 791d) ──→ BiGRU ────────┘
```

**Architecture**:
- Text projection: Linear(768→128) + LayerNorm + ReLU + Dropout(0.3)
- Audio projection: Linear(791→128) + LayerNorm + ReLU + Dropout(0.3)
- Cross-attention: 4-head MultiheadAttention (text↔audio both directions)
- BiGRU(128d, 2 layers, bidirectional)
- MLP(256→128→1) + Sigmoid
- Trainable params: ~1.16M

**Results** (5-fold CV):
| Fold | AUC |
|------|-----|
| 1 | 0.860 |
| 2 | 0.879 |
| 3 | 0.856 |
| 4 | 0.864 |
| 5 | 0.833 |
| **Mean** | **0.858 ± 0.015** |

**Key finding**: Text alone is random (AUC 0.50), but text+audio cross-attention provides +0.016 AUC over audio-only. Text provides semantic context that helps distinguish setup vs punchline structure.

## Research Pipeline

```
StandUp4AI Audio (639 files, 12,780 segments)
    │
    ├── Bridge 1: Prosody extraction → pseudo-labels (0.940 mean)
    │
    ├── Bridge 4: WavLM+prosody → BiGRU arc tracker (AUC 0.842)
    │
    └── v6: Whisper transcription + RoBERTa → TriModal CrossAttn (AUC 0.858)
              ↑
              └── Whisper-base transcription (~10 hours CPU)
                  639 files, 66,851 words, 0 errors
```

## HuggingFace Models

| Repo | Model | AUC | Notes |
|------|-------|-----|-------|
| [`Hayasuki/hahascore-bridge4-arc-tracker`](https://huggingface.co/Hayasuki/hahascore-bridge4-arc-tracker) | Bridge 4 | 0.842 | Audio-only BiGRU |
| `Hayasuki/hahascore-fusion-v5` | Fusion v5 | 0.632 | Superseded |

## Key Findings

1. **Text alone is random** (AUC 0.50) — words carry no humor signal
2. **Audio alone achieves AUC 0.842** — delivery is the dominant signal
3. **Text+audio achieves AUC 0.858** (+0.016) — text provides semantic context
4. **Self-training hurts** — iterative confidence filtering creates distributional shift
5. **Humor ≠ laughter** — pseudo-labels (funniness) ≠ gold labels (laughter)

## Why Audio Dominates

Comedy is fundamentally about **delivery**. The same words with comedic timing,
prosodic emphasis, and vocal inflection are humorous — the words alone are not.

## Gradio Demo

```bash
python3 hahascore_fusion_v5_app.py   # Fusion v5 (AUC 0.632)
```

## GitHub

https://github.com/Das-rebel/HaHaScore
