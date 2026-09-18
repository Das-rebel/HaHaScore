# HaHaScore 😄

**Sentence-level humor strength prediction (0–100) via multimodal fusion.**

> **Bridge 1 complete.** 639 StandUp4AI comedy files (12,780 segments) pseudo-labeled with per-segment prosody. v6 training in progress. Target: AUC ≥ 0.70.

## Key Finding

**Text-only models are RANDOM (AUC ~0.50) for sentence-level humor detection.
Audio prosody is the primary discriminative signal (+13% AUC improvement).**

## Performance

| Model | AUC | Notes |
|-------|-----|-------|
| Text-only (RoBERTa) | 0.499 | Indistinguishable from random |
| Text-only (DeBERTa) | 0.501 | Indistinguishable from random |
| Audio-only (WavLM) | 0.538 | Moderate discrimination |
| **Fusion v5 (bilinear)** | **0.613** | **RoBERTa + WavLM bilinear** |
| 5-Fold CV AUC (v5) | **0.632 ± 0.007** | Stable across video subsets |
| **Bridge 4 (GRU arcs)** | **0.842 ± 0.027** | **BiGRU over WavLM+prosody segments** |
| **v6 target** | **≥ 0.85** | **Cross-attention + cascade gate + self-training** |

## Bridge Status

| Bridge | Status | Description |
|--------|--------|-------------|
| Bridge 0 (Text features) | ✅ | DeBERTa-v3-base features extracted for 639 files |
| Bridge 1 (Pseudo-labels) | ✅ | 639 files × 20 segments = 12,780 pseudo-labels |
| Bridge 2 (Cascade gate) | 🔄 | Audio gated by text confidence |
| Bridge 3 (Incongruity) | ✅ | Validated: audio-only ρ=−0.23, TEXT required |
| Bridge 4 (Humor arcs) | ✅ | **5-Fold CV AUC = 0.8422 ± 0.027** |
| Bridge 5 (Self-training) | 📋 | Iterative label refinement |
| **v6 training** | **🔄** | **TriModal fusion on Modal T4** |

### Bridge 1 Results (Completed Sep 18 2026)
- **639 files** from StandUp4AI (EMNLP 2025) processed
- **12,780 segments** (20 per file, ~3 seconds each)
- **Score distribution**: mean=0.730, std=0.250, range=[0.001, 1.000]
- **66.2%** of segments ≥0.7 (high humor), **8.0%** <0.3 (low humor)
- Per-segment prosody with WavLM + MFCC + pitch features
- 4× downsample (16kHz→4kHz) for efficient WavLM inference
- **0 errors** across all 639 files

### Bridge 4 Results (Completed Sep 19 2026)
- **Bidirectional GRU** over 20 sequential segments per file
- **Input**: 791d per segment (768d WavLM + 23d prosody) + 4d position encoding
- **Architecture**: BiGRU(128) × 2 layers → MLP(256→128→1)
- **5-Fold CV AUC: 0.8422 ± 0.027** (massive improvement over v5's 0.632)
- Fold AUCs: [0.8459, 0.8575, 0.8509, 0.8667, 0.7898]
- Trainable params: ~790K
- **Key insight**: Sequential modeling of humor arcs captures temporal dynamics missed by per-segment fusion
- Segment 20 consistently low (mean=0.291) — model learned end-of-video ≠ funny
- ⚠️ Evaluated on pseudo-labeled data (in-distribution) — real generalization TBD

## v6 Architecture (In Progress)

```
Text (DeBERTa-v3-base 768d)  ──┐
                                  ├──→ Bilinear/CrossAttn → CascadeGate → Sigmoid → Score
Audio (WavLM-base 512d)      ───┤
                                  │
Kinetic Energy (YOLOv8s 7d)  ────┘
     ↑
     └── GRU over 20 sequential segments (Humor Arc Tracker)
```

**New modalities for v6:**
- **HuBERT** vs WavLM comparison (MTLLFM F1=99%)
- **Kinetic energy** from pose keypoints (TIC-TALK r=−0.75 validated)
- **Cascade gate** gating audio by text confidence
- **Humor arc tracker** (GRU over sequential segments)
- **Cross-attention fusion** (DARC-CLIP +4.18 AUROC)

## Live Demo

```bash
python hahascore_fusion_v5_app.py
```
Then open http://127.0.0.1:7860

Features:
- Upload WAV/MP3/M4A audio clip (3–6 seconds)
- Enter transcribed sentence
- Get 0–1 humor strength score with interpretation

## Models on HuggingFace

| Repo | Description |
|------|-------------|
| [`Hayasuki/hahascore-fusion-v5`](https://huggingface.co/Hayasuki/hahascore-fusion-v5) | **Current best** — DeBERTa + WavLM bilinear, AUC 0.613 |
| [`Hayasuki/hahascore-fusion-v3`](https://huggingface.co/Hayasuki/hahascore-fusion-v3) | Superseded — bare MLP (no projections) |
| [`Hayasuki/hahascore-text-v7`](https://huggingface.co/Hayasuki/hahascore-text-v7) | Text-only baseline — AUC ~0.50 (random) |
| [`Hayasuki/chuckle-net`](https://huggingface.co/Hayasuki/chuckle-net) | FusionMLP weights (461KB) |

## Training

```bash
# Local CPU training
python3 train_fusion_v5.py

# Modal cloud (T4 GPU — FREE tier)
modal run modal_train_v6.py
```

## Research Findings

See [`docs/DEEP_RESEARCH_AND_SCALEUP_PLAN.md`](docs/DEEP_RESEARCH_AND_SCALEUP_PLAN.md):
- 35+ datasets catalogued
- 132 humor detection papers analyzed
- 25+ commercial products mapped
- 10 audio foundation models benchmarked

### Key Validated Findings
- **TIC-TALK** (Interspeech 2022): Kinetic energy r=−0.75 with laughter — stillness before punchline predicts more laughter ✅
- **MTLLFM** (WSC Sports): HuBERT + MAE F1=99% on laughter localization ✅
- **DARC-CLIP**: Cross-attention +4.18 AUROC over static fusion ✅
- **Per-segment prosody**: Model trained on 3-second windows; per-file prosody causes BatchNorm saturation ✅
- **No commercial API** for humor strength scoring — HaHaScore is unique ✅

## Dataset

| Source | Files | Labels | Status |
|--------|-------|--------|--------|
| Indian comedy (YouTube) | 48 | 3,774 clips | ✅ Gold standard |
| StandUp4AI (EMNLP 2025) | 639 | 12,780 pseudo | ✅ Bridge 1 done |
| StandUp4AI with gold labels | 12 | 240 clips | ✅ Validation set |

## Architecture

- **Text encoder**: microsoft/deberta-v3-base (768d [CLS] pooler)
- **Audio encoder**: microsoft/wavlm-base-plus (512d mean-pooled last hidden state)
- **Fusion**: Bilinear (hadamard product) → concat(384) → MLP(384→128→32→1)
- **Trainable params**: ~111K (fusion only; encoders frozen)
- **Training**: 3,774 sentence clips from 38 Indian comedy videos
- **Validation**: 9,211 clips from 10 held-out comedy videos

## Why Audio Dominates

Comedy is fundamentally about delivery. The same words delivered with comedic timing,
prosodic emphasis, and vocal inflection are humorous — but the words alone are not.

## Commercial Landscape

| Product | What it does | vs HaHaScore |
|---------|-------------|--------------|
| WSC Sports | Laughter localization in sports | Adjacent (sports vs comedy) |
| Gong.io | Conversation analysis | Adjacent (sales vs comedy) |
| Hume AI | Laughter detection API | Laughter only (not strength) |
| AssemblyAI | Speech emotion | Adjacent (general vs humor) |
| **HaHaScore** | **Humor strength 0–100** | **Unique niche** |

## License

- HaHaScore model: Apache 2.0
- StandUp4AI labels: Research use only (contact authors)
- StandUp4AI videos: YouTube TOS (research exemption)

## GitHub

https://github.com/Das-rebel/HaHaScore
