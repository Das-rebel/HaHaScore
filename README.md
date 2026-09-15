# HaHaScore 😄

**Sentence-level humor strength prediction (0–100) via text-audio fusion.**

> ⚠️ **v6 in development.** Current best: AUC 0.632 ± 0.007 (v5 bilinear). Target: AUC ≥ 0.70 with kinetic energy modality.

## Key Finding

**Text-only models are RANDOM (AUC ~0.50) for sentence-level humor detection.
Audio is the primary discriminative signal (+13% AUC improvement).**

## Performance

| Model | AUC | Notes |
|-------|-----|-------|
| Text-only (RoBERTa) | 0.499 | Indistinguishable from random |
| Text-only (DeBERTa) | 0.501 | Indistinguishable from random |
| Audio-only (WavLM) | 0.538 | Moderate discrimination |
| **Fusion v5 (bilinear)** | **0.613** | **RoBERTa + WavLM bilinear** |
| 5-Fold CV AUC | **0.632 ± 0.007** | Stable across video subsets |
| v6 target | ≥ 0.70 | +HuBERT + kinetic energy |

## v6 Architecture (In Progress)

```
Text (RoBERTa/DeBERTa 768d) ──┐
                                ├──→ Bilinear/Cross-Attention → Sigmoid → Humor Score
Audio (WavLM/HuBERT 512d)  ───┤
                                │
Kinetic Energy (YOLOv8s-pose 7d) ─→ TriModal Fusion
```

**New modalities for v6:**
- **HuBERT** vs WavLM comparison (D-G2: MTLLFM F1=99%)
- **Kinetic energy** from pose keypoints (D-G3: TIC-TALK r=−0.75 validated)
- **Cross-attention fusion** (D-G4: DARC-CLIP +4.18 AUROC)

## Models on HuggingFace

- [`Hayasuki/hahascore-text-v7`](https://huggingface.co/Hayasuki/hahascore-text-v7): RoBERTa text-only (word-level AUC 0.566)
- [`Hayasuki/hahascore-fusion-v5`](https://huggingface.co/Hayasuki/hahascore-fusion-v5): RoBERTa bilinear fusion (**AUC 0.613, CV 0.632**)

## Live Demo

**Running at:** http://127.0.0.1:7860

Features:
- **Text tab**: Enter any joke or text → get 0-100 humor score
- **Fusion tab**: Enter text + upload audio → get combined text+audio humor score

## Training

### Local (CPU/MacBook)
```bash
python3 train_fusion_v5.py  # ~1 hour on MacBook CPU
```

### Modal Cloud (GPU)
```bash
modal run modal_train_v6.py  # Tesla T4 16GB (FREE tier)
```

## Research Findings (10x Deep Research)

See [`docs/DEEP_RESEARCH_AND_SCALEUP_PLAN.md`](docs/DEEP_RESEARCH_AND_SCALEUP_PLAN.md) for:
- 35+ datasets catalogued
- 132 humor detection papers analyzed
- 25+ commercial products mapped
- 10 audio foundation models benchmarked

See [`docs/DECISION_GRAPH.md`](docs/DECISION_GRAPH.md) for 10 decision nodes with confidence matrix.

### Key Validated Findings
- **TIC-TALK** (Interspeech 2022): Kinetic energy r=−0.75 with laughter — stillness before punchline predicts more laughter ✅
- **MTLLFM** (WSC Sports): HuBERT + MAE F1=99% on laughter localization ✅
- **DARC-CLIP**: Cross-attention +4.18 AUROC over static fusion ✅
- **No commercial API exists** for humor strength scoring — HaHaScore is unique ✅

## Commercial Landscape

| Product | What it does | vs HaHaScore |
|---------|-------------|--------------|
| WSC Sports | Laughter localization in sports | Adjacent (sports vs comedy) |
| Gong.io | Conversation analysis | Adjacent (sales vs comedy) |
| Hume AI | Laughter detection API | Laughter only (not strength) |
| AssemblyAI | Speech emotion | Adjacent (general vs humor) |
| **HaHaScore** | **Humor strength 0-100** | **Unique niche** |

## Architecture

- **Text encoder**: microsoft/deberta-v3-base OR roberta-base (768d pooler output)
- **Audio encoder**: microsoft/wavlm-base-plus OR facebook/hubert-base (512d/768d)
- **Fusion**: Bilinear (hadamard product) → MLP(384→128→32→1)
- **Training**: 3,774 sentence clips from 38 Indian comedy videos
- **Validation**: 9,211 sentence clips from 10 held-out comedy videos

## Dataset Sources

- **Primary**: 48 Indian comedy videos (YouTube, ~1.8GB audio)
- **Extended**: StandUp4AI EMNLP 2025 (3,617 videos, 7 languages) — labels licensed, videos from YouTube
- **Semi-supervised**: 467K word-level pseudo-labels from fusion model

## Research Paper

See [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for full methodology, ablation studies, and discussion.

## Why Audio Dominates

Comedy is fundamentally about delivery. The same words delivered with comedic timing,
prosodic emphasis, and vocal inflection are humorous — but the words alone are not.

## Dataset License

- Indian comedy videos: Fair use / YouTube TOS (research exemption)
- StandUp4AI labels: Research use only (contact authors)
- HaHaScore model: Apache 2.0
