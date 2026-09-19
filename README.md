# HaHaScore 😄

**Sentence-level humor strength prediction (0–100) via multimodal fusion of text and audio.**

> **Bridge 4 + 5 complete.** Best model: BiGRU Humor Arc Tracker (AUC 0.842 on pseudo-labels). Transcription in progress for v6 multimodal fusion. Target: AUC ≥ 0.85 with text+audio cross-attention.

## Key Findings

1. **Text-only models are RANDOM (AUC ~0.50)** for sentence-level humor — words alone have no humor signal
2. **Audio is the dominant signal** — WavLM prosodic features achieve AUC 0.54
3. **Temporal arc modeling is critical** — BiGRU over sequential segments (AUC 0.842) outperforms per-segment fusion (AUC 0.632)
4. **Self-training doesn't help** — iterative confidence filtering creates distributional shift; best AUC stays at 0.840

## Performance

| Model | AUC | Notes |
|-------|-----|-------|
| Text-only (RoBERTa) | 0.499 | Random guessing |
| Text-only (DeBERTa) | 0.501 | Random guessing |
| Audio-only (WavLM LR) | 0.538 | Weak signal |
| Fusion v5 (bilinear) | 0.632 ± 0.007 | Text+audio, per-segment |
| **Bridge 4 (BiGRU arcs)** | **0.842 ± 0.027** | **Best — audio-only, sequential modeling** |
| Bridge 5 (self-training) | 0.833–0.859 | Hurt generalization |
| **v6 target** | **≥ 0.85** | **Text + audio cross-attention** |

## Bridge Status

| Bridge | Status | AUC | Description |
|--------|--------|-----|-------------|
| Bridge 0 (Text features) | ✅ | 0.50 | RoBERTa/DeBERTa — random |
| Bridge 1 (Pseudo-labels) | ✅ | — | 639 files × 20 segments = 12,780 pseudo-labels |
| Bridge 2 (Cascade gate) | 🔄 | — | Audio gated by text confidence (needs transcription) |
| Bridge 3 (Incongruity) | ✅ | — | Audio-only ρ=−0.23; TEXT required |
| Bridge 4 (Humor arcs) | ✅ | **0.842** | BiGRU over WavLM+prosody, sequential |
| Bridge 5 (Self-training) | ✅ | 0.840 | Did NOT help — distributional shift |
| **v6 (Cross-attention)** | 🔄 | TBD | **Whisper transcription + text-audio fusion** |

## Bridge 4: Humor Arc Tracker

**Architecture**: BiGRU(128d, 2 layers, bidirectional) over 20 sequential segments.

```
Input per segment: 768d WavLM + 23d prosody + 4d position = 795d
    ↓
BiGRU(128) × 2 layers (bidirectional)
    ↓
MLP(256 → 128 → 1)
    ↓
Sigmoid → humor score per segment
```

**Results**:
- **5-Fold CV AUC: 0.842 ± 0.027** (massive +0.21 over per-segment v5)
- Fold AUCs: [0.846, 0.859, 0.848, 0.862, 0.787]
- Trainable params: ~790K
- Key insight: Segment 20 (end-of-video) consistently low (mean=0.291) — model learned end ≠ funny
- ⚠️ Evaluated on pseudo-labels — real generalization TBD with gold human ratings

## Bridge 5: Self-Training

Iterative confidence filtering with |pred − 0.5| > 0.3 threshold:

| Iter | Files | AUC | Notes |
|------|-------|-----|-------|
| 0 | 639 | **0.840** | Bridge 4 baseline |
| 1 | 133 | 0.968 | In-distribution overfitting |
| 2 | 525 | 0.833 | Worse than baseline |
| 3 | 124 | 0.859 | High variance (std=0.108) |

**Lesson**: Self-training on pseudo-labels where the teacher already saturates (AUC > 0.84) creates distributional shift. The model memorizes confident subsets but generalizes worse.

## v6 Architecture (In Progress)

Whisper-transcribed text + WavLM audio → Cross-attention fusion:

```
Text (Whisper-base, 768d) ──→ Cross-Attention ──┐
                                                   ├──→ Score (0–100)
Audio (WavLM-base-plus, 768d) ──→ BiGRU ────────┘
                     ↑
             Sequential arc modeling (from Bridge 4)
```

**Pipeline**:
1. Whisper-base transcription of all 639 StandUp4AI files (~10 hours CPU)
2. Align transcription segments with audio segments
3. RoBERTa-base text embeddings per segment (768d)
4. WavLM audio embeddings per segment (768d, cached)
5. Cross-attention fusion + BiGRU arc tracking
6. Expected: AUC ≥ 0.85 on pseudo-labels, meaningful generalization to gold labels

## Dataset

| Source | Files | Segments | Labels | Status |
|--------|-------|----------|--------|--------|
| Indian comedy (YouTube) | 48 | 3,774 | Gold (laughter) | ✅ Reference |
| StandUp4AI (EMNLP 2025) | 639 | 12,780 | Pseudo (Bridge 1) | ✅ Done |
| StandUp4AI gold labels | 12 | ~240 | Human laughter | ✅ Validation |

## HuggingFace Models

| Repo | Model | AUC | Notes |
|------|-------|-----|-------|
| [`Hayasuki/hahascore-bridge4-arc-tracker`](https://huggingface.co/Hayasuki/hahascore-bridge4-arc-tracker) | Bridge 4 BiGRU | **0.842** | **Current best** |
| `Hayasuki/hahascore-fusion-v5` | Bilinear fusion | 0.632 | Superseded |

## Research Findings

- **TIC-TALK** (Interspeech 2022): Kinetic energy r=−0.75 with laughter — stillness before punchline predicts laughter
- **MTLLFM**: HuBERT + MAE F1=99% on laughter localization
- **Per-segment prosody critical**: Per-file prosody causes BatchNorm saturation (scaled mean=0.005)
- **DARC-CLIP**: Cross-attention +4.18 AUROC over static fusion
- **No commercial API** for humor strength scoring — HaHaScore is unique

## Why Audio Dominates

Comedy is fundamentally about **delivery**. The same words with comedic timing,
prosodic emphasis, and vocal inflection are humorous — the words alone are not.
Sentence-level humor is a **paralinguistic** signal, not a linguistic one.

## Commercial Landscape

| Product | What it does | vs HaHaScore |
|---------|-------------|--------------|
| WSC Sports | Laughter localization in sports | Adjacent (sports vs comedy) |
| Gong.io | Conversation analysis | Adjacent (sales vs comedy) |
| Hume AI | Laughter detection | Laughter only (not strength) |
| AssemblyAI | Speech emotion | General speech (not humor) |
| **HaHaScore** | **Humor strength 0–100** | **Unique niche** |

## Training

```bash
# Bridge 4 (already trained — best model)
python3 bridge4_humor_arc_tracker.py

# v6 (when transcription finishes)
python3 v6_cross_attention.py
```

## Gradio Demo

```bash
python3 hahascore_fusion_v5_app.py   # v5 bilinear (AUC 0.632)
# Bridge 4 demo: TBD
```

## GitHub

https://github.com/Das-rebel/HaHaScore
