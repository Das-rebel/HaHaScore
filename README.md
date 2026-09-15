# HaHaScore 😄

**Sentence-level humor strength prediction (0–100) via text-audio fusion.**

## Key Finding

**Text-only models are RANDOM (AUC ~0.50) for sentence-level humor detection.
Audio is the primary discriminative signal (+13% AUC improvement).**

## Performance

| Model | AUC | Notes |
|-------|-----|-------|
| Text-only (RoBERTa) | 0.499 | Indistinguishable from random |
| Text-only (DeBERTa) | 0.501 | Indistinguishable from random |
| Audio-only (WavLM) | 0.538 | Moderate discrimination |
| **Fusion v5 (bilinear)** | **0.613** | **DeBERTa + WavLM bilinear** |
| 5-Fold CV AUC | **0.632 ± 0.007** | Stable across video subsets |

## Models on HuggingFace

- [`Hayasuki/hahascore-text-v7`](https://huggingface.co/Hayasuki/hahascore-text-v7): RoBERTa text-only (word-level AUC 0.566)
- [`Hayasuki/hahascore-fusion-v3`](https://huggingface.co/Hayasuki/hahascore-fusion-v3): DeBERTa concat MLP fusion (AUC 0.601)
- [`Hayasuki/hahascore-fusion-v5`](https://huggingface.co/Hayasuki/hahascore-fusion-v5): DeBERTa bilinear fusion (AUC 0.613, CV 0.632)

## Live Demo

**Running at:** http://127.0.0.1:7860

Features:
- **Text tab**: Enter any joke or text → get 0-100 humor score
- **Fusion tab**: Enter text + upload audio → get combined text+audio humor score

## Architecture

- **Text encoder**: microsoft/deberta-v3-base (768d pooler output)
- **Audio encoder**: microsoft/wavlm-base-plus (512d)
- **Fusion**: Bilinear (hadamard product) → MLP(384→128→32→1)
- **Training**: 3,774 sentence clips from 38 Indian comedy videos
- **Validation**: 9,211 sentence clips from 10 held-out comedy videos

## Research Paper

See [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for full methodology, ablation studies, and discussion.

## Why Audio Dominates

Comedy is fundamentally about delivery. The same words delivered with comedic timing,
prosodic emphasis, and vocal inflection are humorous — but the words alone are not.

1. **Timing**: Comedians use pauses,节奏 variations
2. **Tone**: Vocal pitch and energy changes signal punchlines
3. **Emphasis**: Stress on certain words creates comedic effect
4. **Rhythm**: The overall rhythm of delivery makes something funny or not

Text models see only the words — they miss all of this.

## Limitations

- Training data from 38 Indian comedy videos (limited style diversity)
- Binary labels (funny/not funny) — continuous scoring needs validation
- Sentence boundaries via 0.5s gap heuristic
- Audio limited to last 6 seconds per sentence

## Citation

```bibtex
@article{das2026hahascore,
  title={HaHaScore: Sentence-Level Humor Strength Prediction via Text-Audio Fusion},
  author={Das, Subhajit},
  year={2026}
}
```
