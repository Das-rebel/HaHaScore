# v1 HaHaScore — Text-only RoBERTa Regression on Jester

**Canonical path** per council realignment (Oct 5 2026).

## Task

Predict continuous humor strength 0-100 from joke text, using a text-only RoBERTa regression head.

## Dataset

- **Source**: `SeppeV/rated_jokes_dataset_from_jester` (HuggingFace, 1,761,440 rows, Apache-2.0)
- **Verified**: HTTP 200, has jokeText + Z_rating inline
- **Avoids StandUp4AI's 1.16% positive class problem**: continuous ratings, no class imbalance

## Architecture

```
roberta-base (frozen for v1.0; LoRA r=8 in v1.1)
  → Linear(768, 1)  # single scalar regression head
```

## Loss

MSE on continuous 0-100 humor strength.

## Validation protocol (per council realignment)

- 5×3 repeated joke-disjoint CV via `laugho_cv.repeated_joke_disjoint_cv_regression()`
- 3 seeds × 5 folds = 15 measurements
- Metrics: **Spearman ρ** (primary), MAE (secondary), RMSE (tertiary)
- 10,000-resample bootstrap 95% CI
- Pre-registered success gate: **ρ ≥ 0.35 AND CI excludes 0.30**

## Pre-registered falsifiers

| Scenario | Verdict |
|---|---|
| seed=42 alone yields ρ > 0.50 but mean across 15 measurements < 0.35 | single-seed fold-luck (analog of +0.163 paper finding) |
| Any single fold produces ρ > 0.7 | log as outlier, recompute without it |
| 95% CI spans zero | gate fails regardless of point estimate |
| joke-disjoint vs random-split gap exceeds |Δρ| = 0.15 | same-joke leakage detected, redo splits |

## Status

- [x] `laugho_cv.repeated_joke_disjoint_cv_regression()` function added
- [x] `train_jester_regression.py --dry-run` runs end-to-end with synthetic data
- [ ] Real run: download Jester + train RoBERTa + apply 5×3 CV

## Files

| File | Status |
|---|---|
| `train_jester_regression.py` | ✅ At repo root |
| `experiments/v1_jester_regression/dry_run_results.json` | ✅ Toy baseline (constant mean predictor) |
| `experiments/v1_jester_regression/result_<timestamp>.json` | ⏳ Pending real run |
| `experiments/v1_jester_regression/PAPER_OUTLINE.md` | ⏳ Pending |

## Cost

| Item | Estimate |
|---|---|
| Cash | $0 |
| GPU | ~4-6 hours T4 (or ~24h CPU for 1.76M jokes) |
| Human | ~8 hours total |

## Why this is canonical

Per `decisions/002_RETHINK_ALPNEWR_FINDINGS.md`:
- The core project (ChuckleNet/HaHaScore) is label-starved at 620v @ 1.16% pos
- Continuous human ratings (Jester) escape the label-starvation
- v1 = text-only RoBERTa regression is the canonical first step
- v2 (multimodal) and v3 (reverse-gen) come after v1 honest baseline is established

Per `documents/PRD.md`:
- Target Jester gold Spearman ρ ≥ 0.35
- TF-IDF+Ridge baseline at ρ = 0.277

Per `rerun_v1_kaggle results`:
- M3 RoBERTa regression achieved ρ = 0.313 (above baseline, below 0.35 target)

The v1 path is **above the TF-IDF baseline, below the multimodal target**. It's the honest middle ground.

## Sister project

`experiments/v2_laugho/` is the sister project (audio-only laughter detection). It uses the same `laugho_cv.py` methodology but is NOT the canonical v1 path.
