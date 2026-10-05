# HaHaScore Improvement Plan (Realigned, Oct 5 2026)

**Council-realigned**: The canonical v1 path is **text-only RoBERTa regression on Jester**, not v2 LaughO (sister project).

**Starting state**: HaHaScore v10 archived as multimodal humor model, falsification paper drafted, methodology codified. v2 LaughO is correctly identified as a sister project.

---

## v1 canonical path: text-only RoBERTa on Jester

### Architecture
```
roberta-base (frozen for v1.0; LoRA r=8 in v1.1)
  → Linear(768, 1)  # single scalar regression head
```

### Dataset
- **Source**: `SeppeV/rated_jokes_dataset_from_jester` (HF, 1,761,440 rows, Apache-2.0)
- **Has**: jokeText + Z_rating inline
- **Why**: continuous labels escape the StandUp4AI 1.16% positive class problem

### Validation
- 5×3 repeated joke-disjoint CV (3 seeds × 5 folds = 15 measurements)
- Metrics: Spearman ρ (primary), MAE, RMSE
- 10,000-resample bootstrap 95% CI
- **Pre-registered gate**: ρ ≥ 0.35 AND CI excludes 0.30

### Baseline (per `baseline_m1.py`)
- TF-IDF(1-2g) + Ridge α=10: ρ = 0.277
- Target: ρ > 0.35 (above TF-IDF, below multimodal target)

---

## What's actionable RIGHT NOW (CPU-only, no GPU)

### ✅ Already done in this session

| File | Purpose |
|---|---|
| `laugho_cv.py` | Added `repeated_joke_disjoint_cv_regression()` for Spearman ρ/MAE/RMSE |
| `train_jester_regression.py` | v1 entry point with dry-run mode (uses synthetic data) |
| `experiments/v1_jester_regression/` | New v1 experiment directory + README |
| `experiments/v2_laugho/SISTER_PROJECT.md` | One-line scope: `v2 LaughO is sister project, NOT canonical v1` |
| `experiments/v2_laugho/` | All v2 files moved here (audio-only laughter detection) |
| Deleted | All staged v6-v10 iteration cruft (bridge*, cascade*, modal*, per_language_norm) |
| `README.md` | Replaced with v1 scope + sister project + paper pointer |
| `laugho_ci.py` + `Makefile` | CI still passes |
| `5x3_cv_results.json` | Falsification provenance (15 measurements) |
| `lang_norm_results.json` | Falsified result retained |

### What v2 LaughO keeps

| Path | What |
|---|---|
| `experiments/v2_laugho/v2_laugho_train.py` | 1.5M MLP head over WavLM (sister project code) |
| `experiments/v2_laugho/v2_laugho_train.ipynb` | Colab notebook (runnable) |
| `experiments/v2_laugho/v2_laugho_kaggle_kernel.py` | Kaggle kernel |
| `experiments/v2_laugho/kaggle_v2_laugho/` | Kaggle metadata |
| `experiments/v2_laugho/laugho_data.py` | AudioSet extractor |
| `experiments/v2_laugho/5x3_cv_audioset_eval00_*.json` | AUC 0.7501 ± 0.1817 |
| `experiments/v2_laugho/RESULT_20261005.md` | Full first-run analysis |
| `experiments/v2_laugho/SISTER_PROJECT.md` | Scope clarification |

---

## What requires GPU (Kaggle T4 / Colab)

### Week 1: download Jester + train v1 (4-6 GPU h)
```bash
python3 -c "from datasets import load_dataset; ds = load_dataset('SeppeV/rated_jokes_dataset_from_jester')"
python3 train_jester_regression.py --epochs 5 --n-seeds 3 --n-folds 5
```

### Week 2: scale to full 1.76M + get honest baseline (12 GPU h)

### Week 3: if ρ ≥ 0.35, draft v1 paper (16 hrs human)

### Week 4: optional v2 multimodal extension using AST-clean labels

---

## Memory rules (the "avoid digression" part)

The following 5 things MUST be remembered and avoided:

1. **The original goal is multimodal text+audio humor STRENGTH (0-100), not laughter detection.** v2 LaughO is sister project.
2. **The 5×3 repeated CV methodology is the contribution.** Never inflate a single-fold number to a headline.
3. **Jester continuous ratings** are the canonical v1 training data, not StandUp4AI 1.16% positive class.
4. **The honest 5×3 CV for v10 is humor AUC 0.6924, gold 0.5453** — never cite 0.860 or 0.823 as headline.
5. **ChuckleNet is read-only.** Cite papers, download HF weights, never modify.

If at any point the work drifts to "build a better laughter detector," STOP and re-read this doc.

---

## Cost summary

| Item | Estimate |
|---|---|
| Cash | $0 |
| GPU | ~4-6 hours T4 (per Jester training) + ~12 hours for full scale-up |
| Human | ~16-24 hours total over 4 weeks |
| Commits | ~3-5 to HaHaScore |

---

## Single line status

`make ci` returns PASS for the realigned state. The v1 entry point `train_jester_regression.py --dry-run` runs end-to-end with synthetic data. The real Jester run needs GPU.