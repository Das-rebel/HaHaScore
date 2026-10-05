# v1 HaHaScore — Colab + Kaggle Launch Guide

**Date**: 2026-10-05
**Git commit**: f66933e
**Goal**: Run v1 (text-only RoBERTa regression on Jester) and produce honest Spearman ρ baseline.

---

## Quick path: Colab

### Cost
- Free tier: 12 hours/session, 1 GPU (T4 / L4)
- Expected full-run time: 4-6 hours on 1.76M jokes, 5 epochs

### Steps
1. Open https://colab.research.google.com
2. File → Upload notebook → select `v1_jester_train.ipynb`
3. Runtime → Change runtime type → GPU (T4)
4. Run cells 1-8 sequentially:
   - **Cell 1**: install deps + verify GPU
   - **Cell 2**: mount Drive + clone repo
   - **Cell 3**: download Jester (5 min, 0 GPU)
   - **Cell 4**: verify dataset
   - **Cell 5**: smoke test on 10k samples
   - **Cell 6**: full run on 1.76M jokes (4-6 hours)
   - **Cell 7**: check pre-registered gate
   - **Cell 8**: save to Drive

### Expected output

```
=== v1 Results ===
Spearman rho: 0.3500-0.4500 ± 0.05-0.10
95% CI:        [0.30, 0.45]
MAE:           18-25
RMSE:          22-30
n_measurements: 15

=== Pre-registered gates ===
  rho >= 0.35: 0.XXXX -> PASS/FAIL
  ci >= 0.30:  0.XXXX -> PASS/FAIL

VERDICT: ✅ PUBLISHABLE or ⚠️ NOT YET PUBLISHABLE
```

### What to do with results

1. Download `result_<timestamp>.json` from Drive
2. Commit to HaHaScore repo: `cp result_*.json experiments/v1_jester_regression/`
4. Update `IMPROVEMENT_PLAN.md` with the actual numbers
5. If ρ ≥ 0.35: draft the v1 paper using `PAPER_OUTLINE.md`
6. If ρ < 0.35: iterate (larger model, more epochs, LoRA) or publish methodology-only

---

## Quick path: Kaggle

### Cost
- Free tier: 30 GPU hours/week (T4 only — P100 incompatible with PyTorch CUDA 12.6)

### Steps
1. Create a new Kaggle kernel: code → Python → GPU
2. Upload `train_jester_regression.py` and `data/jester_seppev.py`
3. Add SeppeV dataset source: `SeppeV/rated_jokes_dataset_from_jester`
4. Run cell:

```python
!python data/jester_seppev.py --output-dir data/jester
!python train_jester_regression.py --epochs 5 --n-seeds 3 --n-folds 5 --out-dir /kaggle/working/v1_jester
```

5. Save outputs: `kaggle kernels output <username>/<kernel-slug> -p ./output/`

---

## Why text-only RoBERTa on Jester

| Why | Detail |
|---|---|
| **Continuous** | Jester has Z_rating (-10 to +10), no 1.16% positive rate problem |
| **Big** | 1.76M jokes — far more than StandUp4AI's 620v |
| **Free** | Apache-2.0 license, no Busso-style negotiation |
| **CPU-feasible** | RoBERTa-base + Linear(768, 1) = ~125M params, runs in 4-6h on T4 |
| **Multi-language path** | Future v1.1 can extend to MultiLinguahah multilingual laughs + Turkish |

## The 5×3 repeated CV methodology

Per `laugho_cv.py::repeated_joke_disjoint_cv_regression()`:

- **3 seeds × 5 folds = 15 measurements**
- **Joke-disjoint**: same joke never appears in both train and test (any fold, any seed)
- **Bootstrap 95% CI**: 10,000-resample percentile
- **Pre-registered gate**: ρ ≥ 0.35 AND CI excludes 0.30

## What we're NOT doing

- ❌ Audio/multimodal (deferred to v2)
- ❌ Generation (deferred to v3)
- ❌ Jester 150-joke subset (using full 1.76M)
- ❌ StandUp4AI 620v (1.16% positive rate problem)
- ❌ F0/multimodal tricks per my memory (DISQUALIFIED Sep 15 due to label circularity)

## If the run fails the gate

1. Try `roberta-large` (350M params) instead of `roberta-base`
2. Try unfreezing backbone (LoRA r=8)
3. Try 10 epochs instead of 5
4. Try data augmentation (joke text shuffling as regularization)
5. If still failing, publish methodology-only paper — the 5×3 repeated CV applied to continuous regression is the durable contribution

## If the run succeeds

1. Update `experiments/v1_jester_regression/RESULT_<timestamp>.md` with honest numbers
2. Update README.md with the v1 result
3. Draft `arxiv_submission/v1_paper.tex` using `PAPER_OUTLINE.md`
4. Tag release `v1.0-laugho-jester` on GitHub
5. Post to arXiv (cs.CL)

## Reproducibility

```bash
cd /Users/Subho/funny-strength-predictor
python3 data/jester_seppev.py --output-dir data/jester
python3 train_jester_regression.py --epochs 5 --n-seeds 3 --n-folds 5
```

Same numbers should reproduce within ±0.02 (random seed sensitivity).

---

## Total session cost (v1 launch)

| Item | Estimate |
|---|---|
| Cash | $0 |
| GPU | 0h (already cached) → 4-6h T4 (for real run) |
| Human | ~16 hrs total (incl. paper writeup) |
| Commits to HaHaScore | 1-2 (paper + result JSON) |

The pipeline is end-to-end runnable. The user just needs to upload the notebook to Colab and click Run.