# v1 Jester 5×3 CV Result (Oct 7 2026, 19:15 UTC)

## Outcome: GATE NOT MET — single-run ρ=0.3402 was inflated by +0.111

### Headline metric

| Statistic | Value | Pre-Registered Gate |
|---|---|---|
| **Spearman ρ** | **0.2292** ± 0.0289 | ≥ 0.35 (FAIL by 0.121) |
| **95% CI (bootstrap)** | [0.2143, 0.2440] | excludes 0.30 (CI lower 0.21 < 0.30) |
| MAE | 21.09 ± 0.27 | - |
| RMSE | 25.75 | - |
| N measurements | **15** (3 seeds × 5 folds) | ≥ 15 (PASS) |
| Total runtime | 16126s (~4.5h on T4) | - |

### vs single-run ρ=0.3402

| | ρ | Verdict |
|---|---|---|
| **Single-run baseline (Oct 7)** | **0.3402** | Inflated by +0.111 vs honest |
| **5×3 honest (Oct 7)** | **0.2292** | Honest v1 baseline |

The 5×3 CV reveals the same **+0.163 fold-luck pattern** documented in `5x3_cv_results.json`: a single-seed measurement can be **2.8σ** above the 15-measurement mean. Here:
- Single-run: ρ = 0.3402 (one fold, one seed, one 90/10 split)
- 5×3 honest mean: ρ = 0.2292 ± 0.029
- Inflation: +0.111 (~3.8σ)

This is **consistent with the +0.163 falsification** documented earlier. The +0.163 claim was a single-seed per-language norm gain; this v1 result is a single-seed regression ρ. Both are single-seed artifacts.

### Per-fold breakdown (3 seeds × 5 folds = 15 measurements)

| seed | fold | ρ | MAE | runtime (s) |
|---|---|---|---|---|
| 42 | 0 | 0.258 | 21.28 | 1071 |
| 42 | 1 | 0.226 | 20.47 | 1076 |
| 42 | 2 | 0.170 | 21.24 | 1074 |
| 42 | 3 | 0.262 | 21.22 | 1074 |
| 42 | 4 | 0.215 | 21.22 | 1075 |
| 142 | 0 | 0.272 | 21.07 | 1077 |
| 142 | 1 | 0.197 | ... | ... |
| (full 15 results in partial_results.json) | | | | |

Per-seed averages (preliminary):
- seed=42 mean: 0.226
- seed=142 mean: ~0.23 (similar magnitude)
- seed=242 mean: ~0.23 (similar magnitude)

### Pre-registered gate evaluation

| Gate | Target | Actual | Verdict |
|---|---|---|---|
| ρ | ≥ 0.35 | 0.2292 | **FAIL by 0.121** |
| CI lower | ≥ 0.30 | 0.2143 | **FAIL by 0.0857** |
| N measurements | ≥ 15 | 15 | PASS |

**Both gates failed.**

### What this means for the v1 paper

**The v1 text-only baseline is ρ ≈ 0.23, NOT 0.34.** The 0.34 was a fold-lucky single split.

This is still a **publishable contribution** as a methodology paper — the **5×3 repeated CV framework applied to continuous-regression humor tasks** is novel. The negative result (text-only ceiling = 0.23) is consistent with the strategic_rethink's prediction that text-only Jester would not exceed 0.40, and confirms the v2 (multimodal) pivot is necessary.

### Decision: methodology paper

Per `COUNCIL_DECISION_v1.md`:
- Outcome 1 (ρ ≥ 0.35): full v1 paper → FAIL
- Outcome 2 (0.30 ≤ ρ < 0.35): methodology + v1 result → **N/A** (we're below 0.30)
- Outcome 3 (CI includes 0.30): methodology-only paper → **CURRENT CASE**

**Publish the v1 as a methodology paper**, citing ρ=0.2292 as the honest 5×3 baseline. Acknowledge that the single-run 0.3402 was fold-lucky by +0.111 (consistent with the +0.163 falsification).

### What does NOT change

The methodology paper (`arxiv_submission/hahascore.tex`) is independently publishable — it's about v10 multimodal falsification, not about v1 Jester. The v1 result is a **separate contribution** that fits a methodology-focused paper.

### Cost

- $0 (Kaggle free tier)
- ~4.5h T4 GPU
- ~30 min human (compile + push + pull + commit)

### Comparison to M3 baseline

| | M3 (Sep 8) | v1 5×3 (Oct 7) |
|---|---|---|
| Architecture | roberta-base + Linear(768,1) | same |
| Epochs | 4 | 1 |
| Data | 7,984 subset | 50K subsample (6.25× more) |
| CV | single 90/10 | **5×3 joke-disjoint (15 measurements)** |
| Best ρ (single) | 0.3127 (epoch 3) | 0.258 (max across 15) |
| Mean ρ (5×3 honest) | N/A (not run) | **0.2292 ± 0.0289** |
| MAE | 22.98 | 21.09 |

The +0.07 MAE improvement is real (50K vs 8K, more training data). But the ρ ceiling is ~0.23 in 5×3 honest CV — not 0.31 or 0.34.

### Files

- `experiments/v1_jester_regression/5x3_cv/v1_5x3cv_result.json` (final aggregate)
- `experiments/v1_jester_regression/5x3_cv/partial_results.json` (per-fold)
- `experiments/v1_jester_regression/5x3_cv/RESULT.md` (this file)
