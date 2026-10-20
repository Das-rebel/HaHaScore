# v1 Jester 5×3 CV Result v6 (Oct 20 2026, 11:02 UTC)

## Outcome: GATE NOT MET — 5×3 honest ρ = 0.2920 ± 0.0272

### Headline metric

| Statistic | Value | Pre-Registered Gate |
|---|---|---|
| **Spearman ρ** | **0.2920** ± 0.0272 | ≥ 0.35 (FAIL by 0.058) |
| **95% CI (bootstrap)** | [0.2783, 0.3057] | excludes 0.30 (CI lower 0.278 < 0.30) |
| MAE | 20.50 | - |
| RMSE | 25.19 | - |
| N measurements | **15** (3 seeds × 5 folds) | ≥ 15 (PASS) |
| Total runtime | 17086s (~4.75h on T4) | - |

### vs single-run ρ=0.3402 and previous v4 result

| | ρ | Verdict |
|---|---|---|
| **Single-run baseline (Oct 7)** | **0.3402** | Inflated by +0.048 vs honest 5×3 |
| **v4 5×3 (Oct 8)** | 0.2292 | (no fp16, frozen backbone via cached embeddings) |
| **v6 5×3 (Oct 20)** | **0.2920** | This result — full fine-tune, 1 epoch/fold, no fp16 |

### Fold-luck pattern, now 4× confirmed

| Date | Claim | Single-seed | 5×3 honest | Δ |
|---|---|---|---|---|
| Sep 8 | per-language +0.163 | +0.163 | -0.127 | +0.290 |
| Oct 6 | v1 CPU frozen | (5×3 only) | 0.2315 | n/a |
| Oct 7 | v1 M3 recipe | 0.3402 | 0.2292 | +0.111 |
| **Oct 20** | **v1 M3 recipe (v6 kernel)** | **0.3402** | **0.2920** | **+0.048** |

**Single-seed numbers on Jester/StandUp4AI still inflate, but less severely on this run** (+0.048 vs previous +0.111). The key differences:
- v4 used frozen backbone (cached embeddings) → large divergence
- v6 uses full fine-tune (same as M3 recipe) → smaller divergence
- M3 baseline (0.3127) was also single-fold, so is also inflated

**The honest text-only v1 ceiling is now ~0.29** (better than previous estimate of ~0.23).

### Kernel details

- **Slug**: `subhajitdas/v1-hahascore-kernel-v6-no-pyarrow-pin`
- **Status**: COMPLETE
- **Runtime**: 4.75 hours T4 GPU
- **5×3 protocol**: 3 seeds × 5 folds, 1 epoch per fold, batch=32, lr=2e-5, max_len=128
- **Fixes applied** (per v19_canonical): no pyarrow pin (Kaggle pre-installs compatible version), GPU memory check with try/except, hardcoded /kaggle/working/ made env-driven with cwd fallback, empty-input guard in forward()

### Pre-registered gate evaluation

| Gate | Target | Actual | Verdict |
|---|---|---|---|
| ρ | ≥ 0.35 | 0.2920 | **FAIL by 0.058** |
| CI lower | ≥ 0.30 | 0.2783 | **FAIL by 0.0217** |
| N measurements | ≥ 15 | 15 | PASS |

**Both gates still failed** but the gap narrowed:
- Previous (v4): ρ failed by 0.121
- This (v6): ρ failed by 0.058
- **The v6 kernel (full fine-tune, no fp16) is better than the frozen-backbone approach by +0.063**

### Decision: methodology paper

Per `COUNCIL_DECISION_v1.md`:
- Outcome 1 (ρ ≥ 0.35): full v1 paper → FAIL (still)
- Outcome 2 (0.30 ≤ ρ < 0.35): methodology + v1 result → N/A (ρ < 0.30)
- Outcome 3 (CI includes 0.30): methodology-only paper → **CURRENT CASE**

**Publish the v1 as a methodology paper**, citing ρ=0.2920 as the honest 5×3 baseline with full fine-tune. The single-run 0.3402 was fold-lucky by +0.048.

### Cost

- $0 (Kaggle free tier)
- ~4.75h T4 GPU
- ~5 min human (commit + write up)

### Files

- `experiments/v1_jester_regression/5x3_cv_v6/v1_5x3cv_result.json` (final aggregate)
- `experiments/v1_jester_regression/5x3_cv_v6/partial_results.json` (per-fold)
- `experiments/v1_jester_regression/5x3_cv_v6/RESULT.md` (this file)
