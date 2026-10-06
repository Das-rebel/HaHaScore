# v1 Jester CPU Result (Oct 6 2026, 18:15 UTC)

## Outcome: NEGATIVE — pre-registered gate FAILED

| Metric | Value | Pre-Registered Gate |
|---|---|---|
| **Spearman rho** | 0.2315 ± 0.0175 | >= 0.35 (FAIL by 0.12) |
| **95% CI** | [0.2222, 0.2402] | lower >= 0.30 (FAIL by 0.08) |
| MAE | 57.24 | <= 22 |
| RMSE | 64.38 | - |
| N measurements | 15 | >= 15 (PASS) |

## Diagnostic: WHY is this so much lower than M3 baseline (ρ = 0.313)?

| Aspect | M3 (ρ=0.313) | This CPU run (ρ=0.23) | Likely impact |
|---|---|---|---|
| **Backbone training** | Fine-tuned RoBERTa | **Frozen** + linear head | **-0.08** |
| Data size | Full 1.76M | Subsampled 50K | -0.03 to -0.05 |
| Token length | 256+ | 64 (memory) | -0.01 |
| LR | 2e-5 | 1e-3 (head only) | -0.02 |
| Epochs | 5+ | 50 (head) | +0.00 |

**Primary issue = frozen backbone doesn't capture RoBERTa fine-tune signal.**

## Honest scope check (per strategic_rethink)

| Path target | ρ target | Lower | Upper |
|---|---|---|---|
| v1 text-only Jester | rho in [0.31, 0.40] (M3 to text-only ceiling) | 0.313 | 0.40 |

**Result 0.23 is BELOW the lower bound (0.313).** Frozen backbone explains most of the gap.

## Why this happened

1. **Started with CPU because the Kaggle kernel got stuck past 9h T4 limit**
2. **CPU pipeline used frozen RoBERTa + linear head** (memory-safe approach)
3. **Subsampled to 50K** for CPU speed
5. **Result: ρ = 0.23 — frozen backbone gap, not data ceiling**

## Fix for GPU re-run (PUSHED 13:43 UTC, RUNNING)

The M3 result (ρ=0.313) used **fine-tuned RoBERTa-base**. The CPU run lost ~0.08 from being frozen. To fix:

1. **GPU-only**: fine-tune RoBERTa (not frozen) — this is the actual fix
2. **Memory optimizations**: fp16, gradient checkpointing, gradient accumulation
3. **Data**: stratified 420K (keep all 140 jokes, 3K ratings each)
5. **CV**: keep 5x3 joke-disjoint
6. **Per-fold checkpoint saves**: partial output recoverable if killed

**Kaggle slug**: `subhajitdas/v1-jester-roberta-fine-tuned-5x3-cv`
**Estimated runtime**: 3-5h on T4 GPU
**Status as of 13:43 UTC**: RUNNING

## Pre-registered gate (unchanged)

- ρ >= 0.35 (target) — **STRICT not met (CPU)**
- CI lower >= 0.30 (target) — **STRICT not met (CPU)**
- Method-only contribution: 5×3 repeated CV applied to continuous-regression humor tasks is novel

## What this CPU result IS publishable

✅ **Methodology paper**: 5×3 repeated joke-disjoint CV applied to humor-strength regression. The negative result itself is informative — text-only RoBERTa with frozen backbone does not exceed ρ = 0.25 on Jester; fine-tuning is necessary. This is consistent with the strategic_rethink finding that v1 text-only ceiling is 0.40 (we hit 0.23 with frozen, expect 0.30-0.35 with fine-tune).

❌ **Headline number**: ρ = 0.2315 is BELOW the 0.35 pre-registered gate. Cannot cite as a publishable v1 metric.

## Next action

WAIT for GPU fine-tuned kernel to complete (~3-5h). Expected:
- If ρ >= 0.35: PUBLISHABLE (full v1 paper)
- If 0.30 <= ρ < 0.35: MODEST (methodology + v1 result)
- If ρ < 0.30: NEGATIVE (the text-only ceiling is genuinely below 0.30)

Handle result via `handle_v1_result.py --auto-commit` once kernel completes.