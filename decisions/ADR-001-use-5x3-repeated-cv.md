# ADR-001: Use 5×3 repeated joke-disjoint CV for all continuous-regression evaluations

**Status**: Accepted
**Date**: 2026-10-07
**Deciders**: 4-agent council (methodology, engineering, research-strategy, skeptic)

## Context

Single-seed evaluations on Jester and StandUp4AI systematically inflate performance by 0.05–0.30 ρ/AUC relative to the honest 5×3 mean. This has been confirmed 4 times:

| Date | Claim | Single-seed | 5×3 honest | Δ |
|---|---|---|---|---|
| Sep 8 | per-language norm +0.163 | +0.163 | -0.127 (p=0.0004) | +0.290 |
| Oct 7 | v1 M3 recipe ρ=0.3402 | 0.3402 | 0.2292 (v4, frozen) | +0.111 |
| Oct 20 | v1 M3 recipe ρ=0.3402 | 0.3402 | 0.2920 (v6, full fine-tune) | +0.048 |
| — | v1 CPU frozen | (5×3 only) | 0.2315 | n/a |

## Decision

All continuous-regression evaluations MUST use 5×3 repeated group-disjoint CV with 10,000-resample bootstrap 95% confidence intervals. The implementation is `laugho_cv.py::repeated_joke_disjoint_cv_regression()`.

Pre-registered gate for v1 (Jester regression): **ρ ≥ 0.35 AND CI lower bound ≥ 0.30**.

## Consequences

- Single-seed numbers are NOT publishable as headline metrics.
- The v1 text-only baseline is ρ = 0.2920 (honest 5×3, full fine-tune), NOT 0.3127 (M3 single-fold) or 0.3402 (single 90/10 split).
- All future evaluation protocols must be pre-registered before running.
- This ADR should be cited whenever someone proposes using a single-fold evaluation.

## References

- `5x3_cv_results.json` — the original falsification experiment
- `experiments/v1_jester_regression/5x3_cv_v6/v1_5x3cv_result.json` — the v6 full fine-tune 5×3 result
- `arxiv_submission/hahascore.tex` — the methodology paper documenting the +0.163 falsification
- `laugho_cv.py` — the CV harness implementation
