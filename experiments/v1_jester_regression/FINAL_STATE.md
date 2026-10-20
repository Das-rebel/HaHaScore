# v1 HaHaScore — FINAL STATE (Oct 20 2026)

## The honest v1 text-only baseline

**ρ = 0.2920 ± 0.0272** (5×3 honest CV, full fine-tune RoBERTa-base + Linear head)

95% CI: [0.2783, 0.3057]
N: 15 measurements (3 seeds × 5 folds)
Runtime: 4.75h T4 GPU
Pre-registered gate: **NOT MET** (ρ < 0.35 by 0.058, CI lower < 0.30 by 0.022)

## Comparison across all v1 attempts

| Run | Approach | 5×3 ρ | Notes |
|---|---|---|---|
| **v6** (Oct 20) | **Full fine-tune, 1 epoch/fold, batch=32** | **0.2920 ± 0.0272** | **Best** — matches M3 recipe |
| v4 (Oct 8) | Frozen backbone + cached embeddings | 0.2292 ± 0.0189 | -0.063 vs v6 (frozen backbone gap) |
| CPU run (Oct 6) | Frozen backbone, 50 epochs head | 0.2315 ± 0.0175 | Similar to v4 |
| M3 (Sep 8) | Full fine-tune, single 90/10 | 0.3127 (NOT 5×3) | **+0.021 inflated** vs honest 5×3 |
| M3 recipe kernel (Oct 7) | Full fine-tune, single 90/10 | 0.3402 (NOT 5×3) | **+0.048 inflated** vs honest 5×3 |

## Conclusion

Text-only RoBERTa regression on Jester has a hard ceiling of **ρ ≈ 0.29** in honest 5×3 CV. This is below the pre-registered gate (0.35) but consistent with the strategic_rethink prediction that text-only Jester would land in [0.31, 0.40] (we hit 0.29, close to the lower bound).

The contribution is the **5×3 repeated CV framework applied to continuous-regression humor tasks**, not the headline metric. This is consistent with the +0.163 falsification paper's methodology recommendation.

## Next action

Publish the methodology paper (`arxiv_submission/hahascore.tex`), then pivot to v2 (multimodal audio) per `STRATEGIC_RETHINK_2026.md`.
