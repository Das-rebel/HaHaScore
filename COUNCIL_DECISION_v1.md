# Council Decision — v1 rho=0.3402 (Oct 7 2026)

**Date**: 2026-10-07
**Council**: 4 parallel critics (methodology, engineering, research strategy, skeptic)
**Verdict**: ✅ **Unanimous — adopt TIER 2 (apply 5×3 CV), TIER 1 REJECT, rho=0.3402 not publishable as headline**

---

## Per-memeber verdict

### 🟡 Methodologist (commit 62d1cd6 evidence)
- **Gate FAILS**: ρ = 0.3402 < 0.35 by 0.0098
- **CI #2 not evaluable** — kernel was deliberately single-fold (per `v1_jester_m3_recipe.py:24-25`)
- **Within-run std is 0.0021** (gradient noise on one fixed val set), which is NOT informative about fold-luck
- **Honest framing**: gate "failed by 0.01" is exactly the single-fold-inflation pattern the council flagged on Sep 15
- 5×3 CV honestly expected to land in [0.30, 0.32]

### 🔧 Engineer
- Saved model approach = **NO** (Kaggle-output junction creates data-leakage risk)
- **Fresh 1-epoch per-fold training** in a self-contained second kernel = YES
- Identical 50K subsample (`joke_id = abs(hash(joke_text)) % 10^12`, seed=42) to v1
- 1 epoch per fold (v1 showed epoch-1 best — no overtraining)
- **Runtime estimate**: 16 min/fold × 15 folds = ~4h on T4, well under 9h limit
- File written: `kaggle_v1_jester/v1_jester_m3_cv_5x3.py`

### 📚 Research Strategist
- **TIER 1 (ship now) REJECT**: Would violate the paper's own protocol (mandates 5×3 + bootstrap CI)
- **TIER 2 (apply 5×3 CV) ADOPT** — 0.3402 is a single-seed point, not 5×3 verified
- **TIER 3 (pivot to v2) DEFER** per Strategic Rethink (data blocked)
- Outcome scenarios:
  - mean ≥ 0.35 AND CI excludes 0.30: full v1 paper (~30% probability)
  - 0.30 ≤ mean < 0.35: methodology paper with v1 as corroboration
  - CI includes 0.30: methodology-only paper
- v1 paper is NOT blocked by 0.3402 — it's blocked by absence of 5×3 CV
- Expected v1 paper headline range: ρ ≈ 0.34–0.36 (not higher)

### 🔬 Skeptic
- **Most rigorous finding**: per-epoch std=0.0021 is **irrelevant** to fold-variance estimation
- **CPU 5×3 gave 0.2315 ± 0.0175** — same task, weaker model → fold-std empirical anchor
- **Under std=0.022 (optimistic)**: 95% CI = [0.330, 0.350] — gate fails on mean (0.34 < 0.35)
- **Under std=0.035 (likely)**: 95% CI = [0.319, 0.361] — gate fails on mean
- **Under std=0.05 (pessimistic)**: 95% CI = [0.308, 0.372] — gate fails
- **Under NO realistic assumption does gate pass** — the ρ < 0.35 mean is too close
- 0.3402 is structurally identical to the +0.163 number that just got falsified
- Only honest publishable framing: methodology paper OR ceiling finding

---

## Synthesis — 4-agent unanimous verdict

| Decision | Vote |
|---|---|
| Ship rho=0.3402 as v1 headline | ❌ REJECT (4/4) |
| Apply 5×3 CV using fresh kernel | ✅ ADOPT (4/4) |
| Publish methodology paper only (TIER 1) | ❌ WAIT until 5×3 done |
| Pivot to v2 audio | ⏸️ DEFER per Strat Rethink |

---

## Concrete execution plan (unanimous)

### Action 1: Push the 5×3 CV kernel (engineer's `v1_jester_m3_cv_5x3.py`)
- Slug: `subhajitdas/v1-jester-m3-5x3cv-reproduction`
- Self-contained, no Kaggle-output coupling
- 5×3 GroupKFold on joke-disjoint, fresh 1-epoch training per fold
- Expected runtime: ~4h T4 (vs 9h limit)
- Expected rho_mean: 0.30–0.34 with CI [0.27, 0.37]

### Action 2: Once 5×3 completes, evaluate pre-registered gate:
- ρ ≥ 0.35 AND CI excludes 0.30 → full v1 paper
- 0.30 ≤ mean < 0.35 → methodology + v1 result paper
- CI includes 0.30 → methodology-only paper (still publishable)

### Action 3: Update PAPER_OUTLINE.md §5.1 with verified 5×3 numbers
- This requires the 5×3 result before the paper can be drafted

---

## Cost

- $0 (Kaggle free tier)
- ~4h GPU T4 for 5×3 CV
- 0h CPU (the kernel handles CPU fallback if GPU fails)
- +0 GPU for the 5×3 evaluation on saved model (the engineering critique identified this risk, addressed by fresh kernel approach)

---

## What we do NOT do

- ❌ Cite 0.3402 as v1 headline anywhere
- ❌ Publish v1 paper based on single-run result
- ❌ Skip the 5×3 CV
- ❌ Ignore the +0.163 falsification lesson (apply it here)

## What we DO

- ✅ Push 5×3 CV kernel now (self-contained, fresh training per fold)
- ✅ Wait for honest 15-measurement result
- ✅ Cite 0.3402 only as "upper bound, M3 recipe reproduced, single-run pending 5×3"
- ✅ Publish methodology paper regardless of v1 outcome (it has independent value from the falsification findings)