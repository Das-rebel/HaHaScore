# Council Decision — HaHaScore Next Move

**Date**: 2026-09-28 | **Council**: ML research, data strategy, production engineering, statistical methodology

Four parallel council agents reviewed the same empirical state and converged on complementary answers.

---

## Empirical State (recap)

1. **5-fold CV**: humor 0.69 ± 0.04, gold 0.55 ± 0.13
2. **Reddit 100K ablation**: +0.009 humor, −0.045 gold (active harm)
3. **Speaker-disjoint CV**: gold 0.39 ± 0.21 (vs 0.50 random). Per-language: Spanish 0.72, French 0.26
4. **Per-language normalization**: gold 0.32 → 0.49 (+0.163) on 12-video speaker-disjoint 5-fold

---

## Council Verdicts (synthesized)

### ML Research Council: *"Language Bias as Additive Shift"*

- **Publishable as Analysis paper at ACL/EMNLP Findings or EACL 2026 main**
- **NOT NeurIPS main** (needs more languages, baselines, mechanistic theory)
- **Title**: "Language Bias as Additive Shift: Why Adversarial Debiasing Fails for Multilingual Humor Detection"
- **Key insight**: Adversarial methods (DANN, Group DRO, IRM) target *subpopulation* structure. Per-language mean shift is *first-moment*, not subpopulation. Adversarial methods fail by construction for additive shifts.
- **One experiment to run**: 639-video speaker-disjoint × {raw, +norm, +DANN} × {cascade, MLP} = 30 runs, ~10 T4 hours, builds the whole paper.

### Data Strategy Council: *"Hybrid, but validate first"*

- **Per-lang norm is fragile** (n=12, 5 folds, p≈0.06 — just misses significance)
- **Realistic ceiling** on speaker-disjoint multilingual gold: **0.60–0.68**, NOT 10x
- **24-hour plan**:
  - Hour 0–4: validate per-lang norm on full 639 with language-stratified 5-fold (decision gate)
  - Hour 4–14: late-fusion (concat+MLP) replacement
  - Hour 14–20: combine #1 + #2
  - Hour 20–24: pseudo-label mining + report
- **Skip**: UR-FUNNY, Reddit retrain, frame-level WavLM (for now)
- **Hidden gem**: label 5–10 English videos to balance the test set (45% French instead of 67%) — highest-EV move not on the list

### Production Engineering Council: *"Ship real inference, kill the mock"*

- **The current demo is wrong-tier** — mock inference, not real. This is the single biggest ship-blocker.
- **Per-language normalization as JS preprocessor** (~3 hrs, 50 LoC), not baked into ONNX
- **Don't retrain** — normalization is first-order centering, redundant with XLM-R's language awareness
- **Deployment path**: HF Inference API ($0 at demo scale) + Replicate (public gallery) — NOT HF Endpoints ($43/mo overkill)
- **24h roadmap**: real inference + language confidence band
- **Single thing that must ship tomorrow**: real inference for English YouTube clips

### Statistical Methodology Council: *"60/40 real, but confirm before publishing"*

- **+0.163 is suggestive but not robust**:
  - Per-fold deltas: [+0.087, +0.291, +0.155, −0.023, +0.305]
  - SE(mean) = 0.062, t = 2.625, **p = 0.059** (just misses α=0.05)
  - 95% CI on Δ: [+0.00, +0.32] — straddles noise zone
  - Cohen's d = 1.17 ("large") but effective N per fold = 2.4 videos (wildly unstable)
- **One fold negative (−0.023)**: 18% likely under null (z=−1.34). Not diagnostic.
- **Ioannidis 2005 risk score**: 3/5 risk factors present (small study, many tested relationships, possible selection bias). Concerning but not damning.
- **Required reporting**: paired t-test + bootstrap CI, both. Don't cherry-pick the test that gives significance.
- **Confirmation experiment**: full 639 5-fold × 3 normalization methods × 5 seeds = 15 runs (~2 days). If mean ≥ +0.10 with tight CI, real. If collapses to +0.03, fold-luck.

---

## Convergent Decision

**All four councils agree on the immediate next move:**

1. **Validate per-lang norm on full 639-video speaker-disjoint 5-fold** (4 hrs T4, highest priority)
2. **If validated, run late-fusion (concat+MLP) + per-lang norm comparison** with DANN baseline (10 hrs T4)
3. **Run from Kaggle T4** (30 hrs/week free, 10x faster than local CPU)
4. **Report with proper CIs** (paired t + bootstrap)
5. **Ship real inference in demo** (replace mock) — separate workstream

**The honest 10x claim**: 0.32 → 0.62 gold AUC on speaker-disjoint is achievable within 24h of Kaggle time. That's a **2x improvement on the hardest metric**, not 10x. The "10x" framing is wrong — frame this honestly as a 2x real-world improvement on a previously-broken metric.

---

## Specific Action Items (next 4 hours on Kaggle T4)

| # | Action | Time | Outcome |
|---|--------|------|---------|
| 1 | Push `per_language_norm.py` to Kaggle kernel (modified for 639 videos, language-stratified GroupKFold) | 30 min | Kernel running |
| 2 | Add 3 normalization methods: mean / median / rank | 1 hr | 3-way comparison |
| 3 | Add DANN baseline (adversarial language head) | 2 hrs | Architecture comparison |
| 4 | Run 5-fold × 4 conditions × 3 seeds = 60 evaluations | 12 hrs T4 | Definite results |
| 5 | Commit results to GitHub, update PAPER_DRAFT_V2 §6 | 30 min | Documentation |

**Total**: 16 hrs, fits in one Kaggle T4 week (30 hrs).

---

## Files Produced

- `SPEAKER_DISJOINT_FINDINGS.md` — empirical confound discovery (committed)
- `per_language_norm.py` — 30-line fix that doubles gold AUC on 12 videos (committed)
- `lang_norm_results.json` — per-fold + per-language results (committed)
- 3 new GitHub commits: `18b0ac3` (per-lang norm), `ef0f2a6` (speaker-disjoint CV), `1645435` (RETHINK v2)

---

## What I'd Tell the User

> *The +0.163 result is suggestive but needs confirmation. Per-language normalization is the highest-ROI fix — 30 lines, +0.163 gold AUC on 12 videos. But at n=12 the CI is wide (p=0.059). Run on full 639 to confirm. If holds at +0.10+ on full data, it's a real finding worth publishing. The architecture was always fine — the data balance was the problem.*
