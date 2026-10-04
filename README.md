# HaHaScore — REPO DEPRECATED 2026-10-05

## Honest final numbers (5-fold speaker-disjoint CV)

- **humor AUC**: 0.6924 ± 0.0359
- **gold AUC**: 0.5453 ± 0.1313

Earlier single-fold AUC 0.860 (v7) and 0.823 (v10) numbers are leakage artifacts — do not cite.

## Why this repo was archived

1. **0 downloads on HuggingFace** since deployment (2026-09-27 → 2026-10-05).
2. **+0.163 per-language norm claim was fold-luck** — falsified by 5×3 repeated CV (true delta = −0.127, p = 0.0004).
3. **Architecture iterations v7→v8→v9→v10 are statistically indistinguishable** under 5-fold CV.
4. **The model learns language as a proxy for laughter** (Spanish 0.72 vs French 0.26 gold AUC), which is interesting but not a publishable contribution.
5. **The 5×3 self-falsification case study is publishable** — but as a methodology paper, not as a humor detection paper.

## Audit documents

- `COUNCIL_FINAL_DECISION.md` — the 4-week plan and 24-hour fix list
- `RETHINK_2026.md` / `RETHINK_2026_v2.md` — the audit that invalidated v7-v10
- `SPEAKER_DISJOINT_FINDINGS.md` — identity/language confound diagnostic
- `5x3_cv_results.json` — the falsification experiment
- `lang_norm_results.json` — the falsified single 5-fold result (retained as provenance)

## Where the work went

Audio-multimodal research continues under the ChuckleNet "Interaction Signal Research Program" in `~/autonomous_laughter_prediction_essential/`. The sister ChuckleNet-Ten model (WavLM-based, F1 ≈ 0.33 IoU-F1@0.2 on 118v) has 70 HF downloads and a real user base.

This repo is preserved for audit provenance only.