# Cover Letter — INTERSPEECH 2026 / EMNLP Findings

**Title**: Evaluating Speech-Humor Models Without Identity Leakage: A Self-Falsification Case Study

**Authors**: Subhajit Das

**Submission venue**: INTERSPEECH 2026 Short Paper (4 pages) / EMNLP 2026 Short Paper (4 pages + references)

## Summary

This is a **methodology / case-study paper**, not an improvement on the state-of-the-art. The contribution is a single recommendation:

> Speech classification tasks with positive class rate < 2% and gold n ≤ 30 should always report 5×3 repeated speaker-disjoint CV with bootstrap confidence intervals before publishing any headline metric.

The empirical basis is a self-falsification case study: a multimodal text+audio humor-detection model (HaHaScore v10) was trained, deployed to a public model hub, and observed for 8 days with 0 downloads, before a 5×3 repeated CV revealed that its central numerical claim (+0.163 per-language normalization gain) was fold-luck at p=0.0004. The 30-minute CV caught what 8 days of deployment did not.

## Three honest findings

1. **Identity leakage**: Speaker-disjoint CV reveals a −0.115 gold-AUC drop that random CV misses (n=12 videos, single 5-fold).
2. **Language confound**: Spanish 0.72 vs French 0.26 gold-AUC under LOGO CV on a 92.5% English/Portuguese training corpus.
3. **Self-falsification**: A +0.163 single-fold gain reverses to −0.127 (p=0.0004, 95% CI [−0.183, −0.078]) under 5×3 repeated CV with 15 measurements.

## Why this is publishable

- The lesson applies beyond humor: laughter detection, sarcasm detection, deception detection, conversation segmentation — any speech classification with low positive class rate.
- Sister ChuckleNet project (https://github.com/Das-rebel/ChuckleNet) reports the same +0.115 identity-leakage finding under a different evaluation protocol, supporting generalization.
- The recommendation is concrete and actionable: 5×3 repeated speaker-disjoint CV with bootstrap CIs, runtime cost ~30 minutes on CPU for n=12-100 datasets.
- The case-study framing is honest about its own limitations (n=12, no annotation budget for n=50, three languages only).

## Why the humor model itself is not the deliverable

- HaHaScore v10 achieved 5-fold speaker-disjoint AUC 0.69 (humor) / 0.55 (gold) — not competitive with the state-of-the-art (StandUp4AI baseline: 0.51 IoU-F1).
- The HuggingFace deployment (`Hayasuki/hahascore-cascade`) received 0 downloads in 8 days and has been archived.
- The model's headline single-fold number (0.823) was leakage; the paper does not cite it.

## Reproducibility

All experiment scripts, raw results JSON, and the falsification analysis are in the archived repository:
- `github.com/Das-rebel/HaHaScore` (commit `03884e0`)
- `5x3_cv_results.json` (15 raw measurements)
- `speaker_disjoint_cv_results.json` (random + speaker-disjoint + LOGO per-video)
- `lang_norm_results.json` (the falsified single 5-fold result, retained as provenance)
- `run_5x3_cv.py`, `speaker_disjoint_cv.py`, `per_language_norm.py` (reproduction scripts)

## Conflict of interest

The author is also the author of the sister ChuckleNet project. This is not a conflict: both projects report the same identity-leakage finding under different evaluation protocols.

## Suggested reviewers

- Anyone working on speech-humor, laughter detection, or speaker-disjoint CV
- Anyone with experience in cross-validation methodology for low-positive-class tasks
- INTERSPEECH audience: speech/audio community, especially humor and paralinguistics tracks
- EMNLP audience: short paper track on evaluation methodology or low-resource evaluation

## Submission package

- `hahascore.tex` (4 pages, ~2000 words)
- `figure_5x3_falsification.pdf` + `.png` (Figure 1)
- `figure_speaker_disjoint.pdf` + `.png` (Figure 2)
- `references.bib`

**Total budget for paper preparation**: ~16 hours human, 0 GPU.