# PIVOT TO PAPER — HaHaScore 5×3 Falsification Methodology

**Author**: Agent Council (file-audit + claims-ledger + production-demo synthesis)
**Date**: 2026-10-05
**Status**: Repo archived. Path forward: methodology paper, not model paper.

---

## Why this paper exists

A multi-modal humor-detection model (HaHaScore v10) was trained, deployed to HF, and **received 0 downloads over 8 days**. The project's central numerical claim (a +0.163 per-language normalization gain) was refuted by 5×3 repeated cross-validation within 30 minutes of compute. **The falsification is the paper.**

Three honest findings survive the audit:

| # | Finding | Source | Effect size | n |
|---|---|---|---|---|
| 1 | Identity leakage under random 5-fold CV | `speaker_disjoint_cv_results.json` | gold AUC drops 0.501 → 0.386 (Δ = −0.115) | 12 videos |
| 2 | Language channel as proxy for humor | `SPEAKER_DISJOINT_FINDINGS.md:34` | Spanish 0.78 vs French 0.26 gold AUC | 12 videos |
| 3 | Self-falsification of +0.163 via 5×3 CV | `5x3_cv_results.json` | true delta = −0.127 ± 0.103 (p = 0.0004) | 15 measurements |

## Sister-project alignment (ChuckleNet / autonomous_laughter_prediction_essential)

- **ChuckleNet-Ten** (`Hayasuki/ChuckleNet-Ten`): WavLM-based, IoU-F1@0.2 ≈ 0.33 on 118v, **70 downloads** since 2026-09-23.
- **ChuckleNet-Genki** (`Hayasuki/ChuckleNet-Genki`): weak-label, F1=0.27 on the same set, **265 downloads**.
- **HaHaScore-cascade** (`Hayasuki/hahascore-cascade`): text+audio fusion, 0.69 humor / 0.55 gold on 12v, **0 downloads**.
- **ChuckleNet-StandUp4AI baseline**: IoU-F1@0.2 = 0.51 on 118v (cited as upper bound in `WORD_LEVEL_30V_RESULTS.md`).
- **ChuckleNet honest metric** (memory): IoU-F1@0.2 = 0.31 on 118v (5-fold GroupKFold), below the 0.51 baseline.

The intersection: both projects use the same StandUp4AI dataset. ChuckleNet has a small user base (laughter detection has more direct applications: caption timing, podcast editing, call-center monitoring). Humor-strength has zero — there is no real-world task that needs a 0–100 humor score per sentence of stand-up comedy.

**The audit conclusion**: drop the humor model, keep the falsification methodology. Submit a paper that says the cost of failure was small (8 days of dead repo) and the lesson is concrete (5×3 CV catches single-fold fold-luck at p<0.001).

## Paper proposal

### Title

**"Evaluating Speech-Humor Models Without Identity Leakage: A Self-Falsification Case Study"**

### Target venue

**INTERSPEECH 2026 short paper** (4 pages, 1 figure + 1 table + commit). Submission deadline: ~early 2026. Alternative: EMNLP Findings, ACL Short, or workshop.

Alternative: an arXiv preprint (cs.CL) without venue submission. Recommended for fastest reach — 1 week vs 3-6 months.

### Abstract (≤250 words)

> We describe a pre-publication self-falsification case study on HaHaScore v10, a multimodal text+audio humor-detection model trained on 639 stand-up comedy videos with 12 gold-labeled videos. The model was originally claimed to achieve 0.823 humor AUC and 0.613 gold AUC on a single 5-fold CV, and a +0.163 per-language normalization improvement was reported as the project's headline finding. We replicated both results using a 5×3 repeated speaker-disjoint cross-validation protocol and found: (i) the single-fold numbers collapse to 0.6924 ± 0.0359 (humor) and 0.5453 ± 0.1313 (gold) under speaker-disjoint CV, with the gold AUC delta from random to speaker-disjoint splits being −0.115; (ii) the +0.163 per-language normalization gain reverses sign and is significantly negative (delta = −0.127, p = 0.0004, 95% CI [−0.183, −0.078]) — a single-fold fold-luck pattern. We argue that single-fold AUC on imbalanced speech-humor tasks (<1% positive rate) systematically over-states performance via speaker and language-channel confounds. We recommend that all humor-detection evaluations on speech data report at minimum 5×3 repeated speaker-disjoint CV with bootstrap confidence intervals.

### Outline (4 pages)

1. **Introduction** (½ page): Why speech-humor is hard (low positive rate, speaker identity, language as confound). The single-fold trap.
2. **Setup** (½ page): HaHaScore v10 architecture (text 768 + audio 791 + cross 16, multi-task with CORAL domain adaptation). 12-video gold subset on StandUp4AI (591 en, 10 pt, 10 cs, 10 hu, 8 fr, 7 it, 3 es). 8 French, 3 Spanish, 1 unknown overlap.
3. **Finding #1 — Identity leakage** (1 page): Random vs speaker-disjoint CV. Gold AUC 0.501 → 0.386. Per-video breakdown (Spanish 0.78 vs French 0.26). Table.
4. **Finding #2 — Falsification methodology** (1 page): Single-fold +0.163 → 5×3 delta -0.127, p=0.0004. Distribution plot. Bootstrap CI.
5. **Discussion** (½ page): Cost of fold-luck (8-day dead repo, 0 HF downloads). Generalization to other low-positive-rate speech tasks (laughter detection, deception detection, sarcasm detection).
6. **Conclusion** (¼ page): Always 5×3 repeated speaker-disjoint CV with bootstrap CIs for speech-humor and related tasks.

### Figure 1

Two-panel plot:
- Left: gold AUC for each of 15 measurements (3 seeds × 5 folds), with raw vs normalized side-by-side
- Right: delta histogram with bootstrap CI overlay

### Table 1

Random vs speaker-disjoint CV (gold AUC), per-language breakdown (Spanish, French, English).

### Repository for reviewers

`github.com/Das-rebel/HaHaScore` (archived, deprecation notice at root). Includes:
- `5x3_cv_results.json` (raw measurements)
- `speaker_disjoint_cv_results.json`
- `lang_norm_results.json` (the falsified single 5-fold result)
- `run_5x3_cv.py` (the falsification script)
- `speaker_disjoint_cv.py`
- `per_language_norm.py`
- `COUNCIL_FINAL_DECISION.md` (audit trail)
- `RETHINK_2026_v2.md` (audit narrative)
- `SPEAKER_DISJOINT_FINDINGS.md` (per-language diagnostic)

---

## 4-week execution plan

| Week | Action | Cost | Output |
|------|--------|------|--------|
| 1 | Draft paper (4 pages). Request feedback from 1-2 colleagues if available. | 16-20 hrs human, 0 GPU | Paper draft v1 |
| 2 | Run additional experiments: PR-AUC per-language; bootstrap speaker-disjoint 5×3; AUC on full 639 pseudo-labels to bound ceiling. | 4-8 hrs Kaggle T4, $0 | Figure 1 final, Table 1 final |
| 3 | Finalize paper, polish figures, write cover letter if submitting to venue. Post to arXiv (cs.CL). | 4-6 hrs human, 0 GPU | arXiv preprint |
| 4 | Submit to INTERSPEECH 2026 short or EMNLP Findings. Track decisions. | 4 hrs human | Submission |

**Total budget**: ~30 hrs human, ≤8 GPU hrs, 0 new annotations, 0 new data.

## What NOT to do

- **Do not** retry v10 with per-language norm as input. The 5×3 CV proved it's noise.
- **Do not** add more architectures (v11, v12). The 4-arch ablation was null.
- **Do not** try Reddit pretraining again. −0.045 gold harm.
- **Do not** chase the 0.823 number. It was single-fold leakage.
- **Do not** collect more gold annotations. No budget. n=12 is what we have, and the paper acknowledges this as a constraint.
- **Do not** ship the model as a product. The 0.69/0.55 number is not competitive.

## What TO do (overlapping with the 4-week plan above)

- **Use the existing 5x3 CV measurement script** as the primary scientific artifact. It is short, self-contained, and reproducible.
- **Reference ChuckleNet-Ten** as a complementary sibling finding (ChuckleNet also has the speaker-disjoint +0.115 finding per `DECISION_GRAPH_V3.md` Anchor 3).
- **Frame the paper as a "small lesson from a failed model"** rather than "a new humor detector". The lesson (always 5×3 repeated CV) is more durable than the model.
- **Submit the anomaly to a venue that values methodology contributions**: EMNLP Findings, ACL Workshop on Evaluation, NeurIPS Datasets & Benchmarks track. INTERSPEECH is also fine if reframed for the speech-humor community.

## Stop / Start / Continue

**STOP**: training v10 variants; chasing per-language norm; HF deployment; YouTube-download attempts; multi-modal data collection; any "10x better" retraining.

**START**: 5×3 PR-AUC + bootstrap CIs as the project standard; referencing sister ChuckleNet-Ten (10x more users); arXiv preprint in 1 week; INTERSPEECH/EMNLP submission in 4 weeks; framing this as a "self-falsification" contribution.

**CONTINUE**: keeping HF Space live as an honest archive; preserving `COUNCIL_FINAL_DECISION.md` and `5x3_cv_results.json` as the canonical audit trail; the policy "always 5×3 repeated CV with bootstrap CIs on imbalanced classification"; sister-project alignment with ChuckleNet.

---

**One sentence summary**: The HaHaScore v10 humor model was a failed deployment (0 downloads). The 5×3 repeated CV that revealed its central claim was fold-luck is the publishable contribution. Drop the model, write the methodology paper. ~30 hours human + ≤8 GPU hrs over 4 weeks. arXiv in 1 week, INTERSPEECH 2026 in 4 weeks.

— Agent Council (file audit + claims ledger + production demo + sister-project alignment), unanimous