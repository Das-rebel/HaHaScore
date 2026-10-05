# v1 Paper Outline — Text-only RoBERTa Regression on Jester

**Working title**: *Continuous Humor-Strength Prediction from Text: A 5×3 Repeated Cross-Validation Baseline*

**Authors**: Subhajit Das (Independent Researcher)

**Status**: Outline only. To be drafted after v1 runs complete.

---

## 1. Abstract (~250 words)

We establish a continuous humor-strength baseline on the Jester dataset of 1.76M jokes with user-rated scores. A text-only RoBERTa-base model with a single-scalar regression head achieves **Spearman ρ = {to-be-filled}** (95% bootstrap CI [{to-be-filled}]) under 5×3 repeated joke-disjoint cross-validation. This is the canonical v1 step in a multi-version research program targeting multimodal humor understanding; the v1 regression head serves as the baseline against which v2 (multimodal text+audio) and v3 (humor-conditioned generation) will be measured. The 5×3 repeated CV protocol is the methodological contribution of this work: in our pilot experiments on speech-humor detection, single-fold AUC was inflated by 0.15-0.20 relative to repeated CV with bootstrap confidence intervals, leading to false-positive architecture comparisons. We argue that all continuous-regression tasks on user-rated humor should adopt 5×3 repeated group-disjoint CV with bootstrap CIs as the minimum reporting standard. Pre-registered success gate: ρ ≥ 0.35 with CI lower bound ≥ 0.30.

## 2. Introduction

- **2.1 Motivation** — humor strength as a continuous attribute useful for content moderation, recommendation, captioning, parody generation.
- **2.2 The label-starvation problem** — small labeled corpora (12 videos in StandUp4AI) and the 1.16% positive rate that follows from binary classification.
- **2.3 Jester as escape route** — 1.76M user-rated jokes, continuous labels, Apache-2.0 licensed.
- **2.4 The methodological gap** — single-fold AUC inflates; we use 5×3 repeated CV + bootstrap CI.

## 3. Related Work

- 3.1 Humor detection (StandUp4AI, MUMOR, HaHaScore, MultiLinguahah)
- 3.2 Continuous rating prediction (Jester literature)
- 3.3 Repeated cross-validation methodology

## 4. Method

- 4.1 Dataset — SeppeV/rated_jokes_dataset_from_jester, 1.76M jokes, Z_rating scale
- 4.2 Architecture — RoBERTa-base + Linear(768, 1) regression head, MSE loss
- 4.3 Pre-processing — Z_rating normalized to 0-100, joke_id hashed for grouping
- 4.4 Validation protocol — 5×3 repeated joke-disjoint CV with bootstrap 95% CI
- 4.5 Pre-registered success gate and falsifiers

## 5. Experimental Results

- 5.1 Main result — Spearman ρ, MAE, RMSE across 15 measurements
- 5.2 Comparison to baselines — TF-IDF + Ridge (ρ = 0.277), prior RoBERTa regression (ρ = 0.313)
- 5.3 Ablations — backbone freezing, backbone size (base vs large), normalization scheme
- 5.4 Per-fold and per-seed breakdowns (no fold-luck pattern)

## 6. Discussion

- 6.1 What the v1 baseline enables — v2 multimodal can now be measured honestly
- 6.2 Limitations — text-only, single language (English-only Jester), no humor-as-multimodal
- 6.3 Societal / dataset caveats — Jester ratings come from a narrow population

## 7. Related Work (cross-project)

- 7.1 ChuckleNet sister project (audio-only laughter detection) shares the 5×3 CV methodology
- 7.2 HaHaScore v10 archived multimodal attempt (5×3 honest = 0.6924 humor / 0.5453 gold) motivates this v1 step
- 7.3 MultiLinguahah (Callejas et al., 2026, arXiv:2605.06309) — multilingual laughter segmentation baseline for v2 multilingual extension

## 8. Conclusion

v1 establishes the honest text-only baseline for humor-strength prediction. The 5×3 repeated CV protocol documented in `laugho_cv.py` is the methodological contribution. v2 (multimodal) and v3 (generation) follow only after v1 passes the pre-registered gate.

---

## Pre-registered success gate

```
IF Spearman ρ ≥ 0.35 AND bootstrap CI lower bound ≥ 0.30:
    -> v1 publishable as baseline paper (this outline)
    -> proceed to v2 multimodal extension
ELSE:
    -> iterate (larger model, more epochs, LoRA fine-tuning)
    -> if still failing, document as negative result
    -> publish methodology contribution alone
```

## Pre-registered falsifiers

1. seed=42 alone yields ρ > 0.50 but mean < 0.35 -> fold-luck pattern (analog of the +0.163 paper falsification)
2. Any single fold produces ρ > 0.7 -> log as outlier, recompute without it
3. 95% CI spans 0 -> gate fails regardless of point estimate
4. Joke-disjoint vs random-split gap > |Δρ| = 0.15 -> same-joke text overlap, redo splits

## What this paper is NOT

- NOT a humor-detection paper (different from MultiLinguahah, HaHaScore, ChuckleNet)
- NOT a multimodal paper (v2 deferred)
- NOT a generation paper (v3 deferred)
- NOT a 0.86 AUC paper (all single-fold fold-luck numbers excluded)

## Open questions

- Is Jester-rating the right ground truth, or do we need domain-expert ratings?
- Does the ρ ~0.35 ceiling reflect the noise of human ratings, or a model limitation?
- Does the model generalize to non-joke humor (stand-up, sitcom, social media)?

## Expected publication venues

- ACL / EMNLP Findings (NLP)
- RecSys (recommendation context)
- Web Conference (joke classification)

## Cost

- $0 cash
- ~4-6 GPU h on T4 (Colab or Kaggle)
- ~16 human hours

## Companion artifacts

- `train_jester_regression.py` — training script
- `data/jester_seppev.py` — dataset downloader
- `laugho_cv.py::repeated_joke_disjoint_cv_regression()` — CV harness
- `experiments/v1_jester_regression/` — results JSONs
- `arxiv_submission/hahascore.tex` — sister paper (5×3 CV falsification methodology)

## Decision: to write this paper, the v1 result must pass the pre-registered gate.

If ρ < 0.35 after one round of improvements, the paper still exists but as a **methodology contribution** (5×3 repeated CV applied to continuous-regression humor tasks). This is publishable regardless of the headline metric — the methodology is the durable contribution.