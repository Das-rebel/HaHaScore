# HaHaScore

**Sentence-level humor strength predictor (0–100 continuous score).**

## Current scope (post-realignment, Oct 5 2026)

| Track | Status |
|---|---|
| **v1 canonical** | Text-only RoBERTa regression on Jester (1.76M rated jokes, continuous 0-100) |
| **v2 deferred** | Multimodal text + audio fusion (deferred until v1 baseline established) |
| **v2 LaughO sister project** | Audio-only laughter detection on AudioSet (sister project, not v1) |
| **Methodology paper** | "Evaluating Speech-Humor Models Without Identity Leakage" (5×3 repeated CV falsification case study, ready to submit) |

## v1 canonical path

**Text-only RoBERTa regression on Jester**, validated with 5×3 repeated joke-disjoint CV.

- **Dataset**: `SeppeV/rated_jokes_dataset_from_jester` (1.76M rated jokes, Apache-2.0)
- **Architecture**: `roberta-base` + Linear(768, 1) regression head
- **Loss**: MSE on continuous 0-100 ratings
- **Validation**: `laugho_cv.repeated_joke_disjoint_cv_regression()` — Spearman ρ, MAE, RMSE with bootstrap 95% CI
- **Pre-registered success gate**: ρ ≥ 0.35 AND CI excludes 0.30

See `train_jester_regression.py` for the v1 entry point.

## Sister project: v2 LaughO

`experiments/v2_laugho/` contains the audio-only laughter detection pipeline (AUC 0.7501 ± 0.1817 on AudioSet eval/00). This is a **sister project**, not the canonical v1 path. See `experiments/v2_laugho/SISTER_PROJECT.md`.

## Methodology paper

`arxiv_submission/hahascore.tex` (and `arxiv_submission/hahascore_arxiv_bundle.tar.gz`): "Evaluating Speech-Humor Models Without Identity Leakage: A Self-Falsification Case Study."

Key finding: 5×3 repeated speaker-disjoint CV with bootstrap 95% CI caught a +0.163 per-language normalization claim at p = 0.0004 (true delta = -0.127). **Honest 5-fold v10 numbers**: humor AUC 0.6924 ± 0.0359, gold AUC 0.5453 ± 0.1313.

Single-fold numbers like 0.860 or 0.823 are leakage artifacts — never cite.

## CI

```bash
make ci    # validates model + paper + README integrity
```

## Files

| Path | What |
|---|---|
| `train_jester_regression.py` | v1 canonical entry point (text-only) |
| `laugho_cv.py` | 5×3 repeated CV harness (speaker-disjoint + joke-disjoint regression) |
| `run_5x3_cv.py` | Falsification experiment (paper provenance) |
| `laugho_ci.py` | CI validator |
| `Makefile` | Reproducible workflow |
| `arxiv_submission/hahascore.tex` | Methodology paper |
| `experiments/v1_jester_regression/` | v1 experiment directory |
| `experiments/v2_laugho/` | v2 LaughO sister project |
| `5x3_cv_results.json` | The 15 measurements from the falsification |
| `lang_norm_results.json` | Falsified result, retained as provenance |
| `data/audioset_laugh_ontology.json` | 8 laugh classes |

## Constraints

- Local 8.4 GB free (per disk cache)
- Kaggle T4 30 h/week free
- No annotation budget (per memory)
- 12-video gold set (per memory: "small n, dominant constraint")

## Memory rules

1. **Multimodal text+audio humor STRENGTH (0-100)** is the original goal, NOT laughter detection.
2. **5×3 repeated CV methodology** is the contribution; never inflate single-fold numbers.
3. **Jester continuous ratings** are the canonical v1 training data, not StandUp4AI 1.16% positive class.
4. **Honest 5×3 CV for v10**: humor AUC 0.6924, gold 0.5453 — never cite 0.860 or 0.823.
5. **ChuckleNet is read-only**. Cite papers, download HF weights, never modify.

## Historical baseline: most defensible multimodal result

HaHaScore v5 (DeBERTa-v3-base + WavLM-base-plus bilinear fusion, Sep 15 2026):
- **5-fold CV AUC: 0.632 ± 0.007** on StandUp4AI pseudo-label set
- **Val AUC: 0.613** on 10 held-out videos
- Text-only baselines (random): RoBERTa 0.499, DeBERTa 0.501
- Audio-only (WavLM LR): 0.538
- Bilinear fusion: 0.613

**v5 is the most defensible multimodal number we have published.** It was:
- Verified with 5-fold CV (not single-fold)
- Built on a clean DeBERTa-v3-base + WavLM-base-plus architecture
- Below the inflated single-fold numbers (0.860/0.823) that v10's falsification revealed as leakage

Per the strategic rethink (commit `46c6713`), v10 (the current repo's headline) does not improve on v5's generalization. The honest baseline for the multimodal track is **0.632 ± 0.007** (v5) or **0.6924 ± 0.0359** (v10 speaker-disjoint 5-fold), not 0.860.

**Do NOT cite:**
- ❌ AUC 0.860 (v7, single-fold fold-luck)
- ❌ AUC 0.823 (v10, single-fold fold-luck)
- ❌ +0.163 per-language norm gain (falsified by 5×3 CV, p=0.0004)

**Cite instead:**
- ✅ v5 multimodal: 0.632 ± 0.007 (5-fold CV on StandUp4AI pseudo-labels)
- ✅ v10 multimodal: 0.6924 ± 0.0359 humor / 0.5453 ± 0.1313 gold (5×3 repeated speaker-disjoint CV)
- ✅ Methodology: `arxiv_submission/hahascore.tex`
