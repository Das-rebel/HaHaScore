# Council Realization Plan — HaHaScore Realignment to Canonical v1 Path

**Date**: 2026-10-05
**Council**: 4 parallel critics (Architecture, Data, Methodology, Integration)
**Verdict**: ✅ Unanimous on canonical path; v2 LaughO is correctly identified as sister project

---

## Council Synthesis

### What the 4 critics agreed on

1. **Canonical v1 = text-only RoBERTa regression on Jester** (1.76M rated jokes, continuous 0-100, Spearman ρ primary metric)
2. **The 5×3 repeated CV methodology from `laugho_cv.py` needs a regression-extension** (Spearman ρ + MAE + RMSE instead of AUC)
3. **Jester primary = `SeppeV/rated_jokes_dataset_from_jester`** (1,761,440 rows, Apache-2.0, has jokeText + Z_rating inline, verified HTTP 200)
4. **v2 LaughO is correctly identified as sister project** — keep it isolated in `experiments/v2_laugho/`
5. **ChuckleNet is READ-ONLY dependency** — cite papers + download HF model weights, do not fork or modify

### Key file-level changes recommended by council

| File | Action | Reason |
|---|---|---|
| `README.md:1-3` | Replace banner | Was "DEPRECATED", should be "v1=RoBERTa regression on Jester, v2=sister project" |
| `train_jester_regression.py` | **CREATE** | New: text-only RoBERTa regression training (canonical v1) |
| `laugho_cv.py:138` | Add `repeated_joke_disjoint_cv_regression()` | Regression-ready CV (Spearman/MAE/RMSE) |
| `experiments/v1_jester_regression/` | **CREATE** | New: v1 results directory |
| `experiments/v2_laugho/SISTER_PROJECT.md` | **CREATE** | One-line note that v2 is sister project, not v1 |
| `v2_laugho_train.py`, `laugho_train.py`, `laugho_data.py` | Move to `experiments/v2_laugho/` | Isolates sister-project files |
| `IMPROVEMENT_PLAN.md:50-80` | Rewrite | Was scaling sister project; now should be |
| `Makefile:11-14` | Update | Add `v1-jester`, `v1-eval` targets |
| 14 bridge*.py + cascade*.py + modal_*.py + per_language_norm.py | DELETE | Staged v6-v10 iteration, superseded |

---

## Execution Plan (this session)

1. **Move v2 files** to `experiments/v2_laugho/` (5 min)
2. **Delete staged iteration cruft** (5 min)
3. **Add regression CV to `laugho_cv.py`** (10 min)
4. **Create `train_jester_regression.py`** (15 min)
5. **Create `experiments/v1_jester_regression/` skeleton** (5 min)
6. **Update README + Makefile + IMPROVEMENT_PLAN** (15 min)
7. **Run CI** (1 min)
8. **Commit + push** (5 min)

**Total: ~60 minutes of automated work, 0 GPU, 0 cash.**

---

## What gets created in this session

```
data/jester_seppev.py                # Jester downloader, 15 LOC
train_jester_regression.py            # text-only RoBERTa regression, 120 LOC
experiments/v1_jester_regression/
    README.md                          # experiment README
    result.json                         # (empty placeholder)
experiments/v2_laugho/
    SISTER_PROJECT.md                  # one-line scope note
    v2_laugho_train.py                 # moved from root
    laugho_train.py                    # moved from root
    laugho_data.py                     # moved from root
    RESULT_20261005.md                 # already exists, update frontmatter
```

## What gets deleted in this session

```
bridge1_fix_20.py
bridge1_pseudolabel_cpu.py
bridge1_pseudolabel_test.py
bridge1_reprocess_99.py
bridge3_incongruity_test.py
bridge4_gold_eval.py
bridge4_humor_arc_tracker.py
bridge4_inference.py
bridge5_self_training.py
cascade_gold_eval.py
cascade_inference.py
v6_cross_attention.py
v6_demo.py
v6_inference.py
v6_prepare_features.py
modal_pseudolabel_v1.py
modal_train_v6.py
per_language_norm.py           # source of +0.163 fold-luck — falsified
colab_train.py
test_modal.py
download_standup4ai_audio.py
transcribe_all.py
transcribe_resume.py
run_bridge1.sh
run_bridge4.sh
```

## Memory rules (the "remember and avoid digression" part)

The following 5 things MUST be remembered and avoided:

1. **The original goal is multimodal text+audio humor STRENGTH (0-100), not laughter detection**. v2 LaughO is sister project.
2. **The 5×3 repeated CV methodology is the contribution**. Never inflate a single-fold number to a headline.
3. **Jester continuous ratings** are the canonical v1 training data, not StandUp4AI 1.16% positive class.
4. **The honest 5×3 CV for v10 is humor AUC 0.6924, gold 0.5453** — never cite 0.860 or 0.823 as headline.
5. **ChuckleNet is read-only**. Cite papers, download HF weights, never modify.

If at any point the work drifts to "build a better laughter detector," STOP and re-read this doc.

---

## Cost

| Item | Estimate |
|---|---|
| Cash | $0 |
| GPU | 0h |
| Human | ~60 min automated |
| Commits | ~3 to HaHaScore |

---

## After this session: the v1 launch sequence

1. **Week 1**: `train_jester_regression.py` ready, run on Jester subset (~6h CPU)
2. **Week 2**: scale to full 1.76M, get honest Spearman ρ + bootstrap CI
3. **Week 3**: if ρ ≥ 0.35 AND CI excludes 0.30 → publish v1 paper
4. **Week 4**: optional v2 multimodal extension using AST-clean labels + v1's regression head