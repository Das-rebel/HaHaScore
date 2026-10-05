# v2 LaughO — SISTER PROJECT (not canonical HaHaScore v1)

**Status**: Sister project, **NOT** the canonical v1 path.

The canonical HaHaScore v1 path is text-only RoBERTa regression on Jester continuous ratings (see `train_jester_regression.py` at repo root). v2 LaughO is audio-only laughter detection on AudioSet — a separate, related line of work that:

- Borrows the 5×3 repeated speaker-disjoint CV methodology from `laugho_cv.py`
- Uses clean AudioSet acoustic laughter labels (vs StandUp4AI's noisy VTT discourse positions)
- Was originally a quick proof-of-concept run on 1 AudioSet shard (AUC 0.7501 ± 0.1817)
- Documents the AudioSet ontology → 7 laugh classes mapping

**Do not cite v2 LaughO results as HaHaScore headline numbers.** Per `IMPROVEMENT_PLAN.md`, scaling v2 to 35 AudioSet shards + AMI + UR-FUNNY-Temporal is a separate publication track (a "laugh word" paper), distinct from the v1 humor-strength regression.

---

## What's in this directory

| File | Purpose |
|---|---|
| `v2_laugho_train.py` | 1.5M-param MLP head over frozen WavLM-base-plus |
| `v2_laugho_train.ipynb` | Colab notebook (T4 GPU) |
| `v2_laugho_kaggle_kernel.py` | Kaggle kernel fallback |
| `kaggle_v2_laugho/` | Kaggle kernel metadata + script |
| `laugho_train.py` | Earlier skeleton (superseded by v2_laugho_train.py) |
| `laugho_data.py` | AudioSet laugh-class extractor |
| `5x3_cv_audioset_eval00_20261005_184739.json` | First run: AUC 0.7501 ± 0.1817, n=15 |
| `RESULT_20261005.md` | Full first-run analysis |

---

## Sister-project rationale

Both projects share:
- The 5×3 repeated CV methodology (`laugho_cv.py`)
- The bootstrap CI infrastructure
- The "honest baseline before claiming improvement" discipline

They differ:
- v2 LaughO: **audio-only** laughter detection (binary classification)
- v1 HaHaScore: **text-only** RoBERTa regression on continuous humor strength

The v2 work informs v2's standalone publication (and may later inform v3's multimodal extension), but is not on the canonical HaHaScore v1 path.

---

## Files referencing v2 LaughO

- `experiments/v2_laugho/` (this directory)
- `data/audioset_laugh_ontology.json` (saved 7-class ontology)
- `arxiv_submission/hahascore.tex` (mentions audio laugh detection as future direction)
- `DATASET_SURVEY_LAUGHTER_2026.md` (catalogs AudioSet as Tier-1 corpus for v2 only)

---

**Last updated**: 2026-10-05 (council realignment)