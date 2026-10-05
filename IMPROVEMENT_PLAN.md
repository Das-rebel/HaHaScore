# HaHaScore — Improvement Plan (Oct 5 2026)

**Starting state**: HaHaScore v10 archived, falsification paper drafted, methodology codified.

**Goal of this document**: catalog what improvements are actionable right now vs. require external dependencies (Drive, GPU, gold annotations).

---

## What's in the repo RIGHT NOW (actionable without external deps)

### CI pipeline (just built)

| File | Purpose |
|---|---|
| `laugho_ci.py` | Continuous integration: validates model + paper + README integrity |
| `ci_results.json` | Last CI run output (verdict: PASS, 2026-10-05) |
| `Makefile` | Reproducible workflow: `make ci / data / train / eval / paper / all` |

**Verified locally**: `make ci` returns PASS for current state.

### 5×3 CV harness (reusable)

| File | Purpose |
|---|---|
| `laugho_cv.py` | `repeated_speaker_disjoint_cv()` with bootstrap CI + paired comparison |
| `5x3_cv_results.json` | The actual 15 measurements from the falsification |
| `lang_norm_results.json` | The original (now falsified) +0.163 finding, retained as provenance |

**Verified locally**: `python3 laugho_cv.py` runs demo with synthetic data, returns expected outputs.

### v10 model validation

| File | Purpose |
|---|---|
| `run_v10_5x3.py` | Apply laugho_cv to v10 ONNX model |
| `deployment/models/v10_cascade_int8.onnx` | The shipped ONNX (3.26 MB, verified intact) |

**Verified locally**: `python3 run_v10_5x3.py` loads v10, prints inputs/outputs, lists gold videos.

### v2 LaughO training skeleton

| File | Purpose |
|---|---|
| `v2_laugho_train.py` | Trains 768→256→128→1 MLP head (~1.5M params) |
| `laugho_train.py` | Earlier skeleton (superseded by v2_laugho_train.py) |

**Status**: skeleton only; needs Drive features to actually run.

### Data pipeline (reusable but untested)

| File | Purpose |
|---|---|
| `laugho_data.py` | Streams AudioSet parquet, filters 8 laugh ontology IDs |
| `data/audioset_laugh_ontology.json` | Saved 8 laugh ontology IDs |

**Verified locally**: `laugho_data.py` compiles, AudioSet schema confirmed via range request.

### Honest docs (verified)

| File | Status |
|---|---|
| `README.md` | Replaced with DEPRECATED banner + honest 5-fold numbers |
| `arxiv_submission/hahascore.tex` | SciSlop-verified, 2 unresolved refs fixed |
| `arxiv_submission/hahascore_arxiv_bundle.tar.gz` | 182 KB submission bundle ready |
| `RETHINK_2026_v2.md` | 5-fold CV honest number provenance |
| `SPEAKER_DISJOINT_FINDINGS.md` | Identity + language confound diagnostic |
| `COUNCIL_FINAL_DECISION.md` | 4-week plan + 24-hour fix list |

---

## What requires Drive features

The actual CV validation requires `bridge4_features.npz` + `v6_features.npz` from Google Drive (the 12-video gold subset + the 639-video StandUp4AI features). Local disk is 8.4 GB free; Drive has 1.6 GB of training data per memory.

**Action**: download Drive features → run `run_v10_5x3.py` → verify the published 0.69/0.55 numbers reproduce.

**Cost**: 0 GPU; ~5 minutes CPU.

---

## What requires GPU (Kaggle T4)

### Step 1: Download + decode AudioSet laughter family

- **Source**: `agkphysics/AudioSet` (CC-BY-4.0)
- **Size**: 22.5 GB for all 35 eval shards
- **Filter**: 8 laugh ontology IDs (Laughter, Giggle, Snicker, etc.)
- **Output**: ~5k-10k laugh samples at frame level
- **Cost**: 0 cash + 4 GPU h (T4)

### Step 2: Train v2 LaughO MLP

- **Architecture**: 768 → 256 → 128 → 1 (1.5M params, WavLM frozen)
- **Data**: AudioSet laugh samples (positive) + StandUp4AI non-laugh (negative)
- **Loss**: BCEWithLogitsLoss with pos_weight for 1.16% positive rate
- **Optimizer**: AdamW, lr=1e-3
- **Cost**: 0 cash + 2 GPU h (T4)

### Step 3: 5×3 repeated CV evaluation

- **Protocol**: 5 folds × 3 seeds × 15 measurements
- **Output**: AUC mean, std, 95% bootstrap CI, paired comparison vs baseline
- **Cost**: 0 cash + 1 GPU h (T4) or 0 cash + 30 min CPU

### Total cost to v2+ get honest published number

**3 GPU h on Kaggle T4 + 0 cash.** Equivalent to: 30% of one week of free T4 quota.

---

## What requires gold annotation (out of scope)

The 12-video gold set is too small to support any single-fold headline. The user has no annotation budget per the "no-paid-data-spends" directive. This is **out of scope** — but documented in `DATASET_SURVEY_LAUGHTER_2026.md` as the next-step priority if budget becomes available.

---

## 4-week improvement roadmap (concrete)

### Week 1 (already done in this session)
- ✅ CI pipeline (`laugho_ci.py`, `Makefile`)
- ✅ v10 ONNX verification (`run_v10_5x3.py`)
- ✅ v2 LaughO training script (`v2_laugho_train.py`)
- ✅ AudioSet data pipeline (`laugho_data.py`)
- ✅ Honest docs (README, paper, falsification logs)

### Week 2 (requires Drive download)
- Download `bridge4_features.npz` + `v6_features.npz` to local
- Run `run_v10_5x3.py` → confirm 0.69/0.55 numbers reproduce
- Save results to `experiments/v2_laugho/v10_baseline_5x3.json`

### Week 3 (requires Kaggle T4)
- Run `laugho_data.py --decode` on 1 AudioSet eval shard (~650 MB)
- Get ~1500-3000 laugh samples
- Train v2 LaughO MLP (~2 GPU h)
- 5×3 CV → get honest number, compare to v10

### Week 4 (publishable)
- Update PAPER_DRAFT_V3_CONSOLIDATED.md with v2 LaughO number
- Update `DATASET_SURVEY_LAUGHTER_2026.md` with actual training result
- Tag release `v2.0-laugho` on `Das-rebel/HaHaScore`
- Post to arXiv (if falsification paper not yet posted)
- Tweet thread: "We tried to make laughter detection 10x better, ended up 0.05x better but with confidence"

---

## Key file: 1-line runnable

```bash
make all
```

This runs:
1. `make ci` — validate paper + README + model
2. `make eval` — run 5×3 CV harness demo
3. `make paper` — rebuild arXiv submission bundle

If all three pass, the repo is in a publishable state.

---

## Honest scope

This improvement plan is **bounded by**:

| Constraint | Status | Mitigation |
|---|---|---|
| Local disk 8.4 GB free | Blocks Drive download of full features | Use streaming / partial subsets |
| No GPU locally | Blocks training | Kaggle T4 30 h/week free |
| 12-video gold set | Limits ceiling | Use 5×3 CV methodology + bootstrap CI |
| No annotation budget | Limits gold expansion | Use existing 5-fold honest number |
| StandUp4AI labels are discourse positions (per memory) | Limits acoustic generalization | Use AudioSet acoustic labels instead |

The HaHaScore v10 model is **archived**, not deleted. The falsification paper is **drafted, not submitted**. The methodology is **codified, not auto-run**. Each step is gated by the constraints above.

The plan delivers **honest progress**, not inflated numbers.