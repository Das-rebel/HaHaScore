# Agent Council Decision — Post-Audit Path Forward

**Date**: 2026-09-28 | **Three parallel councils** (decision, production-engineering, research-methodology) reviewed the file audit + claims ledger and converged.

---

## TL;DR

The project is in a state of **controlled self-deception**: real science + fake marketing. The 5-fold v10 number (humor 0.69, gold 0.55) IS publishable as a methodology contribution. The README's 0.86 and the demo's 0.823 are NOT. The single highest-EV action in the next 24 hours is to **replace the mock demo with real ONNX inference** — that one action unblocks 80% of the cleanup.

---

## 1. The Council's Unanimous Diagnosis

All three councils independently converged on the same findings:

| Finding | Source | Status |
|---|---|---|
| `app.py` defaults to SHA-1 hash placeholder | Engineering | **Critical** — model never loaded |
| `hf_demo_v2/index.html` runs `mockInference()` JS | Engineering | **Critical** — fake scores |
| README claims "AUC 0.860" (v7, single-fold) | Marketing | **Inflated** — 5-fold honest = 0.69 |
| deployment/README claims "0.823 / 0.613" (v10, single-fold) | Marketing | **Inflated** — 5-fold honest = 0.69 / 0.55 |
| Demo shows "Val AUC 0.823" badge | Engineering | **Fabricated** — no real inference |
| Architecture iterations v7→v8→v9→v10 | Methodology | **No-op** — statistically indistinguishable |
| Reddit 100K pretraining | Methodology | **Net-harmful** on gold (−0.045) |
| Per-language confound (Spanish 0.72 vs French 0.26) | Methodology | **Real** — language, not comedian |
| Per-language norm: +0.163 gold AUC | Methodology | **Suggestive** (n=12, p=0.059, 1 fold negative, French NaN) |
| `hf_release/v8_cascade_fp32.onnx` is 0.28 MB | Engineering | **Stub** (smaller than INT8 sibling) |
| 1.5 GB of stale `.pt` files | Engineering | Wasted disk |

**The ONE rigorous number in the project**: 5-fold v10 on speaker-disjoint CV = **humor AUC 0.6924 ± 0.0359, gold AUC 0.5453 ± 0.1313** (RETHINK_2026_v2.md, from Kaggle T4 GPU).

**The ONE publishable claim**: Speaker-disjoint CV reveals identity confounds hidden by random splitting. The v7→v10 architecture is noise; the evaluation regime is the dominant factor.

---

## 2. The 24-Hour Fix List (Unanimous Priority)

| # | Action | File | Time | Outcome |
|---|--------|------|------|---------|
| 1 | Hard-link v10 in `app.py`, remove SHA-1 fallback | `app.py` | 30 min | Loud error or real v10, never silent mock |
| 2 | Delete `hf_release/v8_cascade_fp32.onnx` (stub) | `hf_release/` | 1 min | No misleading 0.28 MB twin |
| 3 | Delete stale `.pt` files (1.5 GB recovery) | `models/`, `training_output/` | 30 min | +35% disk free |
| 4 | Export per-language mean/std → `per_lang_stats.json` | new | 1 h | Demo can z-normalize in JS |
| 5 | Ship client-side ONNX.js demo with real v10 | `hf_demo_v2/index.html` | 6 h | Paste transcript → real score in browser |
| 6 | Rewrite README + HF model card with honest 5-fold numbers | `README.md`, model card | 2 h | Single 0.69 number with caveat |
| 7 | Smoke test demo + delete 0.860/0.823 from text | manual | 1 h | Demo provably runs model |
| 8 | Pin `requirements.txt` to one `onnxruntime` version | `requirements.txt` | 30 min | No future version drift |
| **Total** | | | **~12 h** | Honest demo live in 24 h |

**Compute cost**: 0 GPU hours (the v10 export is already done).

**What does NOT ship in 24 hours**:
- Retraining v10 with language as auxiliary input (40+ hours)
- Modal/Replicate backend (rejected by Q2)
- More architecture iterations (the data shows they don't matter)
- Chasing the 0.823 number (it is a leakage artifact)

---

## 3. The 4-Week Unified Narrative (Unanimous Across Councils)

### Week 1 — CLEAN + SHIP REAL DEMO
- Delete mocks, stubs, stale .pt files
- Ship real ONNX.js demo with honest 5-fold number visible
- Rewrite README + HF model card

**Cost**: 6-8 hours human, 0 GPU, +1.5 GB disk

### Week 2 — SCALE THE LANGUAGE DIAGNOSTIC
- Today: per-language analysis on 12 videos (p=0.059, French NaN)
- Run on full 162-video gold-labeled subset
- Identify whether confound is language ID or speaker ID
- **Run 5×3 repeated CV first** (free, 4 hours, 15 measurements) — this is the precondition for the 30-video study

**Cost**: 8-12 GPU hours on Kaggle T4, 16 GB disk

### Week 3 — DRAFT THE PAPER
- **Title**: "Speaker-Disjoint Cross-Validation Reveals That Humor Classifiers Learn the Language Channel, Not Humor"
- 7 pages, EMNLP Findings / ACL short / Interspeech format
- Three figures: (1) random vs speaker-disjoint CV delta, (2) per-language confusion matrix, (3) architecture ablation null across v7-v10
- Honest report of +0.163 with full CI, per-fold breakdown, NaN disclosure

**Cost**: 16-20 hours human, 0 GPU

### Week 4 — SUBMIT + ARXIV
- ArXiv preprint (cs.CL)
- Submit to nearest-deadline venue
- Public thread: "We audited our own model. Architecture doesn't matter. Data does."

**Cost**: 8 hours human, 0 GPU

**Month total**: ~50 hours human, 12 GPU hours, 0 new annotations, 0 new data

---

## 4. The Numbers — What Goes Where

| Surface | Old (inflated) | New (honest) |
|---|---|---|
| **README headline** | "AUC 0.860" | "5-fold speaker-disjoint AUC: 0.69 humor, 0.55 gold" |
| **HF model card** | "Val AUC 0.823" | "5-fold 0.69 / 0.55, single-fold 0.823 was leakage" |
| **Demo badge** | "Val AUC 0.823" | "5-fold 0.69/0.55; AUC 0.823 was artifact" |
| **deployment/README** | "v10 = Best Model 0.823/0.613" | "v10 5-fold 0.69/0.55; per-lang norm +0.163" |
| **Paper abstract** | (new) | "Random CV inflates humor AUC by 0.15-0.20 vs speaker-disjoint" |

**One number to delete everywhere**: **0.860** (v7 single-fold). Move to a "Historical claims" callout box with strikethrough and a link to the audit.

---

## 5. The Decision Tree (Council Unanimous)

```
Day 1
├── Fix app.py to hard-link v10 (30 min)
├── Delete mock demo and v8 FP32 stub (5 min)
├── Delete stale .pt files (30 min)
├── Push to HF Space + commit to GitHub (30 min)
└── 24h ship: real demo with honest 5-fold number

Day 2-7
├── Run 5×3 repeated CV (4h Kaggle T4, $0)
├── Run per-fold, per-language diagnostic (30 min, $0)
└── Decision gate at Day 7:
    ├── 15-fold mean >+0.10, CI excludes 0 → proceed to 30-video study
    └── Otherwise → publish methodology paper only, abandon per-lang norm claim

Week 2
├── 30-video × 4-language study if decision gate passed ($300-500, 3 weeks)
├── OR per-language analysis at full 162-video scale (8-12h Kaggle T4)

Week 3-4
├── Draft paper
├── ArXiv preprint
├── Public thread
└── Submit to nearest venue
```

---

## 6. The Single Most Important Action Today

**Run the 5×3 repeated CV.** Every other decision in the next month depends on its result.

- 4 hours on Kaggle T4, $0
- Produces 15 measurements instead of 5
- Determines whether the +0.163 is real or fold-luck
- Decides whether to spend $300-500 on 30-video follow-up

If 15-fold mean >+0.10 with CI excluding 0: continue.
If 15-fold CI includes 0 or any fold shows <−0.05: drop the per-language norm claim, publish only the methodology paper.

---

## 7. The One Sentence the Project README Must Open With (When Rewritten)

> "HaHaScore v10 achieves **5-fold speaker-disjoint AUC 0.69 (humor) and 0.55 (gold)** on a 162-video gold-labeled stand-up subset; standard random-split CV inflates these by 0.15-0.20 via speaker leakage, which is why earlier reported numbers (0.86 v7, 0.82 v10) are not cited here."

---

## Sign-off

**Three councils, unanimous**: stop pretending, start honest. Replace the mock demo today. Run 5×3 CV by Day 7. Write the methodology paper by Week 3. The science is real; the marketing is fake; the fix is straightforward.

— Agent Council, unanimous

---

## What we are EXPLICITLY NOT doing this month
- Iterating on architecture (v11, v12). The 5-fold number is flat.
- Collecting more data. YouTube blocked, no budget. 162 videos is what we have.
- Chasing the 0.823 number. It is a leakage artifact.
- Deploying to "production." The model is not production-ready.
- Pretending per-language norm is "confirmed" when p=0.059.
- Saying AUC 0.860 anywhere visible to a user.
