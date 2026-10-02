# Agent Council Decision — Post-Audit Path Forward (UPDATED)

**Date**: 2026-09-28 (updated 2026-10-02 with 5x3 CV falsification)

Three parallel councils (decision, production-engineering, research-methodology) reviewed the file audit + claims ledger. The +0.163 per-language norm finding was subsequently **FALSIFIED** by 5x3 repeated CV (delta = -0.127, p=0.0004). Updated verdict below.

---

## TL;DR

The project is in a state of **controlled self-deception**: real science + fake marketing. The 5-fold v10 number (humor 0.69, gold 0.55) IS publishable as a methodology contribution. The README's 0.86 and the demo's 0.823 are NOT. The single highest-EV action in the next 24 hours is to **replace the mock demo with real ONNX inference** — that one action unblocks 80% of the cleanup.

**CRITICAL UPDATE (2026-10-02)**: The per-language normalization claim (+0.163 gold AUC) was **falsified by 5x3 repeated CV**. Across 15 measurements, the true effect is **-0.127** (NEGATIVE — normalization HURTS gold AUC, p=0.0004, 95% CI [-0.183, -0.078]). The single 5-fold CV at seed=42 was a lucky split. The +0.163 must be **dropped from all documentation**. The publishable claim reverts to: "Speaker-disjoint CV reveals identity confounds hidden by random splitting" + "Per-language mean subtraction destroys the model's learned language-aware signal."

---

## 1. The Council's Unanimous Diagnosis

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
| **Per-language norm: +0.163 gold AUC** | **Falsified** | **DO NOT CITE** — 5x3 CV: true effect = **-0.127** |
| `hf_release/v8_cascade_fp32.onnx` is 0.28 MB | Engineering | **Stub** (smaller than INT8 sibling) |
| 1.5 GB of stale `.pt` files | Engineering | Wasted disk |

**The ONE rigorous number in the project**: 5-fold v10 on speaker-disjoint CV = **humor AUC 0.6924 ± 0.0359, gold AUC 0.5453 ± 0.1313** (RETHINK_2026_v2.md, from Kaggle T4 GPU).

**The ONE publishable claim**: Speaker-disjoint CV reveals identity confounds hidden by random splitting. The v7→v10 architecture is noise; the evaluation regime is the dominant factor. (Per ChuckleNet v20_gate_forensics: single-fold metrics on imbalanced data are unreliable. ALWAYS use multi-seed repeated CV + bootstrap CIs before any AUC claim.)

---

## 2. The 24-Hour Fix List (Unanimous Priority)

| # | Action | File | Time | Outcome |
|---|--------|------|------|---------|
| 1 | Hard-link v10 in `app.py`, remove SHA-1 fallback | `app.py` | 30 min | Loud error or real v10, never silent mock |
| 2 | Delete `hf_release/v8_cascade_fp32.onnx` (stub) | `hf_release/` | 1 min | No misleading 0.28 MB twin |
| 3 | Delete stale `.pt` files (1.5 GB recovery) | `models/`, `training_output/` | 30 min | +35% disk free |
| 4 | Export per-language mean/std → `per_lang_stats.json` | new | 1 h | **DEFERRED** — claim falsified |
| 5 | Ship client-side ONNX.js demo with real v10 | `hf_demo_v2/index.html` | 6 h | Paste transcript → real score in browser |
| 6 | Rewrite README + HF model card with honest 5-fold numbers | `README.md`, model card | 2 h | Single 0.69 number with caveat |
| 7 | Smoke test demo + delete 0.860/0.823 from text | manual | 1 h | Demo provably runs model |
| 8 | Pin `requirements.txt` to one `onnxruntime` version | `requirements.txt` | 30 min | No future version drift |
| **Total** | | | **~12 h** | Honest demo live in 24 h |

**Compute cost**: 0 GPU hours (the v10 export is already done).

**What does NOT ship in 24 hours**:
- ~~Per-language normalization fix~~ — **FALSIFIED by 5x3 CV**; do not add
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

### Week 2 — SCALE THE LANGUAGE DIAGNOSTIC (with PR-AUC, not AUC)
- Today: per-language analysis on 12 videos (p=0.059, French NaN)
- Per ChuckleNet v20_gate_forensics: **use PR-AUC instead of F1@0.5 or AUC for sparse labels** (1.16% pos rate)
- Run on full 162-video gold-labeled subset
- **Run 5×3 repeated CV first** (free, 4 hours, 15 measurements with PR-AUC) — this is the precondition for any follow-up

**Cost**: 8-12 GPU hours on Kaggle T4, 16 GB disk

### Week 3 — DRAFT THE PAPER
- **Title**: "Speaker-Disjoint Cross-Validation Reveals That Humor Classifiers Learn the Language Channel, Not Humor"
- 7 pages, EMNLP Findings / ACL short / Interspeech format
- Three figures: (1) random vs speaker-disjoint CV delta (PR-AUC), (2) per-language confusion matrix, (3) architecture ablation null across v7-v10
- **DO NOT include the +0.163 finding** (falsified by 5x3 CV)

**Cost**: 16-20 hours human, 0 GPU

### Week 4 — SUBMIT + ARXIV
- ArXiv preprint (cs.CL)
- Submit to nearest-deadline venue
- Public thread: "We audited our own model. Architecture doesn't matter. Data does. And our 'fix' was fold-luck — here's the 5x3 CV that caught it before publication."

**Cost**: 8 hours human, 0 GPU

**Month total**: ~50 hours human, 12 GPU hours, 0 new annotations, 0 new data

---

## 4. The Numbers — What Goes Where

| Surface | Old (inflated) | New (honest) |
|---|---|---|
| **README headline** | "AUC 0.860" | "5-fold speaker-disjoint AUC: 0.69 humor, 0.55 gold" |
| **HF model card** | "Val AUC 0.823" | "5-fold 0.69 / 0.55, single-fold 0.823 was leakage" |
| **Demo badge** | "Val AUC 0.823" | "5-fold 0.69/0.55; AUC 0.823 was artifact" |
| **deployment/README** | "v10 = Best Model 0.823/0.613" | "v10 5-fold 0.69/0.55" |
| **Paper abstract** | (new) | "Random CV inflates humor AUC by 0.15-0.20 vs speaker-disjoint" |
| **~Per-language norm~** | ~~"+0.163 gold AUC"~~ | ~~"FALSIFIED by 5x3 CV; -0.127 true effect"~~ |

**One number to delete everywhere**: **0.860** (v7 single-fold). Move to a "Historical claims" callout box with strikethrough and a link to the audit.

**One more number to delete everywhere**: **+0.163** (single-fold per-language norm). Already marked as falsified in `lang_norm_results.json`.

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
├── Run 5×3 repeated CV with PR-AUC (4h Kaggle T4, $0)
├── Run per-fold, per-language diagnostic (30 min, $0)
└── Decision gate at Day 7:
    ├── 15-fold mean PR-AUC >+0.10, CI excludes 0 → proceed to 30-video study
    └── Otherwise → publish methodology paper only

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

**Run 5×3 repeated CV with PR-AUC (not AUC).** Every other decision in the next month depends on its result.

- 4 hours on Kaggle T4, $0
- Produces 15 measurements instead of 5
- Determines whether ANY claim is real or fold-luck
- Decides whether to spend $300-500 on 30-video follow-up

**If 15-fold mean PR-AUC delta >+0.10 with CI excluding 0: continue.**
**If 15-fold CI includes 0 or any fold shows <−0.05: drop the claim, publish only the methodology paper.**

---

## 7. The One Sentence the Project README Must Open With (When Rewritten)

> "HaHaScore v10 achieves **5-fold speaker-disjoint AUC 0.69 (humor) and 0.55 (gold)** on a 162-video gold-labeled stand-up subset; standard random-split CV inflates these by 0.15-0.20 via speaker leakage, which is why earlier reported numbers (0.86 v7, 0.82 v10) are not cited here. The +0.163 per-language normalization finding (single 5-fold CV) was falsified by 5×3 repeated CV (true effect: -0.127, p=0.0004)."

---

## 8. CRITICAL UPDATE: The +0.163 Lesson

The most important finding from the 5×3 repeated CV (2026-10-02):

| What we thought | What the data says (n=15) |
|---|---|
| Per-language normalization fixes the language confound | Per-language normalization **destroys** the model's learned signal |
| +0.163 gold AUC, p=0.059, "suggestive" | **−0.127 gold AUC, p=0.0004** (the opposite direction, highly significant) |
| Worth pursuing as a research contribution | **Drop the claim entirely**; the v10 model was already language-aware |

**The deeper lesson** (per ChuckleNet v20_gate_forensics in memory): **single-fold metrics on imbalanced data are systematically misleading**. The v20 guard caught a 0.7% positive rate; the v10 5-fold CV would have caught the +0.163 fold-luck. The rule is: **never trust a single-fold AUC claim on <20% positive class data without multi-seed repeated CV + bootstrap CIs.**

---

## Sign-off

**Three councils, unanimous, with 5×3 CV falsification**: stop pretending, start honest. The +0.163 finding was the single most embarrassing trap in the project. Replace the mock demo today. Run 5×3 CV with PR-AUC by Day 7. Write the methodology paper by Week 3. The science is real; the marketing is fake; the fix is straightforward.

— Agent Council, unanimous (with empirical falsification)

---

## What we are EXPLICITLY NOT doing this month
- Iterating on architecture (v11, v12). The 5-fold number is flat.
- **Per-language normalization** (falsified by 5×3 CV; -0.127 true effect).
- More data collection. YouTube blocked, no budget. 162 videos is what we have.
- Chasing the 0.823 number. It is a leakage artifact.
- Deploying to "production." The model is not production-ready.
- Saying AUC 0.860 anywhere visible to a user.
- **Saying +0.163 anywhere visible to a user.** It was fold-luck.
- Using single-fold AUC on imbalanced data. **PR-AUC + 5×3 CV is the minimum honest protocol.**
