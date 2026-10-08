# HaHaScore → Product Vision: Research Roadmap

**Date**: 2026-10-07
**Status**: Living document linking current research state to product vision (PRD v6, recovered Sep 7)
**Constraint**: This terminal handles only HaHaScore repo. ChuckleNet is READ-ONLY context.

---

## 1. Product Vision (per recovered PRD v6)

The product vision is a **3-product platform** for laughter/sarcasm analysis:

| Product | Task | Core Metric | Commercial Use | Current Status |
|---|---|---|---|---|
| **P1: Group Laughter Predictor** ⭐ | Word-level audience laughter discourse positioning | **IoU-F1 (boundary precision)** | Comedian scripts, content scoring, video editing | Foundation research done; 0.50 IoU-F1 ceiling |
| **P2: Individual Laughter + Sarcasm** | Per-person acoustic analysis (requires diarization) | F1, sarcasm detection F1 | Mental health, customer service, accessibility | Deferred (VTT crowd labels unusable) |
| **P3: Sarcasm-Aware Content Scorer** | Prosodic + semantic incongruity | Sarcasm F1 | Comedy content, social media monitoring | F0-based features + Reddit corpus |

**Critical Discovery (verified)**: Word-level labels are **audience-reaction positions on function words after punchlines**, NOT lexical laughter events. This reframes the task: text = region detection; prosody = boundary refinement.

---

## 2. Current HaHaScore State (Oct 7 2026)

### What exists (HaHaScore repo)

| Artifact | Value | Maps to Product |
|---|---|---|
| `arxiv_submission/hahascore.tex` (methodology paper) | 5×3 repeated CV applied to v10 multimodal falsification (+0.163 → −0.127) | **Methodology contribution for ALL products** |
| `arxiv_submission/v1_paper.tex` (v1 Jester paper) | Text-only RoBERTa regression, ρ = 0.2292 honest 5×3 | **P3 (Sarcasm content)** — text + NLP features |
| `experiments/v2_laugho/` (sister project) | Audio-only laugh detection on AudioSet, AUC 0.7501 on 1 shard | **P1 (Group Laughter)** — audio side |
| `kaggle_v1_jester/` (v1 Jester pipeline) | Text-only baseline at ρ=0.2292, gate FAILED | **P3 (Sarcasm)** baseline |
| `kaggle_v1_jester/v1_jester_m3_cv_5x3.py` (5×3 CV harness) | Reusable: 5×3 joke-disjoint CV with bootstrap CI | **Methodology reusable across products** |

### What does NOT exist yet

- **P1 multi-modal text+audio group laughter prediction** (cascade architecture per PRD v6) — closest we have is v10 multimodal (5×3 honest ρ=0.6924 humor / ρ=0.5453 gold), but this is on detection task not word-level positioning
- **P2 individual laughter** (deferred per PRD, requires diarization data)
- **P3 sarcasm scorer** (we have text-only RoBERTa, not F0+text fusion)

---

## 3. Gaps to product vision (per P1)

| Gap | Current State | Required for P1 |
|---|---|---|
| Text region proposal (Stage 1 cascade) | `v1_paper.tex` validates text-only RoBERTa on Jester; not yet trained on StandUp4AI word-level labels | Train XLM-R-base on 620v StandUp4AI word labels (span-level F1) |
| Prosody boundary refinement (Stage 2 cascade) | v2 LaughO (sister project) uses WavLM-base-plus on 1 AudioSet shard; AUC=0.7501 | Apply same pipeline to StandUp4AI audio (620v × 20 segments = 12,400 audio clips) |
| IoU-F1 metric on word-level | We compute F1, not IoU-F1 | Implement IoU-F1 evaluator, run on the 5×3 CV predictions |
| Honest 5×3 baseline before adding complexity | v1 text-only at 0.2292 (rho, not IoU-F1) | Re-train on word-level task with proper IoU-F1 metric |

---

## 4. Research expansion plan (toward P1)

### Week 1-2: Reuse v1 framework on word-level task
- Re-train the M3-recipe RoBERTa-base on word-level labels from StandUp4AI
- Output: word-level binary classification (laugh position yes/no)
- Metric: span-level F1 (NOT IoU-F1 yet — that's Stage 2)
- 5×3 repeated joke-disjoint CV (use `laugho_cv.repeated_joke_disjoint_cv_regression()` as template)
- Target: F1 ≥ 0.50 (current v3 text-only is 0.499)
- **Cost**: 0 cash, 4-6h T4 GPU

### Week 3-4: Add prosody boundary refinement (Stage 2)
- Extract WavLM-base-plus embeddings for the same 12,400 audio clips
- Train Stage 2 model: attention-pooled WavLM + MLP head, refines boundaries
- Output: precise word-level boundaries
- Metric: IoU-F1
- Apply 5×3 CV
- **Cost**: 0 cash, 4-6h T4 GPU (or cached embeddings approach)

### Week 5-6: Cascade integration
- Combine Stage 1 + Stage 2: use Stage 1 to propose regions, Stage 2 to refine boundaries
- Train end-to-end (or use Stage 1 fixed and only fine-tune Stage 2)
- Apply 5×3 CV
- Report IoU-F1 as headline
- **Cost**: 0 cash, 4-6h T4 GPU

### Week 7-8: Scale + comparison
- Compare cascade vs text-only on IoU-F1
- Honest 5×3 baseline: text-only IoU-F1 vs cascade IoU-F1
- If cascade > text-only by ≥ 0.10, that's the empirical basis for P1
- **Cost**: 0 cash, 8-12h T4 GPU

### Total cost
- 0 cash
- ~25-35h T4 GPU (≈ 1-2 weeks of Kaggle free tier 30h/week)
- All within current infrastructure (Kaggle T4, drive data)

---

## 5. Honest scope check (per strategic_rethink)

Per `STRATEGIC_RETHINK_2026.md`:
- v1 text-only (Jester, 0.2292) is the honest baseline
- v2 multimodal (text+audio) is BLOCKED by data alignment
- v3 (generation) is BLOCKED by data

**The above research expansion explicitly addresses v2's data alignment problem**: by reusing the existing 620v StandUp4AI word-level labels, we can build the cascade architecture WITHOUT needing new data.

**P1 is achievable in 6-8 weeks with current data.** The data barrier is lower than the v2 audio-text-Jester barrier.

---

## 6. What stays in this terminal vs sister project

| Terminal | Scope |
|---|---|
| **HaHaScore** (this) | v1 text-only baseline (ρ=0.2292), v10 multimodal methodology (0.6924), v2 LaughO sister project, methodology paper, **NEW: P1 cascade word-level positioning** |
| **ChuckleNet** (READ-ONLY) | Sister product work: P2 individual, P3 sarcasm, commercial launch |

I will NOT modify the ChuckleNet repo. This terminal builds the research foundation that the ChuckleNet product work can use.

---

## 7. Memory rules (avoiding digression per user mandate)

| Rule | Source |
|---|---|
| **5×3 repeated joke-disjoint CV is the methodology standard** | `STRATEGIC_RETHINK_2026.md`, `5x3_cv_results.json`, COUNCIL_FINAL_DECISION.md |
| **v1 ceiling in honest 5×3 = 0.2292** (text-only Jester) | `experiments/v1_jester_regression/5x3_cv/RESULT.md` |
| **P1 is the PRIMARY product (group laughter predictor)** | `docs/recovered/PRD_V6_MULTI_PRODUCT_LAUGHTER_PLATFORM.md` |
| **Word-level labels = audience-reaction positions on function words, NOT acoustic laughter** | PRD v6 "Critical Discovery" H0 VERIFIED |
| **ChuckleNet is READ-ONLY** | user directive 2026-10-05 |
| **Strategic pivot Sep 26: research → product → commercial** | `pivotal_decision_2026_09_26` |

---

## 8. Action plan for this week

| Day | Action | Output |
|---|---|---|
| **Today (Oct 7)** | ✅ M3-recipe v1 5×3 CV complete (ρ=0.2292) | v1 paper draft + methodology paper ready |
| **Mon Oct 13** | Build `train_p1_cascade.py`: text Stage 1 (XLM-R-base, word-level binary classification, span-level F1) on 620v StandUp4AI | Span-level F1 ≥ 0.50 (vs current v3=0.499) |
| **Tue Oct 14** | Apply 5×3 joke-disjoint CV to Stage 1 (use `laugho_cv.repeated_joke_disjoint_cv_regression()` template adapted for binary classification) | 5×3 honest span-level F1 for text-only baseline |
| **Wed-Thu Oct 15-16** | Add Stage 2: WavLM-base-plus embeddings for the same 12,400 audio clips (per cached-embeddings pattern, ~2h CPU) + train prosody boundary refinement head | Stage 2 alone 5×3 honest IoU-F1 |
| **Fri Oct 17** | Cascade integration: Stage 1 proposes regions, Stage 2 refines boundaries; 5×3 CV | Cascade IoU-F1 vs text-only IoU-F1 |
| **Weekend** | Document P1 cascade results, update PRODUCT_VISION_RESEARCH_ROADMAP.md, commit | 6 commits to HaHaScore repo |

**Total cost**: 0 cash, ~10-15h T4 GPU over the week (within Kaggle free tier 30h/week), 0 modifications to ChuckleNet.

---

## 9. What this roadmap does NOT promise

- ❌ P1 launch (PRD says "research → product → commercial"; this roadmap is the research half)
- ❌ P2 (individual laughter) — different problem, deferred per PRD
- ❌ P3 (sarcasm) — covered by v1 Jester work but not focused here
- ❌ Modifying ChuckleNet repo — user directive, READ-ONLY
- ❌ Competing with ChuckleNet sister-project work

**The scope is: research foundation for P1, built incrementally with current data, validated with 5×3 repeated joke-disjoint CV + bootstrap CIs.** All other products are either out of scope (P2) or covered elsewhere (P3 = v1 Jester work).

