# HaHaScore Decision Graph v7
**Date:** 2026-09-16 | **Status:** ACTIVE — Post 10x Research

---

## Decision Nodes

### D-G1: v5 Results — Text-Only Is Random ✅ RESOLVED
**Finding:** Text-only AUC ~0.50 (random) for sentence-level humor. Audio is the primary discriminative signal.
**Decision:** Multimodal fusion (text + audio) is REQUIRED. Text alone cannot solve this task.
**Evidence:** v5 bilinear fusion AUC 0.613 vs text-only 0.4971. 10x research confirms: MTLLFM, TIC-TALK all use audio as primary.
**Status:** Locked — no further text-only experiments.

### D-G2: Audio Encoder Choice — WavLM vs HuBERT 🔄 IN PROGRESS
**Finding:** MTLLFM (WSC Sports) used HuBERT + MAE and got F1=99% on laughter localization. WavLM used in v5.
**Decision:** Compare HuBERT vs WavLM on SAME data split. Do NOT assume WavLM is best.
**Options:**
- A: Use WavLM (current, AUC 0.613)
- B: Use HuBERT (MTLLFM approach) ← likely better
- C: Ensemble both
**Action:** `extract_audio_features_modal(model_name="facebook/hubert-base")` → compare head-to-head.
**Status:** Week 2 — GPU ACCESS CONFIRMED (Modal T4 16GB, PyTorch 2.14+cu130 ✅)

### D-G3: Third Modality — Kinetic Energy 🔄 READY
**Finding:** TIC-TALK (r=-0.75): Kinetic energy NEGATIVELY correlates with laughter. Stillness before punchline = more laughter.
**Decision:** Add kinetic energy as a THIRD input modality alongside text + audio.
**Implementation:** YOLOv8s-pose for skeletal keypoints → compute kinetic energy (RMS of joint velocities, 7 features: mean/std/max/p25/p75/median/sum).
**Code:** `extract_kinetic_from_videos()` in `modal_train_v6.py` ✅
**Expected Impact:** +5-8% AUC (r=-0.75 is very strong signal).
**Status:** READY to run — need StandUp4AI video files (or any comedy videos)

### D-G4: Fusion Architecture — Bilinear vs Cross-Attention 🔄 IN PROGRESS
**Finding:** DARC-CLIP showed adaptive cross-attention > static fusion (+4.18 AUROC). MTLLFM uses adaptive gating.
**Decision:** Replace bilinear with cross-attention + gating in v6b.
**Options:**
- A: Bilinear (current, AUC 0.613) — simple, works
- B: Cross-attention — better for misaligned features
- C: Bilinear + adaptive gating — hybrid
**Action:** Implement B or C in v6b. Evidence suggests C (bilinear + gating) is best balance.
**Status:** Week 3.

### D-G5: Training Data Scaleup 🔄 IN PROGRESS
**Finding:** 35+ datasets found. UR-FUNNY-Temporal (11K videos, public), StandUp4AI (103 videos WavLM cached on GDrive), TIC-TALK 90 specials.
**Decision:** Scale from 3,774 clips to 50K+ clips via:
1. StandUp4AI 103 videos (WavLM features already on GDrive — FREE)
2. Semi-supervised on 467K word-level pseudo-labels
3. UR-FUNNY-Temporal for benchmark comparison
**Status:** Week 1-4.

### D-G6: GPU Compute Strategy ✅ CONFIRMED
**Finding:** MacBook CPU too slow. Modal T4 (16GB) is FREE and confirmed working.
**Result:**
- Modal workspace: `sdas22` (ak-144kQcm6lVNW39gDvUqhlb)
- GPU: Tesla T4 16GB ✅ (PyTorch 2.14.0+cu130)
- Image: `im-WYVBpse7K0g5C7K0BsefFo` (cached, includes torch+transformers+ultralytics)
- A100: blocked (needs payment method — T4 free tier sufficient for v6)
**Status:** ✅ COMPLETE — T4 GPU ready, scaffold deployed

### D-G7: Commercial Opportunity — No Direct Competitor ✅ CONFIRMED
**Finding:** NO commercial product offers humor strength scoring (0-100). WSC Sports does laughter localization (sports). Gong does sales analytics. Hume AI has laughter detection. All adjacent, none direct.
**Decision:** HaHaScore is UNIQUE. Market gap is real. Go to market with "world's first humor strength API."
**Positioning:** "Measure how funny your comedy content is — 0 to 100 — using the same AI that analyzes audience laughter."
**Status:** Locked — proceed to commercial deployment in parallel with research.

### D-G8: Target Metric — AUC vs Spearman ρ 🔄 IN PROGRESS
**Finding:** Current v5 uses AUC (binary: funny/not-funny). Research shows continuous regression is more meaningful.
**Decision:** Pivot to continuous STRENGTH scoring (0-100) using:
- Laughter duration as continuous proxy (from VTT markers)
- Regression loss (MSE) instead of BCE
**Target:** Spearman ρ for continuous score prediction.
**Status:** Week 3 — pending pseudo-label generation for 467K word samples.

### D-G9: CLAP for Zero-Shot Humor 🔄 DEFERRED
**Finding:** CLAP (laion/clap) encodes audio-text similarity. Can encode "this is funny" as text anchor.
**Decision:** Investigate CLAP as auxiliary signal. Low priority vs. kinetic energy + HuBERT.
**Status:** Deferred to v7.

### D-G10: Academic Benchmark Publication 🔄 IN PROGRESS
**Finding:** No standard benchmark for sentence-level humor strength. HaHaScore-Bench would be first.
**Decision:** Publish HaHaScore-Bench-48 as standard benchmark alongside ICASSP/INTERSPEECH paper.
**Target:** INTERSPEECH 2027 (Sep 2026 deadline).
**Status:** Week 4+.

---

## Research Decisions Locked (10x Survey)

| Decision | Value | Confidence |
|----------|-------|------------|
| Audio is primary signal | Text AUC=0.50, Fusion AUC=0.613 | High |
| Kinetic energy r=-0.75 with laughter | TIC-TALK 90 specials | High |
| HuBERT > WavLM for speech | MTLLFM F1=99% | Medium (needs head-to-head) |
| Cross-attention > concat for misaligned | DARC-CLIP +4.18 AUROC | Medium |
| No commercial humor strength API | 25+ products surveyed | High |
| WSC Sports = closest competitor | Sports laughter localization | High |

---

## Action Graph (Next 30 Days)

```
Week 1: Infrastructure
├─ D-G6: Set up Modal (A100 GPU)
├─ D-G5: Download StandUp4AI WavLM features (GDrive)
└─ D-G3: Build video processing pipeline (YOLOv8s-pose)

Week 2: Model v6a
├─ D-G2: HuBERT vs WavLM head-to-head
├─ D-G3: Add kinetic energy as 3rd modality
└─ D-G4: Implement cross-attention + gating

Week 3: Model v6b  
├─ D-G8: Pivot to continuous regression
├─ D-G5: Semi-supervised on 467K pseudo-labels
└─ D-G4: Finalize fusion architecture

Week 4: Commercial + Publication
├─ D-G7: Deploy REST API on Modal
├─ D-G10: Write benchmark paper
└─ D-G5: Integrate all 103 StandUp4AI videos
```

---

## Execution Log (2026-09-16)

| Step | Status | Notes |
|------|--------|-------|
| Modal token setup | ✅ Done | `ak-144kQcm6lVNW39gDvUqhlb`, workspace sdas22 |
| T4 GPU smoke test | ✅ Done | Tesla T4 16GB, PyTorch 2.14+cu130 |
| v6 scaffold | ✅ Done | `modal_train_v6.py` (449 lines) |
| DECISION_GRAPH.md | ✅ Done | v7 with 10 decision nodes |
| DEEP_RESEARCH_AND_SCALEUP_PLAN.md | ✅ Done | 500+ lines, committed |
| GitHub push | ✅ Done | Commits through `101561f` |
| Disk cleanup | ✅ Done | 12GB free (was 8.4GB) |
| Kinetic extraction code | ✅ Ready | YOLOv8s-pose scaffold in modal_train_v6.py |
| Audio extraction code | ✅ Ready | HuBERT + WavLM in modal_train_v6.py |
| StandUp4AI videos | ⏳ Pending | Need to download from Kaggle |
| Feature file sync to Modal | ⏳ Pending | Need to mount/checkpoint volume |

---

## Confidence Matrix

| Finding | Confidence | Source | Validated? |
|---------|-----------|--------|-------------|
| Text-only random | ✅ High | v5 experiments | Yes |
| Kinetic energy r=-0.75 | ✅ High | TIC-TALK paper | Needs own experiment |
| HuBERT > WavLM | ⚠️ Medium | MTLLFM (different task) | Needs head-to-head |
| Cross-attention helps | ⚠️ Medium | DARC-CLIP (memes) | Needs own experiment |
| No commercial competitor | ✅ High | 25+ product survey | Yes |
| +5-8% from kinetic | ⚠️ Medium | r=-0.75 correlation | Needs own experiment |
