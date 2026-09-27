# HaHaScale — Model Scaling Plan (Replan)

**Date**: post-v10 release | **Current best**: v10 (humor 0.8225, gold 0.6131, 1.5ms)

---

## Where We Are vs. Where We Could Be

| Lever | Current | Available | Multiplier | Cost |
|-------|---------|-----------|------------|------|
| Reddit pretrain data | 30K jokes used | 100K stratified on Drive | **3.3×** | T4: ~2 hrs |
| ColBERT pretrain | 0 used | 200K balanced texts | **new stage** | T4: ~1 hr |
| UR-FUNNY multimodal | 0 used | 1,866 TED samples w/ Covarep audio | **gold-AUC lever** | 764MB dl + 4 hrs |
| Standup audio | 639 files | 639 + 273 gillick = 912 | **1.4×** | feature extraction |
| Text backbone | DistilBERT 66M | DeBERTa-v3 184M | +0.02-0.04 AUC | T4 only |
| Eval folds | 1 fold | 5-fold CV | statistical rigor | 5× train time |
| **Compute** | 8-core CPU | **Kaggle T4 ×2, free 30 hr/wk** | **~20× faster** | $0 |

**Honest ceilings**: pseudo AUC 0.82 → **0.85-0.87** (data+backbone); gold AUC 0.613 → **0.65-0.68** (UR-FUNNY + labels; literature ceiling ~0.70). Gold is label-limited (only 12/34 gold videos overlap our set), not architecture-limited.

## Scaling Strategy: 4 Phases

### Phase 1 — Statistical Rigor + Move to GPU (today, ~3 hrs)
1. **5-fold CV of v10** on Kaggle T4 → mean ± std (is 0.8225/0.6131 real?)
2. Push features + scripts to Kaggle dataset (bridge4 + v6 npz = 84MB)
3. Establish Kaggle as the training backend (disk + GPU solved)

### Phase 2 — Data Scale (this week, Kaggle T4 ~4 hrs)
4. **Reddit 100K pretrain** (10.3 hrs CPU → ~1.5 hrs T4)
5. **ColBERT 200K stage-1** binary pretrain (council's S1)
6. Re-fine-tune v10 with 100K-Reddit encoder → target pseudo 0.84+

### Phase 3 — Gold-Laughter Scale (next week)
7. **UR-FUNNY V1 features** (764MB Dropbox → Kaggle, not local; disk is 4.3GB)
   - covarep audio + language SDK + humor labels = 1,866 real multimodal samples
   - Multi-task pretrain → fine-tune → target gold 0.65+
8. **Gillick 273 extra audio files** → extract WavLM features → +43% standup audio

### Phase 4 — Product Scale (per 2026-09-26 pivot: traction-first)
9. **Real Laugh API** on HF Space (replace mock demo inference with real ONNX + transcript)
10. Metered API → first 10 customers → $2K MRR path

## Rules (constraints re-checked)
- Local disk 4.3GB free → **all bulk work goes to Kaggle/Drive, never local**
- YouTube blocked (India IP) → no new audio downloads; use existing + UR-FUNNY
- MPS unavailable in this env → CPU for local smoke tests only, T4 for real runs
