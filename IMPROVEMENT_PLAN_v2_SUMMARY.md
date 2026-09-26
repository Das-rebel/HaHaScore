# Cascade Gate v8 Improvement Plan — Ensemble-Reviewed Summary

**Document version**: v2.0 summary
**Date**: post-ensemble review
**Status**: REVISION — incorporating critic feedback

---

## 🎯 Ensemble Consensus

Three expert critics reviewed the v7 improvement plan. Sharp, actionable consensus:

> **"The 0.27 humor-laughter AUC gap is 80% data, 20% training. Architecture and optimizer are fine. Stop adding capacity. Add bilinear fusion + CORAL domain adaptation + multi-task laughter head."**

---

## 📊 Critic Verdicts

### Architecture Critic
- Drop MultiScaleAttention (–198K params, –15ms)
- Add bilinear fusion (`text_h @ W_b @ audio_h.T`, ~16K params)
- Ship 0.65M model, not 1.19M

### Training/Data Critic
- Remove curriculum (no difficulty axis)
- Augmentation σ=0.05–0.10 (not 0.01)
- Add CORAL domain adaptation
- Multi-task auxiliary laughter head
- Co-teaching for noise
- 5×3 repeated CV

### Deployment Critic
- ONNX + INT8 dynamic quantization
- Replace SHA-1 placeholder in `app.py`
- Ship v7 baseline now, v8 enhanced later

---

## 🚀 Three-Tier Strategy

| Tier | Goal | Target | When |
|------|------|--------|------|
| **1** | Ship baseline ONNX | <10ms, working demo | Week 1 |
| **2** | Cascade Gate v8 (slim) | 0.65M params, AUC ≥ 0.865 | Week 2 |
| **3** | Domain adaptation | Gold AUC 0.59 → 0.63+ | Week 4 |

---

## 📈 Revised Targets

| Metric | v7 Current | v8 Target | v8+Tier3 Stretch |
|--------|------------|-----------|------------------|
| **Params** | 1.04M | 0.65M | 0.75M |
| **Pseudo AUC** | 0.860 | 0.865 | 0.870 |
| **Gold AUC** | 0.590 | 0.600 | **0.630–0.650** |
| **Inference (CPU)** | ~10ms | **~5ms** | ~8ms |
| **Image size** | 4.0MB | **~1MB** (INT8) | ~1.2MB |

---

## 🔧 Key Implementation Tasks

### Tier 1 (Week 1)
- `export_onnx.py` — ONNX export + INT8 quantization
- Modified `app.py` — real inference (not SHA-1 placeholder)
- Updated `requirements.txt` — add onnxruntime

### Tier 2 (Week 2)
- `improvements/bilinear_fusion.py` — Bilinear cross-modal fusion
- Modify `improvements/enhanced_cascade.py` — slim v8 architecture
- Drop MultiScaleAttention, simplify gating, single-layer BiGRU

### Tier 3 (Week 3–4)
- `improvements/coral.py` — CORAL domain adaptation loss
- `improvements/multitask_v8.py` — Multi-task architecture
- Bootstrap loss for pseudo-label noise

---

## 📋 Decision Tree

```
Start
  │
  ├─→ Tier 1: ONNX + Demo Fix  [BLOCKING - ship now]
  │     │
  │     └─→ Tier 2: Cascade Gate v8 (Slim)
  │           │
  │           ├─→ AUC improves → proceed to Tier 3
  │           └─→ AUC regresses → ship v7 ONNX, abandon v8
  │
  └─→ Tier 3: Multi-task + CORAL (parallel research track)
```

---

**Recommendation**: Tier 1 (immediate) + Tier 2 (parallel research) + Tier 3 (gold laughter stretch goal).

**Next Step**: Begin Tier 1 — export ONNX and fix `app.py` placeholder.
