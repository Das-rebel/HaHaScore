# Cascade Gate v8 Improvement Plan — Ensemble-Reviewed Strategy

**Document version**: v2.0 (after ensemble review by 3 specialist agents)
**Date**: post-ensemble review
**Status**: REVISION — incorporating critic feedback

---

## 🎯 Ensemble Synthesis

Three expert critics reviewed the v7 improvement plan. Their consensus is **sharp and actionable**:

### Consensus Verdict
> **"The 0.27 humor-laughter AUC gap is 80% data, 20% training. Architecture and optimizer are fine. Stop adding capacity. Add bilinear fusion + CORAL domain adaptation + multi-task laughter head."**

| Critic | Top Recommendation | Action |
|--------|-------------------|--------|
| **Architecture** | Drop MultiScaleAttention (–198K params, –15ms). Add bilinear fusion. Ship 0.65M model. | ✅ Adopt |
| **Training/Data** | Remove curriculum (no difficulty axis). Add CORAL domain adaptation. Multi-task laughter head. Co-teaching for noise. Use σ=0.05–0.10 augmentation. 5×3 repeated CV. | ✅ Adopt |
| **Deployment** | ONNX + INT8 dynamic quantization. Replace SHA-1 placeholder in `app.py`. Ship v7 baseline now, v8 enhanced later. | ✅ Adopt |

---

## 🔬 Synthesized Findings

### Finding 1: Architecture is 30% over-parameterized
- Multi-scale attention scales [1,2,4] on 20 segments = 75% interpolated positions → wasted capacity
- 4-head attention with head_dim=32 violates the head_dim=64 norm
- Pseudo-label ceiling at 0.860 means **adding capacity won't help** — it overfits noise
- **Recommendation**: Drop MultiScaleAttention entirely. Replace cross-modal attention with **bilinear fusion** (`text_h @ W_b @ audio_h.T`, ~16K params).

### Finding 2: Dynamic Gating with Uncertainty is over-engineered
- 25K params predicting mean+log_var on pseudo-labels → fits pseudo-noise, not gold generalization
- 3-input gate MLP (209 params) is fine; **drop the estimator heads**
- Use **MC-dropout at inference** for free, calibrated uncertainty

### Finding 3: Humor ≠ Laughter (Fundamental Data Issue)
- Literature (Purandare 2006, Bertero 2016): ρ ≈ 0.3–0.5 between perceived funniness and laughter
- Gold AUC ceiling ≈ **0.65–0.70**, not 0.86
- **Stop optimizing pseudo-AUC as proxy for gold AUC**
- Solution: **multi-task auxiliary laughter head** + **CORAL domain adaptation**

### Finding 4: Training Strategy Mismatches
- Curriculum learning inappropriate (no difficulty axis with uniform 20-segment batches)
- Augmentation σ=0.01 is 5–10× too small (use σ=0.05–0.10)
- 5-fold CV variance ±0.018 vs +0.015 target = 55% power (insufficient)
- Pseudo-label noise ~25–30% (need co-teaching or bootstrap loss)

### Finding 5: Inference Bottleneck is NOT the Model
- Model forward pass: **<2ms on CPU**
- Overhead: 31ms in Python feature extraction, padding, normalization
- `build_enhanced_features()` runs per-call with no graph capture
- `app.py` is a SHA-1 hash placeholder — not actually running either model

---

## 🚀 REVISED PLAN: Three-Tier Strategy

### Tier 1: **Immediately Ship Baseline v7 to Production (Day 1–3)**
**Why**: Demo currently broken (SHA-1 placeholder). Users can't evaluate anything.

```
1. Export existing cascade_v7.pt to ONNX with dynamic_axes
2. Apply INT8 dynamic quantization (preserve gate in FP32)
3. Replace SHA-1 placeholder in app.py with onnxruntime.InferenceSession
4. Deploy to Hayasuki/hahascore-cascade-demo
5. Target: ~10ms inference, 4x smaller image
```

**Files needed**:
- `export_onnx.py` (export + quantize)
- Modified `app.py` (real inference, not SHA-1)
- `requirements.txt` (add onnxruntime)

### Tier 2: **Cascade Gate v8 — Slimmer Architecture (Week 1–2)**
**Why**: Drop wasted capacity; add bilinear fusion; target 0.65M params (45% smaller).

```
New model architecture:
- Input projections: text (780→128), audio (823→128), cross (16→128)
- Drop MultiScaleAttention (–198K params, –15ms)
- Add BILINEAR CROSS-MODAL FUSION (text_h @ W_b @ audio_h.T) [~16K params]
- Simplified Dynamic Gating (text_conf → sigmoid, no uncertainty heads)
- Single-layer BiGRU (hidden=128) [–246K params]
- Prediction head: 1 layer MLP
- TOTAL: ~0.65M params, ~18ms inference
```

**Files to modify**:
- `improvements/enhanced_cascade.py` (slim version)
- Add `bilinear_fusion.py` module

### Tier 3: **Domain Adaptation for Gold Laughter (Week 3–4)**
**Why**: 0.27 AUC gap is fundamentally a domain shift problem.

```
Multi-task architecture:
- Shared backbone (v8)
- Primary head: humor strength (continuous) → pseudo-labels
- Auxiliary head: laughter (binary) → gold labels
- Loss: L_humor + 0.3 * L_laughter
- + CORAL domain adaptation on penultimate features (λ_align=0.5)
- + Co-teaching or bootstrap loss for noise (β=0.95)
```

**Expected outcome**: Gold AUC 0.590 → 0.630–0.650 (realistic)

---

## 📊 Revised Targets

| Metric | v7 Current | v8 Target | v8+Tier3 Stretch |
|--------|------------|-----------|------------------|
| **Params** | 1.04M | 0.65M | 0.75M |
| **Pseudo AUC** | 0.860 | 0.865 | 0.870 |
| **Gold AUC** | 0.590 | 0.600 | **0.630–0.650** |
| **Inference (CPU)** | ~10ms | **~5ms** | ~8ms |
| **Image size** | 4.0MB | **~1MB** (INT8) | ~1.2MB |
| **CV Variance** | ±0.018 | ±0.015 | ±0.010 (5×3) |

---

## 🔧 Implementation Tasks

### Task 1: ONNX Export + INT8 Quantization (Day 1)
```python
# export_onnx.py
import torch
from enhanced_cascade import create_enhanced_model

model = create_enhanced_model(...)  # use original 768/791 dims
model.load_state_dict(torch.load("models/cascade_v7.pt"))
model.eval()

dummy = (torch.randn(1, 20, 768), torch.randn(1, 20, 791), torch.randn(1, 20, 16))
torch.onnx.export(model, dummy, "models/cascade_v7.onnx",
                  dynamic_axes={"text": {0:"batch"}, "audio": {0:"batch"}, "cross": {0:"batch"}},
                  opset_version=17)

# Quantize
from torch.ao.quantization import quantize_dynamic
quantized = quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
torch.save(quantized.state_dict(), "models/cascade_v7_int8.pt")
```

### Task 2: Replace SHA-1 Placeholder in app.py (Day 2)
```python
# app.py (real version)
import onnxruntime as ort
import numpy as np

class HaHaScorer:
    def __init__(self):
        self.session = ort.InferenceSession("models/cascade_v7.onnx")
    
    def score(self, text_features, audio_features):
        outputs = self.session.run(None, {
            "text": text_features,
            "audio": audio_features,
            "cross": np.zeros((1, 20, 16), dtype=np.float32)
        })
        return outputs[0]
```

### Task 3: Bilinear Fusion Module (Week 1)
```python
# improvements/bilinear_fusion.py
class BilinearFusion(nn.Module):
    def __init__(self, text_dim, audio_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(text_dim, audio_dim) / np.sqrt(text_dim))
        self.proj = nn.Linear(text_dim + audio_dim, output_dim)
    
    def forward(self, text, audio):
        # text, audio: (B, L, D)
        # Bilinear: (B, L, T) @ W @ (B, L, A).T → (B, L, L)
        bil = torch.einsum('blt,ta,bma→blm', text, self.W, audio)
        # Mean-pool to (B, L)
        bil_pool = bil.mean(dim=-1, keepdim=True).expand_as(text)
        return self.proj(torch.cat([text, bil_pool], dim=-1))
```

### Task 4: CORAL Domain Adaptation (Week 3)
```python
# improvements/coral.py
def coral_loss(source_feat, target_feat):
    """CORAL: align covariance matrices between source (pseudo) and target (gold)."""
    d = source_feat.size(1)
    ns, nt = source_feat.size(0), target_feat.size(0)
    
    # Source covariance
    src_mean = source_feat.mean(dim=0)
    src_c = (source_feat - src_mean).T @ (source_feat - src_mean) / (ns - 1)
    
    # Target covariance
    tgt_mean = target_feat.mean(dim=0)
    tgt_c = (target_feat - tgt_mean).T @ (target_feat - tgt_mean) / (nt - 1)
    
    return (src_c - tgt_c).pow(2).sum() / (4 * d * d)
```

### Task 5: Multi-Task Architecture (Week 3)
```python
# improvements/multitask_v8.py
class MultiTaskCascadeGate(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... shared backbone (v8 slim) ...
        self.humor_head = nn.Linear(hidden*2, 1)  # continuous
        self.laughter_head = nn.Linear(hidden*2, 1)  # binary
    
    def forward(self, text, audio, cross):
        shared_features = self.backbone(text, audio, cross)
        return {
            'humor': torch.sigmoid(self.humor_head(shared_features)),
            'laughter': torch.sigmoid(self.laughter_head(shared_features))
        }
```

---

## 📈 Updated Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| v8 AUC lower than v7 | Medium | Bilinear fusion alone is +0.02 literature precedent; if regression, ship v7 ONNX |
| Domain adaptation fails | Low | CORAL is well-established; fallback to v8 single-task |
| Quantization breaks gate | Medium | Preserve gate in FP32 (mixed precision), validate on held-out set |
| 5×3 CV not enough | Low | Power analysis: 85% sufficient for +0.02 effect |

---

## 🎯 Success Criteria (Revised)

**Tier 1 (Ship in 1 week)**:
- ✅ Production inference working at <10ms
- ✅ Real model behind demo (no SHA-1 placeholder)
- ✅ INT8 quantized model deployed
- ✅ User feedback mechanism

**Tier 2 (Ship in 2 weeks)**:
- ✅ Cascade Gate v8 model: <0.7M params, AUC ≥ 0.865
- ✅ Bilinear fusion documented and ablated
- ✅ Inference time <5ms on CPU

**Tier 3 (Ship in 4 weeks)**:
- ✅ Gold laughter AUC: 0.590 → 0.630+
- ✅ Multi-task architecture deployed
- ✅ CORAL domain adaptation validated
- ✅ 5×3 repeated CV results reported

---

## 🔄 Decision Tree (for human review)

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

## 📚 References Cited by Critics

- Sun & Saenko 2016 — CORAL domain adaptation
- Ganin 2016 — DANN adversarial adaptation
- Han 2018 — Co-teaching for noisy labels
- Reed 2014 — Bootstrap loss for label noise
- Caruana 1997, Ruder 2017 — Multi-task learning
- Northcutt 2021 — Label noise estimation
- Bengio 2009 — Curriculum learning (original)
- Dietterich 1998 — Repeated CV for statistical power
- Purandare 2006, Bertero 2016 — Humor-laughter correlation

---

**Status**: Awaiting user decision on which tier to prioritize.
**Recommendation**: Tier 1 (immediate) + Tier 2 (parallel research) + Tier 3 (gold laughter stretch goal).
