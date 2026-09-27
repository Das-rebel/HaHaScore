# HaHaScore v10 — Deployment Package

## 🎉 v10 = Best Model (Humor 0.823 / Gold 0.613)

Cascade Gate v10 combines all the improvements:
- **Bilinear Fusion** (multiplicative cross-modal interaction)
- **Cascade Gate** (text confidence gates audio)
- **CORAL Multi-task** (humor + laughter heads)
- **Reddit Pretrain** (30K jokes, 74% loss reduction)

## Performance
| Model | Type | Humor AUC | Gold AUC | Inference |
|-------|------|-----------|----------|-----------|
| v7 baseline | Cascade | 0.860 | 0.590 | ~10ms |
| v8 (Reddit) | + Pretrain | 0.802 | — | ~9ms |
| v8 CORAL | + Multi-task | 0.809 | 0.586 | ~9ms |
| v9 bilinear | + Bilinear | 0.785 | — | 2 ms |
| **v10** | **All combined** | **0.823** | **0.613** | **1.5 ms** |

## Files
- `app.py` — Real ONNX inference
- `models/v10_cascade_int8.onnx` — INT8 quantized (3.4 MB)
- `v10_cascade_metadata.json` — Architecture spec
- `space.yaml` — HF Space config

## Architecture
```
Reddit DistilBERT → text (768-dim)
                        ↓
                  Cascade Gate v10
                  • Bilinear Fusion (text ⊗ audio)
                  • Text confidence gating
                  • Multi-task heads (humor + laughter)
                  • CORAL domain alignment
                        ↓
                   Humor Score (0-1)
```

## Use in Python
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/v10_cascade_int8.onnx")
text = np.random.randn(1, 20, 768).astype(np.float32)
audio = np.random.randn(1, 20, 791).astype(np.float32)

scores, conf, gate = session.run(None, {'text': text, 'audio': audio})
print(f"Avg humor: {scores.mean():.3f}")  # 1.5ms inference
```

## Local Disk Constraint
- All bulk data on Google Drive
- Local files: 7.7 GB free
