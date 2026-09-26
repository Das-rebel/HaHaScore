# HaHaScore v8 — Deployment Package (v1)

## Contents
- `app.py` — Real ONNX inference (replaces SHA-1 placeholder)
- `models/v8_cascade_int8.onnx` — INT8 quantized model (2.9 MB)
- `v8_cascade_metadata.json` — Architecture metadata
- `space.yaml` — HF Space config

## Performance
| Model | Size | CPU Inference |
|-------|------|---------------|
| PyTorch FP32 (v1) | 270 MB | ~9 ms |
| ONNX FP32 | 0.3 MB | ~3.3 ms |
| **ONNX INT8** | **2.9 MB** | **~3.3 ms** |

## v1 Architecture & Training
- **Pretrain**: DistilBERT on 30K Reddit jokes (final loss 0.0177)
- **Fine-tune**: Cascade Gate v8 on 639 standup files (Val AUC **0.8021**)
- **ONNX export**: FP32 (0.3MB) + INT8 (2.9MB) quantized
- **Inputs**: text (768-dim), audio (791-dim), cross-modal (16-dim)
- **Output**: scores (0-1) per 20 segments

## Comparison
| Version | Reddit Pretrain | Val AUC | Notes |
|---------|----------------|---------|-------|
| v7 baseline | None | 0.860 (5-fold CV) | pseudo-labels |
| v8 smoke (2K Reddit) | smoke | 0.807 | first end-to-end test |
| **v8 v1 (30K Reddit)** | **full** | **0.802** | **production model** |

## Local Disk Constraint
- Disk at 98% capacity throughout pipeline
- All bulk data on Google Drive
- Local files: 5.3 GB free at end

## CORAL Domain Adaptation (v2)
- **Multi-task**: humor + laughter heads sharing v8 backbone
- **CORAL**: covariance alignment between pseudo (source) and gold laughter (target)
- **Result**: Humor AUC 0.809 (vs 0.802 baseline), Gold AUC 0.586 (vs 0.590 baseline)
- Multi-task learning improves humor slightly while maintaining gold performance
- See `drive_pipeline/train_coral_v2.py` for implementation
