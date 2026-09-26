# HaHaScore v8 — Deployment Package

## Contents
- `app.py` — Real ONNX inference (replaces SHA-1 placeholder)
- `models/v8_cascade_int8.onnx` — INT8 quantized model (2.9 MB)
- `models/v8_cascade_metadata.json` — Architecture metadata
- `space.yaml` — HF Space config

## Performance
| Model | Size | Inference (CPU) |
|-------|------|-----------------|
| PyTorch FP32 | 270 MB | 9.22 ms |
| ONNX FP32 | 0.3 MB | 2.29 ms |
| ONNX INT8 | 2.9 MB | **1.89 ms** |

## Architecture
- Cascade Gate Fusion v8
- 76 ONNX tensors (just the gate, no DistilBERT)
- Inputs: text (768-dim), audio (791-dim), cross-modal (16-dim)
- Output: scores (0-1) per 20 segments

## Training Pipeline (Drive-Based)
1. Downloaded 1M Reddit jokes → gdrive:/HaHaScore_Pretrain/reddit/
2. Trained DistilBERT on 2K Reddit (smoke) + 30K (full)
3. Initialized Cascade Gate v8 text tower with Reddit DistilBERT
4. Fine-tuned on 639 standup files → Val AUC **0.807**
5. Exported to ONNX + INT8 quantization
6. Deployed to HF Space (this)

## Local Disk Constraint
- Disk at 98% capacity throughout pipeline
- All bulk data on Google Drive
- Local files: ~4.3 GB free
