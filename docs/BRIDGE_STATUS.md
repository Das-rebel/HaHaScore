# Bridge Status - 2026-09-14

## Bridge 1: Pseudo-label StandUp4AI (v4, per-segment prosody)

**Status**: RUNNING (started 1:49 AM)
**PID**: 43202
**Location**: `/Users/Subho/funny-strength-predictor/bridge1_pseudolabel_cpu.py`
**Output**: `/tmp/pseudo_labels_641_v4.json`
**Checkpoint**: `/tmp/pseudo_labels_641_v4_checkpoint.json`
**Log**: `/tmp/bridge1_v4_output.log`

**Speed**: ~13s per file (WavLM batched 4s + per-segment prosody ~9s)
**ETA**: ~2.3 hours (639 files at 13s/file)

**Verified working** (3 test files):
- `10U1uggokdg`: mean=0.940, range=[0.043, 0.999]
- `-1FrUOEswOk,fr`: mean=0.866, range=[0.062, 0.984]
- `-9Q122jAAOM`: mean=0.835, range=[0.618, 0.991]

## Key Technical Decisions

### Why per-segment prosody is REQUIRED
- Fusion model was trained on 3-second utterance prosody (per-segment)
- File-level prosody gives wrong pitch statistics → BatchNorm saturates → mean=0.005
- Per-segment prosody gives correct output (mean=0.940)
- PROSODY_MEAN/PROSODY_STD from training code are the CORRECT scaler (not fitted from NPZ)

### Per-segment prosody extraction
- 20 segments per file (3-second windows)
- librosa.yin for pitch (fmin=50, fmax=500)
- librosa.feature.mfcc for MFCCs (n_mfcc=13)
- librosa.feature.rms, zero_crossing_rate, spectral_centroid
- Stored scaler applied per segment

### WavLM 4x downsample
- 16kHz → 4000Hz (Pearson r=0.999 vs native)
- Batched: 20 segments in ONE forward pass
- ~0.8s per file (vs 19s without downsample)

## Bridges 2-5 Plan

### Bridge 2: Cascade Audio Gate
```
Input: text confidence (DeBERTa/RoBERTa) + audio incongruity
Gate: if text_confidence > 0.85 → use fusion; else use audio-only
```

### Bridge 3: Incongruity Modality
- Text incongruity: BERT/cloze-style
- Audio incongruity: F0 perturbation
- Combined score via learned weighting

### Bridge 4: Temporal Arc Tracker
- GRU over sequential segments (causal, forward-only)
- Captures buildup/punchline arc

### Bridge 5: Cross-Modal Attention
- Cross-attention between text and audio tokens
- Replaces bilinear fusion with attention fusion

## v6 Training Plan
- Dataset: ~2K labeled + ~629 pseudo-labeled StandUp4AI segments
- Architecture: CrossAttentionFusion or CascadeAudioFusion
- 5-fold CV with AUC ≥ 0.70 target
- Training: Modal T4 (if available) or local CPU (slow)
