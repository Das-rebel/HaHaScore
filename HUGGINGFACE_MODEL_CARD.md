---
language:
  - en
license: apache-2.0
library_name: pytorch
pipeline_tag: other
tags:
  - humor-detection
  - comedy
  - audio
  - sequential-model
  - gru
  - multimodal
  - wavlm
  - prosody
  - speech
---

# HaHaScore — Sentence-Level Humor Strength Prediction

**Predict how funny a spoken sentence is on a continuous 0–1 scale.**

HaHaScore is a multimodal humor strength predictor that scores the comedic intensity of spoken sentences using audio prosody and temporal arc modeling. It was developed for stand-up comedy video analysis and is uniquely suited for tasks where delivery matters more than content.

> ⚠️ **Single canonical repo**. All other HaHaScore repos have been merged here. Old repos are deprecated.

## Performance

| Model | AUC | Description |
|-------|-----|-------------|
| Text-only (RoBERTa-large) | 0.499 | Indistinguishable from random |
| Text-only (DeBERTa-v3-base) | 0.501 | Indistinguishable from random |
| Audio-only (WavLM, linear probe) | 0.538 | Moderate discrimination |
| **Fusion v5** (per-segment bilinear) | **0.632** | RoBERTa + WavLM bilinear fusion, 5-fold CV |
| **Bridge 4** (humor arc tracker) | **0.842** | BiGRU over WavLM+prosody segments, 5-fold CV |

**Bridge 4** is the primary model and the recommended choice. **Fusion v5** is also available as a lightweight per-segment alternative.

## Why Audio Dominates Comedy Detection

Comedy is fundamentally about **delivery**. The same words delivered with comedic timing,
prosodic emphasis, and vocal inflection are humorous — but the words alone are not.

In a controlled experiment with 2,600 labeled sentence clips:
- Text alone: AUC 0.499 (random)
- Audio alone: AUC 0.538 (moderate)
- Text + Audio fusion: AUC 0.632 (good)

This confirms what comedians know intuitively: timing and vocal expression carry the humor signal.

## Architecture

### Bridge 4 — Humor Arc Tracker (Recommended)

```
Input: 20 sequential segments per file
  └── Per segment: 768d (WavLM-base-plus) + 23d (prosody) + 4d (position) = 795d

Bidirectional GRU(795d → 128d) × 2 layers, bidirectional
MLP(256 → 128 → 1) + Sigmoid
Output: per-segment humor score (0–1)

Trainable parameters: ~790K
```

**Key design**: Bidirectional processing captures both the buildup before a punchline and the payoff after it.

### Fusion v5 — Per-Segment Bilinear Fusion (Lightweight)

```
Text: microsoft/deberta-v3-base → 768d [CLS] pooler
Audio: microsoft/wavlm-base-plus → 512d mean-pooled
       ↓
txt_proj(768→128) + aud_proj(512→128)
hadamard product → concat(384) → MLP(384→128→32→1) + Sigmoid

Trainable parameters: ~111K
```

## Quick Start

### Bridge 4 Inference

```python
import torch, numpy as np
from bridge4_inference import HumorArcTracker

model = HumorArcTracker(input_dim=791, hidden=128, num_layers=2)
state = torch.load("pytorch_model.bin", map_location="cpu", weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()

# features: (batch=1, seq_len=20, 791d) per-segment WavLM+prosody
features = torch.randn(1, 20, 791)
with torch.no_grad():
    scores = model(features)  # (1, 20) per-segment scores
print(f"Humor arc: {scores.squeeze().numpy()}")
```

### Fusion v5 Inference

```python
import torch, numpy as np
from fusion_v5_inference import BilinearFusionMLP
from transformers import AutoModel, AutoTokenizer, WavLMModel
import librosa
from pydub import AudioSegment

# Load models
fusion = BilinearFusionMLP()
fusion.load_state_dict(torch.load("fusion_v5_model.bin", map_location="cpu", weights_only=False), strict=False)
fusion.eval()
text_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
text_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
audio_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()

def score_sentence(text, audio_samples, sr=16000):
    # Text features
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        txt_feat = text_model(**inputs).last_hidden_state[:, 0, :]
    # Audio features (4× downsample)
    audio_4k = librosa.resample(audio_samples, orig_sr=sr, target_sr=4000)
    with torch.no_grad():
        aud_feat = audio_model(audio_4k.unsqueeze(0)).last_hidden_state.mean(1)
    # Score
    with torch.no_grad():
        return fusion(txt_feat, aud_feat).item()
```

### Gradio Demo

```bash
# Install dependencies
pip install gradio torch transformers librosa pydub

# Launch fusion v5 demo (simple per-segment scoring)
python fusion_v5_app.py

# Bridge 4 requires feature extraction — see bridge4_inference.py
```

## How to Generate Input Features

Bridge 4 requires per-segment WavLM embeddings + prosody features:

```python
import librosa
from pydub import AudioSegment
import torch
import numpy as np
from transformers import WavLMModel

PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0, 0.0,6.0,0.0,5.0])
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0, 6.0,3.0,5.0,2.5])
N_SEGMENTS = 20

def extract_prosody(waveform, sr):
    """Extract 23-dim prosody from audio segment."""
    hop = 160
    if len(waveform) < sr: return np.zeros(23, dtype=np.float32)
    rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=hop)[0]
    try:
        pitch = librosa.yin(waveform, fmin=50, fmax=500, sr=sr)
        pv = pitch[~np.isnan(pitch)]
        pitch_mean = float(np.mean(pv)) if len(pv) > 0 else 0.0
        pitch_std = float(np.std(pv)) if len(pv) > 0 else 0.0
        pitch_max = float(np.max(pv)) if len(pv) > 0 else 0.0
        pitch_min = float(np.min(pv)) if len(pv) > 0 else 0.0
    except: pitch_mean = pitch_std = pitch_max = pitch_min = 0.0
    dur = len(waveform) / sr
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    sc = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
    f = []
    f.extend([float(np.mean(rms)), float(np.std(rms))])
    f.extend([float(np.mean(zcr)), float(np.std(zcr))])
    f.extend([pitch_mean, pitch_std, pitch_max, pitch_min])
    f.extend([dur, dur**0.5, np.log1p(dur)])
    for i in range(13): f.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
    f.append(float(np.mean(sc)))
    while len(f) < 23: f.append(0.0)
    return np.array(f[:23], dtype=np.float32)

def extract_file_features(audio_path, wavlm, device="cpu"):
    """Extract (20, 791) features for a file."""
    audio = AudioSegment.from_file(str(audio_path), format='m4a')
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    duration = len(samples) / 16000.0
    if len(samples) < 1600: return None
    samples_4k = librosa.resample(samples, orig_sr=16000, target_sr=4000)
    seg_dur = duration / N_SEGMENTS
    wavlm_feats, prosody_feats = [], []
    for i in range(N_SEGMENTS):
        s, e = int(i*seg_dur*4000), int((i+1)*seg_dur*4000)
        e = min(e, len(samples_4k)); seg_wav = samples_4k[s:e]
        if len(seg_wav) < int(0.5*4000): seg_wav = np.zeros(int(0.5*4000), dtype=np.float32)
        with torch.no_grad():
            emb = wavlm(torch.tensor(seg_wav).unsqueeze(0).to(device)).last_hidden_state.mean(1).cpu().numpy().squeeze()
        s16, e16 = int(i*seg_dur*16000), int((i+1)*seg_dur*16000)
        e16 = min(e16, len(samples))
        prosody = extract_prosody(samples[s16:e16], 16000)
        wavlm_feats.append(emb); prosody_feats.append(prosody)
    wavlm_arr = np.stack(wavlm_feats)
    prosody_scaled = (np.stack(prosody_feats) - PROSODY_MEAN) / (PROSODY_STD + 1e-8)
    return np.concatenate([wavlm_arr, prosody_scaled], axis=1)  # (20, 791)
```

## Training Data

- **Primary**: 3,774 sentence clips from 38 Indian comedy videos (YouTube)
- **Extended**: 639 StandUp4AI comedy videos (EMNLP 2025) with per-segment pseudo-labels
- **Total**: 12,780 labeled segments across 48+ hours of comedy

## What the Models Learned

### Segment Position Analysis

The Bridge 4 model discovered a consistent temporal pattern:

| Segment | Mean Score | Interpretation |
|---------|-----------|----------------|
| 1–5 | 0.70–0.77 | Comedy sets up quickly |
| 6–18 | 0.74–0.77 | Sustained engagement |
| 19 | 0.69 | Building to closer |
| **20** | **0.29** | End credits/outro — not comedic |

This is not hardcoded — the model learned it from the data.

### Text is Random for Comedy Detection

Extensive experiments with RoBERTa-large, DeBERTa-v3-base, and GPT variants all produced AUC ≈ 0.50 on held-out comedy sentences. The words "I told my wife she was drawing her eyebrows too high" are not inherently funnier than "I told my wife she was making breakfast" — the vocal delivery is what makes comedy comedic.

## Limitations

1. **Pseudo-label training**: Bridge 4 was trained on v5 fusion pseudo-labels, not human ratings. True generalization is TBD.
2. **English-only training**: Trained on English and translated comedy. Non-English comedy may perform differently.
3. **Short segments**: 3-second windows may miss cross-segment humor dynamics.
4. **No speaker normalization**: Individual comedian styles are not explicitly modeled.
5. **Single-modality audio**: No speaker embedding or voice-print features.

## Files in This Repo

| File | Description |
|------|-------------|
| `pytorch_model.bin` | **Bridge 4** HumorArcTracker weights (4.1 MB) |
| `bridge4_inference.py` | Bridge 4 inference code |
| `fusion_v5_model.bin` | **Fusion v5** BilinearFusionMLP weights (461 KB) |
| `fusion_v5_inference.py` | Fusion v5 inference code |
| `fusion_v5_app.py` | Gradio demo for Fusion v5 |
| `hahascore_bridge4_arch.png` | Architecture diagram |

## Relationship Between Models

```
Fusion v5 (per-segment bilinear, AUC 0.632)
    │
    └──→ Bridge 4 (BiGRU over v5 pseudo-labels, AUC 0.842)
              ↑
              └── Bridge 1 (per-segment prosody pseudo-labels, 639 files)
                        ↑
                        └── Bridge 0 (WavLM audio encoder, frozen)
```

Bridge 4 is NOT a replacement for Fusion v5 — it builds on top of it. The improvement from 0.632 → 0.842 AUC demonstrates the value of **temporal arc modeling** over per-segment inference.

## Citation

If you use HaHaScore in your research:

```bibtex
@misc{hahascore2026,
  author = {Subhojit Das},
  title = {HaHaScore: Sentence-Level Humor Strength Prediction via Audio-Text Fusion},
  year = {2026},
  url = {https://huggingface.co/Hayasuki/hahascore-bridge4-arc-tracker}
}
```

## Deprecated Repos

The following repos have been merged into this one:

| Deprecated Repo | Reason |
|---------------|--------|
| `Hayasuki/hahascore-fusion-v5` | Superseded by Bridge 4 |
| `Hayasuki/hahascore-fusion-v3` | Old architecture |
| `Hayasuki/hahascore-text-v7` | Text-only baseline (AUC 0.50) |
| `Hayasuki/chuckle-net` | Internal weights file |

## License

Apache 2.0

*Maintained by Subhojit Das. GitHub: github.com/Das-rebel/HaHaScore*
