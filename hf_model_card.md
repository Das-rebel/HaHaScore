---
language:
  - en
license: apache-2.0
library_name: pytorch
pipeline_tag: multimodal
tags:
  - humor detection
  - comedy analysis
  - audio classification
  - multimodal
  - fusion
  - standup comedy
  - sentence-level
  - bilin_fusion
  - wavlm
  - wavlm-base-plus
  - deberta-v3-base
  - microsoft
  - pytorch
  - audio
  - nlp
co2_emissions:
  emissions_per_second: "0.000056"
  source: "https:// Impact per second (GPU-h: 0.0004 kgCO2eq)"
paperswithcode_url: null
---

# HaHaScore Fusion v5 — Multimodal Humor Strength Predictor

**Hugging Face Model ID:** `Hayasuki/hahascore-fusion-v5`

A multimodal neural network that predicts the **humor strength** of a spoken sentence on a continuous scale from 0.0 to 1.0. It fuses **text features** (DeBERTa-v3-base) and **audio prosodic features** (WavLM-base-plus) through a bilinear fusion architecture.

Built for research on sentence-level comedy analysis, joke scoring, and humor detection in transcribed stand-up comedy.

---

## 🎯 Intended Use Cases

> **⚠️ Important:** This model was trained exclusively on **English stand-up comedy** (stand-up segments with transcribed sentences). Performance on non-comedy audio, other languages, or scripted theatrical performance is **not validated**.

| Use Case | Supported? | Notes |
|----------|------------|-------|
| Scoring stand-up comedy segments | ✅ Yes | Primary use case |
| Joke强度ranking within a comedy set | ✅ Yes | Per-segment scoring |
| Detecting funny vs. not-funny sentences | ✅ Yes | Binary threshold ~0.5 |
| General humor detection (memes, puns, sarcasm) | ❌ No | Out-of-distribution |
| Non-English comedy | ❌ No | English-only training data |
| Real-time speech humor scoring | ⚠️ Limited | Designed for pre-segmented sentences |

---

## 📊 Performance

Evaluated on a held-out test set of **1,174 sentence clips** from 10 unseen comedy videos (5-fold cross-validation on training set of 3,774 clips).

| Metric | Score | Notes |
|--------|-------|-------|
| **5-Fold CV AUC** | **0.632 ± 0.007** | Primary metric |
| Held-Out AUC | 0.613 | Independent test set |
| Average Precision | 0.314 | Imbalanced dataset |
| Max F1 | 0.424 | @ threshold ~0.35 |

### Component Ablation

| Model | AUC | vs. Random |
|-------|-----|------------|
| Random baseline | 0.500 | — |
| DeBERTa-v3-base text only | 0.501 | ≈ Random |
| WavLM-base-plus audio only | 0.538 | +0.038 |
| **Fusion v5 (text + audio bilinear)** | **0.613** | **+0.113** |

**Key insight:** Text alone is indistinguishable from random. Audio prosodic features (pitch, energy, timing) carry the primary humor signal. The fusion model combines both for the best performance.

---

## 🧠 Key Research Finding

> **Comedy is fundamentally about delivery.** The same words delivered with comedic timing, prosodic emphasis, and vocal inflection are humorous — but the words alone carry almost no signal.

This explains why:
- Text-only models score ~0.50 AUC (random)
- Audio prosody alone achieves 0.54 AUC
- Fusion achieves 0.63 AUC (+0.13 over text-only)

---

## 🏗️ Architecture

```
Input
 ├── Text: Sentence string (DeBERTa-v3-base → 768d pooler output)
 └── Audio: WAV/16kHz audio array (WavLM-base-plus → 512d mean pooled)
     ↓
 Text Projection: Linear(768 → 128)
 Audio Projection: Linear(512 → 128)
     ↓
 Hadamard Product: element-wise (128 × 128 = 128)
 Concatenation: [hadamard, txt_proj, aud_proj] → 384d
     ↓
 MLP: 384 → 128 (ReLU, BatchNorm, Dropout 0.3)
     → 32 → 1 (Sigmoid)
     ↓
Output: scalar [0.0, 1.0] — humor strength score
```

### Model Weights

| Layer | Shape | Params |
|-------|-------|--------|
| txt_proj | (128, 768) | 98,304 |
| aud_proj | (128, 512) | 65,536 |
| MLP hidden 1 | (32, 384) | 12,288 |
| MLP hidden 2 | (1, 32) | 33 |
| **Total** | | **~111K trainable params** |

> The encoders (DeBERTa, WavLM) are **NOT** included in `pytorch_model.bin`. They are loaded separately from Hugging Face. This file only contains the fusion MLP weights.

### Full Inference Pipeline (all components)

| Component | Source | Dim |
|-----------|--------|-----|
| Text encoder | `microsoft/deberta-v3-base` | 768 |
| Audio encoder | `microsoft/wavlm-base-plus` | 512 |
| Fusion MLP | `pytorch_model.bin` (this repo) | 111K |
| **Total** | | **~150M params** |

---

## 📦 Model Files

```
Hayasuki/hahascore-fusion-v5/
├── pytorch_model.bin   # Fusion MLP weights + metadata
├── README.md           # This file
└── app.py              # Gradio demo app
```

---

## 🚀 Quickstart

### Installation

```bash
pip install torch transformers librosa pydub gradio
```

### Basic Usage

```python
import torch
import librosa
import numpy as np
from transformers import AutoModel, AutoTokenizer, WavLMModel

#############################################
# 1. SETUP — Load all components
#############################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download model from Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download("Hayasuki/hahascore-fusion-v5", "pytorch_model.bin")
checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

# Text encoder (DeBERTa-v3-base)
text_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
text_model.eval()
text_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Audio encoder (WavLM-base-plus)
audio_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
audio_model.eval()

# Fusion MLP (this model's weights)
class BilinearFusionMLP(torch.nn.Module):
    def __init__(self, txt_dim=768, aud_dim=512, hidden=128):
        super().__init__()
        self.txt_proj = torch.nn.Linear(txt_dim, hidden)
        self.aud_proj = torch.nn.Linear(aud_dim, hidden)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden * 3, hidden), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden), torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, 32), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32), torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1), torch.nn.Sigmoid()
        )
    def forward(self, txt_feat, aud_feat):
        t = self.txt_proj(txt_feat)      # (B, 128)
        a = self.aud_proj(aud_feat)      # (B, 128)
        h = t * a                        # Hadamard product
        x = torch.cat([h, t, a], dim=1) # (B, 384)
        return self.net(x)

fusion = BilinearFusionMLP()
fusion.load_state_dict(checkpoint["model_state_dict"], strict=False)
fusion.eval()
fusion.to(DEVICE)

#############################################
# 2. EXTRACT TEXT FEATURES
#############################################

def extract_text_features(sentence: str) -> np.ndarray:
    """Extract DeBERTa pooler features from a sentence."""
    inputs = text_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
        # Use [CLS] token embedding
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return features  # (1, 768)

#############################################
# 3. EXTRACT AUDIO FEATURES
#############################################

def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extract WavLM features from a 16kHz WAV/MP3/M4A file.
    For short segments (< 6 seconds), pads to 6 seconds.
    """
    # Load audio with pydub (handles MP3/M4A)
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    
    # Pad short audios to 6 seconds (16000 * 6 = 96000 samples)
    target_len = 16000 * 6
    if len(samples) < target_len:
        samples = np.pad(samples, (0, target_len - len(samples)))
    
    # 4× downsample for speed (verified: Pearson r=0.999 vs native 16kHz)
    samples_4k = librosa.resample(samples, orig_sr=16000, target_sr=4000)
    
    # WavLM forward
    with torch.no_grad():
        x = torch.tensor(samples_4k, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        out = audio_model(x)
        features = out.last_hidden_state.mean(dim=1).cpu().numpy()  # (1, 512)
    return features  # (1, 512)

#############################################
# 4. COMBINED INFERENCE
#############################################

def predict_humor(text: str, audio_path: str) -> float:
    """Predict humor strength of a spoken sentence [0.0 - 1.0]."""
    txt_feat = extract_text_features(text)      # (1, 768)
    aud_feat = extract_audio_features(audio_path) # (1, 512)
    
    txt_tensor = torch.tensor(txt_feat, dtype=torch.float32).to(DEVICE)
    aud_tensor = torch.tensor(aud_feat, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        score = fusion(txt_tensor, aud_tensor).item()
    
    return score

#############################################
# EXAMPLE USAGE
#############################################

# Score a funny sentence from a comedy clip
score = predict_humor(
    text="I told my wife she was drawing her eyebrows too high.",
    audio_path="/path/to/comedy_clip.wav"
)
print(f"Humor score: {score:.3f}")  # e.g., 0.732
```

---

## 🎤 Audio Formatting Requirements

| Property | Value |
|----------|-------|
| Sample rate | 16,000 Hz |
| Format | WAV, MP3, M4A (pydub auto-detects) |
| Channels | Mono |
| Duration | Optimal: 3–6 seconds per sentence clip |
| Min duration | 0.5 seconds (shorter clips are zero-padded) |
| Max duration | Truncated at 6 seconds |

### Segment-Level Processing

The model was trained on **sentence-level clips** (averaging 3 seconds each). For full video processing:

```python
def score_full_video(sentence_timestamps: list[dict], audio_path: str, video_id: str):
    """
    Args:
        sentence_timestamps: [{"start": 0.5, "end": 3.2, "text": "..."}, ...]
        audio_path: Path to full audio file
        video_id: YouTube video ID
    Returns: [{"start": 0.5, "end": 3.2, "score": 0.732}, ...]
    """
    from pydub import AudioSegment
    import librosa
    
    # Load full audio
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    total_duration = len(samples) / 16000
    
    results = []
    for seg in sentence_timestamps:
        s_samples = int(seg["start"] * 16000)
        e_samples = int(seg["end"] * 16000)
        clip_samples = samples[s_samples:e_samples]
        
        # Save clip temporarily
        clip = AudioSegment(
            (clip_samples * 2**15).astype(np.int16).tobytes(),
            frame_rate=16000, sample_width=2, channels=1
        )
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        clip.export(tmp.name, format="wav")
        
        score = predict_humor(seg["text"], tmp.name)
        results.append({"start": seg["start"], "end": seg["end"],
                       "text": seg["text"], "score": score})
        os.unlink(tmp.name)
    
    return results
```

---

## 🏋️ Training Details

| Parameter | Value |
|-----------|-------|
| Training samples | 3,774 sentence clips |
| Validation samples | 9,211 sentence clips |
| Test samples | 1,174 clips (held-out 10 videos) |
| Audio preprocessing | 4× downsampling (16kHz → 4kHz), Pearson r=0.999 vs native |
| Text preprocessing | DeBERTa tokenizer, max 256 tokens, truncation |
| Optimizer | AdamW (lr=1e-3) |
| Batch size | 32 |
| Epochs | 50 |
| Early stopping | patience=10 (validation AUC) |
| Augmentation | None (no data augmentation) |
| Training time | ~30 min on CPU |

### Training Data Source

StandUp4AI dataset: **641 full comedy sets** (YouTube IDs: `standup4ai_full/` on GDrive `gdrive:standup4ai/audio_1000/`). Sentence boundaries detected via 0.5-second silence gap heuristic. Only 12 of 641 files had human annotations; 629 used pseudo-labels from Bridge 1.

---

## ⚠️ Limitations

1. **English only** — Trained on English stand-up comedy
2. **Delivery-focused** — Text alone ≈ random; model heavily relies on prosodic audio features
3. **Fixed segment length** — Designed for 3–6 second utterances; longer sentences truncated
4. **Limited style diversity** — Training on 1 comedian/show may not generalize
5. **Binary labels** — Training used binary (funny/not-funny) labels; continuous scoring is extrapolated
6. **No speaker normalization** — Speaker-specific prosodic patterns not modeled
7. **No context** — Each sentence scored independently; no sequential/humor arc modeling

---

## 🔬 Bridge Framework (Advanced)

HaHaScore uses a **5-bridge cascade** for label quality control:

| Bridge | Purpose | Status |
|--------|---------|--------|
| Bridge 0 | Clean labeled data pipeline | ✅ Done |
| Bridge 1 | Pseudo-label unlabeled audio (fusion model) | ✅ Done |
| Bridge 2 | Cascade audio gate (text confidence gating) | 🔜 Planned |
| Bridge 3 | Incongruity modality (text + audio mismatch) | 🔜 Planned |
| Bridge 4 | Temporal arc tracker (GRU over segments) | 🔜 Planned |
| Bridge 5 | Cross-modal attention (transformer fusion) | 🔜 Planned |

---

## 📚 Citation

```bibtex
@article{hahascore2026,
  title={HaHaScore: Sentence-Level Humor Strength Prediction in Stand-Up Comedy},
  author={Das, Subhajit},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/Hayasuki/hahascore-fusion-v5}
}
```

---

## 🤝 Contributing & Feedback

- **GitHub:** `Das-rebel/HaHaScore`
- **Model Hub:** `Hayasuki/hahascore-fusion-v5`
- **Issues:** Open at https://github.com/Das-rebel/HaHaScore/issues

For dataset access (StandUp4AI 641 videos), contact the authors or use the GDrive link: `gdrive:standup4ai/audio_1000/`

---

## 📄 License

Apache 2.0 — See [LICENSE](https://www.apache.org/licenses/LICENSE-2.0)

---

*Model card v5 — Updated 2026-09-16*
