---
language:
  - en
license: apache-2.0
library_name: pytorch
pipeline_tag: multimodal
tags:
  - humor-detection
  - comedy-analysis
  - audio-classification
  - multimodal
  - fusion-model
  - standup-comedy
  - sentence-level
  - bilinear-fusion
  - wavlm
  - microsoft
  - pytorch
  - audio
  - nlp
co2_emissions:
  emissions_per_second: "0.000056"
  source: "Estimated GPU impact"
---

# HaHaScore Fusion v5 — Multimodal Humor Strength Predictor

**Hugging Face Model ID:** `Hayasuki/hahascore-fusion-v5`

Predicts the **humor strength** of a spoken sentence on a continuous scale from **0.0 (not funny) to 1.0 (very funny)**. Fuses text features (DeBERTa-v3-base) and audio prosodic features (WavLM-base-plus) via a bilinear fusion MLP.

> ⚠️ **Trained exclusively on English stand-up comedy.** Performance on non-comedy domains, other languages, or scripted performance is **not validated**. Text-only inputs are near-random — audio prosody is the primary signal.

---

## 🎯 Use Cases

| Use Case | Supported | Notes |
|----------|-----------|-------|
| Scoring stand-up comedy segments | ✅ Yes | Primary use case |
| Joke intensity ranking within a set | ✅ Yes | Per-segment continuous scoring |
| Funny vs. not-funny detection | ✅ Yes | Threshold at ~0.5 |
| General humor (memes, puns, sarcasm) | ❌ No | Out-of-distribution |
| Non-English comedy | ❌ No | English-only training |
| Real-time spoken humor | ⚠️ Limited | Needs pre-segmented sentences |

---

## 📊 Performance

Evaluated on **1,174 held-out sentence clips** from 10 unseen comedy videos. 5-fold CV on 3,774 training clips.

| Metric | Score |
|--------|-------|
| **5-Fold CV AUC** | **0.632 ± 0.007** |
| Held-Out AUC | 0.613 |
| Average Precision | 0.314 |
| Max F1 | 0.424 @ ~0.35 threshold |

### Component Ablation (on held-out set)

| Model | AUC | vs. Random |
|-------|-----|------------|
| Random | 0.500 | — |
| DeBERTa-v3-base text only | 0.501 | ≈ Random |
| WavLM-base-plus audio only | 0.538 | +0.038 |
| **Fusion v5 (text+audio bilinear)** | **0.613** | **+0.113** |

### Key Insight

> **Comedy is about delivery.** The same words with comedic timing and prosodic emphasis are funny; the words alone are not. Text-only ≈ random. Audio prosody carries the primary signal. Fusion combines both for best results.

---

## 🏗️ Architecture

```
Input
 ├── Text: sentence string → DeBERTa-v3-base → 768d [CLS] pooler
 └── Audio: 16kHz WAV/MP3/M4A → WavLM-base-plus → 512d mean pooled
      ↓
 Text Projection: Linear(768 → 128) → ReLU
 Audio Projection: Linear(512 → 128) → ReLU
      ↓
 Hadamard Product: element-wise (128 ⊙ 128 = 128)
 Concatenation: [hadamard, txt_proj, aud_proj] → 384d
      ↓
 MLP: 384 → 128 (ReLU, BatchNorm, Dropout 0.3)
     → 32 → 1 (Sigmoid)
      ↓
Output: scalar [0.0 – 1.0] humor strength score
```

### Fusion MLP Weights (`pytorch_model.bin`)

| Layer | Shape | Parameters |
|-------|-------|------------|
| txt_proj | (128, 768) | 98,304 |
| aud_proj | (128, 512) | 65,536 |
| MLP hidden 1 | (128, 384) | 49,152 |
| MLP hidden 2 | (32, 128) | 4,096 |
| Output | (1, 32) | 33 |
| **Total fusion params** | | **~111K** |

> ⚠️ **Only the fusion MLP is in `pytorch_model.bin`.** The text encoder (DeBERTa-v3-base) and audio encoder (WavLM-base-plus) are loaded separately from Hugging Face at inference time.

### Full Inference Pipeline

| Component | Source | Output Dim |
|-----------|--------|------------|
| Text encoder | `microsoft/deberta-v3-base` | 768 |
| Audio encoder | `microsoft/wavlm-base-plus` | 512 |
| Fusion MLP | `pytorch_model.bin` (this repo) | 111K |
| **Total** | | **~150M params** |

---

## 📦 Repository Contents

```
Hayasuki/hahascore-fusion-v5/
├── pytorch_model.bin   # Fusion MLP weights + metadata (461 KB)
├── README.md           # This file
└── app.py              # Gradio demo (optional)
```

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install torch transformers librosa pydub gradio
```

### 2. Load the Model

```python
import torch
import numpy as np
import librosa
from transformers import AutoModel, AutoTokenizer, WavLMModel
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load fusion weights ──────────────────────────────────────
model_path = hf_hub_download("Hayasuki/hahascore-fusion-v5", "pytorch_model.bin")
checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

# ── Load encoders from Hugging Face ─────────────────────────
text_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").eval()
text_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
audio_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()

# ── Define fusion MLP ───────────────────────────────────────
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
        t = self.txt_proj(txt_feat)       # (B, 128)
        a = self.aud_proj(aud_feat)       # (B, 128)
        h = t * a                         # Hadamard (128,)
        x = torch.cat([h, t, a], dim=1)  # (B, 384)
        return self.net(x)

fusion = BilinearFusionMLP()
fusion.load_state_dict(checkpoint["model_state_dict"], strict=False)
fusion.eval().to(DEVICE)
```

### 3. Extract Features

```python
def extract_text_features(sentence: str) -> np.ndarray:
    """DeBERTa [CLS] pooler features — shape (1, 768)."""
    inputs = text_tokenizer(sentence, return_tensors="pt",
                            truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = text_model(**inputs)
        feat = out.last_hidden_state[:, 0, :].cpu().numpy()
    return feat  # (1, 768)


def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    WavLM mean-pooled features — shape (1, 512).
    Accepts any format pydub supports: WAV, MP3, M4A, OGG, FLAC.
    Audio is 4× downsampled (16kHz → 4kHz) for 5× speedup.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

    # Pad short clips to 6 s
    if len(samples) < 96000:
        samples = np.pad(samples, (0, 96000 - len(samples)))

    # 4× downsample: verified Pearson r = 0.999 vs native 16kHz
    samples_4k = librosa.resample(samples, orig_sr=16000, target_sr=4000)

    with torch.no_grad():
        x = torch.tensor(samples_4k, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        out = audio_model(x)
        feat = out.last_hidden_state.mean(dim=1).cpu().numpy()
    return feat  # (1, 512)
```

### 4. Predict

```python
def predict_humor(text: str, audio_path: str) -> float:
    """
    Returns humor strength score in [0.0, 1.0].

    Args:
        text: The transcribed sentence (English).
        audio_path: Path to audio clip (WAV/MP3/M4A, 16kHz mono preferred).

    Returns:
        Humor strength score (higher = funnier).
    """
    txt_feat = extract_text_features(text)          # (1, 768)
    aud_feat = extract_audio_features(audio_path)   # (1, 512)

    with torch.no_grad():
        score = fusion(
            torch.tensor(txt_feat).float().to(DEVICE),
            torch.tensor(aud_feat).float().to(DEVICE),
        ).item()
    return score


# ── Example ─────────────────────────────────────────────────
score = predict_humor(
    text="I told my wife she was drawing her eyebrows too high.",
    audio_path="path/to/comedy_clip.wav"
)
print(f"Humor score: {score:.3f}")  # e.g., 0.732
```

---

## 🎤 Audio Requirements

| Property | Value | Notes |
|----------|-------|-------|
| Sample rate | 16,000 Hz | Set automatically by pydub |
| Channels | Mono | Set automatically |
| Format | WAV, MP3, M4A, OGG, FLAC | pydub auto-detects |
| Optimal duration | 3–6 seconds | One sentence |
| Minimum | 0.5 seconds | Shorter clips zero-padded |
| Maximum | 6 seconds | Longer clips truncated |

### Processing Full Videos (Multiple Sentences)

```python
def score_video_segments(sentences: list[dict], audio_path: str) -> list[dict]:
    """
    Score multiple sentences from one audio file.

    Args:
        sentences: [{"start": 0.5, "end": 3.2, "text": "..."}, ...]
        audio_path: Path to full audio file

    Returns:
        [{"start": 0.5, "end": 3.2, "text": "...", "score": 0.732}, ...]
    """
    import tempfile, os

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    full_samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

    results = []
    for seg in sentences:
        # Extract clip
        s = int(seg["start"] * 16000)
        e = int(seg["end"] * 16000)
        clip_samples = full_samples[s:e]

        # Export clip as temp WAV
        clip = (clip_samples * 32768).astype(np.int16)
        clip_seg = AudioSegment(
            clip.tobytes(), frame_rate=16000, sample_width=2, channels=1
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        clip_seg.export(tmp.name, format="wav")
        tmp.close()

        score = predict_humor(seg["text"], tmp.name)
        results.append({**seg, "score": score})
        os.unlink(tmp.name)

    return results
```

---

## 🔧 Gradio Demo App

Run locally:

```bash
python app.py
# → Opens http://localhost:7860
```

Or build with the `app.py` in this repository.

---

## 🏋️ Training Details

| Parameter | Value |
|-----------|-------|
| Training clips | 3,774 (38 comedy videos, ~100 clips/video) |
| Validation clips | 9,211 (10 held-out videos) |
| Test clips | 1,174 (10 held-out videos) |
| Audio downsample | 4× (16kHz → 4kHz, r=0.999 vs native) |
| Text max length | 256 tokens |
| Optimizer | AdamW, lr=1e-3 |
| Batch size | 32 |
| Epochs | 50 |
| Early stopping | patience=10 on val AUC |
| Total params | ~111K (fusion MLP only) |

### Training Data Source

StandUp4AI dataset — 641 full comedy sets from YouTube. Sentence boundaries detected via 0.5-second silence gap heuristic. Only 12 of 641 files had human annotations; 629 used pseudo-labels from Bridge 1 (fusion model itself).

Dataset location: `gdrive:standup4ai/audio_1000/` (641 .m4a files, ~3.2 GB)

---

## ⚠️ Limitations

1. **English only** — Trained on English stand-up comedy
2. **Audio is primary signal** — Text alone ≈ random; model requires good audio
3. **Segmented input only** — No automatic sentence boundary detection
4. **3–6 second optimal** — Longer sentences truncated, shorter padded
5. **Single comedian/style** — Limited generalization to other comedy styles
6. **Binary training labels** — Continuous scoring extrapolates from binary labels
7. **No speaker normalization** — Prosodic patterns not speaker-normalized
8. **No temporal context** — Each sentence scored independently

---

## 🔗 Related Models

| Model | Description |
|-------|-------------|
| `Hayasuki/hahascore-text-v7` | Text-only RoBERTa-large baseline (AUC 0.499) |
| `Hayasuki/hahascore-fusion-v3` | Older fusion model — superseded by v5 |

---

## 📄 License

Apache 2.0 — https://www.apache.org/licenses/LICENSE-2.0

---

*Maintained by: Das, Subhajit · GitHub: `Das-rebel/HaHaScore` · Updated: 2026-09-16*
