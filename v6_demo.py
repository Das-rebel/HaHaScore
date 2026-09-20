#!/usr/bin/env python3
"""
v6 TriModal Cross-Attention Gradio Demo
========================================
Interactive humor strength scoring using text + audio.

Usage:
    python v6_demo.py
    Then open http://127.0.0.1:7860
"""
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import whisper
import librosa
from pydub import AudioSegment
import json, os
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
AUDIO_DIR = Path("/Users/Subho/data/standup4ai_full")
WAVLM_MODEL = "microsoft/wavlm-base-plus"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEGMENTS = 20

# Prosody scaler
PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0,
    0.0,6.0,0.0,5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0,
    6.0,3.0,5.0,2.5], dtype=np.float32)

# ── v6 Model ────────────────────────────────────────────────────────────────

class TriModalFusion(nn.Module):
    def __init__(self, text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout))
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)
        self.gru = nn.GRU(
            hidden * 4 + 4, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, text, audio, lengths=None):
        batch_size, seq_len, _ = text.shape
        text_h = self.text_proj(text)
        audio_h = self.audio_proj(audio)
        text_attn_out, _ = self.text_to_audio_attn(text_h, audio_h, audio_h)
        audio_attn_out, _ = self.audio_to_text_attn(audio_h, text_h, text_h)
        fused = torch.cat([text_h, audio_h, text_attn_out, audio_attn_out], dim=-1)
        positions = torch.arange(seq_len, device=text.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        fused = torch.cat([fused, pos_emb], dim=-1)
        out, _ = self.gru(fused)
        return self.head(out).squeeze(-1)


# ── Load Models ─────────────────────────────────────────────────────────────
print("Loading models...")

# v6 model
v6_model = TriModalFusion().to(DEVICE)
v6_model.load_state_dict(torch.load(
    "/Users/Subho/funny-strength-predictor/models/v6_trimodal.pt",
    map_location=DEVICE
))
v6_model.eval()

# Whisper
whisper_model = whisper.load_model("tiny", device=DEVICE)

# RoBERTa
from transformers import AutoModel, AutoTokenizer
roberta = AutoModel.from_pretrained("roberta-base").eval().to(DEVICE)
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# WavLM
from transformers import WavLMModel
wavlm = WavLMModel.from_pretrained(WAVLM_MODEL).eval().to(DEVICE)

print(f"Models loaded. Device: {DEVICE}")


# ── Feature Extraction ─────────────────────────────────────────────────────

def extract_audio_features(audio_path, n_segments=N_SEGMENTS):
    """Extract WavLM + prosody features for all segments."""
    audio = AudioSegment.from_file(audio_path, format='m4a')
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    duration = len(samples) / 16000.0

    if len(samples) < 1600:
        return None

    # 4× downsample
    samples_4k = librosa.resample(samples, orig_sr=16000, target_sr=4000)
    seg_dur = duration / n_segments
    min_seg_len = int(0.5 * 4000)

    wavlm_feats, prosody_feats = [], []

    for i in range(n_segments):
        # WavLM segment
        s = int(i * seg_dur * 4000)
        e = min(int((i + 1) * seg_dur * 4000), len(samples_4k))
        seg_wav = samples_4k[s:e] if e > s else np.zeros(min_seg_len, dtype=np.float32)
        if len(seg_wav) < min_seg_len:
            seg_wav = np.zeros(min_seg_len, dtype=np.float32)

        with torch.no_grad():
            inp = torch.tensor(seg_wav.astype(np.float32)).unsqueeze(0).to(DEVICE)
            out = wavlm(inp)
            emb = out.last_hidden_state.mean(1).cpu().numpy().squeeze()

        # Prosody
        s16 = int(i * seg_dur * 16000)
        e16 = min(int((i + 1) * seg_dur * 16000), len(samples))
        seg_16 = samples[s16:e16] if e16 > s16 else np.zeros(int(0.5 * 16000), dtype=np.float32)
        prosody = extract_prosody(seg_16, 16000)

        wavlm_feats.append(emb)
        prosody_feats.append(prosody)

    wavlm_arr = np.stack(wavlm_feats)  # (20, 768)
    prosody_arr = np.stack(prosody_feats)  # (20, 23)
    prosody_scaled = (prosody_arr - PROSODY_MEAN) / (PROSODY_STD + 1e-8)
    audio_feats = np.concatenate([wavlm_arr, prosody_scaled], axis=1)  # (20, 791)
    return audio_feats


def extract_prosody(waveform, sr):
    try:
        hop = 160
        n = len(waveform)
        if n < sr: return np.zeros(23, dtype=np.float32)
        rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=hop)[0]
        try:
            pitch = librosa.yin(waveform, fmin=50, fmax=500, sr=sr)
            pv = pitch[~np.isnan(pitch)]
            pitch_mean = float(np.mean(pv)) if len(pv) > 0 else 0.0
            pitch_std = float(np.std(pv)) if len(pv) > 0 else 0.0
            pitch_max = float(np.max(pv)) if len(pv) > 0 else 0.0
            pitch_min = float(np.min(pv)) if len(pv) > 0 else 0.0
        except:
            pitch_mean = pitch_std = pitch_max = pitch_min = 0.0
        dur = n / sr
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        sc = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        f = []
        f.extend([float(np.mean(rms)), float(np.std(rms))])
        f.extend([float(np.mean(zcr)), float(np.std(zcr))])
        f.extend([pitch_mean, pitch_std, pitch_max, pitch_min])
        f.extend([dur, dur**0.5, np.log1p(dur)])
        for i in range(13):
            f.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
        f.append(float(np.mean(sc)))
        while len(f) < 23: f.append(0.0)
        return np.array(f[:23], dtype=np.float32)
    except:
        return np.zeros(23, dtype=np.float32)


def extract_text_features(text, n_segments=N_SEGMENTS):
    """Extract RoBERTa CLS features for text, distributed across n_segments."""
    if not text.strip():
        return np.zeros((n_segments, 768), dtype=np.float32)

    # Split text into n_segments roughly equal parts
    words = text.split()
    words_per_seg = max(1, len(words) // n_segments)
    segments_text = []
    for i in range(n_segments):
        seg_words = words[i * words_per_seg:(i + 1) * words_per_seg]
        segments_text.append(" ".join(seg_words) if seg_words else "[empty]")

    with torch.no_grad():
        inputs = roberta_tokenizer(
            segments_text, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(DEVICE)
        outputs = roberta(**inputs)
        cls_feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_feats.astype(np.float32)


@torch.no_grad()
def score(audio_path, text):
    """Score humor strength for audio + text."""
    if not text.strip():
        return None, "Please enter transcription text"

    try:
        # Extract features
        audio_feats = extract_audio_features(audio_path)
        text_feats = extract_text_features(text)

        if audio_feats is None:
            return None, "Audio too short"

        # Score
        text_t = torch.tensor(text_feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        audio_t = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        scores = v6_model(text_t, audio_t).squeeze(0).cpu().numpy()

        # Interpretation
        mean_score = float(scores.mean())
        max_score = float(scores.max())
        min_score = float(scores.min())

        interpretation = []
        if mean_score > 0.7:
            interpretation.append("🎭 Highly comedic delivery detected")
        elif mean_score > 0.5:
            interpretation.append("😄 Moderately funny content")
        elif mean_score > 0.3:
            interpretation.append("😐 Mildly amusing at best")
        else:
            interpretation.append("😶 Flat/deadpan delivery")

        if max_score - min_score > 0.4:
            interpretation.append(f"📈 High variance (punchline at {max_score:.2f})")

        return scores, "\n".join(interpretation) + f"\n\nMean: {mean_score:.3f} | Max: {max_score:.3f} | Min: {min_score:.3f}"

    except Exception as e:
        return None, f"Error: {e}"


# ── Gradio UI ──────────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=score,
    inputs=[
        gr.Audio(type="filepath", label="🎤 Audio File (WAV/MP3/M4A)"),
        gr.Textbox(label="📝 Transcription (what's being said)", lines=3,
                   placeholder="Enter the transcription of the audio...")
    ],
    outputs=gr.Label(label="🎯 Humor Strength per Segment (0-1)"),
    title="HaHaScore 😄 — Humor Strength Predictor",
    description="""
**v6 TriModal Cross-Attention** — Sentence-level humor strength prediction (0-1).

This model combines:
- **Text**: RoBERTa CLS embeddings from transcription
- **Audio**: WavLM + prosody features

Results show per-segment scores across the audio file.

Model AUC: **0.858** (5-fold CV on StandUp4AI pseudo-labels).
    """,
    examples=[
        [None, "I told my therapist I have trouble with boundaries. She said I should install some."],
        [None, "Why did the chicken cross the road? To get to the other side."],
        [None, "The weather today is quite pleasant with partly cloudy skies and a chance of rain."],
    ],
    article="""
## How it works

1. Upload an audio file (WAV, MP3, or M4A)
2. Enter the transcription of what's being said
3. Get per-segment humor scores (0-1)

The model predicts whether each segment sounds comedic based on:
- **Audio delivery**: prosody, pitch variation, energy
- **Text semantics**: setup vs punchline structure

**Note**: This is a research prototype. Scores reflect acoustic patterns associated with comedic performance.
    """
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
