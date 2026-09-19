#!/usr/bin/env python3
"""
Bridge 4 — Humor Arc Tracker Inference
======================================
Loads HumorArcTracker and runs inference on audio files.

Usage:
    python bridge4_inference.py audio.m4a

Output:
    Per-segment scores + arc visualization
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

DEVICE = "cpu"
N_SEGMENTS = 20

PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0,
    0.0,6.0,0.0,5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0,
    6.0,3.0,5.0,2.5], dtype=np.float32)


class HumorArcTracker(nn.Module):
    """BiGRU over sequential audio segments for humor arc tracking."""

    def __init__(self, input_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)
        self.gru = nn.GRU(
            input_dim + 4, hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = torch.cat([x, pos_emb], dim=-1)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


def extract_prosody(waveform, sr):
    """Extract 23-dim prosody from a segment."""
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


@torch.no_grad()
def extract_file_features(audio_path, wavlm):
    """Extract (N_SEGMENTS, 791) features from audio file."""
    import librosa
    from pydub import AudioSegment
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
        inp = torch.tensor(seg_wav.astype(np.float32)).unsqueeze(0).to(DEVICE)
        out = wavlm(inp)
        emb = out.last_hidden_state.mean(1).cpu().numpy().squeeze()
        s16, e16 = int(i*seg_dur*16000), int((i+1)*seg_dur*16000)
        e16 = min(e16, len(samples))
        prosody = extract_prosody(samples[s16:e16], 16000)
        wavlm_feats.append(emb); prosody_feats.append(prosody)
    wavlm_arr = np.stack(wavlm_feats)
    prosody_scaled = (np.stack(prosody_feats) - PROSODY_MEAN) / (PROSODY_STD + 1e-8)
    return np.concatenate([wavlm_arr, prosody_scaled], axis=1).astype(np.float32)


def score_file(audio_path, model, wavlm):
    """Return per-segment scores for an audio file."""
    feats = extract_file_features(audio_path, wavlm)
    if feats is None:
        return None
    features = torch.tensor(feats).unsqueeze(0).float().to(DEVICE)
    scores = model(features).squeeze().numpy()
    return scores


def print_arc(scores, duration):
    """Print a text-based visualization of the humor arc."""
    seg_dur = duration / N_SEGMENTS
    print(f"\nHumor Arc ({N_SEGMENTS} segments, {duration:.0f}s total)")
    print("=" * 50)
    for i, s in enumerate(scores):
        bar_len = int(s * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        t = i * seg_dur
        print(f"[{i+1:02d}] {t:5.1f}s |{bar}| {s:.3f}")
    print("=" * 50)
    print(f"Mean: {np.mean(scores):.3f}  Max: {np.max(scores):.3f}  Min: {np.min(scores):.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bridge 4 Humor Arc Tracker")
    parser.add_argument("audio", help="Audio file (m4a/wav/mp3)")
    parser.add_argument("--model", default="pytorch_model.bin", help="Model weights")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Load models
    print(f"Loading Bridge 4 model from {args.model}...")
    model = HumorArcTracker(input_dim=791, hidden=128, num_layers=2)
    state = torch.load(args.model, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    print("Loading WavLM...")
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval().to(DEVICE)

    print(f"Processing {args.audio}...")
    scores = score_file(args.audio, model, wavlm)

    if scores is None:
        print("ERROR: Could not process audio file")
        sys.exit(1)

    # Get duration
    from pydub import AudioSegment
    audio = AudioSegment.from_file(args.audio)
    duration = len(audio) / 1000.0

    if args.json:
        print(json.dumps({"scores": scores.tolist(), "duration": duration}))
    else:
        print_arc(scores, duration)
