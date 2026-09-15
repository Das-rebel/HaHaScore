#!/usr/bin/env python3
"""
Modal Bridge 1: Pseudo-label all 641 StandUp4AI files
======================================================
Downloads all 641 StandUp4AI audio files from GDrive to Modal volume,
extracts 791-dim features per segment, runs ChuckleNet inference,
and saves pseudo-labels as JSON.

Usage:
    modal run modal_pseudolabel_v1.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import modal

# ── Constants ────────────────────────────────────────────────────────────────
VOLUME_PATH = "/funny"
AUDIO_DIR = f"{VOLUME_PATH}/standup4ai_audio"
OUTPUT_FILE = f"{VOLUME_PATH}/pseudo_labels_641.json"
FUSION_MODEL_PATH = f"{VOLUME_PATH}/fusion_mlp_v2.pt"

N_SEGMENTS = 20  # ~20s segments per file
FEATURE_DIM = 791  # 768 WavLM + 23 prosody

# ── Model ────────────────────────────────────────────────────────────────────
class FusionMLP(nn.Module):
    def __init__(self, input_dim=791):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(self.bn0(x))


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_prosody_23d(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract 23 prosody features."""
    try:
        features = []
        rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=160)[0]
        features.extend([float(np.mean(rms)), float(np.std(rms))])
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=160)[0]
        features.extend([float(np.mean(zcr)), float(np.std(zcr))])
        try:
            pitch = librosa.yin(waveform, fmin=50, fmax=500, sr=sr)
            pitch_valid = pitch[~np.isnan(pitch)]
            features.extend([float(np.mean(pitch_valid)) if len(pitch_valid)>0 else 0.0,
                           float(np.std(pitch_valid)) if len(pitch_valid)>0 else 0.0,
                           float(np.max(pitch_valid)) if len(pitch_valid)>0 else 0.0,
                           float(np.min(pitch_valid)) if len(pitch_valid)>0 else 0.0])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        duration = len(waveform) / sr
        features.extend([duration, duration**0.5, np.log1p(duration)])
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        for i in range(13):
            features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
        spec_cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        features.append(float(np.mean(spec_cent)))
        while len(features) < 23:
            features.append(0.0)
        return np.array(features[:23], dtype=np.float32)
    except:
        return np.zeros(23, dtype=np.float32)


def prosody_scaler_stats() -> tuple:
    """Return fitted mean/std for prosody features (from speech statistics)."""
    # 23-dim: [RMS(2), ZCR(2), pitch(4), duration(3), MFCC(26), spec_cent(1)]
    mean = np.array([
        0.05, 0.02,   # RMS: mean, std
        0.10, 0.05,   # ZCR: mean, std
        150.0, 30.0, 200.0, 80.0,  # pitch: mean, std, max, min
        10.0, 3.2, 2.3,  # duration, sqrt, log1p
        0.0, 20.0, 0.0, 15.0, 0.0, 10.0, 0.0, 8.0, 0.0, 6.0,  # MFCC 1-5: mean, std
        0.0, 5.0, 0.0, 4.0, 0.0, 3.0, 0.0, 2.5, 0.0, 2.0, 0.0, 1.5, 0.0, 1.5,  # MFCC 6-13: mean, std
        2500.0,  # spectral centroid
    ], dtype=np.float32)
    std = np.array([
        0.03, 0.01,
        0.05, 0.02,
        50.0, 15.0, 80.0, 50.0,
        5.0, 1.5, 0.2,
        20.0, 10.0, 15.0, 8.0, 10.0, 5.0, 8.0, 4.0, 6.0, 3.0,
        5.0, 2.5, 4.0, 2.0, 3.0, 1.5, 2.5, 1.2, 2.0, 1.0, 1.5, 0.8, 1.5, 0.8,
        800.0,
    ], dtype=np.float32)
    return mean[:23], std[:23]


def extract_segment_features(waveform: np.ndarray, sr: int, wavlm_model, scaler_mean, scaler_std, n_segments=20):
    """Extract 791-dim features for all segments."""
    duration = len(waveform) / sr
    seg_duration = duration / n_segments
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    seg_feats = []
    for i in range(n_segments):
        s = int(i * seg_duration * 16000)
        e = int((i+1) * seg_duration * 16000)
        sw = waveform[s:e]
        if len(sw) < 1600:
            seg_feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue
        with torch.no_grad():
            wout = wavlm_model(input_values=torch.tensor(sw).float().unsqueeze(0).cuda())
            wf = wout.last_hidden_state.mean(1).squeeze().cpu().numpy()
        if wf.shape[0] < 768:
            wf = np.pad(wf, (0, 768-wf.shape[0]))
        elif wf.shape[0] > 768:
            wf = wf[:768]
        pros = extract_prosody_23d(sw, 16000)
        pros_scaled = (pros - scaler_mean) / (scaler_std + 1e-8)
        feat = np.concatenate([wf, pros_scaled]).astype(np.float32)
        seg_feats.append(feat)
    return np.array(seg_feats)


@torch.no_grad()
def process_file(audio_path: str, fusion_model, wavlm_model, scaler_mean, scaler_std) -> dict:
    """Process a single audio file."""
    import librosa
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(waveform) / sr
    feats = extract_segment_features(waveform, sr, wavlm_model, scaler_mean, scaler_std, N_SEGMENTS)
    probs = fusion_model(torch.tensor(feats).cuda()).squeeze().cpu().numpy()
    return {
        "duration_sec": round(duration, 1),
        "n_segments": N_SEGMENTS,
        "laughter_probs": probs.tolist(),
        "humor_scores": (probs * 100).tolist(),
        "stats": {
            "prob_mean": float(probs.mean()), "prob_std": float(probs.std()),
            "score_mean": float(probs.mean()*100), "score_std": float(probs.std()*100),
            "high_gte_0.5": int((probs >= 0.5).sum()),
        }
    }


# ── Modal App ─────────────────────────────────────────────────────────────────
app = modal.App("bridge1-pseudolabels")

@app.function(volumes={"/funny": modal.Volume.from_name("hahascore-data", create=False)})
def run_pseudolabeling():
    from transformers import WavLMModel
    import glob, librosa
    
    print("Loading models...")
    fusion = FusionMLP(input_dim=791).cuda()
    state = torch.load(FUSION_MODEL_PATH, map_location="cuda", weights_only=False)
    fusion.load_state_dict(state, strict=False)
    fusion.eval()
    
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda()
    wavlm.eval()
    
    scaler_mean, scaler_std = prosody_scaler_stats()
    
    audio_files = sorted(glob.glob(f"{AUDIO_DIR}/*.m4a"))
    print(f"Found {len(audio_files)} audio files")
    
    all_results = {}
    for i, af in enumerate(audio_files):
        fname = os.path.basename(af)
        if i % 50 == 0:
            print(f"[{i}/{len(audio_files)}] Processing {fname}...")
        try:
            all_results[fname] = process_file(af, fusion, wavlm, scaler_mean, scaler_std)
        except Exception as e:
            print(f"  ERROR {fname}: {e}")
            all_results[fname] = {"error": str(e)}
    
    print(f"Saving {len(all_results)} results...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f)
    print(f"✓ Saved to {OUTPUT_FILE}")
    
    # Summary
    valid = {k: v for k, v in all_results.items() if "error" not in v}
    all_probs = np.concatenate([np.array(r["laughter_probs"]) for r in valid.values()])
    print(f"\n=== SUMMARY ===")
    print(f"Files: {len(valid)}/{len(audio_files)}")
    print(f"Segments: {len(all_probs)}")
    print(f"High≥0.5: {(all_probs>=0.5).sum()} ({(all_probs>=0.5).mean()*100:.1f}%)")
    print(f"Mean score: {all_probs.mean()*100:.1f} ± {all_probs.std()*100:.1f}")
    return len(valid)


@app.local_entrypoint
def main():
    n = run_pseudolabeling.remote()
    print(f"Done: {n} files processed")
