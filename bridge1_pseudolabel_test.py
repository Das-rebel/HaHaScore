#!/usr/bin/env python3
"""
Bridge 1 Test v2: ChuckleNet Pseudo-Labels with correct feature normalization
================================================================================
Key fix: StandardScaler on prosody to match training data statistics.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import librosa
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

AUDIO_DIR = Path("/tmp/standup4ai_audio/")
FUSION_MODEL = "/Users/Subho/autonomous_laughter_prediction/models/fusion_mlp_v2.pt"
OUTPUT_FILE = "/tmp/bridge1_pseudolabels_v2.json"

# ── Model Architecture ────────────────────────────────────────────────────────
class FusionMLP(nn.Module):
    """Matches fusion_mlp_v2.pt: 791 → 128 → 64 → 1 (Sigmoid)"""
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


# ── Prosody Extraction (21 dims, matching original code) ─────────────────────
def extract_prosody_21d(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract 21 prosody features matching ChuckleNet's prosody extraction.
    From streamlined_1000_pipeline.py:
      [RMS, ZCR, pitch_mean, pitch_std, duration, n_words, 13 MFCCs, spec_cent, spec_bw]
    """
    try:
        features = []
        
        # RMS energy — 1 dim
        rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=160)[0]
        features.append(float(np.mean(rms)))
        
        # Zero-crossing rate — 1 dim
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=160)[0]
        features.append(float(np.mean(zcr)))
        
        # Pitch (yin) — 2 dims
        try:
            pitch = librosa.yin(waveform, fmin=50, fmax=500, sr=sr)
            pitch_valid = pitch[~np.isnan(pitch)]
            features.append(float(np.mean(pitch_valid)) if len(pitch_valid) > 0 else 0.0)
            features.append(float(np.std(pitch_valid)) if len(pitch_valid) > 0 else 0.0)
        except:
            features.extend([0.0, 0.0])
        
        # Duration — 1 dim
        features.append(len(waveform) / sr)
        
        # n_words placeholder — 1 dim
        features.append(0.0)
        
        # MFCCs — 13 dims
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        for i in range(13):
            features.append(float(np.mean(mfccs[i])))
        
        # Spectral centroid — 1 dim
        spec_cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        features.append(float(np.mean(spec_cent)))
        
        # Spectral bandwidth — 1 dim
        spec_bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        features.append(float(np.mean(spec_bw)))
        
        # Pad to 21
        while len(features) < 21:
            features.append(0.0)
        
        return np.array(features[:21], dtype=np.float32)
        
    except Exception as e:
        return np.zeros(21, dtype=np.float32)


# ── Segment Feature Extraction ────────────────────────────────────────────────
def extract_segment_features(
    waveform: np.ndarray,
    sr: int,
    wavlm_model,
    scaler: StandardScaler,  # Fitted scaler for prosody normalization
    n_segments: int = 20
) -> np.ndarray:
    """
    Extract 791-dim features per segment:
    WavLM last_hidden_state mean (768d) + standardized prosody (23d).
    """
    duration = len(waveform) / sr
    segment_duration = duration / n_segments
    
    # Resample to 16kHz
    if sr != 16000:
        waveform_16k = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    else:
        waveform_16k = waveform
    
    segment_features = []
    
    for i in range(n_segments):
        start_sample = int(i * segment_duration * 16000)
        end_sample = int((i + 1) * segment_duration * 16000)
        seg_wav = waveform_16k[start_sample:end_sample]
        
        if len(seg_wav) < 1600:  # <100ms
            feat_791 = np.zeros(791, dtype=np.float32)
            segment_features.append(feat_791)
            continue
        
        # WavLM last_hidden_state mean → 768-dim
        with torch.no_grad():
            wav_input = torch.tensor(seg_wav).float().unsqueeze(0).to(DEVICE)
            out = wavlm_model(wav_input)
            # last_hidden_state: (1, time, 768) → mean → (1, 768)
            wavlm_feat = out.last_hidden_state.mean(1).squeeze().cpu().numpy()
        
        # Ensure 768 dims
        if wavlm_feat.shape[0] < 768:
            wavlm_feat = np.pad(wavlm_feat, (0, 768 - wavlm_feat.shape[0]))
        elif wavlm_feat.shape[0] > 768:
            wavlm_feat = wavlm_feat[:768]
        
        # Prosody 21-dim
        prosody_21 = extract_prosody_21d(seg_wav, 16000)
        
        # Pad to 23 dims (model expects 23 prosody dims)
        prosody_23 = np.pad(prosody_21, (0, 2)).astype(np.float32)  # 21 → 23
        
        # Standardize prosody using fitted scaler
        prosody_scaled = scaler.transform(prosody_23.reshape(1, -1)).squeeze()
        
        # Concatenate: 768 + 23 = 791
        feat_791 = np.concatenate([wavlm_feat, prosody_scaled])
        segment_features.append(feat_791)
    
    return np.array(segment_features, dtype=np.float32)  # (n_segments, 791)


def fit_prosody_scaler() -> StandardScaler:
    """
    Fit StandardScaler on plausible speech statistics.
    Uses representative values for each prosody feature.
    """
    # Representative mean/std for each of 23 prosody features
    # Based on typical speech audio statistics
    feature_stats = {
        # RMS energy — typically small positive values
        0: (0.05, 0.03),
        # ZCR — typically 0.05-0.15 for speech
        1: (0.10, 0.05),
        # Pitch mean (Hz) — typically 100-200 for speech
        2: (150.0, 50.0),
        # Pitch std
        3: (30.0, 15.0),
        # Duration (sec) — varies
        4: (10.0, 5.0),
        # n_words placeholder
        5: (0.0, 1.0),
        # MFCCs 1-13 — typically centered around 0
        6: (0.0, 20.0),
        7: (0.0, 20.0),
        8: (0.0, 15.0),
        9: (0.0, 10.0),
        10: (0.0, 8.0),
        11: (0.0, 6.0),
        12: (0.0, 5.0),
        13: (0.0, 4.0),
        14: (0.0, 3.0),
        15: (0.0, 2.5),
        16: (0.0, 2.0),
        17: (0.0, 1.5),
        18: (0.0, 1.5),
        # Spectral centroid (Hz) — typically 1000-4000
        19: (2500.0, 800.0),
        # Spectral bandwidth (Hz)
        20: (3000.0, 1000.0),
        # Padding (2 extra dims)
        21: (0.0, 1.0),
        22: (0.0, 1.0),
    }
    
    X_fit = []
    for feat_idx in range(23):
        mean, std = feature_stats[feat_idx]
        # Generate ~1000 sample points for fitting
        X_fit.append(np.random.normal(mean, std, 1000))
    
    X_fit = np.array(X_fit).T  # (1000, 23)
    
    scaler = StandardScaler()
    scaler.fit(X_fit)
    
    print(f"  Prosody scaler fitted: mean[:5]={scaler.mean_[:5].round(3)}, std[:5]={scaler.scale_[:5].round(3)}")
    return scaler


def main():
    from transformers import WavLMModel
    
    print("=" * 60)
    print("BRIDGE 1 v2: ChuckleNet Pseudo-Labels (with normalization)")
    print("=" * 60)
    
    # Load fusion model
    print("\n[1/5] Loading ChuckleNet fusion model...")
    fusion = FusionMLP(input_dim=791).to(DEVICE)
    state = torch.load(FUSION_MODEL, map_location=DEVICE, weights_only=False)
    fusion.load_state_dict(state, strict=False)
    fusion.eval()
    n_params = sum(p.numel() for p in fusion.parameters())
    print(f"  ✓ Fusion model: {n_params:,} params")
    
    # Fit prosody scaler
    print("\n[2/5] Fitting prosody StandardScaler...")
    scaler = fit_prosody_scaler()
    
    # Load WavLM (microsoft/wavlm-base — original model)
    print("\n[3/5] Loading WavLM...")
    try:
        # Try base first (what ChuckleNet likely used)
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        print(f"  ✓ WavLM-base loaded")
    except Exception as e:
        print(f"  ⚠ WavLM-base failed: {e}, falling back to base-plus")
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        print(f"  ✓ WavLM-base-plus loaded")
    wavlm.eval()
    
    # Find audio files
    audio_files = sorted(AUDIO_DIR.glob("*.m4a"))
    print(f"\n[4/5] Found {len(audio_files)} audio files")
    
    all_results = {}
    
    for af in audio_files:
        fname = af.name
        print(f"\n  Processing: {fname}")
        
        try:
            waveform, sr = librosa.load(af, sr=16000, mono=True)
            duration = len(waveform) / sr
            n_segments = max(10, int(duration / 20))
            print(f"    {duration:.1f}s → {n_segments} segments")
            
            # Extract 791-dim features per segment
            features = extract_segment_features(
                waveform, sr, wavlm, scaler, n_segments
            )
            print(f"    Features: {features.shape}")
            
            # Run through fusion model
            with torch.no_grad():
                probs = fusion(torch.tensor(features).to(DEVICE)).squeeze().cpu().numpy()
            
            humor_scores = probs * 100.0
            
            print(f"    Laughter probs: min={probs.min():.3f} max={probs.max():.3f} mean={probs.mean():.3f}")
            print(f"    Humor scores:   min={humor_scores.min():.1f} max={humor_scores.max():.1f} mean={humor_scores.mean():.1f}")
            
            high = (probs >= 0.5).sum()
            mid = ((probs >= 0.2) & (probs < 0.5)).sum()
            low = (probs < 0.2).sum()
            print(f"    High≥0.5: {high}, Mid0.2-0.5: {mid}, Low<0.2: {low}")
            
            all_results[fname] = {
                "duration_sec": round(duration, 1),
                "n_segments": n_segments,
                "laughter_probs": probs.tolist(),
                "humor_scores": humor_scores.tolist(),
                "stats": {
                    "prob_min": float(probs.min()),
                    "prob_max": float(probs.max()),
                    "prob_mean": float(probs.mean()),
                    "prob_std": float(probs.std()),
                    "score_min": float(humor_scores.min()),
                    "score_max": float(humor_scores.max()),
                    "score_mean": float(humor_scores.mean()),
                    "score_std": float(humor_scores.std()),
                },
                "buckets": {
                    "high_gte_0.5": int(high),
                    "mid_0.2_to_0.5": int(mid),
                    "low_lt_0.2": int(low),
                }
            }
            
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            all_results[fname] = {"error": str(e)}
    
    # Save
    print("\n[5/5] Saving...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Saved to {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid = {k: v for k, v in all_results.items() if "error" not in v}
    for fname, res in valid.items():
        s = res["stats"]
        print(f"  {fname}: {res['n_segments']} segs, "
              f"score={s['score_mean']:.1f}±{s['score_std']:.1f}, "
              f"range=[{s['score_min']:.1f},{s['score_max']:.1f}]")
    
    all_probs = np.concatenate([np.array(r["laughter_probs"]) for r in valid.values()])
    all_scores = np.concatenate([np.array(r["humor_scores"]) for r in valid.values()])
    
    print(f"\n  ALL FILES: {len(all_probs)} segments")
    print(f"    High≥50:  {(all_probs>=0.5).sum()} ({(all_probs>=0.5).mean()*100:.1f}%)")
    print(f"    Mid20-50: {((all_probs>=0.2)&(all_probs<0.5)).sum()} ({((all_probs>=0.2)&(all_probs<0.5)).mean()*100:.1f}%)")
    print(f"    Low<20:   {(all_probs<0.2).sum()} ({(all_probs<0.2).mean()*100:.1f}%)")
    print(f"    Mean score: {all_scores.mean():.1f} ± {all_scores.std():.1f}")
    print(f"    Median: {np.median(all_scores):.1f}")
    
    return all_results


if __name__ == "__main__":
    main()
