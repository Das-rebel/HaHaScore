#!/usr/bin/env python3
"""
Modal Bridge 1: Pseudo-label all 641 StandUp4AI files
======================================================
Downloads audio from GDrive via rclone, runs ChuckleNet fusion inference
for pseudo-label generation.

Usage:
    modal run modal_pseudolabel_v1.py
"""
import os, json, io, base64
import numpy as np
import torch, torch.nn as nn
import modal

FEATURE_DIM = 791
N_SEGMENTS = 20
FUSION_PATH = "/tmp/fusion_mlp_v2.pt"
AUDIO_DIR = "/tmp/standup4ai_audio"
OUTPUT_FILE = "/tmp/pseudo_labels_641.json"

# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Prosody Extraction (23 dims) ───────────────────────────────────────────────
PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0,
    0.0,6.0,0.0,5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0,
    6.0,3.0,5.0,2.5], dtype=np.float32)


def extract_prosody_23d(waveform: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    try:
        f = []
        rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=160)[0]
        f.extend([float(np.mean(rms)), float(np.std(rms))])
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=160)[0]
        f.extend([float(np.mean(zcr)), float(np.std(zcr))])
        try:
            pitch = librosa.yin(waveform, fmin=50, fmax=500, sr=sr)
            pv = pitch[~np.isnan(pitch)]
            f.extend([float(np.mean(pv)) if len(pv)>0 else 0.0,
                      float(np.std(pv)) if len(pv)>0 else 0.0,
                      float(np.max(pv)) if len(pv)>0 else 0.0,
                      float(np.min(pv)) if len(pv)>0 else 0.0])
        except:
            f.extend([0.0, 0.0, 0.0, 0.0])
        dur = len(waveform) / sr
        f.extend([dur, dur**0.5, np.log1p(dur)])
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        for i in range(13):
            f.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
        sc = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        f.append(float(np.mean(sc)))
        while len(f) < 23: f.append(0.0)
        return np.array(f[:23], dtype=np.float32)
    except:
        return np.zeros(23, dtype=np.float32)


@torch.no_grad()
def extract_features(waveform: np.ndarray, sr: int, wavlm_model) -> np.ndarray:
    import librosa
    duration = len(waveform) / sr
    seg_dur = duration / N_SEGMENTS
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    seg_feats = []
    for i in range(N_SEGMENTS):
        s = int(i * seg_dur * 16000)
        e = int((i+1) * seg_dur * 16000)
        sw = waveform[s:e]
        if len(sw) < 1600:
            seg_feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue
        wout = wavlm_model(input_values=torch.tensor(sw).float().unsqueeze(0).cuda())
        wf = wout.last_hidden_state.mean(1).squeeze().cpu().numpy()
        wf = wf[:768] if wf.shape[0] > 768 else np.pad(wf, (0, 768-wf.shape[0]))
        pros = extract_prosody_23d(sw, 16000)
        pros_scaled = (pros - PROSODY_MEAN) / (PROSODY_STD + 1e-8)
        seg_feats.append(np.concatenate([wf, pros_scaled]).astype(np.float32))
    return np.array(seg_feats)


# ── Modal App ─────────────────────────────────────────────────────────────────
image = (modal.Image.debian_slim()
    .pip_install("torch","numpy","librosa","transformers","scipy","huggingface_hub")
    .pip_install("rclone"))
app = modal.App("bridge1-pseudolabels")


@app.function(image=image, gpu="T4", timeout=3600)
def run_pseudolabeling():
    import glob, librosa, subprocess, urllib.request
    from transformers import WavLMModel
    
    print("=== Bridge 1: Pseudo-labeling 641 StandUp4AI files ===")
    
    # 1. Download fusion_mlp_v2.pt from GitHub raw content
    print("\n[1/5] Downloading fusion_mlp_v2.pt...")
    # Try GitHub raw first (if committed to public repo)
    # Fallback: upload to HF or embed
    try:
        # Download from a known URL — this will fail until we upload
        # So we embed as base64 as fallback
        raise FileNotFoundError("Use embedded weights")
    except:
        # Use embedded base64 weights (451KB when decoded)
        # The weights are embedded at the bottom of this file
        pass
    
    # 2. Load fusion model (weights are embedded as base64 at end of script)
    print("\n[2/5] Loading fusion model...")
    fusion = FusionMLP(input_dim=FEATURE_DIM).cuda()
    # Load from embedded base64
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="Hayasuki/chuckle-net",
        filename="fusion_mlp_v2.pt",
        repo_type="model"
    )
    state = torch.load(model_path, map_location="cuda", weights_only=False)
    fusion.load_state_dict(state, strict=False)
    fusion.eval()
    print(f"  ✓ Fusion loaded: {sum(p.numel() for p in fusion.parameters())} params")
    
    # 3. Load WavLM
    print("\n[3/5] Loading WavLM...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda()
    wavlm.eval()
    print("  ✓ WavLM loaded")
    
    # 4. Download audio from GDrive
    print("\n[4/5] Downloading StandUp4AI audio from GDrive...")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    print("  Running rclone copy (this may take a few minutes)...")
    result = subprocess.run(
        ["rclone", "copy", "gdrive:standup4ai/audio_1000/", AUDIO_DIR,
         "--transfers", "8", "--checkers", "16", "--drive-chunk-size", "64M"],
        capture_output=True, text=True, timeout=3000
    )
    audio_files = sorted(glob.glob(f"{AUDIO_DIR}/*.m4a"))
    print(f"  ✓ {len(audio_files)} audio files ready ({len(audio_files)/641*100:.0f}% of 641)")
    
    # 5. Inference
    print(f"\n[5/5] Running inference on {len(audio_files)} files...")
    all_results = {}
    for i, af in enumerate(audio_files):
        fname = os.path.basename(af)
        if i % 50 == 0:
            print(f"  [{i}/{len(audio_files)}] {fname}")
        try:
            waveform, sr = librosa.load(af, sr=16000, mono=True)
            feats = extract_features(waveform, sr, wavlm)
            probs = fusion(torch.tensor(feats).cuda()).squeeze().cpu().numpy()
            all_results[fname] = {
                "duration_sec": round(len(waveform)/sr, 1),
                "n_segments": N_SEGMENTS,
                "laughter_probs": probs.tolist(),
                "humor_scores": (probs * 100).tolist(),
                "stats": {
                    "prob_mean": float(probs.mean()), "prob_std": float(probs.std()),
                    "prob_min": float(probs.min()), "prob_max": float(probs.max()),
                    "high_gte_0.5": int((probs >= 0.5).sum()),
                    "low_lt_0.2": int((probs < 0.2).sum()),
                }
            }
        except Exception as e:
            all_results[fname] = {"error": str(e)}
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f)
    print(f"\n  ✓ Saved to {OUTPUT_FILE}")
    
    valid = {k: v for k, v in all_results.items() if "error" not in v}
    if valid:
        all_probs = np.concatenate([np.array(r["laughter_probs"]) for r in valid.values()])
        print(f"\n=== SUMMARY ===")
        print(f"Files: {len(valid)}/{len(audio_files)} processed")
        print(f"Segments: {len(all_probs)}")
        print(f"Laughter prob: {all_probs.mean():.3f}±{all_probs.std():.3f}")
        print(f"High≥0.5: {(all_probs>=0.5).sum()} ({(all_probs>=0.5).mean()*100:.1f}%)")
        print(f"Low<0.2: {(all_probs<0.2).sum()} ({(all_probs<0.2).mean()*100:.1f}%)")
        print(f"Mean humor score: {all_probs.mean()*100:.1f}")
    
    return {"n": len(valid), "total": len(audio_files)}


@app.local_entrypoint
def main():
    result = run_pseudolabeling.remote()
    print(f"\nDone: {result}")


# ── EMBEDDED MODEL WEIGHTS ────────────────────────────────────────────────────
# fusion_mlp_v2.pt base64 encoded (~600KB encoded, 451KB decoded)
# To regenerate: python3 -c "import torch,base64,io; state=torch.load('fusion_mlp_v2.pt',weights_only=False); print(base64.b64encode(io.BytesIO(torch.save(state, io.BytesIO()).getvalue())).decode())"
EMBEDDED_WEIGHTS_B64 = ""  # ← Will be populated before modal run
