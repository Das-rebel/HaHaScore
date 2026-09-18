#!/usr/bin/env python3
"""
Bridge 1 v4: Pseudo-label all 639 StandUp4AI files (CPU, per-segment prosody)
===============================================================================
Per-segment prosody is REQUIRED - file-level prosody gives wrong pitch statistics.

Speed: ~44s per file (WavLM 4s + prosody 40s) → 639 files = ~7.4 hours
"""
import os, json, sys, time
import numpy as np
import torch, torch.nn as nn
from pydub import AudioSegment
import librosa
from transformers import WavLMModel
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_DIR = Path("/Users/Subho/data/standup4ai_full")
OUTPUT_FILE = "/Users/Subho/tmp/bridge1_output/pseudo_labels_641_v4.json"
CHECKPOINT_FILE = "/Users/Subho/tmp/bridge1_output/pseudo_labels_641_v4_checkpoint.json"
FUSION_MODEL = "/Users/Subho/autonomous_laughter_prediction/models/fusion_mlp_v2.pt"
N_SEGMENTS = 20
FEATURE_DIM = 791

print(f"Device: {DEVICE}")
audio_files = sorted(AUDIO_DIR.glob("*.m4a"))
print(f"Audio dir: {AUDIO_DIR} ({len(audio_files)} files)")

# PROSODY_MEAN/PROSODY_STD from original training code
PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0,
    0.0,6.0,0.0,5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0,
    6.0,3.0,5.0,2.5], dtype=np.float32)


class FusionMLP(nn.Module):
    def __init__(self, input_dim=791):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 1), torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(self.bn0(x))


def extract_prosody_segment(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract 23-dim prosody for a single audio segment.
    Matches the feature extraction from the original training pipeline."""
    try:
        hop = 160
        n = len(waveform)
        if n < sr:  # less than 1 second
            return np.zeros(23, dtype=np.float32)
        
        # RMS per frame
        rms = librosa.feature.rms(y=waveform, frame_length=400, hop_length=hop)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=400, hop_length=hop)[0]
        
        # Pitch (yin)
        try:
            pitch = librosa.yin(waveform, fmin=50, fmax=500, sr=sr)
            pv = pitch[~np.isnan(pitch)]
            pitch_mean = float(np.mean(pv)) if len(pv) > 0 else 0.0
            pitch_std = float(np.std(pv)) if len(pv) > 0 else 0.0
            pitch_max = float(np.max(pv)) if len(pv) > 0 else 0.0
            pitch_min = float(np.min(pv)) if len(pv) > 0 else 0.0
        except:
            pitch_mean = pitch_std = pitch_max = pitch_min = 0.0
        
        # Duration
        dur = n / sr
        
        # MFCC (13 coefficients)
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        
        # Spectral centroid
        sc = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        
        # Build feature vector
        f = []
        f.extend([float(np.mean(rms)), float(np.std(rms))])
        f.extend([float(np.mean(zcr)), float(np.std(zcr))])
        f.extend([pitch_mean, pitch_std, pitch_max, pitch_min])
        f.extend([dur, dur**0.5, np.log1p(dur)])
        for i in range(13):
            f.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
        f.append(float(np.mean(sc)))
        
        while len(f) < 23:
            f.append(0.0)
        
        return np.array(f[:23], dtype=np.float32)
    except:
        return np.zeros(23, dtype=np.float32)


@torch.no_grad()
def process_file(audio_path: Path, fusion, wavlm) -> dict:
    """Process one file: per-segment prosody + batched WavLM."""
    vid = audio_path.stem
    
    # Load audio with pydub
    try:
        seg_audio = AudioSegment.from_file(str(audio_path), format='m4a')
        seg_audio = seg_audio.set_frame_rate(16000).set_channels(1)
        samples_16k = np.array(seg_audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    except Exception as e:
        return {'video_id': vid, 'segments': [], 'error': f'audio_load_failed: {e}'}
    
    duration = len(samples_16k) / 16000.0
    if len(samples_16k) < 1600:
        return {'video_id': vid, 'segments': [], 'error': 'audio_too_short'}
    
    # 4x downsample for WavLM
    samples_4k = librosa.resample(samples_16k, orig_sr=16000, target_sr=4000)
    sr_wavlm = 4000
    
    seg_dur = duration / N_SEGMENTS
    min_seg_len = int(0.5 * sr_wavlm)  # minimum 0.5 second per segment
    
    # Collect WavLM input tensors (all same length)
    wavlm_inputs = []
    seg_starts = []
    seg_ends = []
    
    for i in range(N_SEGMENTS):
        s = int(i * seg_dur * sr_wavlm)
        e = int((i + 1) * seg_dur * sr_wavlm)
        e = min(e, len(samples_4k))
        seg_wav = samples_4k[s:e]
        seg_starts.append(s / sr_wavlm)
        seg_ends.append(e / sr_wavlm)
        
        if len(seg_wav) < min_seg_len:
            seg_wav = np.zeros(min_seg_len, dtype=np.float32)
        else:
            seg_wav = seg_wav.astype(np.float32)
        
        wavlm_inputs.append(torch.tensor(seg_wav))
    
    # Ensure all tensors same size by padding
    max_len = max(t.shape[0] for t in wavlm_inputs)
    padded = []
    for t in wavlm_inputs:
        if t.shape[0] < max_len:
            t = torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
        padded.append(t)
    batch_wav = torch.stack(padded)  # (20, max_len)
    wavlm_out = wavlm(batch_wav)
    wavlm_emb = wavlm_out.last_hidden_state.mean(dim=1).numpy()  # (20, 768)
    
    # Per-segment prosody
    prosody_list = []
    for i in range(N_SEGMENTS):
        s = int(seg_starts[i] * 16000)
        e = int(seg_ends[i] * 16000)
        e = min(e, len(samples_16k))
        audio_seg = samples_16k[s:e]
        prosody = extract_prosody_segment(audio_seg, 16000)
        prosody_list.append(prosody)
    
    prosody_array = np.stack(prosody_list)  # (20, 23)
    
    # Scale prosody
    prosody_scaled = (prosody_array - PROSODY_MEAN) / (PROSODY_STD + 1e-8)  # (20, 23)
    
    # Build 791-dim features
    features = np.concatenate([wavlm_emb, prosody_scaled], axis=1).astype(np.float32)  # (20, 791)
    
    # Fusion forward
    fusion.eval()
    with torch.no_grad():
        scores = fusion(torch.tensor(features)).squeeze().numpy()  # (20,)
    
    # Build result
    segments = []
    for i in range(N_SEGMENTS):
        segments.append({
            'start': round(seg_starts[i], 2),
            'end': round(seg_ends[i], 2),
            'score': float(scores[i])
        })
    
    return {
        'video_id': vid,
        'segments': segments,
        'duration': round(duration, 2)
    }


def main():
    print("\nLoading WavLM...")
    wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').eval()
    
    print("Loading FusionMLP...")
    fusion = FusionMLP(input_dim=FEATURE_DIM)
    fusion.load_state_dict(
        torch.load(FUSION_MODEL, map_location='cpu', weights_only=False),
        strict=False
    )
    fusion.eval()
    print(f"Models loaded. WavLM: {sum(p.numel() for p in wavlm.parameters()):,} params")
    print(f"Fusion: {sum(p.numel() for p in fusion.parameters()):,} params")
    
    # Check checkpoint
    done = set()
    checkpoint_results = []
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                d = json.loads(line)
                done.add(d['video_id'])
                checkpoint_results.append(d)
        print(f"Checkpoint: {len(done)} files already processed")
    
    total = len(audio_files)
    start_time = time.time()
    
    for idx, audio_path in enumerate(audio_files):
        vid = audio_path.stem
        if vid in done:
            continue
        
        t0 = time.time()
        try:
            result = process_file(audio_path, fusion, wavlm)
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'video_id': vid, 'segments': [], 'error': str(e)}
        
        elapsed = time.time() - t0
        total_elapsed = time.time() - start_time
        
        scores = [s['score'] for s in result.get('segments', [])]
        if scores:
            print(f"[{idx+1}/{total}] {vid}: mean={np.mean(scores):.3f} "
                  f"range=[{np.min(scores):.3f},{np.max(scores):.3f}] "
                  f"({elapsed:.1f}s, {total_elapsed/60:.1f}min total)")
        else:
            err = result.get('error', 'unknown')
            print(f"[{idx+1}/{total}] {vid}: ERROR {err} ({elapsed:.1f}s)")
        
        checkpoint_results.append(result)
        
        # Checkpoint every 20 files
        if (idx + 1) % 20 == 0:
            with open(CHECKPOINT_FILE, 'w') as f:
                for r in checkpoint_results:
                    f.write(json.dumps(r) + '\n')
            remaining = total - idx - 1
            rate = (idx + 1 - len(done)) / total_elapsed if total_elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0
            print(f"  Checkpoint saved. ETA: {eta/60:.1f} min")
    
    # Final save
    with open(CHECKPOINT_FILE, 'w') as f:
        for r in checkpoint_results:
            f.write(json.dumps(r) + '\n')
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(checkpoint_results, f, indent=1)
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(checkpoint_results)} files in {total_elapsed/60:.1f} min")
    
    # Stats
    all_scores = [s['score'] for r in checkpoint_results for s in r.get('segments', [])]
    if all_scores:
        print(f"Scores: mean={np.mean(all_scores):.3f}, std={np.std(all_scores):.3f}")
        print(f"  min={np.min(all_scores):.3f}, max={np.max(all_scores):.3f}")
        print(f"  ≥0.7: {(np.array(all_scores)>=0.7).sum()}/{len(all_scores)}")
        print(f"  <0.3: {(np.array(all_scores)<0.3).sum()}/{len(all_scores)}")


if __name__ == '__main__':
    main()
