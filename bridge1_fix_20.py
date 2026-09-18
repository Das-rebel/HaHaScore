#!/usr/bin/env python3
"""Re-process 20 files that failed with Numba error."""
import json, time
import numpy as np
import torch, torch.nn as nn
from pydub import AudioSegment
import librosa
from transformers import WavLMModel
from pathlib import Path

DEVICE = "cpu"
AUDIO_DIR = Path("/tmp/standup4ai_full")
FUSION_MODEL = "/Users/Subho/autonomous_laughter_prediction/models/fusion_mlp_v2.pt"
N_SEGMENTS = 20

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

def extract_prosody_segment(waveform, sr):
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
def process_one(audio_path):
    vid = audio_path.stem
    try:
        seg_audio = AudioSegment.from_file(str(audio_path), format='m4a')
        seg_audio = seg_audio.set_frame_rate(16000).set_channels(1)
        samples_16k = np.array(seg_audio.get_array_of_samples(), dtype=np.float32) / (2**15)
        duration = len(samples_16k) / 16000.0
        if len(samples_16k) < 1600: return {'video_id': vid, 'segments': [], 'error': 'too_short'}

        samples_4k = librosa.resample(samples_16k, orig_sr=16000, target_sr=4000)
        seg_dur = duration / N_SEGMENTS
        min_seg_len = int(0.5 * 4000)

        wavlm_inputs = []
        seg_times = []
        for i in range(N_SEGMENTS):
            s = int(i * seg_dur * 4000)
            e = int((i + 1) * seg_dur * 4000)
            e = min(e, len(samples_4k))
            seg_wav = samples_4k[s:e]
            seg_times.append((s/4000, e/4000))
            if len(seg_wav) < min_seg_len: seg_wav = np.zeros(min_seg_len, dtype=np.float32)
            wavlm_inputs.append(torch.tensor(seg_wav.astype(np.float32)))

        max_len = max(t.shape[0] for t in wavlm_inputs)
        padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[0])) if t.shape[0] < max_len else t for t in wavlm_inputs]
        batch_wav = torch.stack(padded)
        wavlm_out = wavlm(batch_wav)
        wavlm_emb = wavlm_out.last_hidden_state.mean(dim=1).numpy()

        prosody_list = []
        for i in range(N_SEGMENTS):
            s = int(seg_times[i][0] * 16000)
            e = int(seg_times[i][1] * 16000)
            e = min(e, len(samples_16k))
            audio_seg = samples_16k[s:e]
            prosody_list.append(extract_prosody_segment(audio_seg, 16000))

        prosody_array = np.stack(prosody_list)
        prosody_scaled = (prosody_array - PROSODY_MEAN) / (PROSODY_STD + 1e-8)
        features = np.concatenate([wavlm_emb, prosody_scaled], axis=1).astype(np.float32)
        scores = fusion(torch.tensor(features)).squeeze().numpy()

        segments = [{'start': round(seg_times[i][0], 2), 'end': round(seg_times[i][1], 2), 'score': float(scores[i])} for i in range(N_SEGMENTS)]
        return {'video_id': vid, 'segments': segments, 'duration': round(duration, 2)}
    except Exception as e:
        return {'video_id': vid, 'segments': [], 'error': str(e)}

print("Loading WavLM...")
wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').eval()
print("Loading Fusion...")
fusion = FusionMLP(input_dim=791)
fusion.load_state_dict(torch.load(FUSION_MODEL, map_location='cpu', weights_only=False), strict=False)
fusion.eval()
print("Models loaded.")

# Load current results
with open('/tmp/pseudo_labels_641_v4.json') as f:
    results = json.load(f)

# Build lookup
result_by_vid = {r['video_id']: r for r in results}

# Re-process errors
error_ids = ['qcUsQL3xGVc', 'qcarpl5R7cs', 'qdpcb7X-6VY', 'qmq9XSrZDlU', 'qtw0Tg8y-ig',
              'qtzm1lcBie8', 'qyFw-BnDyes', 'r5exfBcm35k', 'rAPZ26R8on8', 'rHfWihSHAGQ',
              'rK_FOkWcwG4', 'rQKN_sp00O8', 'rRYPyKo_1l8', 'rRYa8Cd8NSE', 'r_CGD9M7Wpk',
              'rckeTSiM9p8,pt', 'rjVWh13QIaY', 'rjiH4VcYrnA', 'rkyW0tA9vio', 'rldvVgHQSvo']

fixed = 0
for vid in error_ids:
    audio_path = AUDIO_DIR / f'{vid}.m4a'
    if not audio_path.exists():
        print(f"  {vid}: file not found")
        continue
    print(f"Re-processing {vid}...", end=" ", flush=True)
    result = process_one(audio_path)
    if 'error' not in result:
        result_by_vid[vid] = result
        scores = [s['score'] for s in result['segments']]
        print(f"OK: mean={np.mean(scores):.3f}")
        fixed += 1
    else:
        print(f"STILL ERROR: {result['error']}")

print(f"\nFixed {fixed}/{len(error_ids)} files")

# Save updated results
with open('/tmp/pseudo_labels_641_v4.json', 'w') as f:
    json.dump(list(result_by_vid.values()), f, indent=1)

# Final stats
all_scores = [s['score'] for r in result_by_vid.values() for s in r.get('segments', [])]
arr = np.array(all_scores)
errors = sum(1 for r in result_by_vid.values() if 'error' in r)
print(f"\nFinal stats:")
print(f"  Total files: {len(result_by_vid)}, Errors: {errors}")
print(f"  Segments: {len(all_scores)}")
print(f"  Scores: mean={arr.mean():.3f} std={arr.std():.3f}")
print(f"  Range: [{arr.min():.3f}, {arr.max():.3f}]")
print(f"  ≥0.7: {np.sum(arr>=0.7)}/{len(arr)} ({100*np.mean(arr>=0.7):.1f}%)")
print(f"  <0.3: {np.sum(arr<0.3)}/{len(arr)} ({100*np.mean(arr<0.3):.1f}%)")
