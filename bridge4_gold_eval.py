#!/usr/bin/env python3
"""
Bridge 4 Gold Label Evaluation
==============================
Evaluate Bridge 4's per-segment humor scores against
human-annotated laughter labels from StandUp4AI.

Binary classification: is there laughter in this segment?
Metrics: AUC, F1, Precision, Recall at various thresholds.
"""
import json, csv, torch
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

DEVICE = "cpu"
N_SEGMENTS = 20

# ── Load Gold Labels ───────────────────────────────────────────────────

GOLD_DIR = Path("/Users/Subho/funny-strength-predictor/data/gold_labels")

def load_gold_labels(video_id):
    """Return per-segment binary labels (laughter=1, no_laughter=0)."""
    # Try exact match first
    label_file = GOLD_DIR / f"{video_id}.csv"
    if not label_file.exists():
        # Try base ID (before comma) — language suffix in audio vs plain in labels
        base = video_id.split(',')[0]
        label_file = GOLD_DIR / f"{base}.csv"
        if not label_file.exists():
            return None

    segments = []
    with open(label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t0, t1 = float(row['t0']), float(row['t1'])
            label = 1 if row['label'].strip() == 'risa' else 0
            segments.append((t0, t1, label))

    return segments


def segments_to_binary_labels(segments, duration, n_segs=N_SEGMENTS):
    """
    Convert segment-level labels (t0, t1, label) to per-segment binary labels.
    A segment gets label=1 if >50% of it overlaps with laughter.
    """
    seg_dur = duration / n_segs
    labels = []
    for i in range(n_segs):
        s = i * seg_dur
        e = (i + 1) * seg_dur
        overlap = 0.0
        for t0, t1, lbl in segments:
            ov = max(0, min(e, t1) - max(s, t0))
            overlap += ov * lbl
        # Label=1 if >30% of segment overlaps with laughter
        labels.append(1 if overlap > 0.3 * seg_dur else 0)
    return labels


# ── Load Bridge 4 Predictions ──────────────────────────────────────────

with open('/Users/Subho/funny-strength-predictor/data/pseudo_labels/pseudo_labels_641_v4.json') as f:
    pseudo_data = {d['video_id']: d for d in json.load(f)}

# ── Load Bridge 4 Model ────────────────────────────────────────────────

import torch.nn as nn

class HumorArcTracker(nn.Module):
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


from transformers import WavLMModel

PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0, 0.0,6.0,0.0,5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0, 6.0,3.0,5.0,2.5], dtype=np.float32)


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


@torch.no_grad()
def get_bridge4_scores(video_id, wavlm, model, audio_dir):
    """Get Bridge 4 per-segment scores for a video."""
    import librosa
    from pydub import AudioSegment

    audio_path = audio_dir / f"{video_id}.m4a"
    if not audio_path.exists():
        return None

    audio = AudioSegment.from_file(str(audio_path), format='m4a')
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    duration = len(samples) / 16000.0
    if len(samples) < 1600: return None

    samples_4k = librosa.resample(samples, orig_sr=16000, target_sr=4000)
    seg_dur = duration / N_SEGMENTS
    min_seg_len = int(0.5 * 4000)
    wavlm_feats, prosody_feats = [], []

    for i in range(N_SEGMENTS):
        s = int(i * seg_dur * 4000)
        e = int((i + 1) * seg_dur * 4000)
        e = min(e, len(samples_4k))
        seg_wav = samples_4k[s:e]
        if len(seg_wav) < min_seg_len:
            seg_wav = np.zeros(min_seg_len, dtype=np.float32)
        inp = torch.tensor(seg_wav.astype(np.float32)).unsqueeze(0).to(DEVICE)
        out = wavlm(inp)
        emb = out.last_hidden_state.mean(1).cpu().numpy().squeeze()
        s16 = int(i * seg_dur * 16000)
        e16 = int((i + 1) * seg_dur * 16000)
        e16 = min(e16, len(samples))
        prosody = extract_prosody(samples[s16:e16], 16000)
        wavlm_feats.append(emb)
        prosody_feats.append(prosody)

    wavlm_arr = np.stack(wavlm_feats)
    prosody_arr = np.stack(prosody_feats)
    prosody_scaled = (prosody_arr - PROSODY_MEAN) / (PROSODY_STD + 1e-8)
    features = np.concatenate([wavlm_arr, prosody_scaled], axis=1).astype(np.float32)
    features_tensor = torch.tensor(features).unsqueeze(0).float().to(DEVICE)
    scores = model(features_tensor).squeeze().cpu().numpy()
    return scores


# ── Main Evaluation ──────────────────────────────────────────────────

print("Loading Bridge 4 model...")
model = HumorArcTracker(input_dim=791, hidden=128, num_layers=2)
state = torch.load(
    "/Users/Subho/funny-strength-predictor/models/bridge4_arc_tracker.pt",
    map_location=DEVICE, weights_only=False
)
model.load_state_dict(state, strict=False)
model.eval()

print("Loading WavLM...")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval().to(DEVICE)

AUDIO_DIR = Path("/Users/Subho/data/standup4ai_full")

# ── Evaluate on Gold Labels ───────────────────────────────────────────

gold_files = list(GOLD_DIR.glob("*.csv"))
print(f"\nGold label files: {len(gold_files)}")

# Build base_id -> full_id mapping from pseudo_data
base_to_full = {}
for vid in pseudo_data:
    base = vid.split(',')[0]
    if base not in base_to_full:
        base_to_full[base] = vid  # take first match

all_y_true = []
all_y_pred = []
per_file_results = []

for lf in gold_files:
    base_id = lf.stem
    if base_id not in base_to_full:
        continue
    vid = base_to_full[base_id]  # full ID with language suffix

    # Get gold labels
    raw_segments = load_gold_labels(vid)
    if raw_segments is None:
        continue

    item = pseudo_data[vid]
    duration = item['duration']

    gold_labels = segments_to_binary_labels(raw_segments, duration)
    if sum(gold_labels) == 0 or sum(gold_labels) == len(gold_labels):
        continue  # skip all-same files

    # Get Bridge 4 scores
    scores = get_bridge4_scores(vid, wavlm, model, AUDIO_DIR)
    if scores is None:
        continue

    all_y_true.extend(gold_labels)
    all_y_pred.extend(scores.tolist())
    per_file_results.append({
        'video_id': vid,
        'duration': round(duration, 1),
        'laughter_segs': sum(gold_labels),
        'total_segs': len(gold_labels),
        'bridge4_mean': round(float(np.mean(scores)), 3),
        'auc': round(roc_auc_score(gold_labels, scores), 3) if len(np.unique(gold_labels)) > 1 else None
    })

print(f"\nFiles evaluated: {len(per_file_results)}")
print(f"Total segments: {len(all_y_true)}")
print(f"Laughter segments: {sum(all_y_true)} ({100*sum(all_y_true)/len(all_y_true):.1f}%)")

# ── Aggregate Metrics ────────────────────────────────────────────────

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# AUC
overall_auc = roc_auc_score(all_y_true, all_y_pred)
print(f"\n=== OVERALL AUC: {overall_auc:.4f} ===")

# Precision/Recall at different thresholds
precisions, recalls, thresholds = precision_recall_curve(all_y_true, all_y_pred)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1s)
best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
best_f1 = f1s[best_f1_idx]
best_prec = precisions[best_f1_idx]
best_rec = recalls[best_f1_idx]

print(f"\nBest F1: {best_f1:.4f} at threshold {best_threshold:.3f}")
print(f"  Precision: {best_prec:.4f}")
print(f"  Recall: {best_rec:.4f}")

# Threshold sweep
print("\n--- Threshold Sweep ---")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    preds = (all_y_pred >= t).astype(int)
    p = sum((all_y_true == 1) & (preds == 1)) / max(sum(preds == 1), 1)
    r = sum((all_y_true == 1) & (preds == 1)) / max(sum(all_y_true == 1), 1)
    f = 2 * p * r / (p + r + 1e-8)
    tp = sum((all_y_true == 1) & (preds == 1))
    fp = sum((all_y_true == 0) & (preds == 1))
    fn = sum((all_y_true == 1) & (preds == 0))
    print(f"  t={t:.1f}: P={p:.3f} R={r:.3f} F1={f:.3f} TP={tp} FP={fp} FN={fn}")

# Per-file AUC
file_aucs = [r['auc'] for r in per_file_results if r['auc'] is not None]
print(f"\nPer-file AUC: mean={np.mean(file_aucs):.3f} std={np.std(file_aucs):.3f}")
print(f"  Min={np.min(file_aucs):.3f} Max={np.max(file_aucs):.3f}")

print("\nPer-file results:")
for r in sorted(per_file_results, key=lambda x: x['bridge4_mean'], reverse=True):
    print(f"  {r['video_id']}: laughter={r['laughter_segs']}/{r['total_segs']} bridge4_mean={r['bridge4_mean']} file_auc={r['auc']}")
