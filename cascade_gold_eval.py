#!/usr/bin/env python3
"""
Bridge 7 on Gold Laughter Labels
================================
Evaluate Bridge 7 on real human laughter annotations (StandUp4AI gold labels).
Gold labels: t0, t1, source, label (risa/no_risa)

Important: Gold labels measure AUDIENCE LAUGHTER (behavioral response),
not PERCEIVED FUNNYNESS (what pseudo-labels measure).
These are DIFFERENT signals.
"""
import json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEGMENTS = 20

# ── Model ─────────────────────────────────────────────────────────────────

class CascadeGateFusion(nn.Module):
    def __init__(self, text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.text_confidence = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)
        self.gru = nn.GRU(
            hidden * 3 + 4, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, text, audio, lengths=None):
        batch_size, seq_len, _ = text.shape
        text_h = self.text_proj(text)
        text_conf = torch.sigmoid(self.text_confidence(text_h))
        audio_h = self.audio_proj(audio)
        gated_audio = audio_h * text_conf
        text_attn_out, _ = self.audio_to_text_attn(audio_h, text_h, text_h)
        fused = torch.cat([text_h, gated_audio, text_attn_out], dim=-1)
        positions = torch.arange(seq_len, device=text.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        fused = torch.cat([fused, pos_emb], dim=-1)
        out, _ = self.gru(fused)
        return self.head(out).squeeze(-1), text_conf.squeeze(-1)


# ── Load Data ──────────────────────────────────────────────────────────────

print("Loading features...")

# Audio features (Bridge 4)
cached = np.load("/Users/Subho/tmp/bridge4_features.npz", allow_pickle=True)
all_audio = [np.asarray(f, dtype=np.float32) for f in cached["features"]]
all_lengths = [int(x) for x in cached["lengths"]]

# Text features (v6) — has video_ids, same order as Bridge 4
v6_data = np.load("/Users/Subho/tmp/v6_features.npz", allow_pickle=True)
all_text = [np.asarray(f, dtype=np.float32) for f in v6_data["text_features"]]
video_ids = list(v6_data["video_ids"])  # Order matches Bridge 4 features

# Match order
# Build index: strip lang suffix to match gold labels
vid_to_idx = {}
for i, v in enumerate(video_ids):
    key = v.split(',')[0]  # strip lang suffix
    vid_to_idx[key] = i

# Load gold labels
gold_dir = Path("/Users/Subho/funny-strength-predictor/data/gold_labels")
gold_files = sorted([f for f in os.listdir(gold_dir) if f.endswith('.csv')])
print(f"Gold label files: {len(gold_files)}")

# Build video_id → gold label mapping
def load_gold_labels(csv_path):
    """Load gold labels, return (t0, t1, is_laugh) list."""
    with open(csv_path) as f:
        lines = f.readlines()[1:]  # skip header
    labels = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) != 4:
            continue
        t0, t1, source, label = parts
        is_laugh = 1.0 if label == 'risa' else 0.0
        labels.append((float(t0), float(t1), is_laugh))
    return labels

# Match gold label files to our video IDs
# Gold files are like "LWYfo_8t5WQ.csv", audio files like "LWYfo_8t5WQ,fr.m4a"
matched = 0
matched_vids = []
for gf in gold_files:
    vid_key = gf.replace('.csv', '')
    if vid_key in vid_to_idx:
        matched += 1
        matched_vids.append(vid_key)

print(f"Matched gold labels to audio: {matched}/{len(gold_files)}")
print(f"Matched videos: {matched_vids[:5]}...")

# ── Build evaluation dataset ────────────────────────────────────────────────

# Load model
model = CascadeGateFusion().to(DEVICE)
model.load_state_dict(torch.load(
    "/Users/Subho/funny-strength-predictor/models/bridge7_cascade.pt",
    map_location=DEVICE))
model.eval()

# Collect predictions and gold labels
all_preds = []
all_gold = []
file_results = []

for vid in matched_vids:
    idx = vid_to_idx[vid]  # idx in our feature arrays
    gold_path = gold_dir / f"{vid}.csv"
    gold_labels = load_gold_labels(gold_path)
    
    if not gold_labels:
        continue
    
    # Get features
    text_feat = all_text[idx]  # (20, 768)
    audio_feat = all_audio[idx]  # (20, 791)
    
    # Get actual video ID with lang suffix from our ordering
    actual_vid = video_ids[idx]  # e.g. '-1FrUOEswOk,fr'
    audio_path = Path(f"/Users/Subho/data/standup4ai_full/{actual_vid}.m4a")
    
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
    except:
        duration = 180.0  # fallback
    
    # Get model prediction
    with torch.no_grad():
        text_t = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        audio_t = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        scores, confs = model(text_t, audio_t)
    
    scores = scores.squeeze(0).cpu().numpy()  # (20,)
    seg_duration = duration / N_SEGMENTS
    
    # Map gold labels to segments
    seg_labels = np.zeros(N_SEGMENTS)
    for t0, t1, is_laugh in gold_labels:
        seg_idx = int(t0 / seg_duration)
        if 0 <= seg_idx < N_SEGMENTS:
            seg_labels[seg_idx] = max(seg_labels[seg_idx], is_laugh)
    
    # Collect
    for i in range(min(N_SEGMENTS, len(scores))):
        all_preds.append(scores[i])
        all_gold.append(seg_labels[i])
    
    file_results.append({
        'vid': vid,
        'duration': duration,
        'pred_mean': scores.mean(),
        'gold_positive': seg_labels.sum(),
        'gold_total': N_SEGMENTS
    })

# ── Evaluate ───────────────────────────────────────────────────────────────

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

all_preds = np.array(all_preds)
all_gold = np.array(all_gold)
binary_gold = (all_gold >= 0.5).astype(float)

print(f"\nEvaluation on {len(matched_vids)} videos, {len(all_preds)} segments:")
print(f"  Positive segments: {binary_gold.sum():.0f} / {len(binary_gold)} ({binary_gold.mean()*100:.1f}%)")
print(f"  Negative segments: {(1-binary_gold).sum():.0f} / {len(binary_gold)}")

if len(np.unique(binary_gold)) > 1:
    auc = roc_auc_score(binary_gold, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_gold, (all_preds >= 0.5).astype(float), average='binary')
    print(f"\n  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    # Per-file breakdown
    print(f"\nPer-file (top 10 by pred_mean):")
    top = sorted(file_results, key=lambda x: x['pred_mean'], reverse=True)[:10]
    for r in top:
        print(f"  {r['vid']}: pred={r['pred_mean']:.3f}, gold_pos={r['gold_positive']:.0f}/{r['gold_total']}")
    
    # AUC by gold positive count
    print(f"\nAUC breakdown:")
    for min_gold in [1, 3, 5]:
        mask = np.array([r['gold_positive'] >= min_gold for r in file_results])
        if mask.sum() > 0:
            file_preds = np.array([r['pred_mean'] for r in file_results])[mask]
            file_golds = np.array([1 if r['gold_positive'] > 0 else 0 for r in file_results])[mask]
            if len(np.unique(file_golds)) > 1:
                f_auc = roc_auc_score(file_golds, file_preds)
                print(f"  Files with ≥{min_gold} gold laughs: {mask.sum()} files, AUC={f_auc:.4f}")
else:
    print("  Not enough class diversity for AUC")

print("\nNOTE: Gold labels measure AUDIENCE LAUGHTER (behavioral response),")
print("      not PERCEIVED FUNNYNESS. These are fundamentally different signals.")
print("      AUC on laughter ≠ AUC on humor strength.")