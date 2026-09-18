#!/usr/bin/env python3
"""
Bridge 4: Humor Arc Tracker (GRU over Sequential Segments)
=========================================================
Trains a sequential model on Bridge 1 pseudo-labels to capture humor arcs.

Architecture:
  - WavLM embeddings per segment (512d)
  - Prosody features per segment (23d)
  - Segment position encoding (1d)
  - GRU over 20 sequential segments
  - Per-segment humor score prediction

Expected: Better captures arc dynamics than independent per-segment scoring.
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────
AUDIO_DIR   = Path("/Users/Subho/data/standup4ai_full")
LABELS_FILE = Path("/Users/Subho/funny-strength-predictor/data/pseudo_labels/pseudo_labels_641_v4.json")
MODEL_OUT   = Path("/Users/Subho/funny-strength-predictor/models/bridge4_arc_tracker.pt")
N_SEGMENTS  = 20
HIDDEN      = 128
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
KFOLDS      = 5

# Prosody scaler (from Bridge 1 training)
PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0,
    0.0,6.0,0.0,5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0,
    6.0,3.0,5.0,2.5], dtype=np.float32)

print(f"Device: {DEVICE}")

# ── Dataset ──────────────────────────────────────────────────────────────

class HumorArcDataset(Dataset):
    """One item = one file with 20 segments × (512 WavLM + 23 prosody + 1 pos) = 536d."""

    def __init__(self, features, labels, lengths):
        self.features = features    # (N, 20, 536)
        self.labels = labels        # (N, 20)
        self.lengths = lengths      # (N,) — actual length per file

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.lengths[idx]
        )


def collate_fn(batch):
    """Pad sequences to max length in batch."""
    features, labels, lengths = zip(*batch)
    max_len = max(lengths)
    features = torch.stack([f[:max_len] for f in features])
    labels   = torch.stack([l[:max_len] for l in labels])
    lengths  = torch.tensor(lengths, dtype=torch.long)
    return features, labels, lengths


# ── Model ─────────────────────────────────────────────────────────────────

class HumorArcTracker(nn.Module):
    """
    GRU over sequential segments with bidirectional processing.
    Input: (batch, seq_len, 791) — 768 WavLM + 23 prosody
    Output: (batch, seq_len) — per-segment humor score
    """

    def __init__(self, input_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)  # position encoding
        self.gru = nn.GRU(
            input_dim + 4,  # +4 for position
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        # x: (batch, seq, 536)
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # (batch, seq, 4)
        x = torch.cat([x, pos_emb], dim=-1)       # (batch, seq, 540)
        out, _ = self.gru(x)                      # (batch, seq, hidden*2)
        out = self.head(out).squeeze(-1)           # (batch, seq)
        return out


# ── Feature Extraction (WavLM + Prosody per file) ─────────────────────────

def extract_file_features(audio_path):
    """Extract WavLM + prosody for all 20 segments of one file."""
    import librosa
    from pydub import AudioSegment
    from transformers import WavLMModel

    # Lazy-load models (first call only)
    if not hasattr(extract_file_features, 'wavlm'):
        extract_file_features.wavlm = WavLMModel.from_pretrained(
            'microsoft/wavlm-base-plus'
        ).eval().to(DEVICE)
        extract_file_features.fusion = None  # Not needed for Bridge 4

    wavlm = extract_file_features.wavlm
    audio = AudioSegment.from_file(str(audio_path), format='m4a')
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    duration = len(samples) / 16000.0

    if len(samples) < 1600:
        return None

    # 4× downsample
    samples_4k = librosa.resample(samples, orig_sr=16000, target_sr=4000)
    seg_dur = duration / N_SEGMENTS
    min_seg_len = int(0.5 * 4000)

    # Collect per-segment features
    wavlm_feats = []
    prosody_feats = []

    for i in range(N_SEGMENTS):
        s = int(i * seg_dur * 4000)
        e = int((i + 1) * seg_dur * 4000)
        e = min(e, len(samples_4k))
        seg_wav = samples_4k[s:e]
        if len(seg_wav) < min_seg_len:
            seg_wav = np.zeros(min_seg_len, dtype=np.float32)

        # WavLM
        with torch.no_grad():
            inp = torch.tensor(seg_wav.astype(np.float32)).unsqueeze(0).to(DEVICE)
            out = wavlm(inp)
            emb = out.last_hidden_state.mean(1).cpu().numpy().squeeze()  # (512,)

        # Prosody
        s16 = int(i * seg_dur * 16000)
        e16 = int((i + 1) * seg_dur * 16000)
        e16 = min(e16, len(samples))
        seg_16 = samples[s16:e16]
        prosody = extract_prosody(seg_16, 16000)

        wavlm_feats.append(emb)
        prosody_feats.append(prosody)

    wavlm_arr = np.stack(wavlm_feats)          # (20, 512)
    prosody_arr = np.stack(prosody_feats)      # (20, 23)
    prosody_scaled = (prosody_arr - PROSODY_MEAN) / (PROSODY_STD + 1e-8)

    # Concatenate: (20, 536)
    features = np.concatenate([wavlm_arr, prosody_scaled], axis=1)
    return features


def extract_prosody(waveform, sr):
    """Extract 23-dim prosody features."""
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


# ── Training ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for features, labels, lengths in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(features, lengths)  # (batch, seq_len)
        # Mask padding
        mask = torch.arange(features.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        loss = (criterion(preds, labels) * mask.float()).sum() / mask.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
    return total_loss / sum(l.item() for l in lengths)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels, all_masks = [], [], []
    for features, labels, lengths in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        preds = model(features, lengths)
        mask = torch.arange(features.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        all_preds.append(preds[mask].cpu().numpy())
        all_labels.append(labels[mask].cpu().numpy())
        all_masks.append(mask.cpu())

    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)
    # Binarize pseudo-labels at 0.5 threshold for AUC
    binary_labels = (flat_labels >= 0.5).astype(float)
    if len(np.unique(binary_labels)) > 1:
        auc = roc_auc_score(binary_labels, flat_preds)
    else:
        auc = 0.5
    return auc


# ── Main ──────────────────────────────────────────────────────────────────

print("Loading pseudo-labels...")
with open(LABELS_FILE) as f:
    labels_data = json.load(f)
print(f"  {len(labels_data)} files with pseudo-labels")

# Check for cached features
CACHE_FILE = Path("/Users/Subho/tmp/bridge4_features.npz")
PARTIAL_FILE = Path("/Users/Subho/tmp/bridge4_features_partial.npz")

if CACHE_FILE.exists():
    print(f"Loading cached features from {CACHE_FILE}...")
    cached = np.load(CACHE_FILE, allow_pickle=True)
    # Load object arrays properly
    raw_features = cached['features']
    raw_labels = cached['labels']
    raw_lengths = cached['lengths']
    all_features = [np.asarray(f, dtype=np.float32) for f in raw_features]
    all_labels = [np.asarray(l, dtype=np.float32) for l in raw_labels]
    all_lengths = [int(x) for x in raw_lengths]
elif PARTIAL_FILE.exists():
    print(f"Loading partial features from {PARTIAL_FILE}...")
    cached = np.load(PARTIAL_FILE, allow_pickle=True)
    raw_features = cached['features']
    raw_labels = cached['labels']
    raw_lengths = cached['lengths']
    all_features = [np.asarray(f, dtype=np.float32) for f in raw_features]
    all_labels = [np.asarray(l, dtype=np.float32) for l in raw_labels]
    all_lengths = [int(x) for x in raw_lengths]
    print(f"  Resuming from {len(all_features)} files")
else:
    print("Extracting features (this takes ~2.5 hours for 639 files on CPU)...")
    print("  Use cached features if available to skip.")
    all_features, all_labels, all_lengths = [], [], []
    start_time = time.time()

    # Track already-extracted videos to support resume
    done_ids = set()
    for idx, item in enumerate(labels_data):
        vid = item['video_id']
        if vid in done_ids:
            continue
        audio_path = AUDIO_DIR / f"{vid}.m4a"
        if not audio_path.exists():
            done_ids.add(vid)
            continue
        feats = extract_file_features(audio_path)
        if feats is None:
            done_ids.add(vid)
            continue
        scores = [s['score'] for s in item['segments'][:N_SEGMENTS]]
        all_features.append(feats)
        all_labels.append(scores)
        all_lengths.append(len(scores))
        done_ids.add(vid)
        # Checkpoint every 10 files
        if len(done_ids) % 10 == 0:
            elapsed = time.time() - start_time
            per_file = elapsed / len(done_ids)
            eta_min = per_file * (len(labels_data) - len(done_ids)) / 60
            print(f"  [{len(done_ids)}/{len(labels_data)}] features extracted ({per_file:.1f}s/file, ETA {eta_min:.1f}min)")
            try:
                np.savez_compressed(PARTIAL_FILE,
                         features=np.array(all_features, dtype=object),
                         labels=np.array(all_labels, dtype=object),
                         lengths=np.array(all_lengths))
            except Exception as e:
                print(f"  Warning: checkpoint save failed: {e}")

    # Save final cache
    print(f"  Saving {len(all_features)} files to cache...")
    np.savez_compressed(CACHE_FILE,
             features=np.array(all_features, dtype=object),
             labels=np.array(all_labels, dtype=object),
             lengths=np.array(all_lengths))
    print(f"  Cached to {CACHE_FILE}")

print(f"\nDataset: {len(all_features)} files, {sum(all_lengths)} total segments")
all_features = np.stack(all_features)
all_labels = np.stack(all_labels)

# ── K-Fold Cross-Validation ───────────────────────────────────────────────
print(f"\n{KFOLDS}-Fold Cross-Validation...")
kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(all_features)):
    print(f"\n--- Fold {fold+1}/{KFOLDS} ---")
    train_ds = HumorArcDataset(all_features[train_idx], all_labels[train_idx], np.array(all_lengths)[train_idx])
    val_ds   = HumorArcDataset(all_features[val_idx],   all_labels[val_idx],   np.array(all_lengths)[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader  = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = HumorArcTracker(input_dim=791, hidden=HIDDEN, num_layers=2, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_auc = evaluate(model, val_loader, criterion)
        scheduler.step()
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_OUT.parent / f"bridge4_fold{fold}.pt")
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={train_loss:.4f} val_auc={val_auc:.4f} best={best_auc:.4f}")

    fold_aucs.append(best_auc)
    print(f"  Fold {fold+1} Best AUC: {best_auc:.4f}")

print(f"\n{'='*50}")
print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")

# ── Train Final Model ────────────────────────────────────────────────────
print("\nTraining final model on all data...")
full_ds = HumorArcDataset(all_features, all_labels, all_lengths)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
final_model = HumorArcTracker(input_dim=791, hidden=HIDDEN, num_layers=2, dropout=0.3).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    train_loss = train_epoch(final_model, full_loader, optimizer, criterion)
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f}")

torch.save(final_model.state_dict(), MODEL_OUT)
print(f"Saved final model to {MODEL_OUT}")
