#!/usr/bin/env python3
"""
v6: TriModal Cross-Attention Fusion
=====================================
Text (RoBERTa) + Audio (WavLM+prosody) → Cross-Attention → BiGRU → Score

Architecture:
  - Text: RoBERTa-base CLS embedding per segment (768d)
  - Audio: 768d WavLM + 23d prosody = 791d
  - Concatenate: 768 + 791 = 1559d per segment
  - BiGRU(128d, 2 layers, bidirectional)
  - MLP(256 → 128 → 1)
  
Expected: AUC ~0.85-0.88 (vs Bridge 4's 0.842)
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────
BRIDGE4_FEATURES = Path("/Users/Subho/tmp/bridge4_features.npz")
V6_TEXT_FEATURES = Path("/Users/Subho/tmp/v6_features.npz")
LABELS_FILE = Path("/Users/Subho/funny-strength-predictor/data/pseudo_labels/pseudo_labels_641_v4.json")
MODEL_OUT = Path("/Users/Subho/funny-strength-predictor/models/v6_trimodal.pt")
N_SEGMENTS = 20
HIDDEN = 128
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KFOLDS = 5

print(f"Device: {DEVICE}")

# Prosody scaler (from Bridge 1 training)
PROSODY_MEAN = np.array([0.05,0.02, 0.10,0.05, 150.0,30.0,200.0,80.0,
    10.0,3.2,2.3, 0.0,20.0,0.0,15.0,0.0,10.0,0.0,8.0,
    0.0,6.0,0.0,5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03,0.01, 0.05,0.02, 50.0,15.0,80.0,50.0,
    5.0,1.5,0.2, 20.0,10.0,15.0,8.0,10.0,5.0,8.0,4.0,
    6.0,3.0,5.0,2.5], dtype=np.float32)

# ── Dataset ─────────────────────────────────────────────────────────────────

class TriModalDataset(Dataset):
    """
    One item = one file with 20 segments × (1559d text+audio)
    """
    def __init__(self, text_features, audio_features, labels, lengths):
        # text_features: (N, 20, 768)
        # audio_features: (N, 20, 791)
        self.text_features = text_features
        self.audio_features = audio_features
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text_features[idx], dtype=torch.float32),
            torch.tensor(self.audio_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.lengths[idx]
        )


def collate_fn(batch):
    text_feats, audio_feats, labels, lengths = zip(*batch)
    max_len = max(lengths)
    text_feats = torch.stack([f[:max_len] for f in text_feats])
    audio_feats = torch.stack([f[:max_len] for f in audio_feats])
    labels = torch.stack([l[:max_len] for l in labels])
    lengths = torch.tensor(lengths, dtype=torch.long)
    return text_feats, audio_feats, labels, lengths


# ── Model ─────────────────────────────────────────────────────────────────────

class TriModalFusion(nn.Module):
    """
    Text + Audio cross-attention fusion with BiGRU.
    
    Text: (batch, seq, 768)
    Audio: (batch, seq, 791)
    
    Cross-attention: text attends to audio, audio attends to text
    """
    
    def __init__(self, text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Audio projection
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention: text → audio
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Cross-attention: audio → text
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Position embedding for sequential modeling
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)
        
        # BiGRU over fused features
        # Fused: text_proj(128) + audio_proj(128) + cross_attn(text)(128) + cross_attn(audio)(128) + pos(4) = 516d
        self.gru = nn.GRU(
            hidden * 4 + 4,  # text_proj + audio_proj + text_attn + audio_attn + pos
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
    
    def forward(self, text, audio, lengths=None):
        # text: (batch, seq, 768)
        # audio: (batch, seq, 791)
        batch_size, seq_len, _ = text.shape
        
        # Project to hidden dim
        text_h = self.text_proj(text)   # (batch, seq, hidden)
        audio_h = self.audio_proj(audio)  # (batch, seq, hidden)
        
        # Cross-attention: text attends to audio
        text_attn_out, _ = self.text_to_audio_attn(text_h, audio_h, audio_h)
        
        # Cross-attention: audio attends to text
        audio_attn_out, _ = self.audio_to_text_attn(audio_h, text_h, text_h)
        
        # Concatenate all: text_proj + audio_proj + text_attn + audio_attn
        fused = torch.cat([text_h, audio_h, text_attn_out, audio_attn_out], dim=-1)
        # (batch, seq, hidden * 3)
        
        # Add position embedding
        positions = torch.arange(seq_len, device=text.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # (batch, seq, 4)
        fused = torch.cat([fused, pos_emb], dim=-1)  # (batch, seq, hidden*3 + 4)
        
        # BiGRU
        out, _ = self.gru(fused)  # (batch, seq, hidden*2)
        
        # Score
        out = self.head(out).squeeze(-1)  # (batch, seq)
        return out


# ── Training ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_count = 0, 0
    for text, audio, labels, lengths in loader:
        text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(text, audio, lengths)
        mask = torch.arange(text.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        loss = ((criterion(preds, labels) * mask.float()).sum() / mask.sum())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        total_count += mask.sum().item()
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for text, audio, labels, lengths in loader:
        text, audio, labels = text.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)
        preds = model(text, audio, lengths)
        mask = torch.arange(text.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        all_preds.append(preds[mask].cpu().numpy())
        all_labels.append(labels[mask].cpu().numpy())
    
    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)
    binary = (flat_labels >= 0.5).astype(float)
    if len(np.unique(binary)) > 1:
        return roc_auc_score(binary, flat_preds)
    return 0.5


# ── Main ───────────────────────────────────────────────────────────────────

print("Loading data...")

# Load Bridge 4 features (audio)
cached = np.load(BRIDGE4_FEATURES, allow_pickle=True)
all_audio = [np.asarray(f, dtype=np.float32) for f in cached["features"]]
all_labels = [np.asarray(l, dtype=np.float32) for l in cached["labels"]]
all_lengths = [int(x) for x in cached["lengths"]]

# Load v6 text features
v6_data = np.load(V6_TEXT_FEATURES, allow_pickle=True)
all_text = [np.asarray(f, dtype=np.float32) for f in v6_data["text_features"]]
transcribed_vids = list(v6_data["video_ids"])
print(f"Text features: {len(all_text)} files, shape={all_text[0].shape}")
print(f"Audio features: {len(all_audio)} files, shape={all_audio[0].shape}")

# Match text features to audio features by video_id
# Build video_id order from labels
with open(LABELS_FILE) as f:
    labels_data = json.load(f)
vid_to_idx = {item["video_id"]: i for i, item in enumerate(labels_data)}

# Verify alignment
labels_vids = [item["video_id"] for item in labels_data]
assert labels_vids == transcribed_vids, f"Order mismatch: {labels_vids[:5]} vs {transcribed_vids[:5]}"
print("✓ Text and audio feature order matches")

# Stack
all_text_arr = np.stack(all_text)   # (639, 20, 768)
all_audio_arr = np.stack(all_audio)  # (639, 20, 791)
all_labels_arr = np.stack(all_labels)
print(f"Text: {all_text_arr.shape}, Audio: {all_audio_arr.shape}")

# ── K-Fold CV ───────────────────────────────────────────────────────────────

print(f"\n{KFOLDS}-Fold Cross-Validation...")
kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(all_text_arr)):
    print(f"\n--- Fold {fold+1}/{KFOLDS} ---")
    train_ds = TriModalDataset(
        all_text_arr[train_idx], all_audio_arr[train_idx],
        all_labels_arr[train_idx], np.array(all_lengths)[train_idx]
    )
    val_ds = TriModalDataset(
        all_text_arr[val_idx], all_audio_arr[val_idx],
        all_labels_arr[val_idx], np.array(all_lengths)[val_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = TriModalFusion(text_dim=768, audio_dim=791, hidden=HIDDEN, num_layers=2, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_auc = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_auc = evaluate(model, val_loader)
        scheduler.step()
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_OUT.parent / f"v6_fold{fold}.pt")
        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:2d}: loss={train_loss:.4f} val_auc={val_auc:.4f} best={best_auc:.4f}")
    
    fold_aucs.append(best_auc)
    print(f"  Fold {fold+1} Best AUC: {best_auc:.4f}")

print(f"\n{'='*50}")
print(f"v6 Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")

# ── Train Final Model ───────────────────────────────────────────────────────

print("\nTraining final model on all data...")
full_ds = TriModalDataset(all_text_arr, all_audio_arr, all_labels_arr, all_lengths)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
final_model = TriModalFusion(text_dim=768, audio_dim=791, hidden=HIDDEN, num_layers=2, dropout=0.3).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    train_loss = train_epoch(final_model, full_loader, optimizer, criterion)
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        print(f"  Ep {epoch+1:2d}: loss={train_loss:.4f}")

torch.save(final_model.state_dict(), MODEL_OUT)
print(f"\nSaved final model to {MODEL_OUT}")