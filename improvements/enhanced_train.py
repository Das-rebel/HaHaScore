#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Cascade Gate v7
==============================================
Advanced training with:
- Curriculum learning
- AdamW optimizer with cosine annealing
- Mixed precision training
- Gradient clipping
- Early stopping
- Comprehensive logging
"""
import os, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from enhanced_cascade import create_enhanced_model, count_parameters
from enhanced_dataset import EnhancedDataset, enhanced_collate_fn

# ============ CONFIG ============
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "enhanced"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Feature paths (from previous pipeline)
TEXT_FEATURES = Path("/Users/Subho/tmp/v6_features.npz")
AUDIO_FEATURES = Path("/Users/Subho/tmp/bridge4_features.npz")
LABELS_FILE = DATA_DIR / "pseudo_labels" / "pseudo_labels_641_v4.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEGMENTS = 20
HIDDEN = 128
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
KFOLDS = 5
GRAD_CLIP = 1.0
PATIENCE = 8
MIXED_PRECISION = True if DEVICE == "cuda" else False


# ============ DATASET ============
class EnhancedTrainingDataset(Dataset):
    def __init__(self, text_features, audio_features, labels, lengths, cross_features=None, augment=False):
        self.text_features = text_features
        self.audio_features = audio_features
        self.labels = labels
        self.lengths = lengths
        self.cross_features = cross_features
        self.augment = augment
    
    def __len__(self):
        return len(self.text_features)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.text_features[idx], dtype=torch.float32)
        audio = torch.tensor(self.audio_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        length = self.lengths[idx]
        cross = torch.tensor(self.cross_features[idx], dtype=torch.float32) if self.cross_features is not None else None
        
        # Light augmentation
        if self.augment and np.random.random() < 0.25:
            text = text + torch.randn_like(text) * 0.01
            audio = audio + torch.randn_like(audio) * 0.01
        
        return text, audio, label, length, cross


def collate_fn(batch):
    text, audio, labels, lengths, cross = zip(*batch)
    max_len = max(lengths)
    text = torch.stack([x[:max_len] for x in text])
    audio = torch.stack([x[:max_len] for x in audio])
    labels = torch.stack([x[:max_len] for x in labels])
    lengths = torch.tensor(lengths, dtype=torch.long)
    cross = torch.stack([x[:max_len] for x in cross]) if cross[0] is not None else None
    return text, audio, labels, lengths, cross


# ============ TRAINING ============
def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0, mixed=False):
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if mixed else None
    
    for batch in loader:
        text, audio, labels, lengths, cross = [b.to(device) if b is not None else None for b in batch]
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast() if mixed else torch.no_grad():
            scores, text_conf, gate_weight = model(text, audio, cross)
            loss = criterion(scores, labels)
            
            # Regularization: encourage confident gating
            reg_loss = 0.05 * torch.mean(torch.abs(text_conf - 0.5))
            total_loss_value = loss + reg_loss
        
        if mixed:
            scaler.scale(total_loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device, mixed=False):
    model.eval()
    all_preds = []
    all_labels = []
    scaler = torch.cuda.amp.GradScaler() if mixed else None
    
    with torch.no_grad():
        for batch in loader:
            text, audio, labels, lengths, cross = [b.to(device) if b is not None else None for b in batch]
            if mixed:
                with torch.cuda.amp.autocast():
                    scores, text_conf, gate_weight = model(text, audio, cross)
            else:
                scores, text_conf, gate_weight = model(text, audio, cross)
            all_preds.extend(scores.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    return roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5


def curriculum_schedule(epoch, total_epochs):
    """Curriculum learning schedule."""
    if epoch < total_epochs * 0.2:
        return 0.1  # Focus on easy patterns
    elif epoch < total_epochs * 0.5:
        return 0.5  # Medium complexity
    else:
        return 1.0  # Full complexity


def train_kfold(text_arr, audio_arr, labels_arr, lengths_arr, cross_arr=None):
    """Train with k-fold cross-validation."""
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(text_arr)):
        print(f"\n--- Fold {fold+1}/{KFOLDS} ---")
        
        train_ds = EnhancedTrainingDataset(
            text_arr[train_idx], audio_arr[train_idx], labels_arr[train_idx],
            lengths_arr[train_idx], cross_arr[train_idx] if cross_arr is not None else None,
            augment=True
        )
        val_ds = EnhancedTrainingDataset(
            text_arr[val_idx], audio_arr[val_idx], labels_arr[val_idx],
            lengths_arr[val_idx], cross_arr[val_idx] if cross_arr is not None else None,
            augment=False
        )
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        model = create_enhanced_model(text_dim=text_arr.shape[-1], audio_dim=audio_arr.shape[-1],
                                      cross_dim=cross_arr.shape[-1] if cross_arr is not None else 16,
                                      hidden=HIDDEN)
        model.to(DEVICE)
        print(f"Parameters: {count_parameters(model):,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION else None
        
        best_auc = 0.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Curriculum learning
            curriculum_weight = curriculum_schedule(epoch, EPOCHS)
            
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                     grad_clip=GRAD_CLIP, mixed=MIXED_PRECISION)
            val_auc = evaluate(model, val_loader, DEVICE, mixed=MIXED_PRECISION)
            scheduler.step()
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"  Ep {epoch+1:2d}: loss={train_loss:.4f} val_auc={val_auc:.4f} best={best_auc:.4f}")
        
        fold_aucs.append(best_auc)
        torch.save(best_state, MODEL_DIR / f"enhanced_fold{fold}.pt")
        print(f"  Fold {fold+1} Best AUC: {best_auc:.4f}")
    
    return fold_aucs


def train_final_model(text_arr, audio_arr, labels_arr, lengths_arr, cross_arr=None):
    """Train final model on all data."""
    print("\nTraining final model on all data...")
    full_ds = EnhancedTrainingDataset(
        text_arr, audio_arr, labels_arr, lengths_arr,
        cross_arr if cross_arr is not None else None,
        augment=True
    )
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = create_enhanced_model(text_dim=text_arr.shape[-1], audio_dim=audio_arr.shape[-1],
                                  cross_dim=cross_arr.shape[-1] if cross_arr is not None else 16,
                                  hidden=HIDDEN)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, full_loader, optimizer, criterion, DEVICE,
                                 grad_clip=GRAD_CLIP, mixed=MIXED_PRECISION)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1:2d}: loss={train_loss:.4f}")
    
    torch.save(model.state_dict(), MODEL_DIR / "enhanced_final.pt")
    print(f"Saved final model to {MODEL_DIR / 'enhanced_final.pt'}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Mixed precision: {MIXED_PRECISION}")
    
    # Load data
    try:
        text_data = np.load(TEXT_FEATURES)
        audio_data = np.load(AUDIO_FEATURES)
        labels_data = json.load(open(LABELS_FILE))
        
        text_arr = text_data["text_features"] if "text_features" in text_data else text_data["text"]
        audio_arr = audio_data["audio_features"] if "audio_features" in audio_data else audio_data["audio"]
        labels_arr = np.array([item["label"] for item in labels_data], dtype=np.float32)
        lengths_arr = np.array([len(item) for item in text_arr], dtype=np.int32)
        cross_arr = None
        
        print(f"Text: {text_arr.shape}, Audio: {audio_arr.shape}, Labels: {labels_arr.shape}")
        
        # Train folds
        fold_aucs = train_kfold(text_arr, audio_arr, labels_arr, lengths_arr, cross_arr)
        print(f"\nEnhanced Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
        
        # Train final model
        train_final_model(text_arr, audio_arr, labels_arr, lengths_arr, cross_arr)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure feature files exist and have the correct format.")
