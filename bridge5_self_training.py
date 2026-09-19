#!/usr/bin/env python3
"""
Bridge 5: Self-Training (Iterative Confidence Filtering)
========================================================
Iterative self-training on Bridge 4 pseudo-labels:
  1. Use Bridge 4 to score all 639 files
  2. Keep files where ALL 20 segments are confident (|score-0.5|>0.3)
  3. Retrain Bridge 4 architecture on confident subset
  4. Repeat for 3 iterations

The intuition: remove noisy pseudo-labels where Bridge 4 is uncertain.
"""
import json, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────
AUDIO_DIR    = Path("/Users/Subho/data/standup4ai_full")
LABELS_FILE  = Path("/Users/Subho/funny-strength-predictor/data/pseudo_labels/pseudo_labels_641_v4.json")
MODEL_DIR    = Path("/Users/Subho/funny-strength-predictor/models")
CACHE_FILE   = Path("/Users/Subho/tmp/bridge4_features.npz")
BRIDGE4_PT   = MODEL_DIR / "bridge4_arc_tracker.pt"
N_SEGMENTS   = 20
HIDDEN       = 128
EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
KFOLDS       = 5
CONF_THRESH  = 0.3  # |score - 0.5| > 0.3 → confident
ITERATIONS   = 3

print(f"Device: {DEVICE}")
print(f"Python: {sys.version}")

# ── Dataset ────────────────────────────────────────────────────────────────

class HumorArcDataset(Dataset):
    """One item = one file with N segments × 791d WavLM+prosody."""

    def __init__(self, features, labels, lengths):
        # features: list of (20, 791) or (N, 791) arrays
        self.features = features
        self.labels = labels
        self.lengths = lengths

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


# ── Model (same as Bridge 4) ───────────────────────────────────────────────

class HumorArcTracker(nn.Module):
    """
    GRU over sequential segments with bidirectional processing.
    Input: (batch, seq_len, 791) — WavLM + prosody
    Output: (batch, seq_len) — per-segment humor score
    """

    def __init__(self, input_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)
        self.gru = nn.GRU(
            input_dim + 4,
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
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = torch.cat([x, pos_emb], dim=-1)
        out, _ = self.gru(x)
        out = self.head(out).squeeze(-1)
        return out


# ── Training ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_count = 0
    for features, labels, lengths in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(features, lengths)
        mask = torch.arange(features.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        loss = (criterion(preds, labels) * mask.float()).sum()
        count = mask.sum().item()
        loss = loss / count
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * count
        total_count += count
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    for features, labels, lengths in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        preds = model(features, lengths)
        mask = torch.arange(features.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        all_preds.append(preds[mask].cpu().numpy())
        all_labels.append(labels[mask].cpu().numpy())

    flat_preds  = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)
    binary_labels = (flat_labels >= 0.5).astype(float)
    if len(np.unique(binary_labels)) > 1:
        auc = roc_auc_score(binary_labels, flat_preds)
    else:
        auc = 0.5
    return auc


@torch.no_grad()
def get_all_predictions(model, features, lengths, batch_size=32):
    """Get per-segment predictions for all files."""
    model.eval()
    ds = HumorArcDataset(features, [np.zeros(l) for l in lengths], lengths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_preds = []
    for batch in loader:
        f, _, _ = batch
        f = f.to(DEVICE)
        preds = model(f, torch.tensor(lengths, device=DEVICE) if isinstance(lengths[0], int) else lengths)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)  # (N, 20)


# ── Load Data ───────────────────────────────────────────────────────────────

print("\n[0] Loading data...")
start = time.time()

with open(LABELS_FILE) as f:
    labels_data = json.load(f)
print(f"  Pseudo-labels: {len(labels_data)} files")

cached = np.load(CACHE_FILE, allow_pickle=True)
all_features = [np.asarray(f, dtype=np.float32) for f in cached['features']]
all_labels   = [np.asarray(l, dtype=np.float32) for l in cached['labels']]
all_lengths  = [int(x) for x in cached['lengths']]
print(f"  Cached features: {len(all_features)} files")
print(f"  Feature dim: {all_features[0].shape}")
print(f"  Labels range: [{min(l.min() for l in all_labels):.3f}, {max(l.max() for l in all_labels):.3f}]")
print(f"  Loaded in {time.time()-start:.1f}s")

# Build video_id → index map (pseudo-labels order)
vid_to_idx = {item['video_id']: i for i, item in enumerate(labels_data)}


# ── Helper: Filter Files by Confidence ─────────────────────────────────────

def get_confident_mask(scores, threshold=CONF_THRESH):
    """
    Return boolean array of file indices where ALL segments are confident.
    scores: list of per-segment scores (length N)
    threshold: |score - 0.5| > threshold → confident
    """
    confident_files = []
    for i, seg_scores in enumerate(scores):
        if all(abs(s - 0.5) > threshold for s in seg_scores):
            confident_files.append(i)
    return confident_files


def run_cv(features, labels, lengths, iteration, fold_models=None):
    """Run 5-fold CV and return (mean_auc, std_auc, fold_aucs)."""
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        train_ds = HumorArcDataset(
            [features[i] for i in train_idx],
            [labels[i] for i in train_idx],
            [lengths[i] for i in train_idx]
        )
        val_ds = HumorArcDataset(
            [features[i] for i in val_idx],
            [labels[i] for i in val_idx],
            [lengths[i] for i in val_idx]
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = HumorArcTracker(input_dim=791, hidden=HIDDEN, num_layers=2, dropout=0.3).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_auc   = evaluate(model, val_loader)
            scheduler.step()
            if val_auc > best_auc:
                best_auc = val_auc
                # Save best model for this fold
                torch.save(model.state_dict(), MODEL_DIR / f"bridge5_iter{iteration}_fold{fold}.pt")
            if (epoch + 1) % 10 == 0:
                print(f"      Ep {epoch+1:2d}: loss={train_loss:.4f}  val_auc={val_auc:.4f}  best={best_auc:.4f}")

        fold_aucs.append(best_auc)
        print(f"      Fold {fold+1}: best_auc={best_auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    return mean_auc, std_auc, fold_aucs


# ── Bridge 4 Baseline ───────────────────────────────────────────────────────

print(f"\n[Baseline] Bridge 4 predictions (all 639 files)...")
bridge4 = HumorArcTracker(input_dim=791, hidden=HIDDEN, num_layers=2, dropout=0.3).to(DEVICE)
bridge4.load_state_dict(torch.load(BRIDGE4_PT, map_location=DEVICE))
bridge4.eval()

# Get predictions for all files
all_features_arr = all_features  # list of (20, 791)
bridge4_preds = get_all_predictions(bridge4, all_features_arr, all_lengths)
print(f"  Bridge 4 predictions shape: {bridge4_preds.shape}")

# Compute baseline CV AUC (using original labels)
baseline_mean, baseline_std, baseline_folds = run_cv(
    all_features_arr, all_labels, all_lengths, iteration=0
)
baseline_confident = len(get_confident_mask([l for l in all_labels], threshold=CONF_THRESH))
print(f"\n{'='*60}")
print(f"Bridge 4 (baseline): {len(all_features_arr)} files, {sum(all_lengths)} segments")
print(f"  Mean AUC: {baseline_mean:.4f} ± {baseline_std:.4f}")
print(f"  Fold AUCs: {[f'{a:.4f}' for a in baseline_folds]}")


# ── Iterative Self-Training ─────────────────────────────────────────────────

# Use Bridge 4 predictions on original pseudo-labels to filter confident files
current_features = all_features_arr
current_labels   = all_labels
current_lengths  = all_lengths

# Bridge 4's confidence assessment (on original pseudo-labels)
# For iter 1, we use Bridge 4 predictions to decide confidence
# But we need to be careful: self-training means we use model predictions to filter
# → use the LATEST model to score, then filter for next iteration

results = []
results.append({
    'iteration': 0,
    'model': 'Bridge4-baseline',
    'n_files': len(current_features),
    'n_segments': sum(current_lengths),
    'n_confident_files': len(get_confident_mask([l for l in current_labels], CONF_THRESH)),
    'mean_auc': baseline_mean,
    'std_auc': baseline_std,
    'fold_aucs': baseline_folds,
})

# For iteration 1, use Bridge4 predictions to filter
# (We already have bridge4_preds computed above)
current_preds = bridge4_preds

for iteration in range(1, ITERATIONS + 1):
    print(f"\n{'='*60}")
    print(f"[Iteration {iteration}] Self-training with confident filtering")
    print(f"  Using model predictions to identify confident segments...")

    # Filter: keep files where ALL segments are confident in current_preds
    # i.e., |pred - 0.5| > CONF_THRESH for all 20 segments
    prev_preds = current_preds  # predictions from previous iteration's model
    confident_files = []
    for i in range(len(prev_preds)):
        seg_scores = prev_preds[i]  # (20,) predicted scores
        if all(abs(s - 0.5) > CONF_THRESH for s in seg_scores):
            confident_files.append(i)

    n_confident = len(confident_files)
    print(f"  Confident files (all 20 segs |pred-0.5|>{CONF_THRESH}): {n_confident}/{len(current_features)}")

    if n_confident < KFOLDS:
        print(f"  WARNING: Too few confident files ({n_confident}) for {KFOLDS}-fold CV!")
        break

    # Subset
    iter_features = [current_features[i] for i in confident_files]
    iter_labels   = [current_labels[i]   for i in confident_files]
    iter_lengths  = [current_lengths[i]  for i in confident_files]

    # Run CV on confident subset
    print(f"  Running {KFOLDS}-fold CV on {n_confident} confident files...")
    iter_mean, iter_std, iter_folds = run_cv(
        iter_features, iter_labels, iter_lengths, iteration=iteration
    )

    print(f"\n  Iteration {iteration} Results:")
    print(f"    Files: {n_confident} (filtered from {len(current_features)})")
    print(f"    Mean AUC: {iter_mean:.4f} ± {iter_std:.4f}")
    print(f"    Fold AUCs: {[f'{a:.4f}' for a in iter_folds]}")

    results.append({
        'iteration': iteration,
        'model': f'Bridge5-iter{iteration}',
        'n_files': n_confident,
        'n_segments': sum(iter_lengths),
        'n_confident_files': n_confident,
        'mean_auc': iter_mean,
        'std_auc': iter_std,
        'fold_aucs': iter_folds,
    })

    # Train final model for next iteration's predictions
    print(f"\n  Training final model (iter {iteration}) on {n_confident} files...")
    full_ds = HumorArcDataset(iter_features, iter_labels, iter_lengths)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    final_model = HumorArcTracker(input_dim=791, hidden=HIDDEN, num_layers=2, dropout=0.3).to(DEVICE)
    optimizer   = torch.optim.Adam(final_model.parameters(), lr=LR)
    criterion   = nn.BCELoss()
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(final_model, full_loader, optimizer, criterion)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Ep {epoch+1:2d}: loss={train_loss:.4f}")

    model_path = MODEL_DIR / f"bridge5_iter{iteration}.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"  Saved to {model_path}")

    # Use this model to generate predictions for ALL files for next iteration's filtering
    current_preds = get_all_predictions(final_model, all_features_arr, all_lengths)
    print(f"  Generated predictions for next iteration filtering")
    # Note: confident_files are always indices into all_features_arr (full 639)
    # We intentionally keep current_preds as the only evolving state


# ── Print Comparison Table ─────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"{'='*70}")
print(f"BRIDGE 5 SELF-TRAINING RESULTS")
print(f"{'='*70}")
print(f"{'Iteration':>10} | {'Confident':>12} | {'Mean AUC':>10} | {'Std':>6} | {'Fold AUCs'}")
print(f"{'-'*70}")
for r in results:
    n_conf = r['n_confident_files']
    auc = r['mean_auc']
    std = r['std_auc']
    folds = ', '.join([f'{a:.3f}' for a in r['fold_aucs']])
    label = f"Bridge 4 baseline" if r['iteration'] == 0 else f"  Iter {r['iteration']}"
    print(f"{label:>10} | {n_conf:>12} | {auc:>10.4f} | {std:>6.4f} | {folds}")
print(f"{'='*70}")

# Also print file counts
print(f"\nFiles per iteration:")
for r in results:
    print(f"  Iter {r['iteration']}: {r['n_files']} files, {r['n_segments']} segments")

# ── Save Results ────────────────────────────────────────────────────────────

results_out = MODEL_DIR / "bridge5_results.json"
with open(results_out, 'w') as f:
    json.dump({
        'config': {
            'conf_threshold': CONF_THRESH,
            'iterations': ITERATIONS,
            'kfolds': KFOLDS,
            'epochs': EPOCHS,
            'hidden': HIDDEN,
            'batch_size': BATCH_SIZE,
            'lr': LR,
        },
        'results': [
            {
                'iteration': r['iteration'],
                'model': r['model'],
                'n_files': r['n_files'],
                'n_segments': r['n_segments'],
                'n_confident_files': r['n_confident_files'],
                'mean_auc': float(r['mean_auc']),
                'std_auc': float(r['std_auc']),
                'fold_aucs': [float(a) for a in r['fold_aucs']],
            }
            for r in results
        ]
    }, f, indent=2)
print(f"\nResults saved to {results_out}")

# ── Analysis ────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"ANALYSIS")
print(f"{'='*70}")
baseline_auc = results[0]['mean_auc']
best_iter = max(results[1:], key=lambda r: r['mean_auc'])
best_auc  = best_iter['mean_auc']

print(f"Baseline (Bridge 4, all 639 files): AUC = {baseline_auc:.4f}")
for r in results[1:]:
    delta = r['mean_auc'] - baseline_auc
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"  Iter {r['iteration']}: AUC = {r['mean_auc']:.4f} ({arrow}{abs(delta):.4f})  [{r['n_files']} files]")

if best_auc > baseline_auc:
    print(f"\n✓ Self-training HELPS: best AUC = {best_auc:.4f} at iteration {best_iter['iteration']} (+{best_auc-baseline_auc:.4f} vs baseline)")
    print(f"  Using {best_iter['n_files']} confident files (vs 639 baseline)")
else:
    print(f"\n✗ Self-training does NOT help: best AUC = {best_auc:.4f} at iteration {best_iter['iteration']}")
    print(f"  All iterations perform {'worse' if best_auc < baseline_auc else 'same as'} baseline")
    if best_auc < baseline_auc:
        print(f"\n  Possible explanations:")
        print(f"  1. Confidence filtering removes useful diversity/edge-case patterns")
        print(f"  2. Bridge 4 pseudo-labels are already well-calibrated")
        print(f"  3. The confident subset has biased distribution (skewed toward extremes)")
        print(f"  4. GRU benefits from full-file context even at low-confidence segments")
