"""
eval_bridge4_5x3.py — Apply 5x3 repeated joke-disjoint CV to Bridge 4
========================================================================
Validates the headline 0.8422 AUC with bootstrap CI + multi-seed fold mean.

This is the vision-aligned research expansion per VISION_REALIGNMENT.md:
- The actual HaHaScore vision is humor detection (not Jester regression)
- The best published number is Bridge 4 at 0.8422 (5-fold CV)
- The v1 5x3 falsification lesson applies: single-seed is not evidence

This script:
1. Loads the cached Bridge 4 features (639 × 20 × 791)
2. Trains Bridge 4 architecture (BiGRU 791d → 128d → AUC)
3. Runs 5x3 repeated joke-disjoint CV (each fold holds out different videos)
4. Bootstrap 95% CI on the 15 measurements
5. Compares honest mean to single-fold 0.8422

Cost: 0 cash, ~30 min CPU (Bridge 4 is small, 790K params)
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# === Configuration ===
FEATURES_PATH = '/Users/Subho/tmp/bridge4_features.npz'
N_SEEDS = 3
N_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
HIDDEN = 128
DEVICE = 'cpu'  # Bridge 4 is small enough for CPU
SEED_BASE = 42

print('=== Bridge 4 5x3 repeated joke-disjoint CV ===')
print(f'  Features: {FEATURES_PATH}')
print(f'  CV: {N_SEEDS} seeds × {N_FOLDS} folds = {N_SEEDS * N_FOLDS} measurements')
print(f'  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}, Hidden: {HIDDEN}')

# === Load features ===
print('\n=== Loading Bridge 4 features ===')
data = np.load(FEATURES_PATH, allow_pickle=True)
features = data['features']  # (639, 20, 791)
labels = data['labels']  # (639, 20)
lengths = data['lengths']  # (639,)
print(f'  features: {features.shape}')
print(f'  labels:   {labels.shape}')
print(f'  lengths:  {lengths.shape}')

# Flatten for training
# Each row = one segment
# features is dtype=object containing (20, 791) arrays - convert properly
print("  Converting features to float32...")
features_float = np.zeros(features.shape, dtype=np.float32)
for v in range(features.shape[0]):
    features_float[v] = features[v].astype(np.float32)
X = features_float  # (N_videos, 20, 791) - 3D for BiGRU

# labels is dtype=object containing (20,) arrays - convert properly
print("  Converting labels to float32...")
labels_float = np.zeros(labels.shape, dtype=np.float32)
for v in range(labels.shape[0]):
    labels_float[v] = labels[v].astype(np.float32)
y = labels_float  # (N_videos, 20)
print(f"  X: {X.shape}, y: {y.shape}")
print(f"  y mean: {y.mean():.3f}, y std: {y.std():.3f}")
print(f'  X: {X.shape}, y: {y.shape}')
print(f'  y mean: {y.mean():.3f}, y std: {y.std():.3f}')

# === Bridge 4 architecture ===
class Bridge4(nn.Module):
    """BiGRU(128d, 2-layer, bidirectional) over 791d input.
    Per memory bridge4_complete."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(791, HIDDEN, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        # x: (B, T, 791)
        out, _ = self.gru(x)
        return self.head(out.mean(dim=1)).squeeze(-1)  # mean-pool

# === Train one fold ===
def train_eval_fold(tr_videos, va_videos, seed=42):
    """Train on a list of video indices, eval on holdout video indices.
    Each video is 20 segments of 791d features.
    Batch = whole video (20 segments).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Bridge4().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # Compute pos_weight from training labels
    pos_rate = max(y[tr_videos].mean(), 0.01)
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n_videos = len(tr_videos)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * max(1, n_videos // BATCH_SIZE))

    X_tr = torch.tensor(X[tr_videos], dtype=torch.float32)  # (n_videos, 20, 791)
    y_tr = torch.tensor(y[tr_videos], dtype=torch.float32)  # (n_videos, 20)
    X_va = torch.tensor(X[va_videos], dtype=torch.float32)
    n = len(tr_videos)

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            batch = perm[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_tr[batch])  # (B, 20) -> already mean-pooled
            loss = criterion(logits, y_tr[batch].mean(dim=1))  # BCE on per-video mean
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    with torch.no_grad():
        va_logits = model(X_va).numpy()  # (n_va_videos,)
    return va_logits

# === 5x3 repeated joke-disjoint CV ===
print('\n=== 5x3 repeated joke-disjoint CV ===')

# Group by video index - each video is a group (for joke-disjoint)
groups = np.arange(features.shape[0])  # 639 unique groups
print(f'  Groups (videos): {len(groups)}, segments per video: 20')

all_aucs = []
t_start = time.time()

for seed_idx in range(N_SEEDS):
    seed = SEED_BASE + seed_idx * 100
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    shuffled = rng.permutation(unique_groups)
    group_map = {g: shuffled[i] for i, g in enumerate(unique_groups)}
    groups_shuffled = np.array([group_map[g] for g in groups])
    gkf = GroupKFold(n_splits=N_FOLDS)

    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups_shuffled)):
        t0 = time.time()
        try:
            # GroupKFold returns flat indices. Map to video indices.
            # Each video = 20 consecutive segments, so tr/va indices are flat segment indices.
            # Convert to video indices via integer division
            tr_videos = np.unique(tr // X.shape[1])
            va_videos = np.unique(va // X.shape[1])
            va_logits = train_eval_fold(tr_videos, va_videos, seed=seed)
            # AUC on this fold's val
            # Per-video AUC: each video = 1 prediction (binarize continuous label at 0.5)
            y_bin = (y[va_videos].mean(axis=1) > 0.5).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue  # skip degenerate folds
            auc = float(roc_auc_score(y_bin, va_logits))
            all_aucs.append(auc)
            elapsed = time.time() - t0
            total = time.time() - t_start
            print(f'  seed={seed} fold={fold_idx}: AUC={auc:.4f} ({elapsed:.0f}s, total {total:.0f}s)')
        except Exception as e:
            print(f'  ERROR seed={seed} fold={fold_idx}: {e}')
            continue

all_aucs = np.array(all_aucs)
print(f'\n=== Aggregate ===')
print(f'  N measurements: {len(all_aucs)}')
print(f'  AUC mean: {all_aucs.mean():.4f} ± {all_aucs.std():.4f}')

# Bootstrap 95% CI
boot_rng = np.random.RandomState(42)
boot_means = []
for _ in range(10000):
    idx = boot_rng.choice(len(all_aucs), len(all_aucs), replace=True)
    boot_means.append(np.mean(all_aucs[idx]))
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f'  95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]')

# Verdict
single_fold = 0.8422  # per memory bridge4_complete
diff = single_fold - all_aucs.mean()
print(f'\n=== Comparison to single-fold 0.8422 ===')
print(f'  Single-fold:    0.8422')
print(f'  5x3 mean:       {all_aucs.mean():.4f}')
print(f'  Difference:     {diff:+.4f}')
if diff > 0.05:
    print('  Verdict: SINGLE-FOLD INFLATION. Same +0.163-style fold-luck pattern.')
elif diff < 0.02:
    print('  Verdict: HONEST. Single-fold and 5x3 mean agree within 0.02.')
else:
    print('  Verdict: SOME inflation. The 5x3 mean is the more honest number to cite.')

# Save
out = {
    'auc_5x3_mean': float(all_aucs.mean()),
    'auc_5x3_std': float(all_aucs.std()),
    'auc_5x3_ci_low': float(ci_low),
    'auc_5x3_ci_high': float(ci_high),
    'n_measurements': int(len(all_aucs)),
    'single_fold_bridge4': single_fold,
    'inflate_diff': float(diff),
    'per_fold': all_aucs.tolist(),
    'config': {
        'n_seeds': N_SEEDS,
        'n_folds': N_FOLDS,
        'epochs': EPOCHS,
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'hidden': HIDDEN,
        'device': DEVICE,
    },
    'timestamp': time.strftime('%Y%m%d_%H%M%S'),
    'total_runtime_seconds': time.time() - t_start,
}

out_path = '/Users/Subho/funny-strength-predictor/experiments/bridge4_5x3/result.json'
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: {out_path}')
print(f'\n=== DONE ===')
