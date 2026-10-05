#!/usr/bin/env python3
"""
v2_laugho_train.py — Actual training pipeline for v2 LaughO
============================================================
Trains the small MLP head on AudioSet laugh family + StandUp4AI weak labels.

Inputs:
- WavLM-base-plus embeddings (768-dim) per segment
- Speaker/video IDs for speaker-disjoint CV
- Binary laughter labels (1 = laughter, 0 = non-laughter)

Architecture: 768 → 256 → 128 → 1 (Linear + LayerNorm + GELU + Dropout)
Trainable params: ~1.5M (WavLM frozen)

Loss: BCEWithLogitsLoss with pos_weight for class imbalance
Optimizer: AdamW, lr=1e-3, weight_decay=0.01
Scheduler: CosineAnnealing
Validation: 5×3 repeated speaker-disjoint CV (laugho_cv.py)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from laugho_cv import repeated_speaker_disjoint_cv, paired_comparison


class LaughOMLP(nn.Module):
    """WavLM-base-plus (768-dim) + MLP head. 1.5M params."""
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(X, y, groups, tr_idx, va_idx, seed=42):
    """Train on tr, eval on va. Returns gold-AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LaughOMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Handle class imbalance
    pos_rate = y[tr_idx].mean()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 0.001)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr = torch.tensor(X[tr_idx], dtype=torch.float32).to(device)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32).to(device)
    X_va = torch.tensor(X[va_idx], dtype=torch.float32).to(device)

    # Mini-batch training
    batch_size = 256
    n_epochs = 10
    n = len(tr_idx)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs * n)

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch = perm[i:i+batch_size]
            optimizer.zero_grad()
            logits = model(X_tr[batch])
            loss = criterion(logits, y_tr[batch])
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Eval
    model.eval()
    with torch.no_grad():
        logits = model(X_va).cpu().numpy()
    scores = 1 / (1 + np.exp(-logits))

    from sklearn.metrics import roc_auc_score
    y_va = y[va_idx]
    if len(np.unique(y_va)) < 2:
        return 0.5
    return float(roc_auc_score(y_va, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True,
                    help='path to .npz with X (N,768), y (N,), groups (N,)')
    ap.add_argument('--out-dir', default='/Users/Subho/funny-strength-predictor/experiments/v2_laugho')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()

    print('=' * 60)
    print('v2 LaughO training pipeline')
    print('=' * 60)
    print(f'Features: {args.features}')
    print(f'Output:   {args.out_dir}')

    # Load features
    print('\\nLoading features...')
    data = np.load(args.features)
    X = data['X']
    y = data['y']
    groups = data['groups'] if 'groups' in data.files else np.arange(len(X))
    print(f'  X: {X.shape}')
    print(f'  y: {y.shape}, positive rate: {y.mean():.3f}')
    print(f'  groups: {len(np.unique(groups))} unique')

    # 5×3 repeated CV
    print(f'\\nRunning {args.n_seeds}×{args.n_folds} repeated CV...')
    result = repeated_speaker_disjoint_cv(
        X, y, groups,
        fold_fn=lambda tr_X, tr_y, va_X, va_y, seed: train_one_fold(
            X, y, groups,
            np.where(np.isin(groups, np.unique(groups[np.isin(np.arange(len(X)), np.where(np.in1d(groups, np.unique(groups)[np.random.RandomState(seed).choice(len(np.unique(groups)), len(np.unique(groups)), replace=False)])[0]))[0])))[0])[0],
            np.where(np.isin(groups, [np.unique(groups)[0]]))[0],
            seed=seed
        ),
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
    )
    print(f'\\nMean AUC: {result["mean"]:.4f} ± {result["std"]:.4f}')
    print(f'95% CI:   [{result["ci_low"]:.4f}, {result["ci_high"]:.4f}]')
    print(f'Verdict: {result["verdict"]}')

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / 'v2_laugho_cv_results.json'
    out_path.write_text(json.dumps({**{k: v for k, v in result.items() if k != 'per_fold'}}, indent=2))
    print(f'\\n✓ Saved: {out_path}')


if __name__ == '__main__':
    main()