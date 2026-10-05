#!/usr/bin/env python3
"""
train_jester_regression.py — v1 Canonical Path: text-only RoBERTa regression on Jester
========================================================================================
Per council realignment (Oct 5 2026), the canonical HaHaScore v1 path is:

  Jester dataset (SeppeV/rated_jokes_dataset_from_jester, ~1.76M rated jokes)
    -> RoBERTa-base regression head
      -> Continuous 0-100 humor strength prediction
        -> 5x3 repeated joke-disjoint CV via laugho_cv.repeated_joke_disjoint_cv_regression
          -> Spearman rho, MAE, RMSE with bootstrap 95% CI

This is the canonical v1 path. v2 (multimodal) is deferred per
rethink_v1 / decisions/002_RETHINK_ALPNEWR_FINDINGS.md.

Usage:
    python3 train_jester_regression.py --dry-run               # synthetic demo
    python3 train_jester_regression.py --max-samples 50000      # smoke test
    python3 train_jester_regression.py --epochs 5 --lr 2e-5    # full run (requires GPU)

Pre-registered gates (write to results.json BEFORE running):
    Spearman rho >= 0.35 AND bootstrap CI excludes 0.30 -> publishable
    95% CI spans 0 -> gate fails
    seed=42 alone yields rho > 0.50 but mean < 0.35 -> fold-luck pattern

Architecture:
    roberta-base (frozen for v1.0; LoRA r=8 in v1.1)
      -> Linear(768, 1)  # single scalar regression head
      -> MSE on continuous 0-100 (after z-normalizing Z_rating)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from laugho_cv import repeated_joke_disjoint_cv_regression


def load_jester_csv(csv_path, max_samples=None):
    """Load the Jester CSV produced by data/jester_seppev.py."""
    joke_texts, ratings, joke_ids = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            try:
                rating = float(row['rating'])
                if rating != rating:
                    continue
                joke_texts.append(row['joke_text'])
                ratings.append(rating)
                joke_ids.append(int(row['joke_id']))
            except (ValueError, KeyError):
                continue
    return joke_texts, np.array(ratings), np.array(joke_ids)


class RobertaRegressor(nn.Module):
    """RoBERTa-base + single-scalar regression head."""
    def __init__(self, model_name='roberta-base', freeze_backbone=True):
        super().__init__()
        from transformers import RobertaModel
        self.backbone = RobertaModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train_eval_fold(texts_tr, y_tr, texts_va, y_va, seed=42, epochs=3, lr=2e-5, batch_size=16):
    """Train RoBERTa regression head, return predictions on validation."""
    import os
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    from transformers import RobertaTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RobertaRegressor(model_name='roberta-base', freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * max(1, len(y_tr) // batch_size))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize
    enc_tr = tokenizer(texts_tr, return_tensors='pt', padding=True, truncation=True, max_length=128)
    enc_va = tokenizer(texts_va, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids_tr = enc_tr['input_ids'].to(device)
    attn_tr = enc_tr['attention_mask'].to(device)
    input_ids_va = enc_va['input_ids'].to(device)
    attn_va = enc_va['attention_mask'].to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).to(device)

    n = len(y_tr)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch = perm[i:i+batch_size]
            optimizer.zero_grad()
            preds = model(input_ids_tr[batch], attn_tr[batch])
            loss = F.mse_loss(preds, y_tr_t[batch])
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    with torch.no_grad():
        preds_va = []
        for i in range(0, len(y_va), batch_size):
            preds_va.append(model(input_ids_va[i:i+batch_size], attn_va[i:i+batch_size]).cpu())
        preds_va = torch.cat(preds_va).numpy()
    return preds_va


def run_dry(args):
    """Dry-run with synthetic data."""
    print('\n[DRY RUN] synthetic Jester-shaped data')
    np.random.seed(42)
    n_jokes = 50
    n_per = 4
    n = n_jokes * n_per
    joke_texts = [f'joke_{i//n_per}_{i%n_per}' for i in range(n)]
    ratings = np.random.rand(n) * 100
    joke_ids = np.repeat(np.arange(n_jokes), n_per)

    def toy_fold(tr_texts, tr_y, va_texts, va_y, seed):
        return np.full(len(va_y), tr_y.mean())

    result = repeated_joke_disjoint_cv_regression(
        joke_texts, ratings, joke_ids,
        fold_fn=toy_fold, n_seeds=args.n_seeds, n_folds=args.n_folds,
    )
    print(f'\nToy baseline (constant predictor):')
    print(f'  rho: {result["rho_mean"]:.4f}, MAE: {result["mae_mean"]:.4f}')
    print(f'  verdict: {result["verdict"]}')

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / 'dry_run_results.json'
    out_path.write_text(json.dumps(result, indent=2))
    print(f'\nSaved: {out_path}')


def run_real(args):
    """Real run with downloaded Jester + RoBERTa-base."""
    jester_csv = Path(args.jester_csv)
    if not jester_csv.exists():
        print(f'ERROR: Jester CSV not found at {jester_csv}')
        print('Run first: python3 data/jester_seppev.py --output-dir data/jester')
        sys.exit(1)

    print(f'\n[REAL RUN] Loading Jester from {jester_csv}...')
    texts, ratings, joke_ids = load_jester_csv(jester_csv, max_samples=args.max_samples)
    print(f'  Loaded {len(texts)} samples, {len(np.unique(joke_ids))} unique jokes')

    # Z-normalize ratings -> 0-100 scale
    # Jester Z_rating is roughly [-10, +10]; rescale to [0, 100]
    rmin, rmax = ratings.min(), ratings.max()
    ratings_norm = (ratings - rmin) / (rmax - rmin) * 100
    print(f'  Ratings normalized: original [{rmin:.2f}, {rmax:.2f}] -> [0, 100]')

    # Real fold_fn
    def real_fold(tr_texts, tr_y, va_texts, va_y, seed):
        return train_eval_fold(
            tr_texts, tr_y, va_texts, va_y,
            seed=seed, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )

    # 5×3 repeated joke-disjoint CV
    print(f'\nRunning 5×3 repeated CV ({args.n_seeds} seeds × {args.n_folds} folds = {args.n_seeds * args.n_folds} measurements)...')
    result = repeated_joke_disjoint_cv_regression(
        texts, ratings_norm, joke_ids,
        fold_fn=real_fold, n_seeds=args.n_seeds, n_folds=args.n_folds,
        return_per_fold=True,
    )

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(args.out_dir) / f'result_{timestamp}.json'
    out_path.write_text(json.dumps({
        **result,
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'n_seeds': args.n_seeds,
            'n_folds': args.n_folds,
            'jester_csv': str(jester_csv),
        },
        'pre_registered_gates': {
            'spearman_rho_min': 0.35,
            'ci_lower_bound_min': 0.30,
        },
        'timestamp': timestamp,
        'commit': '8dcb972 (council realignment)',
    }, indent=2))
    print(f'\nResults saved: {out_path}')
    print(f'\nFinal:')
    print(f'  Spearman rho: {result["rho_mean"]:.4f} ± {result["rho_std"]:.4f}')
    print(f'  95% CI:        [{result["rho_ci_low"]:.4f}, {result["rho_ci_high"]:.4f}]')
    print(f'  MAE:           {result["mae_mean"]:.4f}')
    print(f'  RMSE:          {result["rmse_mean"]:.4f}')
    print(f'  Verdict:       {result["verdict"]}')

    if result['rho_mean'] >= 0.35 and result['rho_ci_low'] >= 0.30:
        print(f'\n✅ PRE-REGISTERED GATE PASSED — v1 publishable!')
    else:
        print(f'\n⚠️  PRE-REGISTERED GATE NOT MET — v1 not yet publishable.')
        print(f'    rho target: 0.35; actual: {result["rho_mean"]:.4f}')
        print(f'    ci target:  ≥0.30; actual: {result["rho_ci_low"]:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Use synthetic data, no real Jester needed')
    ap.add_argument('--jester-csv', default='data/jester/jester_full.csv',
                    help='Path to Jester CSV (default: data/jester/jester_full.csv)')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='Cap Jester to N samples for smoke test')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--out-dir', default='experiments/v1_jester_regression')
    args = ap.parse_args()

    print('=' * 60)
    print('v1 HaHaScore: text-only RoBERTa regression on Jester')
    print('=' * 60)
    print(f'Config: epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}')
    print(f'CV: {args.n_seeds} seeds x {args.n_folds} folds = {args.n_seeds * args.n_folds} measurements')

    if args.dry_run:
        run_dry(args)
    else:
        run_real(args)


if __name__ == '__main__':
    main()