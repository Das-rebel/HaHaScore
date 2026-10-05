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
    python3 train_jester_regression.py --dry-run        # toy data demo
    python3 train_jester_regression.py --max-samples 50000    # smoke test on subset
    python3 train_jester_regression.py --epochs 5      # full run on Jester

Status: SKELETON. Real run requires (1) Jester download, (2) GPU or Colab T4.
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
from laugho_cv import repeated_joke_disjoint_cv_regression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Use synthetic data, no real Jester needed')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='Cap Jester to N samples for smoke test')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch-size', type=int, default=32)
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
        return run_dry(args)
    else:
        return run_real(args)


def run_dry(args):
    """Dry run with synthetic data to verify the pipeline."""
    print('\n[DRY RUN] synthetic data...')
    np.random.seed(42)
    n_jokes, n_per_joke = 50, 4
    n_total = n_jokes * n_per_joke
    joke_texts = [f'joke_{i//n_per_joke}_{i%n_per_joke}' for i in range(n_total)]
    ratings = np.random.rand(n_total) * 100
    joke_ids = np.repeat(np.arange(n_jokes), n_per_joke)

    def toy_fold(tr_texts, tr_y, va_texts, va_y, seed):
        """Toy predictor: predicts mean + 0 noise (baseline)."""
        return np.full(len(va_y), tr_y.mean())

    result = repeated_joke_disjoint_cv_regression(
        joke_texts, ratings, joke_ids,
        fold_fn=toy_fold,
        n_seeds=args.n_seeds, n_folds=args.n_folds,
    )
    print(f'\nToy baseline (constant mean predictor):')
    print(f'  rho: {result["rho_mean"]:.4f} ± {result["rho_std"]:.4f}')
    print(f'  CI:  [{result["rho_ci_low"]:.4f}, {result["rho_ci_high"]:.4f}]')
    print(f'  MAE: {result["mae_mean"]:.4f}')
    print(f'  verdict: {result["verdict"]}')
    print('\nExpected: rho ≈ 0 (constant predictor is uninformative)')
    print('Real v1 should achieve rho > 0.30 (TF-IDF baseline) to > 0.35 target')

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / 'dry_run_results.json'
    out_path.write_text(json.dumps(result, indent=2))
    print(f'\nSaved: {out_path}')


def run_real(args):
    """Real run with Jester dataset. Requires GPU + downloaded Jester data."""
    print('\n[REAL RUN] requires Jester data + GPU...')
    print('TODO: not yet implemented')
    print('\nSteps to enable:')
    print('  1. Download SeppeV/rated_jokes_dataset_from_jester (1.76M jokes)')
    print('     python3 -c "from datasets import load_dataset; ds = load_dataset(\'SeppeV/rated_jokes_dataset_from_jester\')"')
    print('  2. Train RoBERTa-base regression head:')
    print('     from transformers import RobertaTokenizer, RobertaForSequenceClassification')
    print('     model = RobertaForSequenceClassification.from_pretrained(\'roberta-base\', num_labels=1)')
    print('  3. Apply repeated_joke_disjoint_cv_regression (already imported)')
    print('  4. Save to experiments/v1_jester_regression/result_<timestamp>.json')
    return 1


if __name__ == '__main__':
    main()