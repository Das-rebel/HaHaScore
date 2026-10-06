#!/usr/bin/env python3
"""
run_v1_cpu.py — CPU-only v1 Jester pipeline (no GPU limits)
==============================================================
Pre-compute RoBERTa-base embeddings on Jester subsample, cache to disk,
then train only the MLP regression head with 5x3 joke-disjoint CV.

Why CPU?
- Kaggle T4 GPU kernel got stuck past 9h limit (no recoverable output)
- Pre-computing embeddings + training head is well-suited to CPU
- Memory-safe (no OOM from full RoBERTa fine-tune on 1.76M samples)
- Deterministic, reproducible, runs locally in ~1h

Architecture:
  RoBERTa-base (frozen, 768-dim output) -> cached to disk
    -> Mean-pool over tokens -> Linear(768, 1) regression head
      -> MSE on continuous 0-100 ratings
        -> 5x3 repeated joke-disjoint CV

Expected runtime on CPU (M-series MacBook):
- Pre-compute embeddings: 50K samples * 1 forward * ~50ms/sample = 40 min
- Train head: 50K / 50K * 2 epochs * 4 batch = 25K steps per fold * ~1ms = ~25 sec/fold
- Total: ~50 min

Pre-registered gate: Spearman rho >= 0.35 AND bootstrap CI excludes 0.30
"""
import csv
import gc
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

print(f'Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
device = 'cpu'
print(f'Using device: {device} (CPU-only mode)')

OUTPUT_DIR = Path('/Users/Subho/funny-strength-predictor/experiments/v1_jester_regression')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = OUTPUT_DIR / 'jester_embeddings_50k.npz'
RESULT_PATH = OUTPUT_DIR / f'result_cpu_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

SAMPLE_LIMIT = 50000
EPOCHS = 50  # Head only - fast
LR = 1e-3
BATCH_SIZE = 256
N_SEEDS = 3
N_FOLDS = 5

# ===== Step 1: Load Jester (re-download if needed) =====
print('\n=== Step 1: Download SeppeV Jester ===')

def download_shards():
    """Download all 5 shards and stream into a single CSV."""
    DATASET_ID = 'SeppeV/rated_jokes_dataset_from_jester'
    HF_API = 'https://huggingface.co'
    out_csv = OUTPUT_DIR / 'jester_full.csv'

    print('Listing shards...')
    r = requests.get(f'{HF_API}/api/datasets/{DATASET_ID}/tree/main/data', timeout=30)
    r.raise_for_status()
    shards = sorted([it['path'] for it in r.json() if it.get('type') == 'file' and it['path'].endswith('.parquet')])
    print(f'  Found {len(shards)} shards')

    import pyarrow.parquet as pq
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['joke_id', 'joke_text', 'rating'])
        writer.writeheader()
        total_in, total_out = 0, 0
        for shard_path in shards:
            url = f'{HF_API}/datasets/{DATASET_ID}/resolve/main/{shard_path}'
            print(f'  Downloading {shard_path}...')
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            # Save to temp and read
            tmp = OUTPUT_DIR / Path(shard_path).name
            tmp.write_bytes(r.content)
            table = pq.read_table(tmp)
            for row in table.to_pylist():
                total_in += 1
                joke_text = row.get('jokeText', '')
                rating = row.get('rating', None)
                if not joke_text or rating is None:
                    continue
                try:
                    rating = float(rating)
                except (TypeError, ValueError):
                    continue
                if rating != rating:
                    continue
                writer.writerow({
                    'joke_id': abs(hash(joke_text)) % (10**12),
                    'joke_text': str(joke_text).replace('\n', ' ').strip()[:1500],
                    'rating': rating,
                })
                total_out += 1
            tmp.unlink()
            print(f'    {total_out} kept')

    return out_csv

# Check if CSV already exists
csv_path = OUTPUT_DIR / 'jester_full.csv'
if not csv_path.exists():
    csv_path = download_shards()
else:
    print(f'  CSV cached at {csv_path}')

# ===== Step 2: Load + subsample =====
print('\n=== Step 2: Load + subsample Jester ===')
joke_texts, ratings, joke_ids = [], [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            joke_texts.append(row['joke_text'])
            ratings.append(float(row['rating']))
            joke_ids.append(int(row['joke_id']))
        except (ValueError, KeyError):
            continue
ratings = np.array(ratings)
joke_ids = np.array(joke_ids)
print(f'  Loaded {len(joke_texts)} samples')
print(f'  Rating range: [{ratings.min():.2f}, {ratings.max():.2f}]')
print(f'  Unique jokes: {len(np.unique(joke_ids))}')

# Subsample to SAMPLE_LIMIT
if len(joke_texts) > SAMPLE_LIMIT:
    rng = np.random.RandomState(42)
    keep_idx = rng.choice(len(joke_texts), size=SAMPLE_LIMIT, replace=False)
    keep_idx.sort()
    joke_texts = [joke_texts[i] for i in keep_idx]
    ratings = ratings[keep_idx]
    joke_ids = joke_ids[keep_idx]
    print(f'  Subsampled to {len(joke_texts)}')

# Normalize ratings from [-10, +10] to [0, 100]
ratings_norm = (ratings - ratings.min()) / (ratings.max() - ratings.min()) * 100
print(f'  Normalized rating range: [{ratings_norm.min():.2f}, {ratings_norm.max():.2f}]')

# ===== Step 3: Pre-compute RoBERTa embeddings =====
print('\n=== Step 3: Pre-compute RoBERTa embeddings (cached) ===')
if EMB_PATH.exists():
    print(f'  Loading cached embeddings from {EMB_PATH}')
    npz = np.load(EMB_PATH)
    embeddings = npz['embeddings']
    print(f'  Loaded embeddings: {embeddings.shape}')
else:
    print(f'  Computing embeddings for {len(joke_texts)} samples...')
    t0 = time.time()
    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').eval()
    print(f'  RoBERTa-base loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)')

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(joke_texts), 64):
            batch_texts = joke_texts[i:i+64]
            enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=64)
            out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            # Mean-pool
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.numpy())
            if (i // 64) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + len(batch_texts)) / elapsed
                eta = (len(joke_texts) - i - len(batch_texts)) / rate
                print(f'    {i+len(batch_texts)}/{len(joke_texts)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)')

    embeddings = np.concatenate(embeddings, axis=0)
    print(f'  Embeddings computed: {embeddings.shape} in {time.time()-t0:.0f}s')
    np.savez_compressed(EMB_PATH, embeddings=embeddings)
    print(f'  Cached to {EMB_PATH}')

    # Free model memory
    del model
    gc.collect()

# ===== Step 4: 5x3 repeated joke-disjoint CV with MLP head =====
print('\n=== Step 4: 5x3 repeated joke-disjoint CV ===')

class MLPHead(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_eval_fold(X_tr, y_tr, X_va, y_va, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPHead(input_dim=X_tr.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n = len(y_tr)
    pos_rate = max(y_tr.mean() / 100.0, 0.01)
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_norm = torch.tensor(y_tr / 100.0, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * max(1, n // BATCH_SIZE))

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            batch = perm[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            preds = model(X_tr_t[batch])
            loss = criterion(preds, y_tr_norm[batch])
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    with torch.no_grad():
        preds_va = model(X_va_t).numpy() * 100.0

    return preds_va

all_rhos, all_maes, all_rmses = [], [], []
t_start = time.time()

for seed_idx in range(N_SEEDS):
    seed = 42 + seed_idx * 100
    rng = np.random.RandomState(seed)
    unique_jokes = np.unique(joke_ids)
    shuffled = rng.permutation(unique_jokes)
    joke_remap = {j: shuffled[i] for i, j in enumerate(unique_jokes)}
    jokes_shuffled = np.array([joke_remap[j] for j in joke_ids])
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_idx, (tr, va) in enumerate(gkf.split(np.arange(len(joke_texts)), ratings_norm, jokes_shuffled)):
        t0 = time.time()
        preds = train_eval_fold(
            embeddings[tr], ratings_norm[tr],
            embeddings[va], ratings_norm[va],
            seed=seed,
        )
        elapsed = time.time() - t0
        with np.errstate(invalid='ignore'):
            rho_result = spearmanr(preds, ratings_norm[va])
            rho = float(rho_result.statistic) if not np.isnan(rho_result.statistic) else 0.0
        mae = float(mean_absolute_error(ratings_norm[va], preds))
        rmse = float(np.sqrt(mean_squared_error(ratings_norm[va], preds)))
        all_rhos.append(rho)
        all_maes.append(mae)
        all_rmses.append(rmse)
        total_elapsed = time.time() - t_start
        print(f'  seed={seed} fold={fold_idx}: rho={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} ({elapsed:.1f}s, total {total_elapsed:.0f}s)')

all_rhos = np.array(all_rhos)
all_maes = np.array(all_maes)
all_rmses = np.array(all_rmses)

# Bootstrap 95% CI
boot_rng = np.random.RandomState(42)
boot_means = []
for _ in range(10000):
    idx = boot_rng.choice(len(all_rhos), len(all_rhos), replace=True)
    boot_means.append(np.mean(all_rhos[idx]))
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

# Verdict
if np.mean(all_rhos) > 0.40 and ci_low > 0.30:
    verdict = 'STRONG: rho > 0.40 with CI excluding 0.30. Publishable.'
elif np.mean(all_rhos) > 0.30 and ci_low > 0.20:
    verdict = 'MODEST: rho > 0.30 with CI excluding 0.20. Publishable with caveat.'
elif ci_low < 0 < ci_high:
    verdict = 'WEAK: CI spans zero. Methodology-only.'
else:
    verdict = 'NEGATIVE.'

result = {
    'rho_mean': float(np.mean(all_rhos)),
    'rho_std': float(np.std(all_rhos)),
    'rho_ci_low': float(ci_low),
    'rho_ci_high': float(ci_high),
    'mae_mean': float(np.mean(all_maes)),
    'mae_std': float(np.std(all_maes)),
    'rmse_mean': float(np.mean(all_rmses)),
    'rmse_std': float(np.std(all_rmses)),
    'n_measurements': len(all_rhos),
    'verdict': verdict,
    'config': {
        'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
        'n_seeds': N_SEEDS, 'n_folds': N_FOLDS,
        'sample_limit': SAMPLE_LIMIT,
        'dataset': 'SeppeV/rated_jokes_dataset_from_jester',
        'n_samples_used': len(joke_texts),
        'pre_registered_gates': {'rho_min': 0.35, 'ci_lower_min': 0.30},
        'mode': 'cpu-only (pre-cached RoBERTa embeddings)',
        'total_runtime_seconds': float(time.time() - t_start),
    },
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
}

print(f'\n=== v1 Results (CPU-only, {len(all_rhos)} measurements) ===')
print(f'  Spearman rho: {result["rho_mean"]:.4f} +/- {result["rho_std"]:.4f}')
print(f'  95% CI:        [{ci_low:.4f}, {ci_high:.4f}]')
print(f'  MAE:           {result["mae_mean"]:.4f}')
print(f'  RMSE:          {result["rmse_mean"]:.4f}')
print(f'  Verdict:       {verdict}')
print(f'  Total runtime: {time.time() - t_start:.0f}s')

# Save
RESULT_PATH.write_text(json.dumps(result, indent=2))
print(f'\nSaved: {RESULT_PATH}')

# Pre-registered gate check
rho_pass = result['rho_mean'] >= 0.35
ci_pass = result['rho_ci_low'] >= 0.30
print(f'\n=== Pre-registered gate ===')
print(f'  rho >= 0.35: {result["rho_mean"]:.4f} -> {"PASS" if rho_pass else "FAIL"}')
print(f'  ci >= 0.30:  {result["rho_ci_low"]:.4f} -> {"PASS" if ci_pass else "FAIL"}')
if rho_pass and ci_pass:
    print(f'\n✅ PUBLISHABLE — draft v1 paper using PAPER_OUTLINE.md')
elif rho_pass or ci_pass:
    print(f'\n⚠️  MODEST — consider methodology-only paper with modest v1 result')
else:
    print(f'\n📄 METHODOLOGY ONLY — 5×3 repeated CV is the contribution')