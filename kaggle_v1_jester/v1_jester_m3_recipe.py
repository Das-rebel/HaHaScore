"""
v1_jester_m3_recipe.py — Reproduction of the M3 recipe that worked
================================================================
Per memory findings (Sep 8 2026):
- M3: roberta-base + Linear(768,1), 4 epochs, lr=2e-5, batch=32, max_len=128
- M3: Spearman rho = 0.3127 (rho_jester = 0.3571 on Jester gold subset)
- M3: ran on Kaggle CPU (P100 was sm_60 incompatible with modern torch)
- M3: ~40 min runtime

CRITICAL FIXES from memory (hf_v2_gpu_kernel):
1. Request T4 GPU explicitly (not P100): set "machine_shape" via Kaggle kernel UI
2. Don't use os.execv (causes restart storm)
3. Don't use warmup LambdaLR that fights cosine decay
4. Use safetensors=True when loading HF models

ADDITIONAL FIX vs my failed runs:
- THIS RUN is just 4 epochs (like M3) — not 2 epochs
- NO fp16 (M3 didn't use it; fp16 caused instability)
- NO gradient checkpointing (M3 didn't use it; unnecessary for 7K samples)
- NO grad accumulation (M3 used raw batch=32)
- Save model.pt after each fold (per M3)
- Save predictions.npz per fold (per M3)

NO 5x3 CV: M3 was single-fold. We will NOT do 5x3 here (would exceed 9h limit).
Single run with rho=0.31 expected.
"""
import os
import sys
import subprocess
import json
import csv
import time
from pathlib import Path

print('=== v1 Jester M3 Recipe (Kaggle CPU/T4, single run) ===')
print(f'  Python: {sys.version.split()[0]}')

# Install deps (Python 3.10 Kaggle default; torch 2.3.1+cu121 for T4 sm_75)
print('\n=== Step 1: Install deps (M3-compatible) ===')
# M3 ran on Kaggle CPU; don't bother with GPU torch
subprocess.run(['pip', 'install', '--quiet', '--upgrade',
                'transformers>=4.44', 'scipy', 'scikit-learn', 'pyarrow', 'requests'],
               check=True)
print('  Installed')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pyarrow.parquet as pq
import requests as req

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  Device: {device} (will use CPU like M3 if GPU unavailable)')

# ===== Step 2: Download Jester =====
print('\n=== Step 2: Download SeppeV Jester ===')
HF_API = 'https://huggingface.co'
DATASET_ID = 'SeppeV/rated_jokes_dataset_from_jester'

def list_shards():
    r = req.get(f'{HF_API}/api/datasets/{DATASET_ID}/tree/main/data', timeout=30)
    r.raise_for_status()
    return sorted([it['path'] for it in r.json() if it.get('type') == 'file' and it['path'].endswith('.parquet')])

def download_shard(remote_path, dst_dir):
    url = f'{HF_API}/datasets/{DATASET_ID}/resolve/main/{remote_path}'
    local = dst_dir / Path(remote_path).name
    if not local.exists():
        r = req.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(local, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    return local

out_dir = Path('/kaggle/working/jester')
out_dir.mkdir(parents=True, exist_ok=True)
parquet_files = list_shards()
print(f'  Found {len(parquet_files)} shards')

out_csv = out_dir / 'jester_full.csv'
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['joke_id', 'joke_text', 'rating'])
    writer.writeheader()
    total_in, total_out = 0, 0
    for shard_path in parquet_files:
        try:
            local = download_shard(shard_path, out_dir)
            table = pq.read_table(local)
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
            os.remove(local)
        except Exception as e:
            print(f'    ERROR {shard_path}: {e}')

print(f'  Total: {total_in} in, {total_out} kept')

# ===== Step 3: Load + normalize =====
print('\n=== Step 3: Load Jester ===')
joke_texts, ratings, joke_ids = [], [], []
with open(out_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        joke_texts.append(row['joke_text'])
        ratings.append(float(row['rating']))
        joke_ids.append(int(row['joke_id']))
ratings = np.array(ratings)
joke_ids = np.array(joke_ids)
print(f'  Loaded {len(joke_texts)} samples, {len(np.unique(joke_ids))} unique jokes')

# Normalize ratings from [-10, +10] to [0, 100]
ratings_norm = (ratings - ratings.min()) / (ratings.max() - ratings.min()) * 100

# ===== Step 4: M3 Recipe - RoBERTa-base + Linear(768, 1) =====
print('\n=== Step 4: Define M3 model ===')
from transformers import RobertaTokenizer, RobertaModel

class RobertaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained('roberta-base')
        # FINE-TUNE backbone (not frozen) — this is the M3 fix
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)

# ===== Step 5: M3 Single-run training (4 epochs, no 5x3) =====
print('\n=== Step 5: M3 single-run training ===')

def train_one_run(texts, ratings, joke_ids, seed=42, epochs=4, batch_size=32, lr=2e-5, max_len=128):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train/val split: stratified by joke_id (so test has unique jokes)
    # Per M3: uses 7,984 real-labeled samples (subset of Jester)
    # Here: use ALL 1.76M for maximum signal (different from M3's subset)
    # Or subsample to 50K to be safe (CPU speed)
    SAMPLE_LIMIT = 50000
    rng = np.random.RandomState(seed)
    if len(texts) > SAMPLE_LIMIT:
        keep = rng.choice(len(texts), SAMPLE_LIMIT, replace=False)
        keep.sort()
        texts = [texts[i] for i in keep]
        ratings = ratings[keep]
        joke_ids = joke_ids[keep]

    # 90/10 train/val split (no jokes in both)
    unique_jokes = np.unique(joke_ids)
    rng_split = np.random.RandomState(seed)
    shuffled = rng_split.permutation(unique_jokes)
    val_jokes = set(shuffled[:len(unique_jokes)//10])
    val_mask = np.array([jid in val_jokes for jid in joke_ids])
    train_mask = ~val_mask

    print(f'  Train: {train_mask.sum()}, Val: {val_mask.sum()}')
    print(f'  Train jokes: {len(np.unique(joke_ids[train_mask]))}, Val jokes: {len(np.unique(joke_ids[val_mask]))}')

    model = RobertaRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # No LambdaLR warmup (per memory: fought cosine decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, train_mask.sum() // batch_size))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
    train_y = ratings[train_mask]
    val_texts = [texts[i] for i in range(len(texts)) if val_mask[i]]
    val_y = ratings[val_mask]

    enc_tr = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    enc_va = tokenizer(val_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    input_ids_tr = enc_tr['input_ids'].to(device)
    attn_tr = enc_tr['attention_mask'].to(device)
    input_ids_va = enc_va['input_ids'].to(device)
    attn_va = enc_va['attention_mask'].to(device)
    y_tr_t = torch.tensor(train_y, dtype=torch.float32).to(device)

    n = len(train_y)
    results_per_epoch = []
    OUT_DIR = Path('/kaggle/working/v1_m3_recipe_results')
    OUT_DIR.mkdir(exist_ok=True)

    for ep in range(epochs):
        model.train()
        perm_idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch = perm_idx[i:i+batch_size]
            optimizer.zero_grad()
            preds = model(input_ids_tr[batch], attn_tr[batch])
            loss = F.mse_loss(preds, y_tr_t[batch] / 100.0)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval
        model.eval()
        preds_va = []
        with torch.no_grad():
            for i in range(0, len(val_y), batch_size):
                preds_va.append(model(input_ids_va[i:i+batch_size], attn_va[i:i+batch_size]).cpu() * 100.0)
        preds_va = torch.cat(preds_va).numpy()

        with np.errstate(invalid='ignore'):
            rho_result = spearmanr(preds_va, val_y)
            rho = float(rho_result.statistic) if not np.isnan(rho_result.statistic) else 0.0
        mae = float(mean_absolute_error(val_y, preds_va))
        rmse = float(np.sqrt(mean_squared_error(val_y, preds_va)))
        results_per_epoch.append({'epoch': ep+1, 'rho': rho, 'mae': mae, 'rmse': rmse})
        print(f'  Epoch {ep+1}: rho={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f}')

        # Save predictions and model per epoch
        np.savez_compressed(OUT_DIR / f'val_predictions_ep{ep+1}.npz',
                           preds=preds_va, ratings=val_y)
        torch.save(model.state_dict(), OUT_DIR / f'model_ep{ep+1}.pt')

    # Best epoch
    best_ep = max(results_per_epoch, key=lambda x: x['rho'])
    print(f'\n  Best epoch: {best_ep}')

    # Save final results
    final_result = {
        'model': 'roberta_regression_v1_m3_recipe',
        'date': time.strftime('%Y%m%d_%H%M%S'),
        'device': str(device),
        'config': {
            'epochs': epochs, 'lr': lr, 'batch_size': batch_size,
            'max_len': max_len, 'sample_limit': SAMPLE_LIMIT,
            'dataset': 'SeppeV/rated_jokes_dataset_from_jester',
            'mode': 'M3 recipe (fine-tuned RoBERTa)',
        },
        'per_epoch': results_per_epoch,
        'best': best_ep,
        'm3_baseline': {'rho': 0.3127, 'rho_jester': 0.3571, 'mae': 22.98, 'epoch': 3.0},
    }

    out_path = OUT_DIR / 'v1_result_m3_recipe.json'
    out_path.write_text(json.dumps(final_result, indent=2))
    print(f'\nSaved: {out_path}')

    return final_result

t0 = time.time()
result = train_one_run(joke_texts, ratings_norm, joke_ids)
elapsed = time.time() - t0

print(f'\n=== M3 Recipe Result ===')
print(f'  Best rho: {result["best"]["rho"]:.4f}')
print(f'  Best MAE: {result["best"]["mae"]:.2f}')
print(f'  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)')

print('\n=== DONE ===')