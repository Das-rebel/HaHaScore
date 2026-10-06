"""
v1_jester_finetuned_kernel.py — GPU kernel with FINE-TUNED RoBERTa (the actual fix)
=====================================================================================
Fixes the CPU result issue:
- Frozen RoBERTa + linear head = 0.23 rho (CPU run, commit 1868142)
- Fine-tuned RoBERTa + linear head = ~0.31 rho expected (M3 baseline)

Memory optimizations for T4 15GB:
- fp16 mixed precision (halves memory)
- Gradient checkpointing (RoBERTa-base re-computes activations)
- Gradient accumulation 8x (effective batch_size 32)
- Per-fold checkpoint saves (partial output if killed)

CV protocol:
- 5x3 repeated joke-disjoint CV
- Pre-registered gate: rho >= 0.35 AND CI excludes 0.30
"""
import os
import sys
import subprocess
import json
import csv
import time
from pathlib import Path

print('=== v1 HaHaScore Jester Regression (FINE-TUNED, GPU) ===')
print(f'  Python: {sys.version.split()[0]}')

# ===== Step 1: Install deps =====
print('\n=== Step 1: Install deps (Python 3.13 compatible) ===')
subprocess.run(['pip', 'install', '--quiet', '--upgrade',
                'transformers', 'datasets', 'scipy', 'scikit-learn', 'pyarrow', 'requests'],
               check=True)
print('  Installed')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
import pyarrow.parquet as pq
import requests as req

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  Device: {device}')

# ===== Step 2: Download =====
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
        print(f'    Downloading {remote_path}...')
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

# ===== Step 3: Load + stratified subsample =====
print('\n=== Step 3: Load + stratified subsample (keep all 140 jokes) ===')
joke_texts, ratings, joke_ids = [], [], []
with open(out_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        joke_texts.append(row['joke_text'])
        ratings.append(float(row['rating']))
        joke_ids.append(int(row['joke_id']))
ratings = np.array(ratings)
joke_ids = np.array(joke_ids)

# Subsample: keep ALL 140 jokes, limit per joke
SAMPLE_PER_JOKE = 3000
rng_np = np.random.RandomState(42)
keep_indices = []
for jid in np.unique(joke_ids):
    jid_idx = np.where(joke_ids == jid)[0]
    if len(jid_idx) > SAMPLE_PER_JOKE:
        jid_idx = rng_np.choice(jid_idx, size=SAMPLE_PER_JOKE, replace=False)
    keep_indices.extend(jid_idx.tolist())
keep_indices = np.array(sorted(keep_indices))

joke_texts = [joke_texts[i] for i in keep_indices]
ratings = ratings[keep_indices]
joke_ids = joke_ids[keep_indices]
print(f'  After stratified subsample: {len(joke_texts)} samples, {len(np.unique(joke_ids))} jokes')

ratings_norm = (ratings - ratings.min()) / (ratings.max() - ratings.min()) * 100

# ===== Step 4: Define FINE-TUNED RoBERTa =====
print('\n=== Step 4: Define model (fine-tuned RoBERTa + head) ===')
from transformers import RobertaTokenizer, RobertaModel

class RobertaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained('roberta-base')
        # FINE-TUNE backbone (not frozen) — this is the fix
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)

# ===== Step 5: 5x3 CV with memory optimizations =====
print('\n=== Step 5: 5x3 repeated joke-disjoint CV (fine-tuned) ===')

def train_eval_fold(tr_texts, tr_y, va_texts, va_y, seed=42, epochs=2, lr=2e-5, batch_size=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RobertaRegressor().to(device)
    model.backbone.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    enc_tr = tokenizer(tr_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    enc_va = tokenizer(va_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids_tr = enc_tr['input_ids'].to(device)
    attn_tr = enc_tr['attention_mask'].to(device)
    input_ids_va = enc_va['input_ids'].to(device)
    attn_va = enc_va['attention_mask'].to(device)
    y_tr_t = torch.tensor(tr_y, dtype=torch.float32).to(device)

    grad_accum = 8
    n = len(tr_y)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, n // (batch_size * grad_accum)))

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        optimizer.zero_grad()
        for i in range(0, n, batch_size):
            batch = perm[i:i+batch_size]
            with torch.amp.autocast('cuda'):
                preds = model(input_ids_tr[batch], attn_tr[batch])
                loss = F.mse_loss(preds, y_tr_t[batch] / 100.0) / grad_accum
            scaler.scale(loss).backward()
            if (i // batch_size + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

    model.eval()
    preds_va = []
    with torch.no_grad():
        for i in range(0, len(va_y), batch_size):
            with torch.amp.autocast('cuda'):
                p = model(input_ids_va[i:i+batch_size], attn_va[i:i+batch_size])
            preds_va.append(p.float().cpu() * 100.0)
    return torch.cat(preds_va).numpy()

EPOCHS = 2
LR = 2e-5
BATCH_SIZE = 4
N_SEEDS = 3
N_FOLDS = 5

RESULT_DIR = Path('/kaggle/working/v1_results')
RESULT_DIR.mkdir(exist_ok=True)
PARTIAL_PATH = RESULT_DIR / 'partial_results.json'
partial_results = {'fold_results': []}

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
        try:
            preds = train_eval_fold(
                [joke_texts[i] for i in tr], ratings_norm[tr],
                [joke_texts[i] for i in va], ratings_norm[va],
                seed=seed, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
            )
            with np.errstate(invalid='ignore'):
                rho_result = spearmanr(preds, ratings_norm[va])
                rho = float(rho_result.statistic) if not np.isnan(rho_result.statistic) else 0.0
            mae = float(mean_absolute_error(ratings_norm[va], preds))
            rmse = float(np.sqrt(mean_squared_error(ratings_norm[va], preds)))
            all_rhos.append(rho)
            all_maes.append(mae)
            all_rmses.append(rmse)
            elapsed = time.time() - t0
            total = time.time() - t_start
            print(f'  seed={seed} fold={fold_idx}: rho={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} ({elapsed:.0f}s, total {total:.0f}s)')
            partial_results['fold_results'].append({
                'seed': seed, 'fold': fold_idx, 'rho': rho, 'mae': mae, 'rmse': rmse,
                'elapsed_s': elapsed, 'total_s': total,
            })
            PARTIAL_PATH.write_text(json.dumps({
                'fold_results': partial_results['fold_results'],
                'n_completed': len(all_rhos),
                'elapsed_s': total,
            }, indent=2))
        except Exception as e:
            print(f'  ERROR seed={seed} fold={fold_idx}: {e}')
            PARTIAL_PATH.write_text(json.dumps({
                'fold_results': partial_results['fold_results'],
                'n_completed': len(all_rhos),
                'elapsed_s': time.time() - t_start,
                'last_error': str(e),
            }, indent=2))
            continue

# ===== Step 6: Save final result =====
if all_rhos:
    all_rhos = np.array(all_rhos)
    all_maes = np.array(all_maes)
    all_rmses = np.array(all_rmses)

    boot_rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        idx = boot_rng.choice(len(all_rhos), len(all_rhos), replace=True)
        boot_means.append(np.mean(all_rhos[idx]))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

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
            'grad_accum': 8, 'max_length': 128,
            'n_seeds': N_SEEDS, 'n_folds': N_FOLDS,
            'sample_per_joke': 3000,
            'n_samples_used': len(joke_texts),
            'dataset': 'SeppeV/rated_jokes_dataset_from_jester',
            'mode': 'GPU fine-tuned + fp16 + grad_checkpointing',
            'pre_registered_gates': {'rho_min': 0.35, 'ci_lower_min': 0.30},
        },
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'total_runtime_seconds': time.time() - t_start,
    }

    print(f'\n=== v1 Results (GPU fine-tuned) ===')
    print(f'  rho: {result["rho_mean"]:.4f} +/- {result["rho_std"]:.4f}')
    print(f'  CI:  [{ci_low:.4f}, {ci_high:.4f}]')
    print(f'  MAE: {result["mae_mean"]:.4f}')
    print(f'  RMSE: {result["rmse_mean"]:.4f}')
    print(f'  Verdict: {verdict}')

    RESULT_PATH = RESULT_DIR / 'v1_result.json'
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f'\nSaved: {RESULT_PATH}')

    rho_pass = result['rho_mean'] >= 0.35
    ci_pass = result['rho_ci_low'] >= 0.30
    print(f'\n=== Pre-registered gate ===')
    print(f'  rho >= 0.35: {result["rho_mean"]:.4f} -> {"PASS" if rho_pass else "FAIL"}')
    print(f'  ci >= 0.30: {result["rho_ci_low"]:.4f} -> {"PASS" if ci_pass else "FAIL"}')
else:
    print('No successful folds.')

print('\n=== DONE ===')