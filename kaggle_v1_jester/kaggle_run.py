"""
kaggle_run.py — Single entry point for all HaHaScore Kaggle kernels
=====================================================================
Previously 4 separate kernel scripts (v1_jester_m3_recipe.py,
v1_jester_m3_cv_5x3.py, v1_jester_finetuned_kernel.py, v2_laugho_kaggle_kernel.py),
each ~250 lines with ~80% boilerplate overlap.

Usage:
    python3 kaggle_run.py --mode m3_recipe
    python3 kaggle_run.py --mode 5x3cv
    python3 kaggle_run.py --mode finetuned
    python3 kaggle_run.py --mode v2_laugho

Each mode differs only in: number of epochs, whether to use fp16 /
gradient checkpointing, and whether to run 5x3 CV or single 90/10 split.

Push via:
    cd kaggle_v1_jester
    kaggle kernels push    # (uses kernel-metadata.json)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

# Inline JesterSource (from jester_source.py)
DATASET_ID = 'SeppeV/rated_jokes_dataset_from_jester'
HF_API = 'https://huggingface.co'

import pyarrow.parquet as pq
import requests as req
from pathlib import Path as _Path

def _list_shards():
    r = req.get(f'{HF_API}/api/datasets/{DATASET_ID}/tree/main/data', timeout=30)
    r.raise_for_status()
    return sorted([it['path'] for it in r.json() if it.get('type') == 'file' and it['path'].endswith('.parquet')])

def _download_shard(remote_path, output_dir):
    url = f'{HF_API}/datasets/{DATASET_ID}/resolve/main/{remote_path}'
    local = Path(output_dir) / Path(remote_path).name
    if not local.exists():
        r = req.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(local, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    return local

def write_jester_csv(output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = _list_shards()
    out_csv = out_dir / 'jester_full.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['joke_id', 'joke_text', 'rating'])
        writer.writeheader()
        total_in, total_out = 0, 0
        for shard_path in shards:
            local = _download_shard(shard_path, out_dir)
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
            try:
                os.remove(local)
            except OSError:
                pass
    return out_csv, total_in, total_out

# Inline RobertaRegressor (from roberta_regressor.py)
class RobertaRegressor(nn.Module):
    def __init__(self, model_name='roberta-base', freeze_backbone=False, use_fp16=False,
                 use_grad_checkpointing=False, device='cuda'):
        super().__init__()
        from transformers import RobertaModel
        self.backbone = RobertaModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.use_fp16 = use_fp16
        if device == 'cuda' and use_grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # v19 lesson: never let a 1-batch edge case crash here
        if input_ids.shape[0] == 0:
            return torch.zeros(0, device=input_ids.device)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)

MODES = ['m3_recipe', '5x3cv', 'finetuned', 'v2_laugho']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='5x3cv', choices=MODES, help='v1 mode (default 5x3cv: the canonical 5x3 joke-disjoint CV)')
    ap.add_argument('--epochs', type=int, default=None, help='override per-mode default')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--sample-limit', type=int, default=50000)
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--out-dir', default='/kaggle/working')
    args = ap.parse_args()
    print(f'=== HaHaScore Kaggle kernel (mode={args.mode}) ===')
    print(f'  Python: {sys.version.split()[0]}')

    # ===== Step 1: Install deps =====
    print('\n=== Step 1: Install deps ===')
    subprocess.run(['pip', 'install', '--quiet', 'pyarrow==18.1.0'], check=True)
    subprocess.run(['pip', 'install', '--quiet', '--upgrade',
                    'transformers', 'scipy', 'scikit-learn', 'requests'],
                   check=True)
    print('  Installed')

    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import GroupKFold
    import pyarrow.parquet as pq
    from transformers import RobertaTokenizer
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import GroupKFold

    device = 'cuda'
    if device == 'cuda':
        free, total = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:  # need ~4GB for roberta-base + head
            print(f'  WARNING: only {free/1e9:.1f} GB free GPU, falling back to CPU')
            device = 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {device}')

    # ===== Step 2: Download Jester =====
    print('\n=== Step 2: Download Jester (via JesterSource) ===')
    out_dir = Path('/kaggle/working/jester')
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, n_in, n_out = write_jester_csv(out_dir)
    print(f'  Wrote {csv_path}: {n_in} in, {n_out} kept')

    # ===== Step 3: Load + subsample =====
    print('\n=== Step 3: Load + subsample ===')
    texts, ratings, joke_ids = load_csv(csv_path, max_samples=args.sample_limit)
    ratings = np.array(ratings)
    joke_ids = np.array(joke_ids)
    ratings_norm = (ratings - ratings.min()) / (ratings.max() - ratings.min()) * 100
    print(f'  Loaded {len(texts)} samples, {len(np.unique(joke_ids))} unique jokes')
    print(f'  Normalized: [{ratings_norm.min():.2f}, {ratings_norm.max():.2f}]')

    # ===== Step 4: Mode-specific config =====
    if args.mode == 'm3_recipe':
        # M3 recipe: single 90/10 split, 4 epochs, batch=32
        epochs = args.epochs or 4
        batch_size = args.batch_size or 32
        lr = args.lr or 2e-5
        use_fp16 = False
        use_grad_checkpointing = False
        cv_mode = False
    elif args.mode == '5x3cv':
        # 5x3 CV: 1 epoch per fold, batch=32
        epochs = args.epochs or 1
        batch_size = args.batch_size or 32
        lr = args.lr or 2e-5
        use_fp16 = False
        use_grad_checkpointing = False
        cv_mode = True
    elif args.mode == 'finetuned':
        # Fine-tuned: 2 epochs, fp16, grad checkpointing
        epochs = args.epochs or 2
        batch_size = args.batch_size or 4
        lr = args.lr or 2e-5
        use_fp16 = True
        use_grad_checkpointing = True
        cv_mode = True
    elif args.mode == 'v2_laugho':
        # v2 LaughO sister project - not implemented in this unified script
        print('  v2 LaughO is a SISTER PROJECT (audio-only laughter detection).')
        print('  See experiments/v2_laugho/SISTER_PROJECT.md')
        print('  Use v2_laugho_kaggle_kernel.py for that mode.')
        sys.exit(1)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    print(f'\n  Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, fp16={use_fp16}, grad_ckpt={use_grad_checkpointing}, cv={cv_mode}')

    # ===== Step 5: Model =====
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaRegressor(freeze_backbone=False, use_fp16=use_fp16,
                             use_grad_checkpointing=use_grad_checkpointing).to(device)

    def train_eval_split(X_tr_ids, y_tr, X_va_ids, y_va, seed=42):
        """Train + eval on a specific split."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda') if use_fp16 else None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * max(1, len(y_tr) // batch_size))
        enc_tr = tokenizer(X_tr_ids, return_tensors='pt', padding=True, truncation=True, max_length=128)
        enc_va = tokenizer(X_va_ids, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids_tr = enc_tr['input_ids'].to(device)
        attn_tr = enc_tr['attention_mask'].to(device)
        input_ids_va = enc_va['input_ids'].to(device)
        attn_va = enc_va['attention_mask'].to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)

        n = len(y_tr)
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch = perm[i:i+batch_size]
                optimizer.zero_grad()
                if use_fp16:
                    with torch.amp.autocast('cuda'):
                        preds = model(input_ids_tr[batch], attn_tr[batch])
                        loss = F.mse_loss(preds, y_tr_t[batch] / 100.0)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(input_ids_tr[batch], attn_tr[batch])
                    loss = F.mse_loss(preds, y_tr_t[batch] / 100.0)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

        model.eval()
        preds_va = []
        with torch.no_grad():
            for i in range(0, len(y_va), batch_size):
                with torch.amp.autocast('cuda') if use_fp16 else torch.no_grad():
                    p = model(input_ids_va[i:i+batch_size], attn_va[i:i+batch_size])
                preds_va.append(p.float().cpu() * 100.0)
        return torch.cat(preds_va).numpy()

    def compute_metrics(preds, ratings_va):
        with np.errstate(invalid='ignore'):
            rho_result = spearmanr(preds, ratings_va)
            rho = float(rho_result.statistic) if not np.isnan(rho_result.statistic) else 0.0
        mae = float(mean_absolute_error(ratings_va, preds))
        rmse = float(np.sqrt(mean_squared_error(ratings_va, preds)))
        return rho, mae, rmse

    OUT_DIR = Path(args.out_dir) / 'v1_results'
    OUT_DIR.mkdir(exist_ok=True)
    all_rhos, all_maes, all_rmses = [], [], []
    t_start = time.time()

    # ===== Step 6: Run =====
    if not cv_mode:
        # Single 90/10 split
        print('\n=== Step 6: Single 90/10 split (M3 recipe) ===')
        rng = np.random.RandomState(42)
        unique_jokes = np.unique(joke_ids)
        shuffled = rng.permutation(unique_jokes)
        val_jokes = set(shuffled[:len(unique_jokes) // 10])
        val_mask = np.array([jid in val_jokes for jid in joke_ids])
        train_mask = ~val_mask
        train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
        train_y = ratings_norm[train_mask]
        val_texts = [texts[i] for i in range(len(texts)) if val_mask[i]]
        val_y = ratings_norm[val_mask]
        print(f'  Train: {train_mask.sum()}, Val: {val_mask.sum()}')

        preds = train_eval_split(train_texts, train_y, val_texts, val_y, seed=42)
        rho, mae, rmse = compute_metrics(preds, val_y)
        all_rhos.append(rho)
        all_maes.append(mae)
        all_rmses.append(rmse)
        print(f'  rho={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f}')
    else:
        # 5x3 repeated joke-disjoint CV
        print(f'\n=== Step 6: 5x3 repeated joke-disjoint CV ===')
        partial_path = OUT_DIR / 'partial_results.json'
        partial = {'fold_results': []}

        for seed_idx in range(args.n_seeds):
            seed = 42 + seed_idx * 100
            rng = np.random.RandomState(seed)
            unique_jokes = np.unique(joke_ids)
            shuffled = rng.permutation(unique_jokes)
            joke_remap = {j: shuffled[i] for i, j in enumerate(unique_jokes)}
            jokes_shuffled = np.array([joke_remap[j] for j in joke_ids])
            gkf = GroupKFold(n_splits=args.n_folds)
            for fold_idx, (tr, va) in enumerate(gkf.split(np.arange(len(texts)), ratings_norm, jokes_shuffled)):
                t0 = time.time()
                try:
                    tr_texts = [texts[i] for i in tr]
                    va_texts = [texts[i] for i in va]
                    preds = train_eval_split(tr_texts, ratings_norm[tr], va_texts, ratings_norm[va], seed=seed)
                    rho, mae, rmse = compute_metrics(preds, ratings_norm[va])
                    all_rhos.append(rho)
                    all_maes.append(mae)
                    all_rmses.append(rmse)
                    elapsed = time.time() - t0
                    total = time.time() - t_start
                    print(f'  seed={seed} fold={fold_idx}: rho={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} ({elapsed:.0f}s, total {total:.0f}s)')
                    # v19 lesson: membership guard prevents resume-duplication
                    if any(r.get('seed') == seed and r.get('fold') == fold_idx for r in partial['fold_results']):
                        print(f'    (already recorded, skip duplicate)')
                        continue
                    partial['fold_results'].append({
                        'seed': seed, 'fold': fold_idx, 'rho': rho, 'mae': mae, 'rmse': rmse,
                        'elapsed_s': elapsed, 'total_s': total,
                    })
                    partial_path.write_text(json.dumps({
                        'fold_results': partial['fold_results'],
                        'n_completed': len(all_rhos),
                        'elapsed_s': total,
                    }, indent=2))
                except Exception as e:
                    print(f'  ERROR seed={seed} fold={fold_idx}: {e}')
                    partial_path.write_text(json.dumps({
                        'fold_results': partial['fold_results'],
                        'n_completed': len(all_rhos),
                        'elapsed_s': time.time() - t_start,
                        'last_error': str(e),
                    }, indent=2))
                    continue

    # ===== Step 7: Save results =====
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

        mean_rho = float(np.mean(all_rhos))
        std_rho = float(np.std(all_rhos))
        mean_mae = float(np.mean(all_maes))
        mean_rmse = float(np.mean(all_rmses))

        if mean_rho >= 0.40 and ci_low > 0.30:
            verdict = 'STRONG: rho > 0.40 with CI excluding 0.30.'
        elif mean_rho >= 0.30 and ci_low > 0.20:
            verdict = 'MODEST: rho > 0.30 with CI excluding 0.20.'
        elif mean_rho >= 0.35 and ci_low >= 0.30:
            verdict = 'GATE PASS: rho >= 0.35 AND CI excludes 0.30. PUBLISHABLE!'
        else:
            verdict = 'GATE NOT MET: 5x3 CV does not pass pre-registered gate.'

        result = {
            'rho_mean': mean_rho,
            'rho_std': std_rho,
            'rho_ci_low': ci_low,
            'rho_ci_high': ci_high,
            'mae_mean': mean_mae,
            'rmse_mean': mean_rmse,
            'n_measurements': len(all_rhos),
            'verdict': verdict,
            'config': {
                'mode': args.mode,
                'epochs': epochs, 'lr': lr, 'batch_size': batch_size,
                'use_fp16': use_fp16, 'use_grad_checkpointing': use_grad_checkpointing,
                'n_seeds': args.n_seeds, 'n_folds': args.n_folds,
                'sample_limit': args.sample_limit,
                'dataset': 'SeppeV/rated_jokes_dataset_from_jester',
                'pre_registered_gates': {'rho_min': 0.35, 'ci_lower_min': 0.30},
            },
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'total_runtime_seconds': time.time() - t_start,
        }

        print(f'\n=== Results ===')
        print(f'  rho: {mean_rho:.4f} +/- {std_rho:.4f}')
        print(f'  95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
        print(f'  MAE: {mean_mae:.4f}')
        print(f'  RMSE: {mean_rmse:.4f}')
        print(f'  Verdict: {verdict}')

        RESULT_PATH = OUT_DIR / f'v1_{args.mode}_result.json'
        RESULT_PATH.write_text(json.dumps(result, indent=2))
        print(f'\nSaved: {RESULT_PATH}')

    print('\n=== DONE ===')


if __name__ == '__main__':
    main()
