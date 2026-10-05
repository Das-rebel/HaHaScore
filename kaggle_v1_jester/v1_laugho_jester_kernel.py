"""
v1_laugho_jester_kernel.py — Kaggle kernel for v1 HaHaScore regression
======================================================================
Single-file Kaggle kernel: downloads SeppeV Jester, trains text-only RoBERTa
regression head, runs 5x3 repeated joke-disjoint CV with bootstrap 95% CI.

Hardware: T4 (15GB VRAM) — required for roberta-base + full 1.76M dataset
Cost: 0 (Kaggle free tier 30h/week)
Runtime: ~4-6h on full 1.76M jokes, 5 epochs, 3 seeds x 5 folds = 15 measurements

Upload via:
    cd kaggle_v1_jester
    kaggle kernels push

Pull output:
    kaggle kernels output subhajitdas/v1-laugho-jester-regression -p ./output/
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path

# Step 1: Install deps
print('=== Step 1: Install deps ===')
subprocess.run(['pip', 'install', '--quiet', 'transformers==4.44.0', 'datasets==4.8.4', 'scipy', 'scikit-learn'], check=True)
print('  Installed: transformers, datasets, scipy, scikit-learn')

# Step 2: Add scripts dir to path
sys.path.insert(0, '/kaggle/working')
print(f'Working dir: /kaggle/working')

# Step 3: Run the training script with v1 canonical config
print('\n=== Step 2: Run v1 canonical training ===')
config = {
    'epochs': 5,
    'lr': 2e-5,
    'batch_size': 16,
    'n_seeds': 3,
    'n_folds': 5,
    'out_dir': '/kaggle/working/v1_results',
}
print(f'Config: {config}')

# Step 4: Download Jester first (so train_jester_regression.py can find it)
print('\n=== Step 3: Download Jester via jester_seppev.py ===')
import jester_seppev
jester_seppev.main = None  # don't run main; use functions directly
from jester_seppev import list_parquet_files, download_shard, normalize_shard
import csv
import pyarrow.parquet as pq

out_dir = Path('/kaggle/working/jester')
out_dir.mkdir(parents=True, exist_ok=True)

print('  Listing shards...')
parquet_files = list_parquet_files()
print(f'  Found {len(parquet_files)} shards')

out_csv = out_dir / 'jester_full.csv'
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['joke_id', 'joke_text', 'rating', 'source'])
    writer.writeheader()
    for shard_path in parquet_files:
        local = out_dir / Path(shard_path).name
        if not local.exists():
            download_shard(shard_path, local)
        table = pq.read_table(local)
        n_in, n_out = normalize_shard(table, writer, shard_path)
        print(f'    {shard_path}: {n_in} in, {n_out} kept')
        try:
            os.remove(local)
        except OSError:
            pass

print(f'  Saved: {out_csv}')

# Step 5: Run the canonical training script
print('\n=== Step 4: Run train_jester_regression.py ===')
import train_jester_regression
sys.argv = [
    'train_jester_regression.py',
    '--jester-csv', str(out_csv),
    '--epochs', str(config['epochs']),
    '--lr', str(config['lr']),
    '--batch-size', str(config['batch_size']),
    '--n-seeds', str(config['n_seeds']),
    '--n-folds', str(config['n_folds']),
    '--out-dir', config['out_dir'],
]
train_jester_regression.main()

# Step 6: Save results to kernel output (will be downloadable)
print('\n=== Step 5: Save results ===')
result_files = sorted(Path(config['out_dir']).glob('result_*.json'))
latest = result_files[-1] if result_files else None
if latest:
    print(f'Latest result: {latest}')
    # Copy to a known location
    import shutil
    out_path = Path('/kaggle/working/output_result.json')
    shutil.copy(latest, out_path)
    print(f'Copied to: {out_path}')

print('\n=== Done ===')