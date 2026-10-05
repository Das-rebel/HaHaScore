"""
v1_laugho_modal.py — Modal app for v1 HaHaScore regression on Jester
======================================================================
Runs the v1 canonical training pipeline on Modal's T4 GPU.

Modal account: sdas22 (verified working as of Oct 6 2026, T4 available
without payment method per the modal test).

Usage:
    modal run v1_laugho_modal.py    # actual run on T4
    modal serve v1_laugho_modal.py  # serve as endpoint (not used)

Cost: 0 (Modal free tier has T4 quota; check current limits)
Runtime: ~4-6h on full 1.76M jokes, 5 epochs, 3 seeds x 5 folds = 15 measurements
"""
import modal

app = modal.App("haHaScore-v1-jester-regression")

# Image with PyTorch + transformers
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.0",
        "datasets==4.8.4",
        "scipy==1.13.0",
        "scikit-learn==1.5.0",
        "numpy==1.26.4",
        "pyarrow==16.0.0",
        "requests==2.32.0",
    )
)


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=8192,
    timeout=6 * 3600,  # 6 hours
)
def run_v1_jester(
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 16,
    n_seeds: int = 3,
    n_folds: int = 5,
    max_samples: int = None,
) -> dict:
    """Run v1 RoBERTa regression on Jester, return 5x3 CV results."""
    import os, sys
    os.chdir('/root')
    sys.path.insert(0, '/root')

    # Download scripts from HaHaScore repo
    import subprocess
    subprocess.run(['git', 'clone', 'https://github.com/Das-rebel/HaHaScore.git'],
                   check=False, capture_output=True)
    os.chdir('/root/HaHaScore')
    sys.path.insert(0, '/root/HaHaScore')

    # Step 1: Download Jester
    import data.jester_seppev as jester_seppev
    from pathlib import Path
    out_dir = Path('/root/HaHaScore/data/jester')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=== Downloading Jester ===')
    parquet_files = jester_seppev.list_parquet_files()
    print(f'  Found {len(parquet_files)} shards')

    import csv
    import pyarrow.parquet as pq
    out_csv = out_dir / 'jester_full.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['joke_id', 'joke_text', 'rating', 'source'])
        writer.writeheader()
        for shard_path in parquet_files:
            local = out_dir / Path(shard_path).name
            if not local.exists():
                jester_seppev.download_shard(shard_path, local)
            table = pq.read_table(local)
            n_in, n_out = jester_seppev.normalize_shard(table, writer, shard_path)
            print(f'    {shard_path}: {n_in} in, {n_out} kept')
            try:
                os.remove(local)
            except OSError:
                pass

    print(f'\nJester saved: {out_csv}')

    # Step 2: Train
    print('\n=== Running v1 training ===')
    import train_jester_regression as v1
    sys.argv = [
        'train_jester_regression.py',
        '--jester-csv', str(out_csv),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--batch-size', str(batch_size),
        '--n-seeds', str(n_seeds),
        '--n-folds', str(n_folds),
        '--out-dir', '/root/v1_results',
    ]
    if max_samples:
        sys.argv.extend(['--max-samples', str(max_samples)])
    v1.main()

    # Step 3: Return result
    import json
    import glob
    from pathlib import Path
    results = sorted(Path('/root/v1_results').glob('result_*.json'))
    if not results:
        return {'error': 'no results found'}
    latest = results[-1]
    with open(latest) as f:
        result = json.load(f)

    # Print summary
    print(f'\n=== v1 Results ===')
    print(f'Spearman rho: {result.get("rho_mean", 0):.4f} +/- {result.get("rho_std", 0):.4f}')
    print(f'95% CI: [{result.get("rho_ci_low", 0):.4f}, {result.get("rho_ci_high", 0):.4f}]')
    print(f'MAE: {result.get("mae_mean", 0):.4f}')
    print(f'RMSE: {result.get("rmse_mean", 0):.4f}')
    print(f'Verdict: {result.get("verdict", "?")}')

    return result


@app.local_entrypoint()
def main(
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 16,
    n_seeds: int = 3,
    n_folds: int = 5,
    max_samples: int = None,
):
    """Local entrypoint — run via 'modal run v1_laugho_modal.py'."""
    print(f'=== Running v1 on Modal T4 GPU ===')
    print(f'  epochs={epochs}, lr={lr}, bs={batch_size}')
    print(f'  n_seeds={n_seeds}, n_folds={n_folds}')
    if max_samples:
        print(f'  max_samples={max_samples}')
    result = run_v1_jester.remote(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_seeds=n_seeds,
        n_folds=n_folds,
        max_samples=max_samples,
    )
    print(f'\n=== Modal run complete ===')
    if 'error' in result:
        print(f'ERROR: {result["error"]}')
    else:
        print(f'rho: {result["rho_mean"]:.4f}, CI [{result["rho_ci_low"]:.4f}, {result["rho_ci_high"]:.4f}]')
        print(f'MAE: {result["mae_mean"]:.4f}')
        print(f'Verdict: {result["verdict"]}')