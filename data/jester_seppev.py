#!/usr/bin/env python3
"""
data/jester_seppev.py — Jester dataset downloader for v1 HaHaScore
====================================================================
Downloads SeppeV/rated_jokes_dataset_from_jester from HuggingFace,
which contains ~1.76M rated jokes with continuous user ratings.

The dataset has 5 parquet shards (~18 MB each). This script streams
all shards into a single normalized csv/jsonl for v1 training.

Usage:
    python3 data/jester_seppev.py --output-dir data/jester
    python3 data/jester_seppev.py --output-dir data/jester --max-shards 1   # smoke test

Output:
    data/jester/jokes_full.csv (columns: joke_id, joke_text, rating, source)
    data/jester/jester_metadata.json (download stats, schema, license)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pyarrow.parquet as pq
import requests


HF_TOKEN = os.environ.get('HF_TOKEN', None)
HF_API = 'https://huggingface.co'
DATASET_ID = 'SeppeV/rated_jokes_dataset_from_jester'

LAUGH_HEADERS = ['joke_id', 'joke_text', 'rating']


def list_parquet_files():
    """List parquet shards in the SeppeV Jester dataset."""
    headers = {}
    if HF_TOKEN:
        headers['Authorization'] = f'Bearer {HF_TOKEN}'
    url = f'{HF_API}/api/datasets/{DATASET_ID}/tree/main/data'
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    items = r.json()
    parquet_files = [it['path'] for it in items if it.get('type') == 'file' and it['path'].endswith('.parquet')]
    return sorted(parquet_files)


def download_shard(remote_path, local_path):
    """Download one parquet shard from HuggingFace to local_path."""
    headers = {}
    if HF_TOKEN:
        headers['Authorization'] = f'Bearer {HF_TOKEN}'
    url = f'{HF_API}/datasets/{DATASET_ID}/resolve/main/{remote_path}'
    print(f'  Downloading {remote_path} -> {local_path.name} ...')
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    return local_path.stat().st_size


def normalize_shard(table, out_writer, source_shard):
    """Stream rows from a parquet shard into CSV writer, dropping NaN."""
    n_in = 0
    n_out = 0
    for row in table.to_pylist():
        n_in += 1
        joke_text = row.get('jokeText', row.get('joke_text', ''))
        rating = row.get('Z_rating', row.get('rating', None))
        if not joke_text or rating is None:
            continue
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            continue
        if rating != rating:  # NaN check
            continue
        out_writer.writerow({
            'joke_id': abs(hash(joke_text)) % (10**12),
            'joke_text': str(joke_text).replace('\n', ' ').strip()[:2000],
            'rating': rating,
            'source': source_shard,
        })
        n_out += 1
    return n_in, n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='data/jester',
                    help='Where to save Jester dataset (default: data/jester)')
    ap.add_argument('--max-shards', type=int, default=None,
                    help='Cap to N shards (for smoke testing)')
    ap.add_argument('--max-rows', type=int, default=None,
                    help='Cap to N total rows (overrides shard cap)')
    ap.add_argument('--format', choices=['csv', 'jsonl'], default='csv')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'parquet_cache'
    cache_dir.mkdir(exist_ok=True)

    # 1. List shards
    print('Listing parquet shards...')
    parquet_files = list_parquet_files()
    if args.max_shards:
        parquet_files = parquet_files[:args.max_shards]
    print(f'  Found {len(parquet_files)} shards')

    # 2. Open output file
    out_csv = out_dir / f'jester_full.{args.format}'
    out_meta = out_dir / 'jester_metadata.json'

    total_in = 0
    total_out = 0
    rows_processed = 0
    started = datetime.now().isoformat()

    if args.format == 'csv':
        out_f = open(out_csv, 'w', newline='')
        writer = csv.DictWriter(out_f, fieldnames=LAUGH_HEADERS + ['source'])
        writer.writeheader()
    else:
        out_f = open(out_csv, 'w')
        writer = None

    try:
        for shard_path in parquet_files:
            local = cache_dir / Path(shard_path).name
            if not local.exists():
                size = download_shard(shard_path, local)
                print(f'    {size/1024/1024:.1f} MB downloaded')
            print(f'  Reading {local.name} ...')
            table = pq.read_table(local)
            n_in, n_out = normalize_shard(table, writer if writer else out_f, shard_path)
            total_in += n_in
            total_out += n_out
            rows_processed += n_out
            print(f'    {n_in} rows in, {n_out} rows kept (cumulative: {total_out})')

            # Drop NaN rows from this shard to save cache space
            try:
                os.remove(local)
            except OSError:
                pass

            if args.max_rows and total_out >= args.max_rows:
                print(f'  Reached max-rows={args.max_rows}, stopping')
                break
    finally:
        out_f.close()

    # 3. Save metadata
    metadata = {
        'dataset_id': DATASET_ID,
        'license': 'Apache-2.0',
        'shards_total': len(parquet_files),
        'rows_in_total': total_in,
        'rows_out_total': total_out,
        'output_format': args.format,
        'output_file': str(out_csv),
        'columns': LAUGH_HEADERS + ['source'],
        'rating_scale': 'continuous (-10 to +10, raw; we recommend normalizing to 0-100 for the regression head)',
        'started': started,
        'finished': datetime.now().isoformat(),
        'note': 'Pre-registered v1 training data for HaHaScore canonical path. '
                'See COUNCIL_REALIGNMENT.md for the council decision.',
    }
    out_meta.write_text(json.dumps(metadata, indent=2))
    print(f'\nSaves: {out_meta}')

    print(f'\n=== Summary ===')
    print(f'  Total rows in:  {total_in}')
    print(f'  Total rows out: {total_out} (after dropping NaN/empty)')
    print(f'  Output:         {out_csv}')
    print(f'  Metadata:       {out_meta}')


if __name__ == '__main__':
    main()