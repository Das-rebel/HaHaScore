#!/usr/bin/env python3
"""
laugho_data.py — AudioSet laughter class extractor for v2 LaughO
================================================================
Following v19's ffmpeg-subprocess m4a loader pattern, this script:
1. Streams AudioSet parquet shards from agkphysics/AudioSet mirror (CC-BY-4.0)
2. Filters rows whose labels contain any of the 8 laugh ontology IDs
3. Decodes audio bytes via PyArrow + soundfile (or ffmpeg fallback for m4a)
4. Writes a unified manifest CSV + chunked .npz files for training

Usage (low-GPU environments):
    python laugho_data.py --shards /tmp/audioset_laugh --manifest laughs.csv
    python laugho_data.py --shards /tmp/audioset_laugh --manifest laughs.csv --n-rows 10000

Output:
- laugh_manifest.csv (video_id, start_sec, label, ontology_id, shard_path, row_idx)
- laughs_part_0.npz, laughs_part_1.npz, ... (chunks of ~5k audio arrays)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from io import BytesIO

# 8 laugh ontology IDs (AudioSet, CC-BY-4.0)
LAUGH_ONTOLOGY_IDS = [
    '/m/01j3sz',   # Laughter
    '/t/dd00001',  # Baby laughter
    '/m/07r660_',  # Giggle
    '/m/07s04w4',  # Snicker
    '/m/07sq110',  # Belly laugh
    '/m/07rgt08',  # Chuckle, chortle
    '/m/07q0yl5',  # Snort
]

LAUGH_NAMES = {
    '/m/01j3sz': 'Laughter',
    '/t/dd00001': 'Baby_laughter',
    '/m/07r660_': 'Giggle',
    '/m/07s04w4': 'Snicker',
    '/m/07sq110': 'Belly_laugh',
    '/m/07rgt08': 'Chuckle',
    '/m/07q0yl5': 'Snort',
}

# AudioSet eval splits (smallest, ~22k samples, 22.5 GB total)
EVAL_SHARDS = [f'data/eval/{i:02d}.parquet' for i in range(35)]
# Unbalanced training (much larger, ~1.9M)
TRAIN_SHARDS = [f'data/unbal_train/{i:05d}.parquet' for i in range(35)]  # partial list

MIRROR = 'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main'
HF_TOKEN = os.environ.get('HF_TOKEN', '')


def head(url, n_bytes=16*1024):
    """Read first n_bytes of a remote file via Range header."""
    headers = {'Range': f'bytes=0-{n_bytes-1}'}
    if HF_TOKEN:
        headers['Authorization'] = f'Bearer {HF_TOKEN}'
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    return r.content


def fetch_parquet_rows(shard_path, ontology_filter=None, max_rows=None):
    """
    Stream rows from a remote parquet shard.
    Returns list of (row_dict, label_int) tuples matching ontology_filter.
    """
    url = f'{MIRROR}/{shard_path}'
    # Stream the full file (for AudioSet eval, ~650 MB per shard)
    headers = {}
    if HF_TOKEN:
        headers['Authorization'] = f'Bearer {HF_TOKEN}'

    local_path = f'/tmp/audioset_{os.path.basename(shard_path)}'
    if not os.path.exists(local_path):
        print(f'  Downloading {shard_path} (~650 MB)...')
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    else:
        print(f'  Using cached {local_path}')

    table = pq.read_table(local_path)
    print(f'  Loaded {len(table)} rows from {shard_path}')

    keep = []
    for i, row in enumerate(table.to_pylist()):
        if max_rows and len(keep) >= max_rows:
            break
        labels = row.get('labels', [])
        matched = [l for l in labels if l in (ontology_filter or LAUGH_ONTOLOGY_IDS)]
        if matched:
            row['matched_label'] = matched[0]
            row['source_shard'] = shard_path
            row['source_row_idx'] = i
            keep.append(row)

    # Cleanup cache
    try:
        os.remove(local_path)
    except OSError:
        pass

    return keep


def decode_audio_bytes(audio_bytes, fmt='wav'):
    """
    Decode audio bytes to numpy array. Uses soundfile for wav/flac/ogg.
    For m4a (AAC), falls back to ffmpeg-subprocess (per v19 pattern).
    Returns (audio_float32, sample_rate) or None on failure.
    """
    try:
        if fmt == 'm4a':
            # v19 ffmpeg pattern: pipe AAC through ffmpeg -> soundfile
            p = subprocess.run(
                ['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0',
                 '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'pipe:1'],
                input=audio_bytes, capture_output=True
            )
            if p.returncode != 0:
                return None
            data, sr = sf.read(BytesIO(p.stdout))
        else:
            data, sr = sf.read(BytesIO(audio_bytes))
        return data.astype(np.float32), sr
    except Exception as e:
        print(f'    decode failed: {e}')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', default=EVAL_SHARDS[:1],
                    help='parquet shard paths in agkphysics/AudioSet (relative to /data)')
    ap.add_argument('--max-rows', type=int, default=10000,
                    help='cap per-shard rows to keep')
    ap.add_argument('--manifest', default='laugh_manifest.csv')
    ap.add_argument('--out-dir', default='/tmp/audioset_laugh_out')
    ap.add_argument('--decode', action='store_true',
                    help='decode audio bytes (slower, writes .npz)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for shard in args.shards:
        rows = fetch_parquet_rows(shard, max_rows=args.max_rows)
        all_rows.extend(rows)
        print(f'  Kept {len(rows)} laugh rows from {shard} (total so far: {len(all_rows)})')

    # Write manifest CSV
    manifest_path = os.path.join(args.out_dir, args.manifest)
    with open(manifest_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video_id', 'label_name', 'ontology_id', 'source_shard', 'source_row_idx', 'audio_path'])
        for r in all_rows:
            w.writerow([
                r.get('video_id', ''),
                LAUGH_NAMES.get(r['matched_label'], 'Unknown'),
                r['matched_label'],
                r['source_shard'],
                r['source_row_idx'],
                '',  # filled in if --decode
            ])
    print(f'Manifest written: {manifest_path} ({len(all_rows)} rows)')

    if args.decode:
        print('Decoding audio bytes (this is slow)...')
        audio_arrays = []
        labels = []
        for i, r in enumerate(all_rows):
            audio = r.get('audio', {})
            audio_bytes = audio.get('bytes') if isinstance(audio, dict) else None
            if audio_bytes is None:
                continue
            decoded = decode_audio_bytes(audio_bytes, fmt='wav')
            if decoded is None:
                continue
            audio_arrays.append(decoded[0])
            labels.append(r['matched_label'])
            if (i + 1) % 100 == 0:
                print(f'  decoded {i+1}/{len(all_rows)}')

        # Save chunked npz
        npz_path = os.path.join(args.out_dir, 'laughs.npz')
        np.savez_compressed(npz_path, audio=audio_arrays, labels=labels)
        print(f'Audio data written: {npz_path} ({len(audio_arrays)} samples)')

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)