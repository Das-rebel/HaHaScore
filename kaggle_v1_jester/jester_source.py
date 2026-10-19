"""
jester_source.py — Jester dataset loader (SeppeV/rated_jokes_dataset_from_jester)
=================================================================================
Single source-of-truth for the SeppeV Jester data download + normalization.

The downloader was previously duplicated in 4 places (data/jester_seppev.py
removed, plus 3 Kaggle kernel scripts). Bug fixes (e.g. the schema fix in
commit 5378590 - 'rating' column not 'Z_rating') now live here.

Interface:
    src = JesterSource(output_dir=Path('/kaggle/working/jester'))
    csv_path = src.download_csv()  # downloads 5 parquet shards, writes 1 CSV, returns path

The CSV has columns: joke_id, joke_text, rating.
"""
import csv
import os
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import requests as req


DATASET_ID = 'SeppeV/rated_jokes_dataset_from_jester'
HF_API = 'https://huggingface.co'
CSV_COLUMNS = ['joke_id', 'joke_text', 'rating']


class JesterSource:
    """Adapter for SeppeV/rated_jokes_dataset_from_jester (Apache-2.0)."""

    def __init__(self, output_dir, dataset_id=DATASET_ID, hf_api=HF_API, timeout=300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_id = dataset_id
        self.hf_api = hf_api
        self.timeout = timeout

    def list_shards(self):
        """List 5 parquet shards in dataset/data/."""
        r = req.get(f'{self.hf_api}/api/datasets/{self.dataset_id}/tree/main/data', timeout=30)
        r.raise_for_status()
        return sorted([it['path'] for it in r.json() if it.get('type') == 'file' and it['path'].endswith('.parquet')])

    def download_shard(self, remote_path):
        """Download one parquet shard to output_dir, return local Path."""
        url = f'{self.hf_api}/datasets/{self.dataset_id}/resolve/main/{remote_path}'
        local = self.output_dir / Path(remote_path).name
        if not local.exists():
            r = req.get(url, stream=True, timeout=self.timeout)
            r.raise_for_status()
            with open(local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        return local

    def _row_passes(self, joke_text, rating):
        if not joke_text or rating is None:
            return False
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            return False
        if rating != rating:  # NaN
            return False
        return True

    def write_csv(self, csv_path=None, csv_columns=None):
        """Download all shards, write a single CSV, return csv_path.

        Schema (CSV columns):
            joke_id  : int  (deterministic from hash(joke_text))
            joke_text : str  (joke text, max 1500 chars, newlines stripped)
            rating    : float (raw 1-10 user score from 'rating' column;
                                  NOT 'Z_rating' which has inf/nan)
        """
        csv_path = Path(csv_path) if csv_path else (self.output_dir / 'jester_full.csv')
        csv_columns = csv_columns or CSV_COLUMNS

        shards = self.list_shards()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            total_in, total_out = 0, 0
            for shard_path in shards:
                local = self.download_shard(shard_path)
                table = pq.read_table(local)
                for row in table.to_pylist():
                    total_in += 1
                    joke_text = row.get('jokeText', '')
                    rating = row.get('rating', None)
                    if not self._row_passes(joke_text, rating):
                        continue
                    writer.writerow({
                        'joke_id': abs(hash(joke_text)) % (10 ** 12),
                        'joke_text': str(joke_text).replace('\n', ' ').strip()[:1500],
                        'rating': float(rating),
                    })
                    total_out += 1
                # Clean up shard to save disk
                try:
                    os.remove(local)
                except OSError:
                    pass
        return csv_path, total_in, total_out


def load_csv(csv_path, max_samples=None):
    """Load the Jester CSV produced by JesterSource.write_csv.

    Returns (joke_texts, ratings, joke_ids) as Python lists/arrays.
    """
    csv_path = Path(csv_path)
    import numpy as np
    joke_texts, ratings, joke_ids = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            joke_texts.append(row['joke_text'])
            ratings.append(float(row['rating']))
            joke_ids.append(int(row['joke_id']))
    ratings = np.array(ratings)
    joke_ids = np.array(joke_ids)
    if max_samples and len(joke_texts) > max_samples:
        rng = np.random.RandomState(42)
        keep = rng.choice(len(joke_texts), max_samples, replace=False)
        keep.sort()
        joke_texts = [joke_texts[i] for i in keep]
        ratings = ratings[keep]
        joke_ids = joke_ids[keep]
    return joke_texts, ratings, joke_ids


if __name__ == '__main__':
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/jester')
    src = JesterSource(out)
    csv, n_in, n_out = src.write_csv()
    print(f'Wrote {csv} from {n_in} rows in, {n_out} kept')
