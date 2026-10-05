"""
v2_laugho_kaggle_kernel.py — Kaggle fallback for v2 LaughO training
=================================================================
Use when Colab is unavailable. Requires Kaggle T4 GPU.

Push via:
    kaggle kernels push -p kaggle_v2_laugho/

Kernels metadata:
{
    "id": "subhajitdas/v2-laugho-train",
    "title": "v2 LaughO: 5x3 repeated CV training on agkphysics/AudioSet",
    "code_file": "v2_laugho_kaggle_kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": true,
    "keywords": ["laughter-detection", "wavlm", "speaker-disjoint-cv", "audioset"],
    "dataset_sources": ["agkphysics/audioset"],
    "competition_sources": [],
    "kernel_sources": []
}

Hardware: T4 (15 GB VRAM, sufficient for 1.5M-param MLP + WavLM-base-plus frozen)
Cost: Kaggle T4 free tier = 30 hours/week
Expected runtime: ~3 hours (1 shard decode + Train + 5x3 CV)
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pypa
import requests
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from transformers import WavLMModel


LAUGH_IDS = {
    '/m/01j3sz': 'Laughter',
    '/t/dd00001': 'Baby_laughter',
    '/m/07r660_': 'Giggle',
    '/m/07s04w4': 'Snicker',
    '/m/07sq110': 'Belly_laugh',
    '/m/07rgt08': 'Chuckle',
    '/m/07q0yl5': 'Snort',
}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # === Step 1: Download AudioSet eval/00.parquet ===
    AUDIOSET_URL = 'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval/00.parquet'
    dest = Path('/kaggle/working/audioset_eval_00.parquet')
    if not dest.exists():
        print(f'Downloading {AUDIOSET_URL}...')
        r = requests.get(AUDIOSET_URL, stream=True)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print(f'Downloaded: {dest.stat().st_size/1024/1024:.1f} MB')

    # === Step 2: Load + decode audio via ffmpeg-subprocess (v19 pattern) ===
    print('Loading AudioSet eval/00.parquet...')
    table = pypa.read_table(dest)
    rows = table.to_pylist()
    print(f'  Total rows: {len(rows)}')

    laugh_rows = [r for r in rows if any(l in LAUGH_IDS for l in r.get('labels', []))]
    print(f'  Laugh rows: {len(laugh_rows)}')

    # WavLM
    print('Loading WavLM-base-plus (frozen)...')
    wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').to(device).eval()
    for p in wavlm.parameters():
        p.requires_grad = False

    def extract_embedding(audio_bytes):
        try:
            p = subprocess.run(
                ['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0',
                 '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'pipe:1'],
                input=audio_bytes, capture_output=True
            )
            if p.returncode != 0:
                return None
            audio, sr = sf.read(BytesIO(p.stdout))
            if sr != 16000:
                return None
            audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = wavlm(audio_t).last_hidden_state.mean(dim=1)
            return emb.cpu().numpy().squeeze()
        except Exception:
            return None

    # Extract laughs
    print('Extracting laugh embeddings...')
    embeddings = []
    labels = []
    groups = []
    for i, r in enumerate(laugh_rows):
        audio_field = r.get('audio', {})
        audio_bytes = audio_field.get('bytes') if isinstance(audio_field, dict) else None
        if audio_bytes is None:
            continue
        emb = extract_embedding(audio_bytes)
        if emb is None:
            continue
        embeddings.append(emb)
        labels.append(1)
        groups.append(r.get('video_id', f'laugh_{i}'))
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(laugh_rows)} decoded')

    # Add negative samples (5x positive count)
    non_laugh = [r for r in rows if not any(l in LAUGH_IDS for l in r.get('labels', []))]
    n_neg = min(len(non_laugh), len(embeddings) * 5)
    import random
    random.seed(42)
    neg_sample = random.sample(non_laugh, n_neg)
    for r in neg_sample:
        audio_field = r.get('audio', {})
        audio_bytes = audio_field.get('bytes') if isinstance(audio_field, dict) else None
        if audio_bytes is None:
            continue
        emb = extract_embedding(audio_bytes)
        if emb is None:
            continue
        embeddings.append(emb)
        labels.append(0)
        groups.append(r.get('video_id', f'neg_{len(embeddings)}'))

    X = np.stack(embeddings)
    y = np.array(labels)
    groups_arr = np.array(groups)
    print(f'\\nDataset: {len(X)} samples, {sum(y)} laugh ({sum(y)/len(y):.3f})')
    print(f'  Unique groups (videos): {len(np.unique(groups_arr))}')

    # === Step 3: Define MLP ===
    class LaughOMLP(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_eval_fold(tr_X, tr_y, va_X, va_y, seed=42, n_epochs=10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LaughOMLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        pos_rate = max(tr_y.mean(), 0.001)
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        X_tr = torch.tensor(tr_X, dtype=torch.float32).to(device)
        y_tr = torch.tensor(tr_y, dtype=torch.float32).to(device)
        X_va = torch.tensor(va_X, dtype=torch.float32).to(device)
        bs = 256
        n = len(tr_X)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs * n)
        for ep in range(n_epochs):
            model.train()
            perm = np.random.permutation(n)
            for i in range(0, n, bs):
                batch = perm[i:i+bs]
                optimizer.zero_grad()
                loss = criterion(model(X_tr[batch]), y_tr[batch])
                loss.backward()
                optimizer.step()
                sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_va).cpu().numpy()
        scores = 1 / (1 + np.exp(-logits))
        if len(np.unique(va_y)) < 2:
            return 0.5
        return float(roc_auc_score(va_y, scores))

    # === Step 4: 5×3 repeated speaker-disjoint CV ===
    print('\\nRunning 5×3 repeated CV...')
    results = []
    for seed_offset in range(3):
        seed = 42 + seed_offset * 100
        rng = np.random.RandomState(seed)
        unique_groups = np.unique(groups_arr)
        shuffled = rng.permutation(unique_groups)
        group_map = {g: shuffled[i] for i, g in enumerate(unique_groups)}
        groups_shuffled = np.array([group_map[g] for g in groups_arr])
        gkf = GroupKFold(n_splits=5)
        for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups_shuffled)):
            auc = train_eval_fold(X[tr], y[tr], X[va], y[va], seed=seed)
            results.append({'seed': seed, 'fold': fold_idx, 'auc': auc})
            print(f'  seed={seed} fold={fold_idx}: AUC={auc:.4f}')

    aucs = [r['auc'] for r in results]
    print(f'\\n5×3 CV results:')
    print(f'  Mean: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  Min:  {np.min(aucs):.4f}')
    print(f'  Max:  {np.max(aucs):.4f}')

    # === Step 5: Save to Kaggle output ===
    out_path = Path('/kaggle/working/v2_laugho_5x3_cv.json')
    out_path.write_text(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'source': 'AudioSet eval/00.parquet (agkphysics)',
        'n_samples': len(X),
        'n_laugh': int(sum(y)),
        'n_non_laugh': int(len(y) - sum(y)),
        'positive_rate': float(sum(y) / len(y)),
        'n_unique_videos': int(len(np.unique(groups_arr))),
        'aucs': aucs,
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'auc_min': float(np.min(aucs)),
        'auc_max': float(np.max(aucs)),
        'protocol': '5-fold GroupKFold by video_id, 3 seeds, frozen WavLM + 1.5M MLP',
        'haHaScore_commit': '7352527',
    }, indent=2))
    print(f'\\nSaved: {out_path}')


if __name__ == '__main__':
    main()