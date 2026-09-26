#!/usr/bin/env python3
"""
Fine-tune Cascade Gate v8 + Reddit on 641 Standup Files
========================================================

Pulls:
- Reddit-pretrained DistilBERT from Drive (text tower)
- Pre-computed standup audio features (791-dim) from local or Drive
- Text transcripts from local standup directory
- Pseudo-labels from local (641 files)

Trains:
- Fine-tune v8's audio + gate layers (freeze text encoder for stability)
- Optionally unfreeze text encoder for end-to-end fine-tuning

Output:
- v8_finetuned.pt pushed to Drive
- Metrics pushed to Drive/results
"""
import os
os.environ['HF_HOME'] = '/tmp/hf_distilbert'
import argparse
import json
import subprocess
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import sys
sys.path.insert(0, '/Users/Subho/funny-strength-predictor/drive_pipeline')
sys.path.insert(0, '/Users/Subho/funny-strength-predictor/improvements')

from v8_with_reddit import V8WithReddit
from enhanced_cascade import create_enhanced_model

warnings.filterwarnings('ignore')

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain"
STANDUP_DIR = Path('/Users/Subho/funny-strength-predictor/data')


def pull_reddit_model(remote_name):
    """Pull Reddit model from Drive to local temp."""
    temp = Path('/tmp/reddit_pull')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    print(f"📥 Pulling {remote_name} from Drive...")
    result = subprocess.run(
        ['rclone', 'copyto', f'{DRIVE_BASE}/models/{remote_name}',
         str(temp / 'reddit.pt'),
         '--retries=3', '--retries-sleep=10s'],
        capture_output=True, text=True
    )
    path = temp / 'reddit.pt'
    if path.exists():
        print(f"✅ Pulled {path.stat().st_size/1e6:.1f} MB")
        return path
    return None


def find_standup_data():
    """Find 641 standup audio features + labels + texts."""
    # Check for v6 features (text features we already have)
    candidates = [
        STANDUP_DIR / 'v6_features.npz',
        STANDUP_DIR / 'bridge4_features.npz',
        Path('/Users/Subho/tmp/v6_features.npz'),
        Path('/Users/Subho/tmp/bridge4_features.npz'),
    ]
    
    text_features = None
    audio_features = None
    
    for path in candidates:
        if path.exists():
            data = np.load(path)
            if 'text' in data.files or 'text_features' in data.files:
                text_features = data
            if 'audio' in data.files or 'audio_features' in data.files:
                audio_features = data
    
    # Labels
    labels_path = STANDUP_DIR / 'pseudo_labels' / 'pseudo_labels_641_v4.json'
    labels = []
    if labels_path.exists():
        with open(labels_path) as f:
            labels_data = json.load(f)
        labels = [item['label'] for item in labels_data]
    
    # Transcripts - look for any json/csv with text
    transcripts_path = STANDUP_DIR / 'transcripts.json'
    transcripts = []
    if transcripts_path.exists():
        with open(transcripts_path) as f:
            transcripts_data = json.load(f)
        if isinstance(transcripts_data, list):
            transcripts = [item.get('text', '') for item in transcripts_data]
    
    print(f"text_features: {'✓' if text_features is not None else '✗'}")
    print(f"audio_features: {'✓' if audio_features is not None else '✗'}")
    print(f"labels: {len(labels)} entries")
    print(f"transcripts: {len(transcripts)} entries")
    
    return text_features, audio_features, labels, transcripts


def tokenize_transcripts(transcripts, tokenizer, max_len=128):
    """Tokenize each transcript (or per-segment text)."""
    if not transcripts:
        return None
    
    # If we have segments, tokenize each segment
    encoded = []
    for t in transcripts:
        if not t:
            encoded.append({'input_ids': torch.zeros(max_len, dtype=torch.long),
                           'attention_mask': torch.zeros(max_len, dtype=torch.long)})
        else:
            enc = tokenizer(t, max_length=max_len, padding='max_length',
                          truncation=True, return_tensors='pt')
            encoded.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            })
    return encoded


def fine_tune(args):
    """Fine-tune v8 on standup data."""
    print("=" * 60)
    print(f"🎯 Fine-tuning Cascade Gate v8 + Reddit on Standup")
    print(f"   Reddit model: {args.reddit_model}")
    print(f"   Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    # Load Reddit
    reddit_path = pull_reddit_model(args.reddit_model)
    if not reddit_path:
        return False

    # Find standup data
    text_features, audio_features, labels, transcripts = find_standup_data()

    if audio_features is None or len(labels) == 0:
        print("❌ Cannot fine-tune: missing audio features or labels")
        print("   Need: /Users/Subho/tmp/bridge4_features.npz + pseudo_labels_641_v4.json")
        shutil.rmtree(reddit_path.parent, ignore_errors=True)
        return False

    # Build v8 + Reddit
    print(f"\nBuilding v8 + Reddit architecture...")
    v8_base = create_enhanced_model(
        text_dim=768, audio_dim=audio_features['audio'].shape[-1] if 'audio' in audio_features else 791,
        cross_dim=16, hidden=128
    )
    print(f"v8 base: {sum(p.numel() for p in v8_base.parameters()):,} params")

    model = V8WithReddit(v8_base, reddit_state_path=str(reddit_path), freeze_text=args.freeze_text)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {n_total:,} params, Trainable: {n_train:,} params")

    # Tokenize transcripts
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized = tokenize_transcripts(transcripts, tokenizer, max_len=128)
    if tokenized is None:
        print("❌ No transcripts found")
        shutil.rmtree(reddit_path.parent, ignore_errors=True)
        return False

    # Build dataset
    audio_arr = audio_features['audio']  # (N, seq_len, 791)
    n_samples = min(len(tokenized), len(audio_arr), len(labels))
    print(f"\nBuilding dataset: {n_samples} samples")

    class StandupDataset(Dataset):
        def __init__(self, tokenized, audio, labels, n):
            self.tok = tokenized[:n]
            self.audio = audio[:n]
            self.labels = labels[:n]
        def __len__(self):
            return len(self.tok)
        def __getitem__(self, i):
            return {
                'input_ids': self.tok[i]['input_ids'],
                'attention_mask': self.tok[i]['attention_mask'],
                'audio': torch.tensor(self.audio[i], dtype=torch.float32),
                'label': torch.tensor(self.labels[i], dtype=torch.float32),
                'cross': torch.zeros(20, 16, dtype=torch.float32),  # dummy cross
            }

    # Convert labels to numpy array
    if isinstance(labels, list):
        labels_arr = np.array(labels[:n_samples])
    else:
        labels_arr = labels[:n_samples]

    # If labels are per-segment (list of lists), flatten
    if labels_arr.ndim == 2:
        # Each row is (seq_len,) — use directly
        pass
    else:
        # Maybe labels are per-sample (single value), broadcast to segments
        labels_arr = np.tile(labels_arr[:, None], (1, audio_arr.shape[1]))

    dataset = StandupDataset(tokenized, audio_arr, labels_arr, n_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    total_steps = len(dataloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 10, total_steps)
    criterion = nn.BCELoss()

    # Train
    print(f"\n🚀 Training {total_steps} steps...")
    model.train()
    best_auc = 0
    step = 0
    losses = []

    for epoch in range(args.epochs):
        for batch in dataloader:
            ids = batch['input_ids']
            am = batch['attention_mask']
            audio = batch['audio']
            labels_b = batch['label']
            cross = batch['cross']

            optimizer.zero_grad()
            scores, conf, gate = model(ids, am, audio, cross)
            # Flatten scores to compute loss
            scores_flat = scores.view(-1, 1)
            labels_flat = labels_b.view(-1, 1)
            loss = criterion(scores_flat.clamp(1e-6, 1-1e-6), labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % 5 == 0:
                avg = np.mean(losses[-20:])
                print(f"  Step {step}/{total_steps} | loss={avg:.4f}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    print(f"\n✅ Training done. Final loss: {np.mean(losses):.4f}")

    # Save model
    out_dir = Path('/tmp/v8_finetuned')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    model_path = out_dir / f'{args.output_name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'epochs': args.epochs, 'lr': args.lr, 'batch': args.batch_size},
        'reddit_model': args.reddit_model,
        'n_samples': n_samples,
    }, model_path, _use_new_zipfile_serialization=True)
    print(f"💾 Saved: {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")

    # Save metrics
    metrics = {
        'final_loss': float(np.mean(losses)),
        'n_samples': n_samples,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'reddit_model': args.reddit_model,
        'total_steps': step,
        'total_params': n_total,
        'trainable_params': n_train,
        'output_name': args.output_name,
    }
    metrics_path = out_dir / f'{args.output_name}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Push to Drive
    print(f"\n📤 Pushing model to Drive...")
    subprocess.run(
        ['rclone', 'copyto', str(model_path),
         f'{DRIVE_BASE}/models/{args.output_name}.pt',
         '--drive-chunk-size=64M', '--transfers=1', '--retries=3'],
        capture_output=False
    )
    print(f"📤 Pushing metrics to Drive...")
    subprocess.run(
        ['rclone', 'copyto', str(metrics_path),
         f'{DRIVE_BASE}/results/{args.output_name}.json',
         '--retries=3'],
        capture_output=False
    )

    # Cleanup
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(reddit_path.parent, ignore_errors=True)
    print(f"\n✅ v8 fine-tuned model on Drive: {args.output_name}.pt")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reddit-model', default='reddit_distilbert_smoke.pt')
    p.add_argument('--output-name', default='v8_finetuned_v1')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--freeze-text', action='store_true', default=True)
    args = p.parse_args()

    fine_tune(args)


if __name__ == "__main__":
    main()
