#!/usr/bin/env python3
"""
Reddit Text Baseline Trainer — STREAMING VERSION
=================================================

Memory-conscious trainer that streams Reddit jokes directly from HuggingFace
without ever materializing the full 299MB CSV locally.

Strategy:
1. Streaming dataset loader — never download full CSV to disk
2. Tokenize on-the-fly (small batches, memory-light)
3. Save model checkpoints directly via rclone to Drive
4. Local file footprint: ~200MB (model only)

Local disk budget at start of training: ~600MB free.
"""
import os
os.environ['HF_HOME'] = '/tmp/hf_cache_stream'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache_stream'

import argparse
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score

DRIVE_MODEL_BASE = "gdrive:/HaHaScore_Pretrain/models"
DRIVE_RESULTS_BASE = "gdrive:/HaHaScore_Pretrain/results"


def is_valid_joke(text: str, min_len: int = 30) -> bool:
    """Filter criteria for usable jokes."""
    if not text or not isinstance(text, str):
        return False
    if len(text) < min_len:
        return False
    if len(text) > 600:  # Truncate long jokes
        return False
    # Quality filters
    bad_substrings = ['[deleted]', '[removed]', 'http://', 'https://']
    return not any(s in text.lower() for s in bad_substrings)


def normalize_funniness(scores_iter):
    """Online normalization of log scores to [0, 1]."""
    # We can't compute exact min/max online, so use log compression with fixed scale
    # Reddit scores are heavy-tailed: log1p(x) ≈ 0..12
    def _norm(s):
        try:
            v = float(s)
        except (ValueError, TypeError):
            return None
        return np.log1p(v) / 12.0  # log(1+142733)/log(e) ≈ 11.87
    return _norm


class StreamingJokeDataset(IterableDataset):
    """Stream jokes from HF, filter, tokenize, yield batches."""

    def __init__(self, tokenizer, max_samples=None, max_len=128, seed=42):
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_len = max_len
        self.seed = seed
        self._counter = 0

    def __iter__(self):
        # Stream dataset
        ds = load_dataset(
            'SocialGrep/one-million-reddit-jokes',
            streaming=True,
            split='train'
        )
        norm = normalize_funniness(None)

        # Shuffle buffer for variety
        buffer = []
        buffer_size = 10000

        import random
        random.seed(self.seed)

        for sample in ds:
            # Build text from title + body
            title = sample.get('title', '') or ''
            body = sample.get('selftext', '') or ''
            text = (title + '. ' + body).strip()
            score_str = sample.get('score', '0')

            if not is_valid_joke(text):
                continue

            funniness = norm(score_str)
            if funniness is None:
                continue

            buffer.append((text, funniness))

            if len(buffer) >= buffer_size:
                random.shuffle(buffer)
                for t, f in buffer:
                    if self.max_samples and self._counter >= self.max_samples:
                        return
                    yield self._encode(t, f)
                    self._counter += 1
                buffer = []

        # Drain buffer
        random.shuffle(buffer)
        for t, f in buffer:
            if self.max_samples and self._counter >= self.max_samples:
                return
            yield self._encode(t, f)
            self._counter += 1

    def _encode(self, text, funniness):
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(funniness, dtype=torch.float32)
        }


class HumorRegressor(nn.Module):
    """DistilBERT regressor with mean pooling + sigmoid head."""

    def __init__(self, backbone_name='distilbert-base-uncased'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(pooled).squeeze(-1)


def train_streaming(
    output_name: str = 'reddit_distilbert_v1',
    max_samples: int = 30000,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 2e-5,
    backbone: str = 'distilbert-base-uncased',
    max_len: int = 128,
    eval_steps: int = 500,
):
    """Stream-train a humor regressor on Reddit upvotes."""
    print("=" * 60)
    print(f"🚀 Streaming Reddit Trainer")
    print(f"   Backbone: {backbone}")
    print(f"   Samples: {max_samples}, Epochs: {epochs}, LR: {lr}")
    print(f"   Batch: {batch_size}, MaxLen: {max_len}, Eval every: {eval_steps} steps")
    print("=" * 60)

    # Load tokenizer
    print(f"Loading tokenizer: {backbone}")
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    # Build datasets
    train_ds = StreamingJokeDataset(
        tokenizer, max_samples=max_samples, max_len=max_len
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=0
    )

    # Build model
    print(f"Loading model: {backbone}")
    model = HumorRegressor(backbone)
    device = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    model.to(device)

    # Optimizer + scheduler
    # Approximate steps for LR schedule
    steps_per_epoch = max_samples // batch_size
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=total_steps
    )
    criterion = nn.MSELoss()

    # Eval buffer (small, fixed size from streamed data)
    print(f"\n📊 Training (streaming, {total_steps} steps)...")
    best_loss = float('inf')
    step = 0
    t0 = time.time()
    train_loss_sum = 0
    train_loss_n = 0

    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attn_mask)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_loss_n += 1
            step += 1

            if step % 50 == 0:
                avg = train_loss_sum / train_loss_n
                rate = step / (time.time() - t0 + 1e-9)
                mem_pct = 0  # placeholder
                print(f"  Step {step:5d}/{total_steps} | loss={avg:.4f} | {rate:.1f} it/s")

            if step >= total_steps:
                break

        if step >= total_steps:
            break

    final_loss = train_loss_sum / train_loss_n
    elapsed = time.time() - t0
    print(f"\n✅ Training complete in {elapsed/60:.1f} min, final loss: {final_loss:.4f}")

    # Save to local temp
    temp_dir = Path(tempfile.gettempdir()) / "reddit_stream_save"
    temp_dir.mkdir(exist_ok=True)

    model_path = temp_dir / f"{output_name}.pt"
    metrics_path = temp_dir / f"{output_name}.json"

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'backbone': backbone,
        'final_loss': final_loss,
        'max_samples': max_samples,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'max_len': max_len,
        'total_steps': step,
        'training_time_min': elapsed / 60,
    }, model_path)

    # Save metrics
    import json
    metrics = {
        'final_loss': final_loss,
        'max_samples': max_samples,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'backbone': backbone,
        'training_steps': step,
        'training_time_min': round(elapsed / 60, 2),
        'device': device,
        'note': 'Trained on log-normalized upvote scores as continuous funniness proxy'
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Saved to: {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")

    # Push to Drive
    print(f"\n📤 Pushing model to Drive: {DRIVE_MODEL_BASE}/")
    subprocess.run(
        ['rclone', 'copy', str(model_path), f"{DRIVE_MODEL_BASE}/", '--progress'],
        check=True
    )
    print(f"📤 Pushing metrics to Drive: {DRIVE_RESULTS_BASE}/")
    subprocess.run(
        ['rclone', 'copy', str(metrics_path), f"{DRIVE_RESULTS_BASE}/", '--progress'],
        check=True
    )

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    # Clean HF cache to free disk
    hf_cache = '/tmp/hf_cache_stream'
    if os.path.exists(hf_cache):
        shutil.rmtree(hf_cache, ignore_errors=True)
        print(f"🧹 Cleaned HF cache")

    print(f"\n✅ DONE — Model on Drive, local cleaned")
    print(f"   Drive: {DRIVE_MODEL_BASE}/{output_name}.pt")


def main():
    parser = argparse.ArgumentParser(description="Streaming Reddit trainer")
    parser.add_argument('--output-name', default='reddit_distilbert_v1')
    parser.add_argument('--max-samples', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--backbone', default='distilbert-base-uncased')
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--eval-steps', type=int, default=500)
    args = parser.parse_args()

    train_streaming(
        output_name=args.output_name,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        max_len=args.max_len,
        eval_steps=args.eval_steps,
    )


if __name__ == "__main__":
    main()
