"""Train Reddit → save state_dict → stream upload to Drive"""
import os
os.environ['HF_HOME'] = '/tmp/hf_distilbert'  # Use pre-populated cache
os.environ['PYTHONUNBUFFERED'] = '1'
import argparse
import subprocess
import sys
import time
import warnings
import tempfile
import shutil
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

DRIVE_MODEL_BASE = "gdrive:/HaHaScore_Pretrain/models"

def is_valid(text, min_len=30):
    if not text or not isinstance(text, str) or len(text) < min_len or len(text) > 600:
        return False
    return not any(s in text.lower() for s in ['[deleted]', '[removed]', 'http://', 'https://'])

class StreamDS(IterableDataset):
    def __init__(self, tok, max_samples=10000, max_len=96):
        self.tok = tok
        self.max_samples = max_samples
        self.max_len = max_len
        self._n = 0
    def __iter__(self):
        ds = load_dataset('SocialGrep/one-million-reddit-jokes', streaming=True, split='train')
        buf = []
        import random; random.seed(42)
        for s in ds:
            t = ((s.get('title') or '') + '. ' + (s.get('selftext') or '')).strip()
            if not is_valid(t):
                continue
            try:
                f = np.log1p(float(s.get('score', 0))) / 12.0
            except:
                continue
            buf.append((t, f))
            if len(buf) >= 5000:
                random.shuffle(buf)
                for tt, ff in buf:
                    if self._n >= self.max_samples:
                        return
                    enc = self.tok(tt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
                    yield {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'label': torch.tensor(ff, dtype=torch.float32)}
                    self._n += 1
                buf = []

class Reg(nn.Module):
    def __init__(self, bb='distilbert-base-uncased'):
        super().__init__()
        self.bb = AutoModel.from_pretrained(bb)
        h = self.bb.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, ids, am):
        out = self.bb(input_ids=ids, attention_mask=am)
        m = am.unsqueeze(-1).float()
        p = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(p).squeeze(-1)

def upload_single(local_path, remote_subfolder, timeout=300):
    """Upload with shorter timeout per attempt + retry."""
    cmd = [
        'rclone', 'copyto', str(local_path),
        f'{DRIVE_MODEL_BASE}/{remote_subfolder}',
        '--progress', '--transfers=1', '--checkers=1',
        '--retries=2', '--retries-sleep=10s',
        '--low-level-retries=1', '--timeout=60s',
        '--drive-chunk-size=64M'
    ]
    print(f"📤 Uploading: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=timeout)
    return result.returncode == 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--max-len', type=int, default=96)
    p.add_argument('--name', default='reddit_distilbert_v1')
    args = p.parse_args()
    
    print(f"🚀 Training {args.samples} samples, {args.epochs} epoch(s)", flush=True)
    
    tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = Reg()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 50, args.samples//args.batch * args.epochs)
    crit = nn.MSELoss()
    
    dl = DataLoader(StreamDS(tok, args.samples, args.max_len), batch_size=args.batch, num_workers=0)
    
    print(f"📊 Training...", flush=True)
    step = 0
    total = args.samples // args.batch * args.epochs
    t0 = time.time()
    losses = []
    for batch in dl:
        ids = batch['input_ids']
        am = batch['attention_mask']
        lab = batch['label']
        opt.zero_grad()
        p_ = model(ids, am)
        loss = crit(p_, lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        losses.append(loss.item())
        if step % 50 == 0:
            avg = np.mean(losses[-50:])
            print(f"  Step {step}/{total} | loss={avg:.4f}", flush=True)
        if step >= total:
            break
    
    final_loss = np.mean(losses)
    print(f"\n✅ Training done. Final loss: {final_loss:.4f}, time: {(time.time()-t0)/60:.1f}min", flush=True)
    
    # Save state dict only (smaller than full torch.save)
    model_dir = Path(f'/tmp/{args.name}_save')
    model_dir.mkdir(exist_ok=True)
    state_path = model_dir / f'{args.name}.pt'
    
    # Save with weights_only=False (PyTorch 2.6 compat)
    torch.save(model.state_dict(), state_path, _use_new_zipfile_serialization=True)
    print(f"💾 Saved state_dict: {state_path} ({state_path.stat().st_size/1e6:.1f}MB)", flush=True)
    
    # Save metadata
    import json
    meta = {'final_loss': final_loss, 'samples': args.samples, 'epochs': args.epochs, 'steps': step}
    with open(model_dir / f'{args.name}.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Upload model
    print(f"\n📤 Uploading model to Drive...", flush=True)
    if upload_single(state_path, f'{args.name}.pt'):
        print(f"✅ Model uploaded", flush=True)
        state_path.unlink()
        print(f"🧹 Local model deleted", flush=True)
    else:
        print(f"❌ Upload failed", flush=True)
        sys.exit(1)
    
    # Upload metadata
    meta_path = model_dir / f'{args.name}.json'
    if upload_single(meta_path, f'{args.name}.json'):
        print(f"✅ Metrics uploaded", flush=True)
        meta_path.unlink()
    
    shutil.rmtree(model_dir, ignore_errors=True)
    print(f"\n🎉 DONE", flush=True)

if __name__ == "__main__":
    main()
