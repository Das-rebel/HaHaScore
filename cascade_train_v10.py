#!/usr/bin/env python3
"""
Cascade Gate v10 — Bilinear + CORAL Multi-task
==============================================

Best architecture combination:
- v9 BilinearFusion: multiplicative cross-modal interaction
- Cascade Gate: text confidence gates audio
- CORAL: covariance alignment between pseudo (source) and gold (target)
- Multi-task: humor + laughter heads share backbone

Initialize from v9_bilinear_v1 weights.
"""
import os
os.environ['HF_HOME'] = '/tmp/hf_distilbert'
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '/Users/Subho/funny-strength-predictor')
from cascade_train_v9 import CascadeGateFusionV9

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain"


def pull_v9_model(remote_name="v9_bilinear_v1.pt"):
    temp = Path('/tmp/v10_pull')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    subprocess.run(
        ['rclone', 'copyto', f'{DRIVE_BASE}/models/{remote_name}',
         str(temp / 'v9.pt'), '--retries=3', '--retries-sleep=10s'],
        capture_output=True, text=True, timeout=120
    )
    path = temp / 'v9.pt'
    if path.exists():
        print(f"✅ Pulled {path.stat().st_size/1e6:.1f} MB")
        return path
    return None


def load_standup_data():
    print("📊 Loading standup data...")
    b4 = np.load('/Users/Subho/tmp/bridge4_features.npz', allow_pickle=True)
    v6 = np.load('/Users/Subho/tmp/v6_features.npz', allow_pickle=True)

    audio_obj = b4['features']
    labels_obj = b4['labels']
    text = v6['text_features']

    n_use = 639
    audio = np.zeros((n_use, 20, 791), dtype=np.float32)
    pseudo = np.zeros((n_use, 20), dtype=np.float32)
    for i in range(n_use):
        for j in range(20):
            try:
                audio[i, j] = np.array(audio_obj[i][j], dtype=np.float32)
                pseudo[i, j] = float(labels_obj[i][j])
            except (ValueError, TypeError):
                seg = labels_obj[i][j]
                if isinstance(seg, dict) and 'score' in seg:
                    pseudo[i, j] = float(seg['score'])
                else:
                    pseudo[i, j] = 0.5

    # Load gold laughter
    gold_path = Path('/tmp/gold_laughter.npy')
    if gold_path.exists():
        gold = np.load(gold_path)
    else:
        gold = np.zeros((n_use, 20), dtype=np.float32)

    print(f"  Audio: {audio.shape}")
    print(f"  Text: {text.shape}")
    print(f"  Pseudo: {pseudo.shape}, mean={pseudo.mean():.3f}")
    print(f"  Gold: {gold.shape}, mean={gold.mean():.4f}, nonzero={(gold > 0).sum()}")
    return audio, text, pseudo, gold


def coral_loss(source_feat, target_feat):
    """CORAL: align covariance matrices."""
    d = source_feat.size(1)
    ns, nt = source_feat.size(0), target_feat.size(0)

    src_mean = source_feat.mean(dim=0, keepdim=True)
    src_centered = source_feat - src_mean
    src_cov = (src_centered.T @ src_centered) / max(ns - 1, 1)

    tgt_mean = target_feat.mean(dim=0, keepdim=True)
    tgt_centered = target_feat - tgt_mean
    tgt_cov = (tgt_centered.T @ tgt_centered) / max(nt - 1, 1)

    return ((src_cov - tgt_cov) ** 2).sum() / (4 * d * d)


class MultiTaskV10(nn.Module):
    """v9 architecture + multi-task (humor + laughter) heads."""

    def __init__(self, v9_state):
        super().__init__()
        self.backbone = CascadeGateFusionV9(text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3)
        self.backbone.load_state_dict(v9_state)

        # Multi-task heads (after backbone BiGRU, output = 256-dim)
        self.humor_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.laughter_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, text, audio, lengths=None):
        # v9 forward returns scores, text_conf
        # We need intermediate features for CORAL
        # Manually replicate backbone forward to capture features

        batch_size, seq_len, _ = text.shape

        text_h = self.backbone.text_proj(text)
        audio_h = self.backbone.audio_proj(audio)
        bilinear_out = self.backbone.bilinear(text_h, audio_h)
        text_conf = torch.sigmoid(self.backbone.text_confidence(text_h))
        gated_audio = audio_h * text_conf
        text_attn_out, _ = self.backbone.audio_to_text_attn(audio_h, text_h, text_h)

        # Fuse (v9 uses bilinear, not gated_audio in concat)
        fused = torch.cat([text_h, bilinear_out, text_attn_out], dim=-1)

        # Position embedding
        positions = torch.arange(seq_len, device=text.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.backbone.pos_embedding(positions)
        fused = torch.cat([fused, pos_emb], dim=-1)

        # BiGRU
        temporal_output, _ = self.backbone.gru(fused)  # (batch, seq, 256)

        # Multi-task heads
        humor_score = self.humor_head(temporal_output).squeeze(-1)
        laughter_score = self.laughter_head(temporal_output).squeeze(-1)

        return {
            'humor': humor_score,
            'laughter': laughter_score,
            'features': temporal_output,
            'text_conf': text_conf.squeeze(-1),
        }


def train_v10(args):
    print("=" * 60)
    print("🎯 Cascade Gate v10 — Bilinear + CORAL Multi-task")
    print("=" * 60)

    # Pull v9
    v9_path = pull_v9_model(args.v9_model)
    if not v9_path:
        return False

    v9_state = torch.load(v9_path, map_location='cpu')['model_state_dict']

    # Load data
    audio, text, pseudo, gold = load_standup_data()
    n_use = min(args.max_samples, len(audio))

    # Build v10
    model = MultiTaskV10(v9_state)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable_params)
    print(f"\nv10 model: {n_total:,} total, {n_train:,} trainable")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * (n_use // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Convert audio
    audio_arr = np.zeros((n_use, 20, 791), dtype=np.float32)
    for i in range(n_use):
        for j in range(20):
            try:
                audio_arr[i, j] = np.array(audio[i][j], dtype=np.float32)
            except:
                audio_arr[i, j] = 0.0

    print(f"\n🚀 Training {total_steps} steps (CORAL λ={args.coral_weight}, laugh λ={args.laughter_weight})...")
    step = 0
    losses = []

    for epoch in range(args.epochs):
        indices = np.random.permutation(n_use)
        for batch_start in range(0, n_use - args.batch_size, args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]

            audio_b = torch.tensor(audio_arr[batch_idx])
            text_b = torch.tensor(text[batch_idx])
            pseudo_b = torch.tensor(pseudo[batch_idx])
            gold_b = torch.tensor(gold[batch_idx])

            optimizer.zero_grad()
            out = model(text_b, audio_b)

            # Humor loss (continuous)
            humor_loss = F.binary_cross_entropy(
                out['humor'].clamp(1e-6, 1-1e-6), pseudo_b
            )
            # Laughter loss (sparse binary)
            laughter_loss = F.binary_cross_entropy(
                out['laughter'].clamp(1e-6, 1-1e-6), gold_b.clamp(0, 1)
            )
            # CORAL on features
            feat_flat = out['features'].reshape(-1, out['features'].size(-1))
            half = feat_flat.size(0) // 2
            coral = coral_loss(feat_flat[:half], feat_flat[half:half*2])

            total = humor_loss + args.laughter_weight * laughter_loss + args.coral_weight * coral
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append({
                'humor': humor_loss.item(),
                'laughter': laughter_loss.item(),
                'coral': coral.item(),
            })
            step += 1
            if step % 20 == 0:
                avg = {k: np.mean([x[k] for x in losses[-20:]]) for k in losses[0]}
                print(f"  Step {step}/{total_steps} | h={avg['humor']:.3f} l={avg['laughter']:.3f} c={avg['coral']:.3f}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    # Validation
    print(f"\n📊 Validating...")
    model.eval()
    all_humor, all_pseudo = [], []
    all_laugh, all_gold = [], []

    with torch.no_grad():
        for i in range(min(50, n_use)):
            text_i = torch.tensor(text[i:i+1])
            audio_i = torch.tensor(audio_arr[i:i+1])
            out = model(text_i, audio_i)
            all_humor.append(out['humor'].cpu().numpy().flatten())
            all_pseudo.append(pseudo[i])
            all_laugh.append(out['laughter'].cpu().numpy().flatten())
            all_gold.append(gold[i])

    y_humor = np.concatenate(all_humor)
    y_pseudo = np.concatenate(all_pseudo)
    y_laugh = np.concatenate(all_laugh)
    y_gold = np.concatenate(all_gold)

    # Humor AUC
    median_p = np.median(y_pseudo)
    humor_auc = roc_auc_score((y_pseudo > median_p).astype(int), y_humor)

    # Laughter AUC
    if (y_gold > 0).any():
        median_g = np.median(y_gold)
        y_gold_bin = (y_gold > median_g).astype(int)
        if y_gold_bin.sum() > 0 and y_gold_bin.sum() < len(y_gold_bin):
            laugh_auc = roc_auc_score(y_gold_bin, y_laugh)
        else:
            laugh_auc = 0.5
    else:
        laugh_auc = 0.5

    print(f"\n🎯 Final Results:")
    print(f"   Humor AUC (pseudo): {humor_auc:.4f}")
    print(f"   Laughter AUC (gold): {laugh_auc:.4f}")

    # Save
    out_dir = Path('/tmp/v10_save')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    model_path = out_dir / f'{args.output_name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'humor_auc': float(humor_auc),
        'laugh_auc': float(laugh_auc),
        'config': vars(args),
    }, model_path, _use_new_zipfile_serialization=True)

    metrics = {
        'humor_auc': float(humor_auc),
        'laugh_auc': float(laugh_auc),
        'config': vars(args),
        'n_params': n_total,
        'trainable_params': n_train,
        'architecture': 'v9 Bilinear Fusion + CORAL Multi-task',
    }
    metrics_path = out_dir / f'{args.output_name}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Push to Drive
    print(f"\n📤 Pushing to Drive...")
    subprocess.run(
        ['rclone', 'copyto', str(model_path),
         f'{DRIVE_BASE}/models/{args.output_name}.pt',
         '--drive-chunk-size=64M', '--retries=3'],
        capture_output=False
    )
    subprocess.run(
        ['rclone', 'copyto', str(metrics_path),
         f'{DRIVE_BASE}/results/{args.output_name}.json',
         '--retries=3'],
        capture_output=False
    )

    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(v9_path.parent, ignore_errors=True)
    print(f"\n✅ v10 model on Drive: {args.output_name}.pt (humor={humor_auc:.4f}, gold={laugh_auc:.4f})")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v9-model', default='v9_bilinear_v1.pt')
    p.add_argument('--output-name', default='v10_bilinear_coral')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--max-samples', type=int, default=639)
    p.add_argument('--coral-weight', type=float, default=0.5)
    p.add_argument('--laughter-weight', type=float, default=0.3)
    args = p.parse_args()
    train_v10(args)


if __name__ == "__main__":
    main()
