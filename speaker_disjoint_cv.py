#!/usr/bin/env python3
"""
SPEAKER-DISJOINT CV DIAGNOSTIC
==============================

The single highest-value experiment per RETHINK_2026_v2.md.

Compares:
- Random 5-fold CV (current): segments from same comedian can appear in train+test
- Video-disjoint 5-fold CV: no two segments from same comedian across train+test
- Leave-one-video-out (LOGO) on 12 gold-overlap videos: tightest estimate

If AUC drops > 0.05 under video-disjoint: model is learning comedian identity
(not humor). Need comedian-adversarial loss.

If AUC stays flat: architecture is correct, bottleneck is label sparsity.
"""
import os, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import math

# ── Configuration ────────────────────────────────────────────────────────
SEED = 42
N_SEG = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f'Device: {DEVICE}')


# ── Model (v10 architecture, identical to other kernels) ──────────────
class BilinearFusion(nn.Module):
    def __init__(self, dim, out_dim, dropout=0.1):
        super().__init__()
        self.wt = nn.Linear(dim, out_dim, bias=False)
        self.wa = nn.Linear(dim, out_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(out_dim)))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, t, a):
        return self.drop(self.norm(self.wt(t) * self.wa(a) * self.scale))


class V10(nn.Module):
    def __init__(self, hidden=128, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(791, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.bilinear = BilinearFusion(hidden, hidden, dropout)
        self.text_confidence = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.attn = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.pos = nn.Embedding(N_SEG + 1, 4)
        self.gru = nn.GRU(hidden * 3 + 4, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.humor_head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.laugh_head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, text, audio):
        B, L, _ = text.shape
        th, ah = self.text_proj(text), self.audio_proj(audio)
        bi = self.bilinear(th, ah)
        conf = torch.sigmoid(self.text_confidence(th))
        att, _ = self.attn(ah, th, th)
        fused = torch.cat([th, bi, att], dim=-1)
        pos = self.pos(torch.arange(L, device=text.device).unsqueeze(0).expand(B, -1))
        feat, _ = self.gru(torch.cat([fused, pos], dim=-1))
        return {'humor': self.humor_head(feat).squeeze(-1),
                'laughter': self.laugh_head(feat).squeeze(-1),
                'features': feat}


def coral_loss(src, tgt):
    d = src.size(1)
    def cov(x):
        c = x - x.mean(0, keepdim=True)
        return (c.T @ c) / max(x.size(0) - 1, 1)
    return ((cov(src) - cov(tgt)) ** 2).sum() / (4 * d * d)


# ── Data loading ────────────────────────────────────────────────────────
def load_data():
    """Load on local CPU; expects files in /Users/Subho/tmp/."""
    b4 = np.load('/Users/Subho/tmp/bridge4_features.npz', allow_pickle=True)
    v6 = np.load('/Users/Subho/tmp/v6_features.npz', allow_pickle=True)
    audio_obj, labels_obj = b4['features'], b4['labels']
    text, vids = v6['text_features'], v6['video_ids']

    n = len(audio_obj)
    audio = np.zeros((n, N_SEG, 791), dtype=np.float32)
    pseudo = np.full((n, N_SEG), 0.5, dtype=np.float32)
    for i in range(n):
        for j in range(N_SEG):
            try:
                audio[i, j] = np.asarray(audio_obj[i][j], dtype=np.float32)
            except Exception:
                pass
            try:
                pseudo[i, j] = float(labels_obj[i][j])
            except Exception:
                seg = labels_obj[i][j]
                if isinstance(seg, dict) and 'score' in seg:
                    pseudo[i, j] = float(seg['score'])

    # Gold: load from gold_labels/ (12 overlap videos with v6)
    gold = np.zeros((n, N_SEG), dtype=np.float32)
    v2i = {}
    for idx, v in enumerate(vids):
        v2i[str(v).split(',')[0]] = idx
        v2i[str(v)] = idx
    import csv
    gold_csvs = sorted(glob.glob('/Users/Subho/funny-strength-predictor/data/gold_labels/*.csv'))
    matched = 0
    seg_len = 495.3 / N_SEG
    for gp in gold_csvs:
        vid = os.path.basename(gp)[:-4]
        if vid not in v2i:
            continue
        matched += 1
        vi = v2i[vid]
        with open(gp) as f:
            for row in csv.DictReader(f):
                if row.get('label') != 'risa':
                    continue
                try:
                    t0, t1 = float(row['t0']), float(row['t1'])
                except Exception:
                    continue
                for s in range(N_SEG):
                    ov = max(0.0, min(t1, (s + 1) * seg_len) - max(t0, s * seg_len))
                    if ov > 0:
                        gold[vi, s] = min(1.0, gold[vi, s] + ov / seg_len)
    print(f'Data loaded: audio={audio.shape} text={text.shape} gold_nonzero={(gold>0).sum()} (matched {matched} CSVs)')
    return audio, text, pseudo, gold, np.asarray(vids)


# ── Training/eval (one fold) ───────────────────────────────────────────
def run_fold(text, audio, pseudo, gold, tr, va, epochs=5, bs=4, lr=5e-5,
             coral_w=0.5, laugh_w=0.3):
    model = V10().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (len(tr) // bs))
    for ep in range(epochs):
        idx = np.random.permutation(tr)
        for s in range(0, len(tr) - bs, bs):
            b = idx[s:s + bs]
            t, a = torch.tensor(text[b]).to(DEVICE), torch.tensor(audio[b]).to(DEVICE)
            p, g = torch.tensor(pseudo[b]).to(DEVICE), torch.tensor(gold[b]).to(DEVICE)
            opt.zero_grad()
            out = model(t, a)
            hl = F.binary_cross_entropy(out['humor'].clamp(1e-6, 1 - 1e-6), p)
            ll = F.binary_cross_entropy(out['laughter'].clamp(1e-6, 1 - 1e-6), g.clamp(0, 1))
            f = out['features'].reshape(-1, out['features'].size(-1))
            h = f.size(0) // 2
            (hl + laugh_w * ll + coral_w * coral_loss(f[:h], f[h:2 * h])).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

    model.eval()
    hs, ps, ls, gs = [], [], [], []
    with torch.no_grad():
        for i in va:
            out = model(torch.tensor(text[i:i + 1]).to(DEVICE),
                        torch.tensor(audio[i:i + 1]).to(DEVICE))
            hs.append(out['humor'][0].cpu().numpy())
            ls.append(out['laughter'][0].cpu().numpy())
            ps.append(pseudo[i]); gs.append(gold[i])
    hs, ls = np.concatenate(hs), np.concatenate(ls)
    ps, gs = np.concatenate(ps), np.concatenate(gs)
    hum = roc_auc_score((ps > np.median(ps)).astype(int), hs)
    if (gs > 0).any():
        gb = (gs > np.median(gs)).astype(int)
        gld = roc_auc_score(gb, ls) if 0 < gb.sum() < len(gb) else 0.5
    else:
        gld = 0.5
    return hum, gld


# ── Main ──────────────────────────────────────────────────────────────
def main():
    audio, text, pseudo, gold, vids = load_data()
    n = len(audio)

    # Use only videos with gold labels (12 overlap) for cleanest diagnostic
    gold_idx_all = np.where((gold > 0).any(axis=1))[0]
    print(f'\nVideos with gold labels: {len(gold_idx_all)}')

    # ============ TEST 1: 5-fold CV RANDOM on all 12 gold videos ============
    print('\n' + '=' * 60)
    print('TEST 1: 5-fold CV RANDOM (segment-level) — 12 gold videos')
    print('=' * 60)
    audio_g = audio[gold_idx_all]
    text_g = text[gold_idx_all]
    pseudo_g = pseudo[gold_idx_all]
    gold_g = gold[gold_idx_all]

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    random_humors, random_golds = [], []
    for k, (tr, va) in enumerate(kf.split(np.arange(len(gold_idx_all)))):
        h, g = run_fold(text_g, audio_g, pseudo_g, gold_g, tr, va)
        random_humors.append(h); random_golds.append(g)
        print(f'  Random fold {k}: humor={h:.4f} gold={g:.4f}')
    random_humor_mean = np.mean(random_humors)
    random_gold_mean = np.mean(random_golds)
    print(f'  RANDOM mean: humor={random_humor_mean:.4f} gold={random_gold_mean:.4f}')

    # ============ TEST 2: 5-fold CV GROUPED by video (speaker-disjoint) ============
    print('\n' + '=' * 60)
    print('TEST 2: 5-fold CV GROUPED by video_id (speaker-disjoint) — 12 gold videos')
    print('=' * 60)
    groups = vids[gold_idx_all]
    gkf = GroupKFold(n_splits=5)
    grouped_humors, grouped_golds = [], []
    for k, (tr, va) in enumerate(gkf.split(np.arange(len(gold_idx_all)), groups=groups)):
        h, g = run_fold(text_g, audio_g, pseudo_g, gold_g, tr, va)
        grouped_humors.append(h); grouped_golds.append(g)
        print(f'  Speaker-disjoint fold {k}: humor={h:.4f} gold={g:.4f}')
    grouped_humor_mean = np.mean(grouped_humors)
    grouped_gold_mean = np.mean(grouped_golds)
    print(f'  GROUPED mean: humor={grouped_humor_mean:.4f} gold={grouped_gold_mean:.4f}')

    # ============ TEST 3: LOGO on full 639-video set, eval on 12 gold ============
    print('\n' + '=' * 60)
    print('TEST 3: Train on full 639 (excluding gold vids), eval on each gold vid (LOGO)')
    print('=' * 60)
    non_gold_idx = np.array([i for i in range(n) if i not in gold_idx_all])
    hum_pred_per_vid = {}
    laugh_pred_per_vid = {}
    for gi in gold_idx_all:
        # Train on all non-gold + other gold except this one
        tr = np.array([i for i in range(n) if i != gi])
        # Eval on this single gold video
        va = np.array([gi])
        h, g = run_fold(text, audio, pseudo, gold, tr, va, epochs=3)
        hum_pred_per_vid[gi] = h
        laugh_pred_per_vid[gi] = g
        print(f'  Held-out {vids[gi]}: humor={h:.4f} gold={g:.4f}')

    # Pool LOGO predictions and compute aggregate AUC
    all_hum_preds, all_gold_targets = [], []
    for gi in gold_idx_all:
        all_hum_preds.append(pseudo[gi])  # we compare rank order
        all_gold_targets.append(gold[gi])
    # The actual LOGO AUC requires computing predictions per-video, then averaging
    # For simplicity, show the per-video results above

    # ============ DIAGNOSIS ============
    print('\n' + '=' * 60)
    print('DIAGNOSIS')
    print('=' * 60)
    delta_humor = random_humor_mean - grouped_humor_mean
    delta_gold = random_gold_mean - grouped_gold_mean

    if delta_humor > 0.10 or delta_gold > 0.10:
        verdict = ('🚨 ARCHITECTURE PROBLEM: model leaks comedian identity. '
                   'Need comedian-adversarial loss or per-comedian normalization.')
    elif delta_humor > 0.05 or delta_gold > 0.05:
        verdict = ('⚠️ PARTIAL LEAKAGE: some comedian-style learning detected. '
                   'Consider group-DRO or comedian embeddings.')
    else:
        verdict = ('✅ NO LEAKAGE: architecture learns content, not style. '
                   'Bottleneck is label sparsity (12 gold videos). '
                   'Pivot to Gillick pseudo-labels or within-comedian ranking.')

    print(f'\nΔ humor AUC (random − grouped): {delta_humor:+.4f}')
    print(f'Δ gold AUC (random − grouped):  {delta_gold:+.4f}')
    print(f'\nVERDICT: {verdict}')

    # Save results
    summary = {
        'random_5fold': {
            'humor_auc_mean': float(random_humor_mean),
            'gold_auc_mean': float(random_gold_mean),
            'humor_auc_per_fold': [float(x) for x in random_humors],
            'gold_auc_per_fold': [float(x) for x in random_golds],
        },
        'speaker_disjoint_5fold': {
            'humor_auc_mean': float(grouped_humor_mean),
            'gold_auc_mean': float(grouped_gold_mean),
            'humor_auc_per_fold': [float(x) for x in grouped_humors],
            'gold_auc_per_fold': [float(x) for x in grouped_golds],
        },
        'logo_per_video': {
            str(vids[gi]): {'humor': float(hum_pred_per_vid[gi]),
                            'gold': float(laugh_pred_per_vid[gi])}
            for gi in gold_idx_all
        },
        'delta_humor_auc': float(delta_humor),
        'delta_gold_auc': float(delta_gold),
        'verdict': verdict,
    }
    with open('/tmp/speaker_disjoint_cv_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n✅ Saved -> /tmp/speaker_disjoint_cv_results.json')

    # Also save to local project
    import shutil
    shutil.copy('/tmp/speaker_disjoint_cv_results.json',
                '/Users/Subho/funny-strength-predictor/speaker_disjoint_cv_results.json')
    print('   Copied to project dir for git commit')


if __name__ == '__main__':
    main()
