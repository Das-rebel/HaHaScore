#!/usr/bin/env python3
"""
HaHaScore v10 — 5-Fold Cross-Validation (Kaggle GPU)
=====================================================
Self-contained: loads features from /kaggle/input/hahascore-features/
Architecture: CascadeGateFusionV9 (bilinear) + multi-task heads + CORAL
Outputs: per-fold + mean±std humor/gold AUC → /kaggle/working/results
"""
import os, json, csv, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_SEG = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'Device: {DEVICE}')

# Auto-discover data files under /kaggle/input (mount path varies)
import glob
def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    assert hits, f'{name} not found; input tree: ' + '\n'.join(glob.glob('/kaggle/input/**', recursive=True)[:40])
    print(f'{name} -> {hits[0]}')
    return hits[0]

B4_PATH = find('bridge4_features.npz')
V6_PATH = find('v6_features.npz')
GOLD_DIR = os.path.dirname(find('gold_labels/LWYfo_8t5WQ.csv')) if glob.glob('/kaggle/input/**/gold_labels/*.csv', recursive=True) else None
print(f'gold_labels dir: {GOLD_DIR}')


# ── Data ────────────────────────────────────────────────────────────────
def load_data():
    b4 = np.load(B4_PATH, allow_pickle=True)
    v6 = np.load(V6_PATH, allow_pickle=True)
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

    # Gold laughter: distribute 'risa' intervals over 20 segments
    gold = np.zeros((n, N_SEG), dtype=np.float32)
    v2i = {}
    for idx, v in enumerate(vids):
        v2i[str(v).split(',')[0]] = idx
        v2i[str(v)] = idx
    gdir = GOLD_DIR
    if os.path.isdir(gdir):
        total_dur, seg_len = 495.3, 495.3 / N_SEG
        for gf in os.listdir(gdir):
            if not gf.endswith('.csv'):
                continue
            vid = gf[:-4]
            if vid not in v2i:
                continue
            vi = v2i[vid]
            with open(os.path.join(gdir, gf)) as f:
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
    print(f'Data: audio={audio.shape} text={text.shape} pseudo={pseudo.shape} '
          f'gold_nonzero={(gold > 0).sum()}')
    return audio, text, pseudo, gold


# ── Architecture (v9 backbone + multi-task) ─────────────────────────────
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
        th = self.text_proj(text)
        ah = self.audio_proj(audio)
        bi = self.bilinear(th, ah)
        conf = torch.sigmoid(self.text_confidence(th))
        att, _ = self.attn(ah, th, th)
        fused = torch.cat([th, bi, att], dim=-1)
        pos = self.pos(torch.arange(L, device=text.device).unsqueeze(0).expand(B, -1))
        fused = torch.cat([fused, pos], dim=-1)
        feat, _ = self.gru(fused)
        return {'humor': self.humor_head(feat).squeeze(-1),
                'laughter': self.laugh_head(feat).squeeze(-1),
                'features': feat}


def coral_loss(src, tgt):
    d = src.size(1)
    def cov(x):
        c = x - x.mean(0, keepdim=True)
        return (c.T @ c) / max(x.size(0) - 1, 1)
    return ((cov(src) - cov(tgt)) ** 2).sum() / (4 * d * d)


# ── Train / Eval ────────────────────────────────────────────────────────
def run_fold(audio, text, pseudo, gold, tr, va, epochs=5, bs=4, lr=5e-5,
             coral_w=0.5, laugh_w=0.3):
    model = V10().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for ep in range(epochs):
        idx = np.random.permutation(tr)
        for s in range(0, len(tr) - bs, bs):
            b = idx[s:s + bs]
            t = torch.tensor(text[b]).to(DEVICE)
            a = torch.tensor(audio[b]).to(DEVICE)
            p = torch.tensor(pseudo[b]).to(DEVICE)
            g = torch.tensor(gold[b]).to(DEVICE)
            opt.zero_grad()
            out = model(t, a)
            hl = F.binary_cross_entropy(out['humor'].clamp(1e-6, 1 - 1e-6), p)
            ll = F.binary_cross_entropy(out['laughter'].clamp(1e-6, 1 - 1e-6), g.clamp(0, 1))
            f = out['features'].reshape(-1, out['features'].size(-1))
            h = f.size(0) // 2
            cl = coral_loss(f[:h], f[h:2 * h])
            (hl + laugh_w * ll + coral_w * cl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

    # Eval on held-out fold (ALL videos in it)
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


def main():
    audio, text, pseudo, gold = load_data()
    n = len(audio)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for k, (tr, va) in enumerate(kf.split(np.arange(n))):
        hum, gld = run_fold(audio, text, pseudo, gold, tr, va)
        results.append({'fold': k, 'humor_auc': float(hum), 'gold_auc': float(gld),
                        'n_train': int(len(tr)), 'n_val': int(len(va))})
        print(f'Fold {k}: humor={hum:.4f} gold={gld:.4f}', flush=True)

    hums = [r['humor_auc'] for r in results]
    glds = [r['gold_auc'] for r in results]
    summary = {
        'model': 'v10 (bilinear + CORAL multi-task)',
        'humor_auc_mean': float(np.mean(hums)), 'humor_auc_std': float(np.std(hums)),
        'gold_auc_mean': float(np.mean(glds)), 'gold_auc_std': float(np.std(glds)),
        'folds': results,
        'baseline': {'v7_pseudo': 0.860, 'v8': 0.802, 'v10_single_fold': 0.8225},
    }
    os.makedirs('/kaggle/working', exist_ok=True)
    with open('/kaggle/working/cv5_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n' + '=' * 50)
    print(f"HUMOR AUC: {summary['humor_auc_mean']:.4f} ± {summary['humor_auc_std']:.4f}")
    print(f"GOLD  AUC: {summary['gold_auc_mean']:.4f} ± {summary['gold_auc_std']:.4f}")
    print('=' * 50)


if __name__ == '__main__':
    main()
