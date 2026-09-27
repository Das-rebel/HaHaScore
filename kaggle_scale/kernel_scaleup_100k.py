#!/usr/bin/env python3
"""
HaHaScore SCALE-UP (Kaggle GPU): Reddit-100K pretrain -> segment re-encoding -> v10 5-fold CV
=============================================================================================
Stage A: Pretrain DistilBERT on 100K Reddit jokes (upvote funniness) — 3.3x more data
Stage B: Encode all 639x20 standup segment texts with the pretrained encoder
Stage C: Fine-tune v10 (bilinear + CORAL multi-task) with new features, 5-fold CV
Outputs: encoder weights, new text features, CV results
"""
import os, json, csv, math, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

SEED = 42
N_SEG = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'Device: {DEVICE}')


def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    assert hits, f'{name} not found'
    print(f'{name} -> {hits[0]}')
    return hits[0]


# ═══════════════════════ STAGE A: Reddit-100K pretrain ═══════════════════════
class JokeDS(Dataset):
    def __init__(self, df, tok, max_len=128):
        self.texts = df['text'].tolist()
        self.labels = df['funniness'].tolist()
        self.tok, self.ml = tok, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        e = self.tok(self.texts[i], max_length=self.ml, padding='max_length',
                     truncation=True, return_tensors='pt')
        return {'input_ids': e['input_ids'][0], 'attention_mask': e['attention_mask'][0],
                'label': torch.tensor(self.labels[i], dtype=torch.float32)}


class Regressor(nn.Module):
    def __init__(self, bb='distilbert-base-uncased'):
        super().__init__()
        self.bb = AutoModel.from_pretrained(bb)
        h = self.bb.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, ids, am):
        out = self.bb(input_ids=ids, attention_mask=am)
        m = am.unsqueeze(-1).float()
        p = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(p).squeeze(-1)


def stage_a(epochs=2, bs=32, lr=2e-5):
    print('\n' + '=' * 60)
    print('STAGE A: Reddit-100K pretrain')
    print('=' * 60)
    df = pd.read_csv(find('reddit_100k.csv'))
    if 'funniness' not in df.columns:
        ls = np.log1p(df['score'].astype(float).values)
        df['funniness'] = (ls - ls.min()) / (ls.max() - ls.min() + 1e-8)
    df = df.dropna(subset=['text', 'funniness'])
    df['text'] = df['text'].astype(str).str.slice(0, 512)
    print(f'Jokes: {len(df):,}, funniness mean={df.funniness.mean():.3f}')

    tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tr, va = df.sample(frac=0.95, random_state=SEED), df.drop(df.sample(frac=0.95, random_state=SEED).index)
    tl = DataLoader(JokeDS(tr, tok), batch_size=bs, shuffle=True, num_workers=2)
    vl = DataLoader(JokeDS(va, tok), batch_size=bs * 2, num_workers=2)

    model = Regressor().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(tl) * epochs
    sched = get_cosine_schedule_with_warmup(opt, 200, steps)
    crit = nn.MSELoss()

    for ep in range(epochs):
        model.train(); tot = 0
        for st, b in enumerate(tl):
            ids, am, y = b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE), b['label'].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(ids, am), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
            if (st + 1) % 500 == 0:
                print(f'  ep{ep+1} step {st+1}/{len(tl)} loss={tot/(st+1):.4f}', flush=True)

        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for b in vl:
                ps += model(b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE)).cpu().tolist()
                ys += b['label'].tolist()
        ps, ys = np.array(ps), np.array(ys)
        auc = roc_auc_score((ys > np.median(ys)).astype(int), ps)
        corr = np.corrcoef(ps, ys)[0, 1]
        print(f'  ep{ep+1} val_auc={auc:.4f} val_corr={corr:.4f}', flush=True)

    torch.save(model.bb.state_dict(), '/kaggle/working/reddit100k_encoder.pt')
    print('✅ Saved encoder -> reddit100k_encoder.pt')
    return model.bb, tok


# ═══════════════ STAGE B: Re-encode standup segments ═══════════════
def stage_b(encoder, tok):
    print('\n' + '=' * 60)
    print('STAGE B: Encode 639x20 standup segments')
    print('=' * 60)
    v6 = np.load(find('v6_features.npz'), allow_pickle=True)
    texts = v6['aligned_texts']  # (639, 20) strings
    encoder.eval()

    # optional projection-free: mean-pooled 768-dim per segment
    feats = np.zeros((texts.shape[0], N_SEG, 768), dtype=np.float32)
    B = 64
    with torch.no_grad():
        for i in range(texts.shape[0]):
            segs = [str(texts[i, j])[:512] for j in range(N_SEG)]
            for s in range(0, N_SEG, 8):
                chunk = segs[s:s + 8]
                e = tok(chunk, max_length=128, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                out = encoder(**e)
                m = e['attention_mask'].unsqueeze(-1).float()
                p = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
                feats[i, s:s + 8] = p.cpu().numpy()
            if (i + 1) % 100 == 0:
                print(f'  encoded {i+1}/639 videos', flush=True)
    np.savez_compressed('/kaggle/working/reddit100k_text_features.npz', features=feats)
    print('✅ Saved -> reddit100k_text_features.npz')
    return feats


# ═══════════ STAGE C: v10 fine-tune 5-fold CV (new features) ═══════════
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


def load_audio_gold():
    b4 = np.load(find('bridge4_features.npz'), allow_pickle=True)
    v6 = np.load(find('v6_features.npz'), allow_pickle=True)
    audio_obj, labels_obj = b4['features'], b4['labels']
    vids = v6['video_ids']
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
    gold = np.zeros((n, N_SEG), dtype=np.float32)
    v2i = {}
    for idx, v in enumerate(vids):
        v2i[str(v).split(',')[0]] = idx; v2i[str(v)] = idx
    gcsv = sorted(glob.glob('/kaggle/input/**/gold_labels/*.csv', recursive=True))
    seg_len = 495.3 / N_SEG
    for gp in gcsv:
        vid = os.path.basename(gp)[:-4]
        if vid not in v2i:
            continue
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
    print(f'audio={audio.shape} gold_nonzero={(gold > 0).sum()}')
    return audio, pseudo, gold


def run_fold(text, audio, pseudo, gold, tr, va, epochs=5, bs=4, lr=5e-5, coral_w=0.5, laugh_w=0.3):
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
            out = model(torch.tensor(text[i:i + 1]).to(DEVICE), torch.tensor(audio[i:i + 1]).to(DEVICE))
            hs.append(out['humor'][0].cpu().numpy()); ls.append(out['laughter'][0].cpu().numpy())
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


def stage_c(text_feats):
    print('\n' + '=' * 60)
    print('STAGE C: v10 5-fold CV with Reddit-100K features')
    print('=' * 60)
    audio, pseudo, gold = load_audio_gold()
    n = len(audio)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for k, (tr, va) in enumerate(kf.split(np.arange(n))):
        hum, gld = run_fold(text_feats, audio, pseudo, gold, tr, va)
        results.append({'fold': k, 'humor_auc': float(hum), 'gold_auc': float(gld)})
        print(f'Fold {k}: humor={hum:.4f} gold={gld:.4f}', flush=True)
    hums = [r['humor_auc'] for r in results]
    glds = [r['gold_auc'] for r in results]
    return {'model': 'v10 + Reddit-100K encoder',
            'humor_auc_mean': float(np.mean(hums)), 'humor_auc_std': float(np.std(hums)),
            'gold_auc_mean': float(np.mean(glds)), 'gold_auc_std': float(np.std(glds)),
            'folds': results, 'baseline_v10': {'humor': 0.8225, 'gold': 0.6131}}


if __name__ == '__main__':
    bb, tok = stage_a()
    feats = stage_b(bb, tok)
    summary = stage_c(feats)
    with open('/kaggle/working/scaleup_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n' + '=' * 60)
    print(f"HUMOR AUC: {summary['humor_auc_mean']:.4f} ± {summary['humor_auc_std']:.4f} (v10 was 0.8225)")
    print(f"GOLD  AUC: {summary['gold_auc_mean']:.4f} ± {summary['gold_auc_std']:.4f} (v10 was 0.6131)")
    print('=' * 60)
