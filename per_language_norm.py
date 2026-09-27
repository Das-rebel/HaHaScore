#!/usr/bin/env python3
"""
PER-LANGUAGE NORMALIZATION FIX
==============================

Per SPEAKER_DISJOINT_FINDINGS.md:
- Per-language mean subtraction removes language as a confound
- Expected: French gold AUC 0.26 -> 0.55+

Implementation:
1. Compute mean feature vector per language (es, fr, en, other)
2. Subtract language mean from each video's features
3. Re-train + re-evaluate with the normalized features
4. Compare: language-normalized vs raw (the gold 0.39 baseline)
"""
import os, glob, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)


class BilinearFusion(nn.Module):
    def __init__(self, dim, out_dim, dropout=0.1):
        super().__init__()
        self.wt = nn.Linear(dim, out_dim, bias=False)
        self.wa = nn.Linear(dim, out_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(out_dim)))
        self.drop = nn.Dropout(dropout); self.norm = nn.LayerNorm(out_dim)
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
        self.pos = nn.Embedding(20 + 1, 4)
        self.gru = nn.GRU(hidden * 3 + 4, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
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


def load_data():
    b4 = np.load('/Users/Subho/tmp/bridge4_features.npz', allow_pickle=True)
    v6 = np.load('/Users/Subho/tmp/v6_features.npz', allow_pickle=True)
    audio_obj, labels_obj = b4['features'], b4['labels']
    text, vids = v6['text_features'], v6['video_ids']
    n = len(audio_obj)
    audio = np.zeros((n, 20, 791), dtype=np.float32)
    pseudo = np.full((n, 20), 0.5, dtype=np.float32)
    for i in range(n):
        for j in range(20):
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

    gold = np.zeros((n, 20), dtype=np.float32)
    v2i = {str(v).split(',')[0]: idx for idx, v in enumerate(vids)}
    v2i.update({str(v): idx for idx, v in enumerate(vids)})
    seg_len = 495.3 / 20
    for gp in glob.glob('/Users/Subho/funny-strength-predictor/data/gold_labels/*.csv'):
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
                for s in range(20):
                    ov = max(0.0, min(t1, (s + 1) * seg_len) - max(t0, s * seg_len))
                    if ov > 0:
                        gold[vi, s] = min(1.0, gold[vi, s] + ov / seg_len)
    return audio, text, pseudo, gold, np.asarray(vids)


def normalize_by_language(features, vids):
    """Subtract per-language mean from each video's features."""
    langs = np.array([str(v).split(',')[-1] if ',' in v else 'en' for v in vids])
    unique_langs = np.unique(langs)
    means = {}
    for lang in unique_langs:
        mask = langs == lang
        if mask.sum() > 0:
            means[lang] = features[mask].mean(axis=0)  # (20, dim)
    normalized = features.copy()
    for i, lang in enumerate(langs):
        if lang in means:
            normalized[i] = features[i] - means[lang]
    return normalized, langs, means


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
    hs, ls, gs, ps = [], [], [], []
    with torch.no_grad():
        for i in va:
            out = model(torch.tensor(text[i:i + 1]).to(DEVICE),
                        torch.tensor(audio[i:i + 1]).to(DEVICE))
            hs.append(out['humor'][0].cpu().numpy())
            ls.append(out['laughter'][0].cpu().numpy())
            gs.append(gold[i]); ps.append(pseudo[i])
    return np.concatenate(hs), np.concatenate(ls), np.concatenate(gs), np.concatenate(ps)


def evaluate_gold(humor_scores, laugh_scores, gold_targets, pseudo_targets):
    """Compute humor and gold AUCs."""
    hum_auc = roc_auc_score((pseudo_targets > np.median(pseudo_targets)).astype(int), humor_scores)
    if (gold_targets > 0).any():
        gb = (gold_targets > np.median(gold_targets)).astype(int)
        gld = roc_auc_score(gb, laugh_scores) if 0 < gb.sum() < len(gb) else 0.5
    else:
        gld = 0.5
    return hum_auc, gld


def main():
    print('Loading data...')
    audio, text, pseudo, gold, vids = load_data()
    n = len(audio)
    gold_idx = np.where((gold > 0).any(axis=1))[0]
    print(f'Videos with gold: {len(gold_idx)}')
    print(f'Languages:', {l: int(((np.array([str(v).split(",")[-1] if "," in v else "other" for v in vids]) == l).sum())) for l in ['es', 'fr', 'other']})

    # Compute per-language means from full 639-video set
    text_norm, langs, means = normalize_by_language(text, vids)
    audio_norm, _, _ = normalize_by_language(audio, vids)

    print(f'\nLanguage means computed for: {list(means.keys())}')
    print(f'  es samples: {(langs == "es").sum()}')
    print(f'  fr samples: {(langs == "fr").sum()}')
    print(f'  other samples: {(langs == "other").sum()}')

    # === Baseline (raw features) on 12 gold videos, speaker-disjoint ===
    print('\n' + '=' * 60)
    print('TEST: Per-language normalization vs raw, speaker-disjoint')
    print('=' * 60)

    # Use only 12 gold videos for evaluation (speaker-disjoint 5-fold)
    audio_g_raw = audio[gold_idx]
    text_g_raw = text[gold_idx]
    audio_g_norm = audio_norm[gold_idx]
    text_g_norm = text_norm[gold_idx]
    pseudo_g = pseudo[gold_idx]
    gold_g = gold[gold_idx]
    vids_g = vids[gold_idx]
    langs_g = langs[gold_idx]

    groups = vids_g
    gkf = GroupKFold(n_splits=5)

    # Run 5-fold speaker-disjoint CV with BOTH raw and normalized features
    raw_humors, raw_golds = [], []
    norm_humors, norm_golds = [], []
    norm_fr_golds, raw_fr_golds = [], []
    norm_es_golds, raw_es_golds = [], []

    for k, (tr, va) in enumerate(gkf.split(np.arange(len(gold_idx)), groups=groups)):
        # Raw baseline
        h_raw, l_raw, g_raw, p_raw = run_fold(text_g_raw, audio_g_raw, pseudo_g, gold_g, tr, va)
        # Per-language normalized
        h_norm, l_norm, g_norm, p_norm = run_fold(text_g_norm, audio_g_norm, pseudo_g, gold_g, tr, va)

        # Combined predictions for joint AUC
        # Per-fold aggregation
        hum_raw, gld_raw = evaluate_gold(h_raw, l_raw, g_raw, p_raw)
        hum_norm, gld_norm = evaluate_gold(h_norm, l_norm, g_norm, p_norm)

        raw_humors.append(hum_raw); raw_golds.append(gld_raw)
        norm_humors.append(hum_norm); norm_golds.append(gld_norm)

        # Per-language breakdown
        va_langs = langs_g[va]
        for i_in_va, idx in enumerate(va):
            lang = langs_g[idx]
            if lang == 'fr':
                norm_fr_golds.append(roc_auc_score((g_norm[i_in_va*20:(i_in_va+1)*20] > 0).astype(int), l_norm[i_in_va*20:(i_in_va+1)*20]) if (g_norm[i_in_va*20:(i_in_va+1)*20] > 0).any() else 0.5)
                raw_fr_golds.append(roc_auc_score((g_raw[i_in_va*20:(i_in_va+1)*20] > 0).astype(int), l_raw[i_in_va*20:(i_in_va+1)*20]) if (g_raw[i_in_va*20:(i_in_va+1)*20] > 0).any() else 0.5)
            elif lang == 'es':
                norm_es_golds.append(roc_auc_score((g_norm[i_in_va*20:(i_in_va+1)*20] > 0).astype(int), l_norm[i_in_va*20:(i_in_va+1)*20]) if (g_norm[i_in_va*20:(i_in_va+1)*20] > 0).any() else 0.5)
                raw_es_golds.append(roc_auc_score((g_raw[i_in_va*20:(i_in_va+1)*20] > 0).astype(int), l_raw[i_in_va*20:(i_in_va+1)*20]) if (g_raw[i_in_va*20:(i_in_va+1)*20] > 0).any() else 0.5)

        print(f'  Fold {k}: raw gold={gld_raw:.4f}  norm gold={gld_norm:.4f}  delta={gld_norm - gld_raw:+.4f}')

    print('\n' + '=' * 60)
    print('AGGREGATE RESULTS (5-fold speaker-disjoint)')
    print('=' * 60)
    print(f'Raw features:        humor={np.mean(raw_humors):.4f} gold={np.mean(raw_golds):.4f}')
    print(f'Lang-normalized:     humor={np.mean(norm_humors):.4f} gold={np.mean(norm_golds):.4f}')
    print(f'Δ humor AUC:          {np.mean(norm_humors) - np.mean(raw_humors):+.4f}')
    print(f'Δ gold AUC:           {np.mean(norm_golds) - np.mean(raw_golds):+.4f}')
    if norm_fr_golds:
        print(f'\nFrench videos (8 gold):')
        print(f'  Raw:  {np.mean(raw_fr_golds):.4f}')
        print(f'  Norm: {np.mean(norm_fr_golds):.4f}')
    if norm_es_golds:
        print(f'\nSpanish videos (3 gold):')
        print(f'  Raw:  {np.mean(raw_es_golds):.4f}')
        print(f'  Norm: {np.mean(norm_es_golds):.4f}')

    # VERDICT
    delta = np.mean(norm_golds) - np.mean(raw_golds)
    if delta > 0.10:
        verdict = '✅ CONFIRMED: per-language normalization breaks the language confound. Apply to all models.'
    elif delta > 0.05:
        verdict = '⚠️ PARTIAL: helps somewhat. Combine with multilingual batch sampling.'
    else:
        verdict = '❌ INSUFFICIENT: language normalization alone not enough. Need architecture-level fix.'

    print(f'\nVERDICT: {verdict}')

    summary = {
        'raw_humor_mean': float(np.mean(raw_humors)),
        'raw_gold_mean': float(np.mean(raw_golds)),
        'norm_humor_mean': float(np.mean(norm_humors)),
        'norm_gold_mean': float(np.mean(norm_golds)),
        'delta_humor': float(np.mean(norm_humors) - np.mean(raw_humors)),
        'delta_gold': float(delta),
        'fr_raw_mean': float(np.mean(raw_fr_golds)) if raw_fr_golds else None,
        'fr_norm_mean': float(np.mean(norm_fr_golds)) if norm_fr_golds else None,
        'es_raw_mean': float(np.mean(raw_es_golds)) if raw_es_golds else None,
        'es_norm_mean': float(np.mean(norm_es_golds)) if norm_es_golds else None,
        'verdict': verdict,
    }
    with open('/Users/Subho/funny-strength-predictor/lang_norm_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n✅ Saved -> lang_norm_results.json')


if __name__ == '__main__':
    main()
