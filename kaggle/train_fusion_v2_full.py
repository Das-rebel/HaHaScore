# v2 fusion: full 48-video training on CPU
import json, os, random, torch, torchaudio, numpy as np, time
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaModel, WavLMModel

random.seed(42); torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUD = os.path.expanduser('~/autonomous_laughter_prediction_essential/data/fusion_audio')
LAB = '/tmp/fusion_aligned_labels.jsonl'
SR = 16000; WIN = 1.5; MAX_SAMPLES_PER_VID = 1200
print(f"Device: {DEVICE} | Start: {time.strftime('%H:%M:%S')}")

# Load all labels
rows = [json.loads(l) for l in open(LAB)]
print(f"Total rows: {len(rows)}")

# Group by video, sample up to MAX_SAMPLES_PER_VID each
by_vid = {}
for r in rows:
    if r.get('start') is None: continue
    by_vid.setdefault(r['video_id'], []).append(r)
print(f"Videos with labels: {len(by_vid)}")

# Sample per video for balanced coverage
import re
audio_files = {}
for f in os.listdir(AUD):
    m = re.match(r'^([A-Za-z0-9_-]{11})\.\w+$', f)
    if m: audio_files[m.group(1)] = os.path.join(AUD, f)
print(f"Audio files: {len(audio_files)}")

by_vid = {v: rs for v, rs in by_vid.items() if v in audio_files}
print(f"Videos with audio+labels: {len(by_vid)}")

# Sample MAX_SAMPLES_PER_VID per video
sampled = []
for v, rs in by_vid.items():
    random.shuffle(rs)
    sampled.extend(rs[:MAX_SAMPLES_PER_VID])
random.shuffle(sampled)
rows = sampled
print(f"Sampled: {len(rows)} rows from {len(by_vid)} videos")

# Rebuild by_vid for split
by_vid = {}
for r in rows:
    by_vid.setdefault(r['video_id'], []).append(r)

# Train/val/test split by video
vids = sorted(by_vid)
random.shuffle(vids)
n_tr = int(0.8*len(vids)); n_va = int(0.1*len(vids))
tr_v, va_v, te_v = vids[:n_tr], vids[n_tr:n_tr+n_va], vids[n_tr+n_va:]
def mkrows(vs): return [r for v in vs for r in by_vid[v]]
tr_r, va_r, te_r = mkrows(tr_v), mkrows(va_v), mkrows(te_v)
print(f"Split: train {len(tr_v)}v/{len(tr_r)}s, val {len(va_v)}v/{len(va_r)}s, test {len(te_v)}v/{len(te_r)}s")

# Dataset
tok = RobertaTokenizerFast.from_pretrained('roberta-base')

class FusionDS(Dataset):
    def __init__(self, rows, audio_files):
        self.rows, self.audio_files = rows, audio_files
        self.cache = {}
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; vid = r['video_id']
        ctx = r.get('context_words') or []
        text = ' '.join(ctx[-8:] + [r['word']])[-200:]
        enc = tok(text, truncation=True, max_length=48, padding='max_length', return_tensors='pt')
        wave = self.cache.get(vid)
        if wave is None:
            wave, sr = torchaudio.load(self.audio_files[vid])
            wave = torchaudio.functional.resample(wave.mean(0), sr, SR)
            self.cache[vid] = wave
        c = int(SR * WIN / 2); s = max(0, int(r['start']*SR)-c); e = min(len(wave), int(r['end']*SR)+c)
        seg = wave[s:e]
        if len(seg) < SR*WIN: seg = torch.nn.functional.pad(seg, (0, SR*WIN-len(seg)))
        seg = seg[:int(SR*WIN)]
        return enc['input_ids'][0], enc['attention_mask'][0], seg, float(r['label'])

# Model
class Fusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.txt = RobertaModel.from_pretrained('roberta-base')
        self.aud = WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(768+768, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1))
    def forward(self, ids, mask, wav):
        t = self.txt(ids, attention_mask=mask).pooler_output
        a = self.aud(wav).extract_features.mean(1)
        return self.fuse(torch.cat([t, a], -1)).squeeze(-1)

model = Fusion().to(DEVICE)
npos = sum(r['label'] for r in tr_r); nneg = len(tr_r)-npos
pw = nneg/max(npos,1); crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
opt = torch.optim.AdamW([
    {'params': model.fuse.parameters(), 'lr': 1e-3},
    {'params': model.txt.parameters(), 'lr': 2e-5},
    {'params': model.aud.parameters(), 'lr': 1e-5}], weight_decay=0.01)

def evaluate(ds):
    model.eval(); dl = DataLoader(ds, batch_size=32, num_workers=0)
    yp, yt = [], []
    with torch.no_grad():
        for ids, mask, wav, y in dl:
            p = torch.sigmoid(model(ids.to(DEVICE), mask.to(DEVICE), wav.to(DEVICE))).cpu()
            yp += p.tolist(); yt += y.tolist()
    yp, yt = np.array(yp), np.array(yt)
    from sklearn.metrics import f1_score, roc_auc_score
    best = max(f1_score(yt, yp > t) for t in np.arange(0.2, 0.8, 0.05))
    return best, roc_auc_score(yt, yp)

def run(rows, epochs=3, bs=16):
    ds = FusionDS(rows, audio_files); dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    model.train()
    for ep in range(epochs):
        tot = 0; t0 = time.time()
        for i, (ids, mask, wav, y) in enumerate(dl):
            opt.zero_grad()
            loss = crit(model(ids.to(DEVICE), mask.to(DEVICE), wav.to(DEVICE)), y.float().to(DEVICE))
            loss.backward(); opt.step(); tot += loss.item()
            if (i+1) % 500 == 0:
                print(f"  ep{ep} batch {i+1}/{len(dl)} loss {tot/(i+1):.4f} {time.time()-t0:.0f}s")
        print(f"  ep{ep}: loss {tot/len(dl):.4f} ({time.time()-t0:.0f}s)")

print("=== TRAIN (full 48 videos) ===")
run(tr_r, epochs=3)
f1, auc = evaluate(FusionDS(va_r, audio_files))
print(f"VAL: F1 {f1:.4f} AUC {auc:.4f}")
print("Saving final model...")
torch.save(model.state_dict(), os.path.expanduser('~/models/chuckle_predictor/fusion_v2_final.pt'))
print(f"Done. fusion_v2_final.pt saved at {time.strftime('%H:%M:%S')}")