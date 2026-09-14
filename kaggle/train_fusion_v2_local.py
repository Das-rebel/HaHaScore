# v2 fusion: local test on 10K samples
import json, os, random, torch, torchaudio, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaModel, WavLMModel

random.seed(42); torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUD = os.path.expanduser('~/autonomous_laughter_prediction_essential/data/fusion_audio')
LAB = '/tmp/fusion_aligned_labels.jsonl'
SR = 16000; WIN = 1.5
print(f"Device: {DEVICE}")

# Load labels (sample 10K)
rows = [json.loads(l) for l in open(LAB)]
random.shuffle(rows)
rows = rows[:10000]
by_vid = {}
for r in rows:
    if r.get('start') is None: continue
    by_vid.setdefault(r['video_id'], []).append(r)
print(f"Videos: {len(by_vid)}, samples: {sum(len(v) for v in by_vid.values())}")

# Map video_id → audio file
import re
audio_files = {}
for f in os.listdir(AUD):
    m = re.match(r'^([A-Za-z0-9_-]{11})\.\w+$', f)
    if m: audio_files[m.group(1)] = os.path.join(AUD, f)
by_vid = {v: rs for v, rs in by_vid.items() if v in audio_files}
print(f"Videos with audio: {len(by_vid)}, samples: {sum(len(v) for v in by_vid.values())}")

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

vids = sorted(by_vid)
random.shuffle(vids); n_tr = int(0.8*len(vids)); n_va = int(0.1*len(vids))
tr_v, va_v, te_v = vids[:n_tr], vids[n_tr:n_tr+n_va], vids[n_tr+n_va:]
def mkrows(vs): return [r for v in vs for r in by_vid[v]]
tr_r, va_r, te_r = mkrows(tr_v), mkrows(va_v), mkrows(te_v)
print(f"Split: train {len(tr_v)}v/{len(tr_r)}s, val {len(va_v)}v/{len(va_r)}s, test {len(te_v)}v/{len(te_r)}s")

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
        tot = 0
        for ids, mask, wav, y in dl:
            opt.zero_grad()
            loss = crit(model(ids.to(DEVICE), mask.to(DEVICE), wav.to(DEVICE)), y.float().to(DEVICE))
            loss.backward(); opt.step(); tot += loss.item()
        print(f"  ep{ep}: loss {tot/len(dl):.4f}")

print("=== TRAIN (10K sample test) ===")
run(tr_r, epochs=3)
f1, auc = evaluate(FusionDS(va_r, audio_files))
print(f"VAL: F1 {f1:.4f} AUC {auc:.4f}")
print("Saving test model...")
torch.save(model.state_dict(), os.path.expanduser('~/models/chuckle_predictor/fusion_v2_test.pt'))
print("Done. fusion_v2_test.pt saved.")
