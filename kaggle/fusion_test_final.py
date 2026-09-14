# v2 fusion: minimal test (2 videos, 200 samples) for architecture verification
import json, os, random, torch, torchaudio, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaModel, WavLMModel

random.seed(42); torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUD = os.path.expanduser('~/autonomous_laughter_prediction_essential/data/fusion_audio')
LAB = '/tmp/tiny_rows.jsonl'  # <--- will create from subset below
SR = 16000; WIN = 1.5
print(f"Device: {DEVICE}", flush=True)

# Get all videos with audio and pick 2 random ones
import re
vids_with_audio = []
for f in os.listdir(AUD):
    m = re.match(r'^([A-Za-z0-9_-]{11})\.\w+$', f)
    if m: vids_with_audio.append(m.group(1))
random.shuffle(vids_with_audio)
target_vids = vids_with_audio[:2]
print(f"Audio files: {len(vids_with_audio)} -> target: {target_vids}", flush=True)

# Load all labels and filter to 2 videos + 200 samples max
rows = [json.loads(l) for l in open('/tmp/fusion_aligned_labels.jsonl')]
subset = [r for r in rows if r['video_id'] in target_vids and r.get('start') is not None]
random.shuffle(subset)
subset = subset[:200]
print(f"Subset: {len(subset)} samples", flush=True)

# Write to temp file for script convenience
with open('/tmp/tiny_rows.jsonl', 'w') as f:
    for r in subset: f.write(json.dumps(r) + '\n')

# Dataset
tok = RobertaTokenizerFast.from_pretrained('roberta-base')

class FusionDS(Dataset):
    def __init__(self, rows, audio_files):
        self.rows = rows
        self.audio_files = audio_files
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; vid = r['video_id']
        ctx = r.get('context_words') or []
        text = ' '.join(ctx[-8:] + [r['word']])[-200:]
        enc = tok(text, truncation=True, max_length=48, padding='max_length', return_tensors='pt')
        # Efficient audio slice loading
        c = int(SR * WIN / 2)
        s = max(0, int(r['start']*SR)-c)
        e = min(int(r['end']*SR)+c, s+int(SR*WIN*1.5))
        wave, sr = torchaudio.load(self.audio_files[vid], frame_offset=s, num_frames=e-s)
        if wave.shape[0] > 1: wave = wave.mean(0)
        if sr != SR: wave = torchaudio.functional.resample(wave, sr, SR)
        if len(wave) < int(SR*WIN): wave = torch.nn.functional.pad(wave, (0, int(SR*WIN)-len(wave)))
        wave = wave[:int(SR*WIN)]
        return enc['input_ids'][0], enc['attention_mask'][0], wave, float(r['label'])

# Map audio files
audio_files = {v: os.path.join(AUD, next(f for f in os.listdir(AUD) if f.startswith(v))) for v in target_vids}

dl = DataLoader(FusionDS(subset, audio_files), batch_size=8, shuffle=True, num_workers=0)
print(f"DataLoader: {len(dl)} batches", flush=True)

class Fusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.txt = RobertaModel.from_pretrained('roberta-base')
        self.aud = WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(768+512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1))
    def forward(self, ids, mask, wav):
        t = self.txt(ids, attention_mask=mask).pooler_output
        a = self.aud(wav).extract_features.mean(1)
        return self.fuse(torch.cat([t, a], -1)).squeeze(-1)

model = Fusion().to(DEVICE)
print("Models loaded", flush=True)
npos = sum(r['label'] for r in subset); nneg = len(subset)-npos
pw = nneg/max(npos,1); crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
opt = torch.optim.AdamW([
    {'params': model.fuse.parameters(), 'lr': 1e-3},
    {'params': model.txt.parameters(), 'lr': 2e-5},
    {'params': model.aud.parameters(), 'lr': 1e-5}], weight_decay=0.01)

model.train()
for ep in range(3):
    tot = 0; n = 0
    for ids, mask, wav, y in dl:
        opt.zero_grad()
        loss = crit(model(ids.to(DEVICE), mask.to(DEVICE), wav.to(DEVICE)), y.float().to(DEVICE))
        loss.backward(); opt.step(); tot += loss.item(); n += 1
    print(f"ep{ep}: loss {tot/n:.4f}", flush=True)

torch.save(model.state_dict(), os.path.expanduser('~/models/chuckle_predictor/fusion_v2_test.pt'))
print("Saved fusion_v2_test.pt")
