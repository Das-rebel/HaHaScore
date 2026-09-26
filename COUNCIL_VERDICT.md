# Agent Council Verdict: Data Strategy for HaHaScore

**Date**: post-council debate
**Council**: 3 senior ML specialists (Data Strategy, Transfer Learning, Production)
**Status**: CONSENSUS REACHED

---

## 🎯 Council Consensus (Strong Positions)

> **"Reddit today. Pretrain, don't deploy Reddit-only. UR-FUNNY next week for gold AUC lift. Keep v7 cascade gate architecture."**

### Consensus Verdict Matrix

| Question | Data Strategist | Transfer Learning | Production |
|----------|-----------------|-------------------|------------|
| Highest leverage for gold gap | UR-FUNNY multimodal | Joint multi-task | Reddit first, then v8 |
| 1M Reddit useful? | Yes for pretraining | Yes (100K subset) | Pretrain only, don't deploy |
| ColBERT 200K or Reddit? | Reddit first | ColBERT Stage 1, Reddit S2 | Reddit first |
| Realistic ceiling | Pseudo: 0.93, Gold: 0.66–0.70 | +0.04–0.08 gold AUC | Ship encoder upgrade |
| Lowest friction | Reddit (minutes) | Reddit | Reddit (1 hr) |
| Teacher distillation? | Skip binary, only audio teacher | Multi-task > distillation | Pretrain, not distill |
| Download today? | Reddit | Reddit (100K subset) | Reddit |

**ALL THREE AGREE**: Download Reddit first. Pretrain on Reddit, fine-tune on standup, ship v8 = v7 + Reddit-pretrained text encoder.

---

## 📋 Concrete Action Plan (Synthesized)

### Day 1 (Today): Reddit Acquisition
```python
from datasets import load_dataset
ds = load_dataset("SocialGrep/one-million-reddit-jokes")  # 2.6GB, CC-BY-4.0
df = ds["train"].to_pandas()
# Stratify by score (upvotes) — top 100K with diverse subreddits
df_top = df.nlargest(100_000, "score")
df_top.to_csv("data/reddit_jokes_100k.csv", index=False)
```

### Week 1: Reddit Text Baseline (Proof of Value)
- Train text-only regression model on 100K Reddit jokes
- Target: continuous funniness prediction from upvotes
- **Ship to HF as `hahascore-reddit-pretrain`** (proves the data has signal)
- Use `microsoft/deberta-v3-base` or `distilbert-base-uncased` as backbone
- Head: single linear → sigmoid → continuous funniness score

### Week 2: Cascade Gate v8 — Reddit-Initialized Text Tower
- Use Reddit-trained text encoder weights to initialize v7's text tower
- Keep cascade gate architecture intact
- Fine-tune on 641 standup files with curriculum:
  - Stage A: freeze text encoder, train audio + gate (5 epochs)
  - Stage B: full fine-tune with low LR (1e-5) on text branch (5 epochs)
- **Target AUC: 0.865–0.880 (pseudo), 0.60–0.62 (gold)**

### Week 3: UR-FUNNY Multimodal Pretraining (Gold AUC Lift)
- Download UR-FUNNY v2 (MIT, 1,866 TED samples)
- Extract audio (Whisper or WavLM features)
- Pretrain fusion backbone on UR-FUNNY binary labels
- Fine-tune on standup with multi-task (humor + laughter heads)
- **Target gold AUC: 0.66–0.70**

---

## 🔬 Detailed Stage Schedule (Transfer Learning Council)

| Stage | Data | Epochs | LR | Purpose |
|-------|------|--------|-----|---------|
| **S1** | ColBERT 200K | 3 | 2e-5 | Binary humor detection backbone (optional) |
| **S2** | Reddit 100K | 2 | 1e-5 | Regression head warmup, drop head after |
| **S3** | UR-FUNNY 1,866 | 5 | 5e-6 | Fusion lock (freeze text branch) |
| **S4** | Multi-task: 100K Reddit + 200K ColBERT + 1,866 UR + 641 standup (12× oversampled) | 8 | 1e-5 | Joint with task-specific heads |
| **S5** | Standup 641 only | 5 | 5e-6 | Domain lock, gold-AUC target |

---

## ⚠️ Critical Warnings from Council

### Production Engineer's Warnings
1. **DON'T ship Reddit-only model** — Reddit upvotes ≠ professional standup humor
2. **Distribution shift trap**: Reddit skews puns/dark humor/in-jokes; will look "tone-deaf" on standup
3. **License landmines**: StandUp4AI and HAult IberLEF are **research-only** — kill from commercial roadmap
4. **Add `ATTRIBUTIONS.md`** to repo for Reddit (CC-BY-4.0) and ColBERT (CC-BY-2.0)
5. **Token 401 issue still blocks deploys** — fix HF token in parallel

### Data Strategist's Warnings
1. **Skip binary teachers** (`mohameddhiab/humor-no-humor`, `humor-detection-comb-23`) — strip continuous signal
2. **Only distill acoustic representations** from `truongkp/asr-laughter-whisper` (not outputs)
3. **Reddit won't move gold AUC** (same label distribution as pseudo-labels)

### Transfer Learning's Warnings
1. **Joint training with 1,866 UR-FUNNY drowns text signal** in 1M Reddit — text-first hierarchy
2. **Reddit overfit**: model learns one-liner pattern, ignores standup context
3. **UR-FUNNY prosody bias**: expects TED pacing, rejects standup cadence
4. **Mitigation**: per-stage head reinitialization, gradual unfreezing, gradient clipping

---

## 📊 Realistic Performance Targets (Honest Numbers)

| Model | Pseudo AUC | Gold AUC | Notes |
|-------|-----------|----------|-------|
| **v7 (current)** | 0.860 | 0.590 | Baseline |
| **v8 Reddit-only pretrain** | 0.870–0.880 | 0.60–0.62 | Ship to HF in Week 2 |
| **v9 + UR-FUNNY multimodal** | 0.880–0.900 | **0.66–0.70** | Gold laughter ceiling |
| **v9 + StandUp4AI gold** | 0.900 | **0.70–0.75** | Stretch goal |

**Total expected lift**: +0.07–0.16 AUC across stages.

---

## 🎬 Immediate Next Steps

### Step 1: Download Reddit (1 hour)
```bash
cd /Users/Subho/funny-strength-predictor
mkdir -p data/pretrain
python3 -c "
from datasets import load_dataset
import pandas as pd
ds = load_dataset('SocialGrep/one-million-reddit-jokes')
df = ds['train'].to_pandas()
# Stratified sample: top 100K by score with diversity
df_top = df.nlargest(100_000, 'score')
df_top.to_csv('data/pretrain/reddit_jokes_100k.csv', index=False)
print(f'Saved {len(df_top)} jokes, score range [{df_top.score.min()}-{df_top.score.max()}]')
"
```

### Step 2: Train Reddit Baseline (6 hours)
```bash
# Use distilbert or deberta for text regression on upvotes
python3 train_reddit_baseline.py
```

### Step 3: Ship Reddit Pretrain to HF (30 min)
```bash
# Upload as hahascore-reddit-pretrain (separate repo for proof)
huggingface-cli upload Hayasuki/hahascore-reddit-pretrain .
```

### Step 4: Initialize v8 with Reddit Weights (Week 2)
```python
# In cascade_train.py:
text_encoder = RedditPretrainedEncoder()
text_encoder.load_state_dict(torch.load("models/reddit_pretrained_text.pt"))
# Fine-tune on standup
```

---

## 🔑 Key Decisions (Council Unanimous)

1. **Reddit first**: download today, train baseline this week
2. **Pretrain, don't deploy Reddit-only**: fine-tune on standup
3. **Keep cascade gate architecture**: Reddit improves components, not system
4. **UR-FUNNY in Week 3**: only for gold AUC lift
5. **Skip binary teacher distillation**: they strip continuous signal
6. **Fix HF token 401 issue**: in parallel to unblock deploys
7. **Add ATTRIBUTIONS.md**: Reddit CC-BY-4.0 + ColBERT CC-BY-2.0

---

## 📁 Output Artifacts

- `DATA_DISCOVERY_REPORT.md` — initial data search
- `COUNCIL_VERDICT.md` — this file (council synthesis)
- Pending: Reddit download script, baseline training script, v8 fine-tuning script

**Next Step**: Download Reddit jokes and ship the text baseline by end of this week.
