# HaHaScore Data Discovery Report

**Date**: post-ensemble review + multi-platform data search
**Platforms searched**: Kaggle, HuggingFace, GitHub, academic papers
**Purpose**: Find training data that could drastically improve Cascade Gate v7

---

## 🎯 Top Recommendations (Ranked by Impact)

### 🥇 #1: **One Million Reddit Jokes** (HF + Kaggle)
- **Sources**:
  - HF: `SocialGrep/one-million-reddit-jokes` (CC-BY-4.0, 2.6GB CSV)
  - Kaggle: `pavellexyr/one-million-reddit-jokes`
- **Why**: **1M jokes with upvotes = continuous "funniness" proxy**
- **Direct match** for our 0-100 scoring (regression target)
- **1,500× our 641 files** — massive scale
- **License**: CC-BY-4.0 ✅

### 🥈 #2: **UR-FUNNY v2** (Multimodal humor dataset)
- **GitHub**: `ROC-HCI/UR-FUNNY` (MIT license, 156 stars)
- **Paper**: EMNLP D19-1211
- **Why**: **Only released public multimodal humor dataset with text+audio+video**
- 1,866 punchline + context samples (TED talks)
- Same architecture target as Cascade Gate v7
- **MIT license** ✅

### 🥉 #3: **ColBERT Humor Detection 200K** (Pretraining corpus)
- **GitHub**: `Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection`
- **HF**: `CreativeLang/ColBERT_Humor_Detection` (CC-BY-2.0)
- **Why**: **200K balanced humorous/non-humorous short texts**
- Binary classification, designed to defeat length/format shortcuts
- ESWA 2024 publication

### 4. **StandUp4AI** (Multilingual, 7 languages)
- **GitHub**: `sofia-callejas/seq-Standup4AI`
- 330 hours, 7 languages, 118 videos
- Already used in our ChuckleNet memory

### 5. **Gillick Laughter Detection** (Audio backbone)
- **GitHub**: `jrgillick/laughter-detection` (MIT, 293 stars)
- Interspeech 2021 — pre-trained audio model
- Switchboard dataset + AudioSet

### 6. **Memotion Dataset 7k** (Ordinal humor labels)
- **Kaggle**: `williamscott701/memotion-dataset-7k`
- 7K memes with **4-level humor labels** (not/funny/very/hilarious)
- Maps to our 0-100 scale via ordinal ranking

### 7. **Humor Genome** (Academic gold standard)
- **Kaggle**: `taylorsamarel/humor-genome-open-controls`
- Sentence-level annotations with humor style taxonomy

### 8. **UCI YouTube Comedy Slam** (Pairwise funniness)
- **Kaggle**: `uciml/youtube-comedy-slam`
- ~6K clips with pairwise winner labels
- Bradley-Terry → continuous scores

---

## 🎓 Teacher Models (for distillation)

| Model | URL | Purpose |
|-------|-----|---------|
| **mohameddhiab/humor-no-humor** | HF models | DistilBERT binary classifier, 3,580 downloads, Apache-2.0 |
| **Humor-Research/humor-detection-comb-23** | HF models | RoBERTa on 23 humor datasets merged |
| **truongkp/asr-laughter-whisper** | HF models | Whisper-large-v2 finetuned for laughter-aware ASR |

---

## 📊 Strategic Implications

### What These Datasets Unlock

| Asset | Current Pain | New Capability |
|-------|-------------|----------------|
| **1M Reddit Jokes** | 641 files, 12,780 segments | 1M+ text samples for regression pretraining |
| **UR-FUNNY v2** | Limited multimodal training data | Reference architecture for fusion model |
| **ColBERT 200K** | Small binary pretraining set | Pretrain text encoder with real humor |
| **Gillick model** | New audio backbone | Drop-in audio encoder instead of WavLM |
| **Humor Genome** | Single pseudo-label source | Academic-grade sentence-level labels |

### Critical Gap Analysis

| Gap | Status | Solution |
|-----|--------|----------|
| Audio + text + continuous labels | **MISSING** | Build from Reddit score + UR-FUNNY audio |
| Real laughter labels | **MISSING** (we have 1.2% sparse) | Combine StandUp4AI + Gillick annotations |
| Large training corpus | **MISSING** (641 files) | 1M Reddit jokes + 200K ColBERT |

---

## 🚀 Recommended Three-Phase Data Strategy

### Phase 1: Quick Wins (Week 1)
1. **Download `SocialGrep/one-million-reddit-jokes`** (CC-BY-4.0)
2. **Train text-only baseline** on Reddit scores → AUC benchmark
3. **Use `mohameddhiab/humor-no-humor`** as soft-label teacher for our 641 utterances

### Phase 2: Multimodal Pretraining (Week 2–3)
4. **Download UR-FUNNY v2** (TED text+audio+video, 1,866 samples)
5. **Pretrain fusion backbone** on UR-FUNNY binary labels
6. **Fine-tune** on our 641 standup comedy files (cascade gate v8)

### Phase 3: Scale Up (Week 4)
7. **Combine StandUp4AI laughter annotations** (118 videos, 7 langs) with gold labels
8. **Train multi-task model** (humor + laughter) with CORAL adaptation
9. **Target**: Gold AUC 0.59 → 0.70+ with multi-task + new data

---

## 📈 Expected Performance Improvements

| Improvement Lever | Expected AUC Gain |
|-------------------|-------------------|
| Reddit score pretraining (1M samples) | +0.01–0.02 (text baseline) |
| UR-FUNNY multimodal pretraining | +0.02–0.05 (transfer learning) |
| Teacher distillation (humor-no-humor) | +0.01–0.03 |
| StandUp4AI + CORAL gold laughter | +0.03–0.06 (gold AUC) |
| **Combined effect** | **+0.07–0.16** |

This could move us from AUC 0.86 to **0.93+** on pseudo-labels and from 0.59 to **0.70+** on gold laughter.

---

## ⚠️ License & Compliance Notes

- **CC-BY-4.0**: Free with attribution
- **CC-BY-2.0**: Free with attribution
- **MIT**: Permissive, attribution
- **Research-only** (HAult IberLEF, StandUp4AI): Non-commercial research only
- **Unspecified** (CodecBench laughterscape): Investigate before use
- **On request** (Pun of the Day): Contact authors

---

## 🛠️ Implementation Tasks

### Task 1: Download Reddit Jokes (1 day)
```bash
# Install
pip install datasets kaggle

# HuggingFace
python3 -c "
from datasets import load_dataset
ds = load_dataset('SocialGrep/one-million-reddit-jokes')
print(ds)
df = ds['train'].to_pandas()
df.to_csv('data/reddit_jokes_1m.csv', index=False)
print(f'Saved {len(df)} jokes')
"
```

### Task 2: Download UR-FUNNY (1 day)
```bash
git clone https://github.com/ROC-HCI/UR-FUNNY.git
cd UR-FUNNY
python3 download_videos.py  # requires YouTube access
# Or use pre-extracted features
```

### Task 3: Download ColBERT (1 day)
```bash
# HF
python3 -c "
from datasets import load_dataset
ds = load_dataset('CreativeLang/ColBERT_Humor_Detection')
"
```

---

## 📁 Files Created
- This report: `DATA_DISCOVERY_REPORT.md`
- Sources: Kaggle (25 candidates), HF (8 datasets + 3 models), GitHub (10 datasets)

**Next Step**: Begin Phase 1 — download and integrate Reddit jokes corpus.
