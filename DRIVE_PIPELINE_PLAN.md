# HaHaScore Drive-Based Data Pipeline

**Date**: post-council verdict
**Constraint**: Local disk at 98% capacity (only 4.4GB available)
**Solution**: All data downloads + processing routed directly to Google Drive

---

## ⚠️ Storage Constraint

| Resource | Status | Implication |
|----------|--------|-------------|
| **Local disk** | 98% full (4.4GB free) | Cannot store 2.6GB Reddit CSV locally |
| **Google Drive** | Unlimited (rclone configured) | All bulk data goes here |
| **Existing memory** | `tools.rclone.gdrive`: always use rclone | Pattern established |

---

## 📂 Drive Folder Structure

```
gdrive:/HaHaScore_Pretrain/
├── reddit/          # 1M Reddit jokes → 100K stratified
├── ur_funny/        # UR-FUNNY multimodal (text only — YouTube blocked)
├── colbert/         # ColBERT 200K
├── models/          # Trained checkpoints
└── results/         # Metrics, plots, attributions
```

---

## 🔧 Pipeline Scripts

All scripts are in `/Users/Subho/funny-strength-predictor/drive_pipeline/`:

| Script | Purpose |
|--------|---------|
| `download_reddit_to_drive.py` | Streams 1M Reddit jokes → Drive (CC-BY-4.0) |
| `download_ur_funny_to_drive.py` | Streams UR-FUNNY metadata → Drive (MIT) |
| `download_colbert_to_drive.py` | Streams ColBERT 200K → Drive (CC-BY-2.0) |
| `train_reddit_text_baseline.py` | Pull CSV → train → push model to Drive |
| `status.py` | Quick check of Drive contents |
| `rclone_helper.sh` | Helper functions |

---

## 🚀 Execution Plan (Drive-Aware)

### Day 1: Download Reddit (1 hour drive-time, 0 local storage)
```bash
cd /Users/Subho/funny-strength-predictor/drive_pipeline
python3 download_reddit_to_drive.py
```

What happens:
1. HF downloads Reddit to `/tmp/reddit_jokes_download/` (transient)
2. Stratified 100K sample created
3. rclone pushes both CSVs to `gdrive:/HaHaScore_Pretrain/reddit/`
4. Local temp directory auto-cleaned
5. **Local disk impact: minimal (only transient /tmp files)**

### Day 2: Train Reddit Text Baseline (4–8 hours, 0 net local storage)
```bash
python3 train_reddit_text_baseline.py --epochs 2
```

What happens:
1. rclone pulls 100K CSV to temp dir
2. Trains DistilBERT regressor on upvotes (continuous funniness)
3. Saves model to temp dir
4. rclone pushes model + metrics to Drive
5. Auto-cleanup temp

### Day 3: Train Larger Backbone (3 hours)
```bash
python3 train_reddit_text_baseline.py \
    --backbone microsoft/deberta-v3-base \
    --epochs 2 \
    --output hahascore-reddit-deberta-v1.pt
```

### Day 4: Download UR-FUNNY (best-effort, YouTube blocked)
```bash
python3 download_ur_funny_to_drive.py
```

Note: UR-FUNNY videos require YouTube downloads which are **blocked from India IP** (403 Forbidden per memory). We extract metadata + text labels only. Audio extraction can be done later if proxies work.

### Day 5: Download ColBERT 200K (30 min)
```bash
python3 download_colbert_to_drive.py
```

---

## 📊 Storage Accounting

| Data | Size | Location |
|------|------|----------|
| Reddit 1M full | 2.6 GB | Drive (reddit/) |
| Reddit 100K stratified | ~250 MB | Drive (reddit/) |
| UR-FUNNY metadata | ~5 MB | Drive (ur_funny/) |
| ColBERT 200K | ~300 MB | Drive (colbert/) |
| DistilBERT model | ~250 MB | Drive (models/) |
| DeBERTa-v3-base model | ~450 MB | Drive (models/) |
| **Total Drive** | **~3.9 GB** | (Drive has 15GB free on free tier) |
| **Total Local** | **~0 GB** (transient /tmp only) | ✅ |

---

## 🎯 Council Verdicts (Recap)

Three specialist agents unanimously recommended:
1. **Data Strategist**: Reddit first, UR-FUNNY next week
2. **Transfer Learning**: Text-first pretraining hierarchy
3. **Production**: Don't deploy Reddit-only — fine-tune on standup

Concrete execution:
- **Week 1**: Reddit text baseline (proof of value)
- **Week 2**: v8 = v7 + Reddit-pretrained text encoder
- **Week 3**: UR-FUNNY multimodal (gold AUC lift)
- **Week 4**: Multi-task + CORAL (stretch goal)

---

## 📜 License Compliance

| Dataset | License | Attribution Required |
|---------|---------|----------------------|
| Reddit 1M | CC-BY-4.0 | Yes — credit r/Jokes |
| UR-FUNNY | MIT | Yes — credit ROC-HCI |
| ColBERT | CC-BY-2.0 | Yes — credit Moradnejad |

`ATTRIBUTIONS.md` to be added to repo before any deployment.

---

## ✅ Status

- [x] Drive folders created (`gdrive:/HaHaScore_Pretrain/`)
- [x] Download scripts ready (3 datasets)
- [x] Training script ready (DistilBERT, DeBERTa)
- [x] Status check script ready
- [x] Local disk preserved (only transient /tmp)

**Ready to execute Day 1**: `python3 download_reddit_to_drive.py`
