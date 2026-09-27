# Kaggle GPU Scale-Up Pipeline

Free T4/P100 GPUs (30 hrs/week) replace local CPU training (20x faster).

## Kernels
1. `kernel_v10_5fold_cv.py` → `subhajitdas/hahascore-v10-5fold-cv`
   - 5-fold CV of v10 architecture (statistical rigor)
2. `kernel_scaleup_100k.py` → `subhajitdas/hahascore-scaleup-100k`
   - Stage A: DistilBERT pretrain on 100K Reddit jokes (3.3x more data)
   - Stage B: Re-encode all 639x20 standup segments with pretrained encoder
   - Stage C: v10 5-fold CV with new features

## Dataset: `subhajitdas/hahascore-features`
- bridge4_features.npz (audio, 639x20x791)
- v6_features.npz (text + aligned_texts + video_ids)
- reddit_100k.csv (100K stratified jokes with funniness)
- gold_labels/ (34 CSVs with 'risa' laughter intervals)
