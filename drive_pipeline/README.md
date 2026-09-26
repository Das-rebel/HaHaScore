# Drive-Based Data Pipeline for HaHaScore

**Constraint**: Local disk at 98% capacity (only 4.4GB available).
**Solution**: All data downloads + processing routed directly to Google Drive.

## Drive Structure
```
gdrive:/HaHaScore_Pretrain/
├── reddit/          # 1M Reddit jokes (CC-BY-4.0)
├── ur_funny/        # UR-FUNNY multimodal (MIT)
├── colbert/         # ColBERT 200K (CC-BY-2.0)
├── models/          # Trained model checkpoints
└── results/         # Eval results, plots
```

## Key Principle
**Never write large files to local disk.** Always stream to Drive via rclone.

## Scripts
- `download_reddit_to_drive.py` — Streams Reddit jokes CSV directly to Drive
- `download_ur_funny_to_drive.py` — Streams UR-FUNNY audio features to Drive
- `train_reddit_text_baseline.py` — Trains text model, saves to Drive
- `rclone_helper.sh` — Convenience functions
