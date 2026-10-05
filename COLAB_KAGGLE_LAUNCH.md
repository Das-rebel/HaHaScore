# Colab / Kaggle Launch Guide — v2 LaughO Training

**Date**: 2026-10-05
**Git commit**: 7352527
**Goal**: Apply the falsification methodology to actually build v2 LaughO — a small (1.5M-param), audio-only laughter detector that uses AudioSet acoustic laughter labels (NOT VTT discourse positions).

---

## Why AudioSet, not StandUp4AI?

StandUp4AI's `[laughter]` VTT markers are **audience reaction positions**, not acoustic laughter events (per memory `labels_are_discourse_not_acoustic`). AudioSet's `Laughter/Giggle/Snicker/Belly laugh/Chuckle/Snort` family is the only open-source corpus with acoustic event labels at scale (5.8k hours, 8 explicit classes).

The v2 LaughO training:
- **Inputs**: AudioSet laugh classes (acoustic) + AudioSet non-laugh frames (negative samples)
- **Architecture**: WavLM-base-plus (frozen, 768-dim) + 256→128→1 MLP head (1.5M trainable)
- **Validation**: 5×3 repeated speaker-disjoint CV with bootstrap CI (laugho_cv.py)
- **No label leakage** by design (acoustic labels, not transcript markers)

---

## Option A: Colab (recommended)

### Cost
- Free tier: 12 hours/session, 1 GPU (T4 / L4)
- Expected runtime: 3 hours for 1 AudioSet eval shard (650 MB) + train + 5×3 CV

### Steps
1. Open https://colab.research.google.com
2. File → Upload notebook → select `v2_laugho_train.ipynb`
3. Runtime → Change runtime type → GPU (T4)
4. Run cells 1-9 sequentially
5. Results saved to Drive at `/content/drive/MyDrive/LaughterOnAudio/HaHaScore/v2_laugho_runs/`

### What each cell does

| Cell | Action | Time |
|---|---|---|
| 1 | Mount Drive + install deps | 30 sec |
| 2 | Verify v10 ONNX integrity (SHA) | 5 sec |
| 3 | Clone HaHaScore repo (pulls laugho_cv.py + scripts) | 10 sec |
| 4 | Download 1 AudioSet eval/00.parquet (650 MB) | 1-5 min |
| 5 | Load parquet, filter 8 laugh ontology IDs | 5 sec |
| 6 | Extract WavLM embeddings via ffmpeg-subprocess + WavLM-base-plus | 2-5 min |
| 7 | Add 5x negative samples from non-laugh rows | 5-10 min |
| 8 | Train MLP + 5×3 repeated CV | 10-15 min |
| 9 | Save results JSON to Drive | 5 sec |

### Expected output (single shard)
```
Dataset: ~500-2000 samples
5×3 CV: 15 measurements
  Mean: 0.55-0.75 (depends on WavLM quality + label distribution)
  Std:  0.10-0.20 (depends on n_unique_videos)
```

### What to do with results
1. Compare to **HaHaScore v10** (5-fold 0.69 humor / 0.55 gold on 12-video StandUp4AI)
2. Compare to **ChuckleNet-Ten** (WavLM-only F1=0.280 on 71v; Ensemble F1=0.587 on 71v — both StandUp4AI pseudo-labels)
3. If v2 LaughO ≥ 0.65 on AudioSet, **consider arXiv submission of methodology + v2 result**
4. If v2 LaughO < 0.55, **the architecture is not the bottleneck** — re-prioritize to data collection (more AudioSet shards, AMI, UR-FUNNY-Temporal)

---

## Option B: Kaggle (fallback when Colab unavailable)

### Cost
- Free tier: 30 GPU hours/week (T4 or P100)
- Per memory: P100 incompatible with PyTorch CUDA 12.6 (sm_60 too old) — use T4

### Push kernel

```bash
cd /Users/Subho/funny-strength-predictor/kaggle_v2_laugho
kaggle kernels push
```

Expected to take 2-5 min for upload, then auto-runs on Kaggle GPU.

**Memory warning**: `SaveKernel API completely rate-limited after ~15+ rapid pushes (v7-v17)`. If you've pushed recently, wait 2+ hours before pushing.

### Check status
```bash
kaggle kernels status subhajitdas/v2-laugho-train
```

### Pull output
```bash
kaggle kernels output subhajitdas/v2-laugho-train -p ./kaggle_output/
```

---

## Why the notebook structure follows v19

Memory: "v19 is canonical Colab notebook. Found+fixed via TRIPLE check: (1) Cell 7 conditional continue, (2) processed_idx guard, (3) BatchNorm1d crash on batch 1, (4) start_idx>0 guard."

The v2 LaughO notebook applies these same guards:
- ffmpeg-subprocess AAC loader (libsndfile can't decode AAC)
- per-fold processed_idx membership guard (not duplicating features on resume)
- BatchNorm1d-safe for batch size 1 edge case (len%256==1)
- start_idx>0 guard (skipping data load on completed runs)

If the notebook is interrupted mid-run, re-run from cell 6 — it skips already-extracted embeddings.

---

## Honest scope

This is **one shard, one model, one CV run**. It produces a number that:
- Is reproducible (script in repo)
- Is validated with proper methodology (5×3 speaker-disjoint CV)
- Is comparable to existing baselines (HaHaScore v10, ChuckleNet-Ten)
- Is NOT a headline number (need ≥ 5 shards + n>500 for a publishable claim)

If v2 LaughO ≥ 0.65 on AudioSet eval/00, the architecture + methodology is validated. Next step: scale to all 35 eval shards + AMI + UR-FUNNY-Temporal.

---

## Run instructions summary

**Fastest path** (5 min from cold start to running notebook):
1. Open `v2_laugho_train.ipynb` in Colab
2. Set runtime to T4 GPU
3. Run all 9 cells
4. Check Drive for `5x3_cv_audioset_eval00_<timestamp>.json`
5. Send back the JSON file (commit results to HaHaScore repo as a follow-up commit)

**Estimated total**: 3-4 hours wall-clock for full pipeline.