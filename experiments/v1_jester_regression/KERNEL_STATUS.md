# v1 Jester Kernel Status (Oct 6 2026, 09:59 UTC)

## Kaggle Kernel State

- **Slug**: `subhajitdas/v1-jester-roberta-5x3-cv`
- **Status**: KernelWorkerStatus.RUNNING (stuck past 9h limit)
- **Started**: Oct 5 23:15:18 UTC
- **Current**: Oct 6 09:59 UTC (10h 44m)
- **Versions pushed**: 6 (v1 had Python 3.9 tokenizers wheel build error; v2 had ModuleNotFoundError; v4-v5 had SAMPLE_LIMIT NameError; v5 had ratings_norm bug; v6 successfully started with SAMPLE_LIMIT=50000 + max_length=64)

## What worked in v6
- Python 3.13 compatible deps (tokenizers, transformers, datasets, scipy, scikit-learn, pyarrow)
- All dependencies inline (no separate .py files needed)
- Memory-conscious: SAMPLE_LIMIT=50000, max_length=64, batch_size=4
- RoBERTa-base loaded to GPU
- 1.76M Jester ratings subsampled to 50K
- Started training on fold 1 seed 1

## What didn't work / what's blocked
- Kernel hit Kaggle T4 9h wall-clock limit at 08:15 UTC
- Kernel still in RUNNING state at 09:59 UTC (Kaggle hasn't killed it)
- Log endpoint only returns final output when status is COMPLETE or ERROR
- No way to get partial output mid-run

## Estimated v1 runtime (realistic)
- 50K samples × 2 epochs / 4 batch = 25K steps per fold
- ~25-30 min per fold × 15 measurements = ~7h total
- This was designed to fit in 9h but actual runtime may have been longer due to overhead
- Even subsampled 50K may have been too much for 9h budget

## Lessons for future runs
1. **Subsample MORE aggressively**: 10K-20K samples is safer for 9h T4
2. **Use smaller model**: roberta-base with 125M params + 15 measurements
   × 2 epochs × 10K samples might fit in 3-4h
3. **Pre-cache embeddings**: extract RoBERTa embeddings once, train
   regression head only (much faster than fine-tuning full model)
4. **Add checkpoint saves**: save fold results as soon as each fold completes
   so partial output is recoverable

## Decision

The kernel is effectively dead - no output available without completion.
Recommend re-running with smaller scope:
- 10K-20K samples (memory + time safe)
- 3 seeds × 3 folds = 9 measurements (faster)
- Save per-fold results to disk for partial recovery

Cost to redo: 0 cash, ~3h T4 GPU
Status: WAITING FOR MANUAL KILL OR RERUN
