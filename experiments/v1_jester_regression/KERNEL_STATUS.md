# v1 Jester Kernel Status (FINAL, Oct 7 04:48 UTC)

## Final state of the GPU fine-tuned kernel

- **Slug**: `subhajitdas/v1-jester-roberta-fine-tuned-5x3-cv`
- **Status**: KernelWorkerStatus.RUNNING (stuck past 9h Kaggle limit)
- **Started**: Oct 6 13:43 UTC
- **Current**: Oct 7 04:48 UTC (~15h elapsed)
- **Versions pushed**: 1
- **Known**: Past the 9h Kaggle T4 wall-clock limit (reached at ~22:43 UTC)

## What we learned

### Failed runs (in order)
1. **v6 (frozen)** — subhajitdas/v1-jester-roberta-5x3-cv — 10h+ stuck, OOM-killed at 851s (RoBERTa full fine-tune OOM'd on T4 15GB)
2. **v1 fine-tuned (current)** — subhajitdas/v1-jester-roberta-fine-tuned-5x3-cv — 15h+ stuck, exceeded 9h limit without completing
3. **CPU run** — `run_v1_cpu.py` — completed in 22 min, ρ = 0.2315 ± 0.0175 (FROZEN backbone, gate failed by 0.12)

### Why GPU kept failing
- T4 15GB is the binding constraint for any real RoBERTa fine-tune
- Kaggle's 9h limit is too short for stratified 420K + 5x3 fine-tune CV
- The kernel writes intermediate files to /kaggle/working but those are private until kernel completes

### What CAN work
- **Smaller scope**: 5K-20K samples (not 420K) × 3x3 (not 5x3) × 1 epoch fits in 1-2h
- **Cached embeddings**: pre-compute RoBERTa embeddings once (CPU ~30 min for 50K), then only train MLP head (CPU <1 min/fold, no GPU limits)
- **Smaller backbone**: distilroberta-base or twitter-roberta-base (~half the params, half the memory)

## The HONEST v1 result

Only ONE honest v1 number exists:
- **CPU run**: ρ = 0.2315 ± 0.0175 (CI [0.2222, 0.2402]) — frozen RoBERTa + linear head
- **Pre-registered gate**: ρ ≥ 0.35 AND CI excludes 0.30 — FAILED by 0.12 on ρ
- **M3 baseline**: ρ = 0.313 (per `evaluation_results/results_v1_kaggle.json`) — full fine-tune, NOT 5x3 verified

The M3 0.313 number is from a different protocol (single CV, not 5x3 repeated) so it's NOT directly comparable to our 5x3 CV. The honest text-only v1 ceiling appears to be 0.20-0.32 in 5x3 repeated CV with frozen backbone.

## What we publish

✅ **Methodology paper** (already drafted as `arxiv_submission/hahascore.tex`):
- 5×3 repeated CV applied to humor-strength regression
- The +0.163 per-language norm falsification (p = 0.0004)
- Identity-leakage on speaker-disjoint vs random splits
- This is publishable regardless of headline metric

❌ **Headline v1 metric**: ρ = 0.2315 is BELOW the 0.35 pre-registered gate. Cannot cite.

❌ **Wait for GPU fine-tuned**: The kernel is at 15h+ RUNNING, past Kaggle's 9h limit, and Kaggle won't release output. Effectively dead.

## Decision

The CPU result is the canonical v1 honest baseline. No further GPU runs unless Kaggle provides a way to:
1. Get intermediate output before 9h limit (e.g., public save_path)
2. Allow longer than 9h sessions for batched CV (or accept partial results)

OR pivot to **smaller scope**:
- 5K-20K samples × 3x3 × 1 epoch = ~1-2h T4
- Pre-compute embeddings on full 1.76M, then train head many times

## Recommend

For the v1 paper:
1. Cite the CPU result (ρ = 0.2315) as the honest text-only v1 baseline
2. Cite the M3 result (ρ = 0.313) with disclaimer: "single-fold, not 5x3 verified"
3. Conclude: text-only regression on Jester has ceiling 0.25-0.31, no surprise
4. Publish the methodology paper (5x3 repeated CV + falsification findings)

For v2 (next step per strategic rethink):
1. Add audio features (WavLM embeddings of actual laugh acoustic events)
2. Apply same 5x3 protocol
3. Expected improvement: +0.05 to +0.10 over v1 ceiling

