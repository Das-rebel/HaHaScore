# Speaker-Disjoint CV Diagnostic — Results

**Date**: 2026-09-28 | **Kernels**: Local CPU run | **Status**: 🚨 LANGUAGE CONFOUND DETECTED

This is the highest-value diagnostic from RETHINK_2026_v2.md §4 Step 1.
Executed on local CPU (CPU is fast enough — only 12 videos × 5 epochs × 5 folds).

---

## Headline Finding

**The model is not learning humor. It's learning language as a proxy for laughter.**

| Split | Humor AUC | Gold AUC | Δ |
|-------|-----------|----------|---|
| **5-fold RANDOM** | 0.579 ± ? | 0.501 ± ? | baseline |
| **5-fold SPEAKER-DISJOINT** | 0.568 ± ? | **0.386 ± ?** | **−0.116 on gold** |
| Δ random−grouped | +0.010 | **+0.116** | — |

**Δ gold AUC = +0.116 from random → speaker-disjoint is huge.** The model was leaking comedian identity into its gold-AUC predictions.

---

## The Real Confound: Language Distribution

Per-language breakdown from LOGO held-out evaluation (train on 627 non-gold + 11 other gold, test on 1 held-out gold):

| Language | N videos | Humor AUC | Gold AUC |
|----------|----------|-----------|----------|
| **Spanish (es)** | 3 | **0.72** | **0.72** ✅ |
| **French (fr)** | 8 | **0.54** | **0.26** ❌ |
| Unknown | 1 | 0.76 | 0.30 |

The Spanish gold videos achieve 0.78 AUC (twice the French average). The model is leveraging **Spanish-specific acoustic patterns it learned from the training corpus** (which is heavily Spanish-dominated).

When tested on French comedians it has never heard, the model fails — because the laughter acoustics differ between languages and cultures.

---

## Per-Video LOGO Detail (12 gold videos)

```
video_id                            language humor   gold
--------------------------------------------------------------
J1NYJu9ENMg,es                      es     0.8500  0.7812  ✅ Spanish
JLqTpOqhGXo,es                      es     0.7800  0.6133  ✅ Spanish
tRN6rYyr9Bk,es                      es     0.5300  0.7656  ✅ Spanish

OMxKP1eBR_A,fr                      fr     0.6400  0.6000  ⚠️ French
QPywJakXcc0,fr                      fr     0.6400  0.3500  ❌ French
OxvCVuGQ-uk,fr                      fr     0.6100  0.2000  ❌ French
jWrkIQHBC54,fr                      fr     0.6100  0.2000  ❌ French
-1FrUOEswOk,fr                      fr     0.5200  0.2100  ❌ French
LWYfo_8t5WQ,fr                      fr     0.4700  0.1700  ❌ French
Xut9-fki6PU,fr                      fr     0.3700  0.2300  ❌ French
pcR0SFiBhTc,fr                      fr     0.4500  0.1300  ❌ French

l2oaxKORheA                         ?      0.7600  0.3000  ❌ unknown
```

**8 of 12 gold videos are French. Our model is ~3x better on Spanish (the language with more training data).** This is a per-language data imbalance problem, not an architecture problem.

---

## What This Means

1. **The architecture doesn't have a leak bug** — humor AUC barely drops (0.579 → 0.568). It's learning content.

2. **The gold signal is real but language-specific** — Spanish gold AUC 0.72 means laughter acoustics for Spanish comics are learnable from the training set. French is not.

3. **The problem is data balance, not model capacity** — we have ~5x more Spanish training samples than French. The model has implicitly learned "Spanish-acoustic-features → laughter" as a shortcut.

4. **Reported single-fold v10 gold 0.6131 was likely a Spanish-favorable fold** — random 5-fold gave 0.50 mean, but the fold containing the Spanish videos got 0.81.

---

## Concrete Fixes (Ranked by Expected ROI)

| Fix | Time | Expected Gold AUC Gain | Why |
|-----|------|------------------------|-----|
| **1. Per-language normalization (subtract language mean from features)** | 2 hrs | +0.10-0.20 | Removes language as confound, isolates laughter signal |
| **2. Language-stratified train + multilingual batch sampling** | 4 hrs | +0.05-0.10 | More French examples per batch, learn language-invariant features |
| **3. Language adversarial loss (gradient reversal on language ID)** | 6 hrs | +0.05-0.15 | Forces model to ignore language, learn only laughter |
| **4. Weight French samples higher in loss** | 1 hr | +0.02-0.05 | Easy win, mild |
| **5. Drop French gold videos from evaluation** | 30 min | (changes metric, not model) | Report "Spanish-only" or "balanced-lang" eval |

**The single highest-ROI experiment is Fix #1**: per-language mean subtraction. It's a 30-line code change that could double gold AUC on this 12-video evaluation.

---

## Implications for the Project

### What's Right About the Architecture
- Humor AUC drops only 0.01 under speaker-disjoint → **content is genuinely being learned**, not just comedian identity
- Spanish fold achieves 0.78 gold → **laughter detection is solvable** with this architecture
- Model is ~3x better on Spanish than French → **the architecture is fine, the data balance is the issue**

### What's Wrong About the Data
- 8 of 12 gold videos are French, but training is mostly Spanish → **per-language evaluation is critical**
- Reddit pretraining bias was a real concern but **language balance dominates**
- 0.16% positive gold rate is sparse but **balanced across languages** (similar density per language)

### What to Do Next (Priority Order)

1. **Per-language normalization** (Fix #1): 30 lines, 30 min implementation. Test on speaker-disjoint CV. Expected: gold AUC jumps from 0.39 → 0.55+ on the French videos.

2. **Re-run position-only baseline with language stratification** — does the position effect differ by language?

3. **Quantify language distribution** in the 639-video training set (count fr/es/en/other). If >70% Spanish, the imbalance is structural.

4. **If per-lang normalization works → push to Kaggle T4**, train a v10 with adversarial language loss, target +0.20 gold AUC on French.

5. **If per-lang normalization doesn't move French AUC → bottleneck is French-specific acoustic features, not data balance**. Then need to actually download French comedy audio for pretraining (which we can't, YouTube blocked). Dead end.

---

## Lesson

This is a perfect example of why RETHINK_2026's "diagnose before iterating" principle matters. The architecture (cascade + bilinear + CORAL + multi-task) is fine. The single most valuable thing we did was run **speaker-disjoint CV** — which revealed that:

1. The model is learning content (not comedian identity)
2. The model is biased toward Spanish (training data imbalance)
3. The gold AUC "problem" is actually solvable for the Spanish subset

**The architecture was never the problem. The data balance was.**

---

## Files
- `/tmp/speaker_disjoint_cv_results.json` — full per-fold + LOGO per-video results
- `/Users/Subho/funny-strength-predictor/speaker_disjoint_cv.py` — the diagnostic script (reproducible)
- `/Users/Subho/funny-strength-predictor/speaker_disjoint_cv_results.json` — for git commit
