# HaHaScore Domain Glossary

**Purpose**: Anchor the vocabulary used in architecture discussions and paper drafts. Add terms lazily as they crystallize.

## Datasets

- **Jester** — 1.76M rated jokes from `SeppeV/rated_jokes_dataset_from_jester` (Apache-2.0). Schema: `joke_id`, `jokeText`, `rating` (raw 1-10, range [-10, +10]). 140 unique jokes. Used for v1 (continuous humor-strength regression). **Avoids StandUp4AI's 1.16% positive class problem.**
- **StandUp4AI** — 639 stand-up comedy videos, 591 English + 48 non-English. Has 12 gold-labeled videos (8 French, 3 Spanish, 1 unlabeled). Used for v10 multimodal humor detection. Has a 1.16% positive class rate that makes binary classification unreliable.
- **AudioSet** — 5.8k hours, 8 laugh ontology classes. Used for v2 LaughO sister project (audio-only laughter detection). Sister project to HaHaScore.

## Models (all 5×3 verified unless noted)

- **v1** — Text-only RoBERTa-base + Linear(768, 1) regression head on Jester. **ρ = 0.2292 ± 0.0289** (5×3 honest, Oct 7). Pre-registered gate: ρ ≥ 0.35 AND CI excludes 0.30. **GATE NOT MET.**
- **v5** — Multimodal: DeBERTa-v3-base + WavLM-base-plus bilinear fusion. **ρ = 0.632 ± 0.007** on StandUp4AI pseudo-labels. Predecessor to v10.
- **v10** — Multimodal Cascade Gate: text confidence gates audio stream. **humor ρ = 0.6924 ± 0.0359, gold ρ = 0.5453 ± 0.1313** on 12-video StandUp4AI gold subset. Most defensible multimodal result.
- **v2 LaughO** — **SISTER PROJECT** (not canonical v1). Audio-only laughter detection on AudioSet, WavLM-base-plus + 1.5M-param MLP head. **ρ = 0.7501 ± 0.1817** on 1 of 35 AudioSet eval shards. Isolated in `experiments/v2_laugho/`.
- **M3** — The canonical v1 recipe (Sep 8): RoBERTa-base + Linear(768, 1), 4 epochs, lr=2e-5, batch=32, max_len=128. **ρ = 0.3127** on 7,984 samples (single-fold, CPU).

## Protocols

- **5×3 repeated CV** — 3 random seeds × 5 folds = 15 measurements. GroupKFold by either speaker (detection) or joke_id (regression). 10,000-resample bootstrap 95% CI. The project's methodological contribution.
- **Joke-disjoint CV** — GroupKFold by `joke_id` (for continuous regression tasks like Jester). No joke appears in both train and val.
- **Speaker-disjoint CV** — GroupKFold by comedian/speaker_id (for binary humor-detection tasks like StandUp4AI).
- **Pre-registered gate** — Written BEFORE running. v1 gate: ρ ≥ 0.35 AND CI excludes 0.30. Applied BEFORE results, not after.

## Known pitfalls

- **Fold-luck** — Single-seed numbers on Jester/StandUp4AI inflate by 0.10-0.30 ρ relative to the honest 5×3 mean. Confirmed 3 times (see `5x3_cv_results.json`, `experiments/v1_jester_regression/5x3_cv/`).
- **Single-fold AUC inflation** — On imbalanced speech-humor tasks (<2% positive rate), single-fold AUC overstates by 0.15-0.20. See `arxiv_submission/hahascore.tex` abstract.
- **Identity leakage** — Speaker-disjoint CV reveals a −0.115 gold-AUC drop that random CV misses.
- **VTT markers are discourse positions, not acoustic events** — StandUp4AI's `[laughter]` VTT markers are audience reaction positions, not laugh acoustic events. Use AudioSet acoustic labels for v2.
