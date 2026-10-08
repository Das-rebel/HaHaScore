# arXiv Submission Package — v1 Jester Humor-Strength Paper

## Files

| File | Purpose |
|---|---|
| `v1_paper.tex` | Main paper (4 pages, ~2300 words). Target: EMNLP 2026 short / arXiv preprint. |
| `figure_v1_5x3.png` + `.pdf` | Figure 1: per-fold 5×3 + bootstrap CI + single-seed baseline marker |
| `references.bib` | Bibliography (shared with hahascore paper) |
| `README.md` | This file |
| `hahascore.tex` | Companion methodology paper (5×3 CV applied to multimodal v10) |

## Paper at a glance

- **Title**: *Continuous Humor-Strength Prediction from Text: A 5×3 Repeated Cross-Validation Baseline on Jester*
- **Core result**: Spearman ρ = 0.2292 ± 0.0289 (95% CI [0.2143, 0.2440], n=15)
- **Pre-registered gate NOT met** (ρ ≥ 0.35) — text-only ceiling confirmed at ~0.23
- **Methodological contribution**: 5×3 repeated joke-disjoint CV with bootstrap CIs, applied to continuous-regression humor tasks

## Build

```bash
pdflatex v1_paper.tex   # twice for cross-references
bibtex v1_paper
pdflatex v1_paper.tex
pdflatex v1_paper.tex
```

## Honest scope

This paper is a **methodology + baseline** contribution, not an improvement on the state-of-the-art. The text-only ceiling is honestly reported (0.23, not 0.34). The publishable claim is the 5×3 framework, not the headline metric.

## Companion work

- **hahascore.tex** (sister paper in same directory): methodology paper for multimodal humor detection (5×3 CV on StandUp4AI v10). Independent publication track.
- **ChuckleNet** (`github.com/Das-rebel/ChuckleNet`): sister project for laughter detection using WavLM.
