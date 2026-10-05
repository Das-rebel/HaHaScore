# arXiv Submission Package — HaHaScore Self-Falsification Paper

## Files

| File | Purpose |
|---|---|
| `hahascore.tex` | Main paper (4 pages, ~2000 words). Target: INTERSPEECH 2026 short / EMNLP Findings. |
| `references.bib` | Bibliography |
| `figure_5x3_falsification.png` + `.pdf` | Figure 1: per-fold gold-AUC + delta histogram, $5\times3$ repeated CV |
| `figure_speaker_disjoint.png` + `.pdf` | Figure 2: random vs speaker-disjoint CV |
| `COVER_LETTER.md` | Cover letter for INTERSPEECH / EMNLP submission |

## Paper at a glance

- **Title**: *Evaluating Speech-Humor Models Without Identity Leakage: A Self-Falsification Case Study*
- **Core recommendation**: 5×3 repeated speaker-disjoint CV with bootstrap CIs is the minimum reporting standard for speech classification with positive class rate < 2% and gold n ≤ 30.
- **Empirical basis**: HaHaScore v10 self-falsification. +0.163 single-fold gain reversed to −0.127 under 5×3 repeated CV (p=0.0004, CI [−0.183, −0.078]).
- **Authorship**: Subhajit Das (solo author). No external data collection. No annotation budget.

## Build

Requires `pdflatex` (not installed locally). On a machine with TeX Live:

```bash
cd arxiv_submission/
pdflatex hahascore.tex   # twice for cross-references
bibtex hahascore
pdflatex hahascore.tex
pdflatex hahascore.tex
```

Or upload to arXiv directly — they accept `.tex` source + figures.

## Submission timeline

| Week | Action |
|---|---|
| 0 (now) | Paper draft v1 + cover letter ready |
| 1 | Internal review / 1-2 external reviewers (colleagues, advisor, sister-project author) |
| 2 | arXiv preprint (cs.CL) — fastest dissemination |
| 4 | Submit to INTERSPEECH 2026 / EMNLP Findings |

## Honest scope

This paper is a **methodology contribution**, not an improvement on humor detection. The headline AUC numbers are deliberately un-cited. The publishable claim is the validation recommendation, not the model itself.

## Companion work

- **ChuckleNet** (`github.com/Das-rebel/ChuckleNet`): sister project on laughter detection using WavLM. Reports a similar +0.115 identity-leakage finding under a different evaluation protocol (`docs/DECISION_GRAPH_V3.md`).
- **HaHaScore v10** (`github.com/Das-rebel/HaHaScore`): archived repo with the full audit trail, raw measurements, and falsification experiment scripts.