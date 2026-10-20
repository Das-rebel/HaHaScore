<!--
Gold-label submission? Start from data/gold_labels/CONTRIBUTING.md.
CI must pass: validate-gold-labels (format, ranges, labels).
-->

## What does this PR do?

- [ ] **Gold labels** — new `<YouTubeID>.csv` file(s) in `data/gold_labels/`
- [ ] Docs / code / other

## For gold-label submissions

| field | value |
|---|---|
| Video ID(s) | `<!-- 11-char YouTube ID(s) -->` |
| Language (ISO 639-1) | `<!-- en / fr / es / hi / ... -->` |
| Segments added | `<!-- count; must include both risa AND no_risa -->` |
| Labeling method | `<!-- manual listening / player timestamping -->` |
| Credit (optional) | `<!-- name/handle for paper acknowledgments -->` |

**Submitter attestation**

- [ ] I am not the uploader of this video (avoids self-labeling bias)
- [ ] Labels were made by watching/listening, not copied from another dataset
- [ ] `validate-gold-labels` CI check passes

## Reviewer notes

- New files must contain **both** `risa` and `no_risa` (legacy 34 files are
  grandfathered).
- One CSV per video, header exactly `t0,t1,source,label`, seconds as floats.
- After merge: recovered videos feed the next honest benchmark
  (`run_5x3_cv.py` / `laugho_cv.py`, 5×3 speaker-disjoint + bootstrap CI).
  Honest numbers only — never cite single-fold AUC.
