# Repo Completeness Audit — Both Projects

**Date**: 2026-10-05
**Repos audited**: `Das-rebel/HaHaScore`, `Das-rebel/HaHaScore-page`, all 100 `Das-rebel/*` repos
**Local check**: `/Users/Subho/funny-strength-predictor`, `/Users/Subho/autonomous_laughter_prediction_essential`, `/Users/Subho/chucklenet-local`, `/Users/Subho/ChuckleNet`, `/Users/Subho/HaHaScore-page`

---

## 1. Repository Map (Public + Local)

### Live `Das-rebel/*` repos relevant to laughter/humor

| Repo | Status | Last push | Notes |
|---|---|---|---|
| `Das-rebel/HaHaScore` | **LIVE** | 2026-10-05 | 11 commits this session. SciSlop-passed falsification paper draft. |
| `Das-rebel/HaHaScore-page` | **LIVE** (just fixed) | 2026-10-05 | Launch pack with HONEST 5-fold numbers now (was 0.86/0.590). |
| `Das-rebel/arxiv_endorsement_request_chuckle` | LIVE | 2026-06-21 | Endorsement request with paper PDFs (109KB + 167KB). Contains MultiLinguahah citation — flagged fabrication. |
| `Das-rebel/chuck-audio-notebooks` | LIVE | 2026-09-02 | 17 Colab notebooks for WavLM+Prosody extraction + training. Latest: `Colab_Complete_Pipeline.ipynb`. |
| `Das-rebel/chuckle_data` | LIVE | 2026-05-26 | Training data for Colab (utterances_clean.jsonl.gz, 10MB). |
| `Das-rebel/chuckle-sample-5videos` | LIVE | 2026-07-15 | labels.jsonl (63MB). |
| `Das-rebel/funny-strength-predictor` | LIVE | 2026-09-04 | Older duplicate of HaHaScore (pushed Sep 4, before HaHaScore existed). |
| ~~`Das-rebel/ChuckleNet`~~ | **DELETED** | — | 404 — entire research record offline |
| ~~`Das-rebel/autonomous_laughter_prediction`~~ | **DELETED** | — | 404 — entire research record offline |
| ~~`Das-rebel/chucklenet` (lowercase)~~ | **DELETED** | — | 404 — `chucklenet-local` clone's remote |

### Local clones with dead remotes

| Local dir | Remote | Last commit | Branches | Size |
|---|---|---|---|---|
| `/Users/Subho/autonomous_laughter_prediction_essential/` | `Das-rebel/autonomous_laughter_prediction` (404) | `e36db26` (Sep 23) | `main`, `add-colab-notebook`, `clean-fixes`, `colab-notebook` | 2.9 GB |
| `/Users/Subho/chucklenet-local/` | `Das-rebel/chucklenet` (404) | `ebe3ba2` | `main` | 99 MB |
| `/Users/Subho/ChuckleNet/` | none | — | stub | empty |
| `/Users/Subho/autonomous_laughter_prediction/` | none | — | stub | docs + models |
| `/Users/Subho/hf_chucklenet_verified/` | none | — | stub | empty |
| `/Users/Subho/chuckle-e2e-v14/` | none | — | stub | empty |
| `/Users/Subho/chuckle-e2e-v30b/` | none | — | stub | empty |
| `/Users/Subho/chuckleNet-FINAL-v6/` | none | — | stub | empty |

---

## 2. Critical Findings

### Finding 1: `HaHaScore-page` was distributing "AUC 0.860" launch copy

**This contradicts the main repo's falsification logs** (commit `171928b`: +0.163 per-language norm refuted, `5x3_cv_results.json`: 5-fold honest 0.69 / 0.55). Specifically:

- `show-hn.md` (Sep 27): claimed "AUC 0.860 ± 0.018" for Cascade Gate v7
- `linkedin.txt` (Sep 27): "AUC 0.860 on its shipped eval set"
- `youtube-description.txt` (Sep 27): "HaHaScore · sentence-level humor strength predictor · 0.860 AUC"
- `x-thread.md` (Sep 27): "AUC 0.860 (pseudo) / 0.590 (gold)"
- `index.html` (Sep 27): stat boxes showed `0.860` and `0.590`; logo said `v0.86`

**Fixed**: All 5 files rewritten + index.html patched + v0.69 logo (commit `6b104ba` pushed).

**Not fixed yet**: `hahascore-thumbnail.jpg` (image with "AUC 0.860" baked in) + demo videos (hahascore-demo.mp4 / vertical / square) — these need manual re-edit / re-record.

### Finding 2: Both canonical ChuckleNet repos are DELETED

`Das-rebel/ChuckleNet` and `Das-rebel/autonomous_laughter_prediction` both return 404. The `chucklenet-local` clone has the lowercase `Das-rebel/chucklenet` remote, also 404.

**The 8-month research record is offline.** `autonomous_laughter_prediction_essential/` has 757 entries (2.9 GB), 60 commits in `chucklenet-local`, and is the closest thing to canonical. Restoring to GitHub would require:

```bash
cd /Users/Subho/autonomous_laughter_prediction_essential
# Recreate the deleted repo on GitHub (private or public)
# Then:
git remote set-url origin https://github.com/Das-rebel/ChuckleNet.git
git push -u origin --all
git push -u origin --tags   # v0.1-data, v2.13.18, v2.14.0, v221-embeddings
```

**Recommendation**: prioritize restoring `Das-rebel/ChuckleNet` because all paper PDFs and reproducer scripts in the canonical papers cite `github.com/Das-rebel/ChuckleNet` (PAPER_DRAFT_V3_CONSOLIDATED.md uses it, JENNI_PAPER_CORE uses it, the arXiv endorsement request uses it). The dead URL will cause reviewer confusion.

### Finding 3: 17 Colab notebooks in `chuck-audio-notebooks` (Sep 2)

`/Users/Subho/autonomous_laughter_prediction_essential` has v18-v19 + 5 variants in `docs/` and v3.0+ notebooks in `chuck-audio-notebooks/`. **The canonical notebook per memory is v19 (`gist 188a3bc5d4346c8189372f00c8bc2d39`)**, but the local repos may have newer versions that haven't been committed.

**Action**: verify which notebook in `chuck-audio-notebooks/` corresponds to v19, and confirm the v18/v19 split is honored.

### Finding 4: `arxiv_endorsement_request_chuckle` contains a **fabricated citation**

The endorsement request README (Jun 2026) cites "MultiLinguahah (Callejas et al., 2026), arXiv:2605.06309" — this was flagged as a fabrication in the user's memory (`claim_audit_20260615`). The repo also contains a paper PDF that uses this citation. If the paper was ever submitted to arXiv, this would be an integrity issue.

**Action**: tag this repo for archive/rewrite before any submission.

### Finding 5: `chucklenet-local` has untracked work

`chucklenet-local` has 3 untracked items not in any commit:
- `HF_MODEL_CARD_FIX.md` (Sep 5)
- `hf_chucklenet_v2/` (Sep 5) — 16 files: README.md, README.md.backup, architecture_diagram.png (113KB), cv_results.png, fusion_analysis.png, performance_comparison.png, roc_comparison.png, generate_architecture.py, generate_charts.py, generate_improved_charts.py, requirements.txt, usage_example.py
- `manim_architecture/` (Sep 5) — plan.md + script.py for architecture visualization

These are stale Sep 5 work predating the Sep 13 audit + canonical paper promotion. Probably should be deleted (the canonical `paper/PAPER_DRAFT_V3_CONSOLIDATED.md` and `paper/figures/` supersede them).

### Finding 6: `chucklenet-local` has 5 branches that haven't been pushed

```
origin/add-colab-notebook
origin/clean-fixes
origin/colab-notebook
origin/main
```

Last commit `ebe3ba2`: "8-agent validation, cross-cultural 75.9%, autonomous research loop". Branches likely have work predating the canonical paper. Could merge `clean-fixes` into main if relevant; otherwise archive.

### Finding 7: 4 commits encode incomplete work

- `e36db26` adds a GitHub Actions workflow to auto-publish releases to Zenodo, but **no release tag exists** (so the workflow has nothing to publish).
- `e7f0843` uploads a chucklenet-verified model to Hayasuki namespace; **0 downloads**.
- `7ee008d` renames HF models with anime names; the renamed models are fine but with 0 downloads suggests this was preparatory, not load-bearing.
- `8163eb3` in HaHaScore: "HF model 0 downloads due to 7 critical issues" — **the 7 issues are still unresolved** as of 2026-10-05.

---

## 3. Missed Work vs Canonical Papers

### In `hahascore.tex` (HaHaScore falsification paper)
- All 5×3 CV findings (Finding #2) — ✅ verified against `5x3_cv_results.json`
- Speaker-disjoint identity leakage (Finding #1) — ✅ verified against `speaker_disjoint_cv_results.json`
- Recommendation section — ✅ verified, 5×3 repeated speaker-disjoint CV is the right standard
- Reproducibility appendix — ✅ lists all scripts and JSON

### In `PAPER_DRAFT_V3_CONSOLIDATED.md` (ChuckleNet canonical)
- Beyond-words +0.112 F1 at 5s horizon — ✅ RESULTS_LOG row 13
- +0.1147 comedian-disjoint — ✅ row 16
- +0.222 in bleed-impossible stratum — ✅ row 14
- +0.2157 horizon curve 15s — ✅ row 15
- MELD NULL Δ=0.0000 — ✅ row 12 (Tier-3 anchor)
- v30e Tier-2 transfer failure F1=0.029 — ✅ row 18
- v32 cited as case study — ⚠️ **v32 has no on-disk JSON artifact** (RESULTS_LOG pending flag)

### Not yet cited in either paper
- **HaHaScore-page launch pack** (commit `6b104ba`) — now cites honest 5-fold numbers
- **chuck-audio-notebooks v18/v19 split** — canonical notebook per memory is v19 gist `188a3bc5d4346c8189372f00c8bc2d39`
- **MSP-Podcast license** — was in flight, dropped per user directive
- **Dataset survey** (`DATASET_SURVEY_LAUGHTER_2026.md`) — research artifact, not paper claim

---

## 4. Top Recommendations

### CRITICAL (do now)
1. ✅ **Fix `HaHaScore-page` launch pack** — DONE (commit `6b104ba`)
2. ⚠️ **Restore `Das-rebel/ChuckleNet`** before submitting any paper that cites the repo URL
3. ⚠️ **Fix `hahascore-thumbnail.jpg`** to bake honest 5-fold number (manual)
5. ⚠️ **Audit `arxiv_endorsement_request_chuckle` MultiLinguahah citation** — never submit if it's still cited

### MEDIUM (this week)
6. Delete 4 stub dirs (chuckle-e2e-v14, chuckle-e2e-v30b, chuckleNet-FINAL-v6, hf_chucklenet_verified) — they have only `.gitkeep` / empty
7. Delete untracked `chucklenet-local/hf_chucklenet_v2/` + `manim_architecture/` — superseded by canonical paper
8. Decide on 4 stub branches in `chucklenet-local` — merge `clean-fixes` if useful, else delete
9. Tag `HaHaScore` release `v1.0-falsification` at `d471802` (the SciSlop-passed commit) — triggers Zenodo workflow

### LOW (later)
10. Consolidate 17 Colab notebooks in `chuck-audio-notebooks` — keep `Colab_Complete_Pipeline.ipynb`, archive the rest
11. Delete 1.6GB stale `.pt` `*.pt` files in `autonomous_laughter_prediction_essential/`
12. Restore `autonomous_laughter_prediction_essential` remote URL to `Das-rebel/ChuckleNet` (after step 2)

---

## 5. Files Created / Updated This Session

| File | Action |
|---|---|
| `/Users/Subho/HaHaScore-page/README.md` | REPLACED with deprecation notice + file fix list |
| `/Users/Subho/HaHaScore-page/show-hn.md` | REPLACED with honest 5-fold story |
| `/Users/Subho/HaHaScore-page/x-thread.md` | REPLACED with honest 5-fold |
| `/Users/Subho/HaHaScore-page/linkedin.txt` | REPLACED with honest 5-fold |
| `/Users/Subho/HaHaScore-page/youtube-description.txt` | REPLACED with honest 5-fold |
| `/Users/Subho/HaHaScore-page/index.html` | PATCHED: stat boxes 0.860→0.6924, 0.590→0.5453 |
| `/Users/Subho/funny-strength-predictor/REPO_AUDIT_2026.md` | THIS FILE |
| `Das-rebel/HaHaScore-page` | commit `6b104ba` pushed to origin |

## 6. What was MISSED that should be done today

1. **Manual thumbnail fix**: edit `hahascore-thumbnail.jpg` to bake "5-fold AUC 0.69" instead of "AUC 0.860". (Or delete the thumbnail until video is re-recorded.)
2. **Manual demo video re-record**: the 4 video files (.mp4) have inflated claims in the actual audio/visuals. Until done, the pack should be marked as not-yet-published.
3. **Confirm canonical notebook**: open `chuck-audio-notebooks/Colab_Complete_Pipeline.ipynb`, check whether it corresponds to v19 (has the 4 conditional continue / processed_idx guard / BatchNorm1d / start_idx>0 fixes). If yes, name it v19. If no, find the v19 notebook.
4. **Restore `Das-rebel/ChuckleNet`**: see Finding 2 — this is the most important infrastructure fix because the canonical papers all cite it.

## 7. Key Takeaway

The HaHaScore-page launch pack was the **single largest source of unsuppressed inflated claims** still on the internet. It was generated Sep 27 by the Duno pipeline (per the README), before the 5×3 CV falsification was completed. Now fixed.

The other big risk is `Das-rebel/ChuckleNet` being 404 — this is cited in every canonical ChuckleNet paper. Restoring this should be the next infrastructure task.

No other "missed work" of this magnitude. Most of the remaining inconsistencies are 1.6GB of stale `.pt` files and 17 Colab notebooks that need pruning, not falsified metrics.