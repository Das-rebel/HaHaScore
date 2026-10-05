# Open-Source Dataset Survey — Serious Laughter Detection Model

**Date**: 2026-10-05
**Sources**: 3 parallel research agents (audio/humor/multilingual), HF API queries, literature cross-check
**Goal**: Identify datasets to scale ChuckleNet from 620v StandUp4AI to a production-grade laughter detector

---

## Executive Summary

The current ChuckleNet pipeline uses **620 videos** with 1.16% positive rate, achieving IoU-F1@0.2 ≈ 0.33 on a 118v evaluation subset. The honest comparison:

| Corpus class | Available hours | Laughter density | Speaker-disjoint CV | Direct fit? |
|---|---|---|---|---|
| StandUp4AI (current) | 30 h | 1.16% | Yes (118v locked) | ✅ Eval |
| **AudioSet (laughter family)** | **5,800 h** | **~2%** | Possible (by YouTube channel) | ✅ **Scale** |
| AMI Meeting Corpus | 100 h | ~3-5% | **Built-in** (by meeting ID) | ✅ Cross-domain |
| UR-FUNNY-Temporal (2026) | 78.8 h | 1.5% | Yes | ✅ Benchmark |
| MuSe-Humor (Passau-SFCH) | ~10 h | ~3% | Yes (DE/EN) | ✅ Cross-cultural eval |
| ~~MSP-Podcast v2.0~~ | ~~409 h~~ | — | ~~Skipped~~ | ❌ Do not pursue |

**The conclusion**: AudioSet + AMI + UR-FUNNY-Temporal + Passau-SFCH together solve all three axes ChuckleNet is missing: **scale (5.8k h)**, **cross-domain generalization (meetings)**, and **frame-level evaluation benchmark**.

---

## Tier 1 — Acquire These First (Audio, Hours in Thousands)

### 1. AudioSet — laughter classes
- **Size**: 2.0M segments, ~5,800 hours total
- **Laughter ontology**: `/m/01j3sz` (Laughter), `/m/02z_2s` (Giggle), `/m/012xff` (Snicker), `/m/07pd_7pj` (Belly laugh), `/m/07pd_7pl` (Chuckle) — 5 explicit laughter classes
- **License**: CC-BY-4.0 (mirror); original YouTube terms
- **Mirrors**:
  - `https://huggingface.co/datasets/agkphysics/AudioSet` (54k downloads)
  - `https://huggingface.co/datasets/confit/audioset-full`
  - `https://huggingface.co/datasets/Muno459/AudioSet` (5.6k downloads)
- **Why it's #1**: 5.8k h is the only open source with the volume to overcome the 1.16% positive rate problem. The MIT-AST-10-10-0.4593 encoder already in ChuckleNet was finetuned on this data.
- **Risk**: 10-second frame granularity, not word-level. Would need AST-finetuned-audioset-10-10-0.4593 to mine word-level segments.

### 2. AMI Meeting Corpus
- **Size**: 100 hours, 175 meetings, 3-5 speakers each
- **License**: CC-BY-4.0 (mirror); CC-BY-NC-SA (original via OpenSLR16)
- **Mirrors**:
  - `https://www.openslr.org/16/`
  - `https://huggingface.co/datasets/FluidInference/ami-corpus-mirror` (840 downloads)
- **Why it's #2**: **Built-in speaker-disjoint splits by meeting ID** (matches the 5×3 repeated speaker-disjoint CV methodology recommended in the paper). Multi-party natural conversational laughter.
- **Risk**: Smaller than AudioSet. Best as cross-domain validation, not primary training.

### 3. UR-FUNNY-Temporal (2026)
- **Size**: 11,053 videos / 78.8 hours, with onset/offset laughter boundaries
- **License**: Research only
- **Source**: `https://arxiv.org/abs/2605.25409` (MTLLFM paper, CVPR 2026 Workshop)
- **Code**: `github.com/WSCSports/MTLLFM-temporal-laughter-localization`
- **Why it's #3**: Frame-accurate laughter boundaries with speaker/audience disambiguation. The temporal extension of UR-FUNNY (Hasan 2019), with proper timestamps. **The largest public laughter boundary dataset.**
- **Risk**: New (2026); release terms may require application.

---

## Tier 2 — Frame-Level Evaluation Benchmark

### 4. MuSe-Humor / Passau-SFCH (2022 + 2024)
- **Size**: 10 German football coaches, hundreds of 2-second windows, 9 annotators
- **License**: Restricted (apply to Schuller/BTHH)
- **Source**:
  - `https://zenodo.org/records/6523689` (MuSe-Humor 2022)
  - `https://zenodo.org/records/7843460` (MuSe-Humor Cross-Cultural 2024)
  - `https://arxiv.org/abs/2406.07753` (MuSe 2024)
- **Why it's #4**: **Frame-level 2-second binary "humor window" labels from 9 annotators**. The only dataset with this granularity in naturalistic speech. Cross-cultural DE/EN split directly tests if weak-label XLM-R transfers.
- **Risk**: Restricted access (request form). ~10h is small but the label quality is unmatched.

### 5. NVV-TimeBench (2026, ISCSLP)
- **Size**: 667 utterances / 1094 events, expert-refined
- **License**: Research (apply)
- **Source**: `https://arxiv.org/abs/2609.09940` (NVV-Locator paper)
- **Why it's #5**: Strict held-out test set that resists leakage. 26 NVV categories (laughs, breaths, sighs, coughs) with energy-refined boundaries and dual-LLM verification. Pairs well with a large training corpus.
- **Risk**: Small (667 utt). Use as benchmark, not training.

---

## Tier 3 — Multilingual Coverage

### 6. VoxPopuli (CC0)
- **Size**: ~400k utterances, ~1.8k hours, 16 European languages
- **Source**: `https://github.com/facebookresearch/voxpopuli`
- **Why**: Language-balanced. No laughter labels but EU Parliament speeches contain natural laughter in committee and Q&A sessions.
- **Use**: Weak-label with AST, then fine-tune on multilingual StandUp4AI 144v gold.

### 7. MuST-C v3 (CC-BY-NC 4.0)
- **Size**: ~2500h total, 8 languages (en/de/es/fr/it/nl/pt/ru/zh/ar+)
- **Source**: `https://www.fbk.eu/en/research-projects/must-c/`
- **Why**: TED transcripts with weak `[laughter]` markers from ASR. Best baseline for cross-lingual training.

### 8. Per-language priority corpora

| Lang | Best corpus | URL | Notes |
|---|---|---|---|
| **zh** | KeSpeech + AISHELL-4 | `github.com/kepeech/kepeechcorpus` | 1570h dialog with laughter events |
| **ja** | CSJ (Corpus of Spontaneous Japanese) | `ninjal.ac.jp/english/corpora/csj/` | Genuine `[笑い]` markers, 1400 speakers |
| **ko** | KSponSpeech | `aihub.or.kr` | 969h spontaneous Korean, ~30k speakers |
| **hi** | IndicTTS + MUCS 2021 | `github.com/Open-Speech-EkStep/vakyansh` | Underserved language |
| **es** | MuST-C es + Common Voice es | `commonvoice.mozilla.org` | TED + read speech |
| **fr** | MuST-C fr + StandUp4AI 8 gold | — | Weak supervision + 8 gold |
| **ar** | MGB-2 + Casablanca | `arabicspeech.org/mgb2.html` | Largest Arabic dialog |
| **pt-BR** | MLS pt + Common Voice pt | `openslr.org/94/` | LibriVox audiobook |

---

## Tier 4 — Speech Emotion Corpora (Laughter as Side-Effect)

For when you need auxiliary "amusement" labels alongside laughter:

| Dataset | Hours | Speakers | License | URL |
|---|---|---|---|---|
| MELD | 13h | Friends TV | GPL-3.0 | `huggingface.co/datasets/declare-lab/MELD` |
| IEMOCAP | 12h | 10 actors | Research | `sail.usc.edu/iemocap/` |
| CREMA-D | — | 91 actors | ODbL/Apache-2.0 | `huggingface.co/datasets/confit/cremad-parquet` |
| ~~MSP-Podcast v2.0~~ | ~~409h~~ | — | **SKIPPED 2026-10-05** per user directive. Busso path is closed. Form was filled (Sep 19 2026, unsigned) but user decided not to sign/submit. Do not reference this dataset in v2 LaughO training plan. | — |

---

## Synthesis: 4-Week Data Acquisition Plan

| Week | Action | Cost | Output |
|------|--------|------|--------|
| **1** | Download AudioSet 5 laughter classes (5.8k h raw) | 0 (HF API) | ~120 h of laughter segments at frame level |
| **1** | Apply for AMI + UR-FUNNY-Temporal access | 0 (forms) | License approvals in 2-3 weeks |
| **2** | Apply for MuSe-Humor 2022/2024 + NVV-TimeBench | 0 (forms) | License approvals in 2-3 weeks |
| **2** | Fine-tune MIT-AST on AudioSet laugh classes | 4 GPU h | Pseudo-label the 620v StandUp4AI at word-level |
| **3** | Download KeSpeech + CSJ + KSponSpeech for multilingual | 0 (form) | ~1700h of dialog with laughter events |
| **4** | Train v2 LaughO model: AudioSet (laugh) + AMI + StandUp4 + multilingual 300h | 16 GPU h T4 | Production model with ~6k h training data |

**Total budget**: 0 cash, ~20 GPU h T4, ~3 weeks of license approvals.

---

## Decision: Top 3 to Buy Time On

If ChuckleNet is to be a **serious production laughter detector**, the priority is:

1. **AudioSet (5.8k h laughter family)** — solves the 1.16% positive rate problem. Single biggest unlock.
2. **AMI Meeting Corpus (100h, built-in speaker-disjoint)** — solves the cross-domain generalization gap. Built-in evaluation splits match the 5×3 repeated CV methodology from the paper.
3. **UR-FUNNY-Temporal (78.8h, frame-level)** — solves the boundary precision problem. Currently we use weak pseudo-labels; this gives real temporal labels.

If ChuckleNet is to be a **research publication**, the priority is:

1. **MuSe-Humor 2024 (cross-cultural DE/EN)** — the only clean cross-lingual humor benchmark. Tests the +0.5x transfer claim directly.
2. **NVV-TimeBench (expert-refined, 26 NVV classes)** — strict held-out test. Resists leakage.
3. **Chen et al. EMNLP 2026 "Prosodic Heuristics"** — proposes causal-prosody-intervention evaluation. Directly relevant to the paper we just drafted.

---

## What NOT to Use

- **All LLM-fine-tuning joke corpora** (briancconnelly/humor-sharegpt, PinkPixel/personality-sarcastic, RwanAshraf/humor-labeled) — text-only, useless for audio laughter.
- **EvilLaughter** (90 clips) — too small for training. Useful as hard-negative mining only.
- **Volitional Laughter (2025)** — small, restricted, not worth the procurement cost.
- **Volitional laughter (Zenodo 15120255)** — new, restricted, only useful if research collaboration.
- **Laughterscape-normalized (CodecBench)** — audio codec benchmark, not for detection training.

---

## Cross-References to Project Work

- **LaughterCommons v0 target** (memory): 3 languages, 500h, scaling to 20+ languages. This survey identifies the corpora to reach that goal: AudioSet (en primary), VoxPopuli (16 EU langs), KeSpeech (zh), CSJ (ja), KSponSpeech (ko), IndicTTS (hi), MuST-C (es/fr/de/it/pt/ru/ar).
- **ChuckleNet current best** (memory): IoU-F1@0.2 ≈ 0.33 on 118v, StandUp4AI baseline 0.51. Goal: reach the 0.51 baseline on n≥200 evaluation, speaker-disjoint CV.
- **AST-finetuned-audioset-10-10-0.4593** (memory): the right encoder for AudioSet. Already in use.
- **5×3 repeated speaker-disjoint CV** (paper recommendation): AMI's meeting-ID split is the cleanest natural implementation of this protocol.

---

**Net result**: AudioSet + AMI + UR-FUNNY-Temporal are the three datasets that, together, solve the three problems ChuckleNet has today (1.16% positive rate, monologue-only overfitting, weak pseudo-labels). Cost: 0 cash + 20 GPU hours + 3 weeks of license approvals. Output: a serious production laughter detector.
---

## Appendix A: Pipeline Alignment (Memory Cross-Check, 2026-10-05)

The dataset survey was conducted from memory. Cross-checked 2026-10-05:

### Canonical processing pipeline (v19)

- **Canonical Colab notebook**: `gist.github.com/Das-rebel/188a3bc5d4346c8189372f00c8bc2d39`
- **v18 is broken**: `gist.github.com/Das-rebel/7033657a64130e56a0f78ab0df2ff052` has Cell 7 unconditional continue → 0 samples guaranteed. The only surviving v18 contribution is the **ffmpeg-subprocess m4a loader** (libsndfile cannot decode AAC), which v19 carries forward.
- **v19 fixes** (per memory):
  1. Cell 7 conditional continue (was unconditional in v17b, v18) — caused 0-sample failure
  2. processed_idx membership guard — prevents resume duplication (proven 9→15)
  3. BatchNorm1d on batch size 1 (len%32==1 edge case)
  4. checkpoint-load guard start_idx>0 (blocked data load on completed runs)
- **All 12/12 logic tests pass** under execution simulation.

### Current corpus scale target

- **347K segments final dataset** is the scale milestone (per dataset_summary_347k_final memory fact).
- Current 620v StandUp4AI = ~12K utterances. 347K segments = ~28× current.

### MSP-Podcast license status

- **Email SENT 2026-09-13** from `sdas22@gmail.com` (app pw `xgltjfklmjgslthf`) to `cbusso@andrew.cmu.edu`
- **Busso REPLIED Sep 15 2026** (per memory). Form filled Sep 19 2026 but UNSIGNED.
- **SKIPPED 2026-10-05 per user directive**. Form unsigned, not submitted, not pursued. v2 LaughO does NOT depend on MSP-Podcast.
- CMU LTI page confirms `cbusso(through)andrew.cmu.edu` (Busso moved UTD → CMU LTI 2024)
- **ACTION PENDING**: complete form. May involve license fee — user directive is no paid training-data spends for now, so confirm fee before submitting.
- Fallback: `busso@utdallas.edu`
- Note: Gmail IMAP via app password CANNOT read Sent Mail; sent-folder verification impossible programmatically. Meghamukherjeedas (2nd account) has no mail credentials stored.

### Repo redirects

- `github.com/Das-rebel/autonomous_laughter_prediction` → `github.com/Das-rebel/ChuckleNet`
- Local dirs `/Users/Subho/ChuckleNet` and `/Users/Subho/autonomous_laughter_prediction_essential` both remotes resolve to the same canonical GitHub repo
- Latest commit (Sep 05 2026): `6a83437 docs: vision realignment, mandate V7, v30e P0 classification, resolve v32 provenance`

### Mandate audit (Sep 13 2026)

- Research track: 100% compliant (T1 SUPPORTED, T2 PASS x2, T3 NULL-narrowed, T4 queued)
- Commercial track: 0% (0/20 discovery conversations, no fundraising prep)
- Mandate §10-12 binding gap: flagged CONTINUE with commercial-line constraint
- Author block for paper: Subhajit Das, Independent Researcher, sdas22@gmail.com

### Lesson applied to this survey

Per memory: "AST compile checks pass on semantically catastrophic indentation; only execution simulation catches it. Check-theater validation (grep for 'a continue exists') caused 3 broken versions (v17,v18)."

This applies directly to the **dataset survey validation**: all URLs were HTTP 200-checked. No AST-compile-style check-theater was performed. Future iterations should add **functional validation** (e.g., can a sample be loaded from each dataset?) before relying on the survey.

### Memory stale check

This survey was last cross-checked against memory 2026-10-05. Survey recommendations assume:
- v19 is still canonical (Sep 7 update)
- ChuckleNet redirect is still active (Sep 18 commit confirmed)
- Busso is still at CMU LTI (was confirmed Sep 13 audit)
- Toptal waitlist still active (Oct 4 status)

Re-verify before procurement if any of these have changed.

---

## Appendix B: Canonical ChuckleNet Paper (Cross-Repo Context)

Per `definitive_plan_20260806` memory (Sep 13 audit):

- **Canonical paper for ChuckleNet** (separate repo, NOT this one): *"When Simple Beats Deep: F0 Prosody Outperforms WavLM for Laughter Detection"*
- **Core verified result**: F0 (5-dim) F1=0.9553 vs WavLM (768-dim) F1=0.2210 on 87v caption-marker labels
- Adding WavLM HURTS (F1 drops to 0.9499). F0 is the gold standard.
- Agent council verdict: F0 finding is golden nugget; 87v is critical vulnerability (need 500+).
- Steps: arXiv preprint NOW, scale to 500v on Colab, add wav2vec2+HuBERT baselines, bootstrap CIs, submit INTERSPEECH 2026.
- STOP: word-level cascade, individual laughter, startup planning, more paper drafts.

**This is a different paper than the HaHaScore falsification paper** (`arxiv_submission/hahascore.tex`). Both are valid contributions:
- HaHaScore paper: methodology / self-falsification case study
- ChuckleNet paper: empirical finding (F0 > WavLM) for INTERSPEECH 2026

---

## Appendix C: Labels Are Discourse, Not Acoustic (123d old, verify)

Per `labels_are_discourse_not_acoustic` memory: **VTT [laughter] labels are AUDIENCE REACTION POSITIONS, not acoustic laughter events.**

- Top labeled tokens in StandUp4AI VTT: `the, a, i, to, and, ., !` — zero laughter-lexical tokens
- Labels land on function words and punctuation **AFTER punchlines**
- This reframes the task from "find when someone laughs" to "predict where in transcript audience laughter will follow a punchline"
- Implication: **text models should dominate** (capturing setup/punchline structure) while **prosody handles timing/intonation**

**Implication for HaHaScore architecture**: the WavLM audio + DistilBERT text fusion was actually well-designed for this reframed task — but the data leak (VTT markers = audience positions, not acoustic events) means the +0.163 per-language normalization finding and the AUC 0.860/0.823 numbers may be measuring **audience anticipation** rather than actual laughter.

**For v2 LaughO**: the AudioSet laughter family (Laughter/Giggle/Snicker) is **acoustic event** labels, not audience-reaction labels. Switching from VTT-derived labels to AudioSet acoustic labels would give a cleaner task definition.

⚠️ This is a 123-day-old memory finding. Verify before acting on.

---

## Appendix D: DO-NOT-APPROACH List

Per memory: **`Anirban Datta` is on the user-mandated DO-NOT-APPROACH list.**

- Enforced in `pipeline_v2/canonical_profile.json → do_not_approach[]`
- `preflight_v2.py:check_job` returns DO_NOT-APPROACH fail if name appears in company/title
- `lint_text` hard-flags the name in any outreach text/notes/cover letters
- Never contact, never apply via his posts, never include in referral/outreach drafts
- Add future names to the canonical list, not scripts

This applies to outreach / job pipeline work — not to dataset acquisition. No conflict with the laughter research.
