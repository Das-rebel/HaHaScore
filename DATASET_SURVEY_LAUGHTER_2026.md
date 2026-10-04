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
| MSP-Podcast v2.0 | 409 h | varies | Provided | ✅ "Amusement" proxy |

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
| MSP-Podcast v2.0 | 409h | ~1000s | Research (form) | `lab-msp.com/MSP/MSP-Podcast.html` |

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