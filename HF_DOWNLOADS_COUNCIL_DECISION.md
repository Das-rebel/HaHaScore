# Agent Council: Why 0 Downloads & How to Fix

**Date**: 2026-10-04 | **Council**: 3 specialist agents (HF investigation, HF ecosystem expert, DevRel)

## 🔍 Root Cause Diagnosis (Evidence-Based)

The model `Hayasuki/hahascore-cascade` is **technically published** (302 MB, 0 downloads, 0 likes) but every layer that drives adoption is broken or misconfigured.

| # | Issue | Evidence | Council fix |
|---|-------|----------|-------------|
| 1 | **Inflated single-fold metrics in card** | README says "Val AUC 0.802"; `hf_release/README.md` headline "v7 0.860"; audit confirms all are single-fold leakage. Honest 5-fold: **0.69/0.55** | Rewrite card with honest 5-fold numbers, add SpeakerDisjoint CV link |
| 2 | **Quick Start uses `np.random.randn` placeholders** | `hf_release/README.md` example: `text = np.random.randn(1, 20, 768).astype(np.float32)` | Add real `predict_humor(wav_path)` with WavLM feature extraction |
| 3 | **Stub ONNX shipped as recommended** | `v8_cascade_fp32.onnx` = 0.28 MB (smaller than INT8 2.9 MB). Metadata says `v8_cascade_v1.onnx` which doesn't exist. | Delete stub, rename to clear canonical file, add file legend |
| 4 | **Wrong/missing `pipeline_tag`** | Current tags: "bilin_fusion" (typo, spaces), "humor detection" (broken spaces), no pipeline_tag | Set `pipeline_tag: audio-classification`, fix tags: 15 curated, lowercase-hyphenated |
| 5 | **No demo Space** | HF API: `"spaces": []`. Every viral HF model has a Space; the listed `Hayasuki/hahascore-cascade-demo` is **HTTP 401** | Create `Hayasuki/hahascore-cascade-demo` Space with `sdk: gradio` |
| 6 | **Demo downloads from URL that doesn't exist** | `hf_demo_v2/index.html` fetches `v10_cascade_int8.onnx` from `Hayasuki/hahascore-cascade` repo, but only v8 files are there | Upload v10 ONNX (3.4 MB) to HF Hub repo, then demo works |
| 7 | **3 inconsistent README files, broken citation** | `HUGGINGFACE_MODEL_CARD.md` cites `hahascore-bridge4-arc-tracker` (doesn't exist); multiple model versions (v7, v8, v10) bundled in one repo | Delete 5 redundant card files, pick one canonical model (v10), single consistent card |

## 📊 7-Day Council Plan

| Day | Action | Time | Why |
|---|---|---|---|
| 1 | Rewrite model card with honest 5-fold numbers + fix tags | 2 h | **#1 priority** — without honest metadata, nothing else compounds |
| 2 | Fix `app.py` SHA-1 placeholder; upload v10 ONNX to HF Hub | 1 h | Removes silent failure mode |
| 3 | Deploy `Hayasuki/hahascore-cascade-demo` Space (gradio) | 1.5 h | #1 download-driver on HF |
| 4 | Create `notebooks/score_a_joke.ipynb` | 3 h | Researchers gate-download on "can I run it without setup?" |
| 5 | Add HF badges to GitHub README + back-link Space | 1 h | Bidirectional traffic |
| 6 | Post to 5 community channels (r/ML, HF Discord, Papers with Code, 3 WavLM repos) | 4 h | Cross-platform reach, zero cost |
| 7 | Open 3 PRs to WavLM-using repos adding HaHaScore as downstream reranker | 4 h | Only compounding channel |

**Total**: 16.5 hours, 7 days, $0 compute, 0 new data.

## 🎯 Single Highest-EV Action (Day 1)

Rewrite the model card with **honest 5-fold speaker-disjoint CV numbers**:

```yaml
---
license: mit
library_name: pytorch
pipeline_tag: audio-classification
tags:
  - humor-detection
  - comedy
  - audio-classification
  - multimodal
  - multimodal-fusion
  - cascade-gate
  - pytorch
  - onnx
  - wavlm
  - speech
  - stand-up-comedy
  - prosody
  - text-audio-fusion
  - speaker-disjoint-cv
  - reproducibility
language: en
datasets:
  - SocialGrep/one-million-reddit-jokes
---

# §1 Honest Disclosure (top of card)
> **Single-fold AUC of 0.860 was found to be a speaker-identity leakage artifact. 
> 5-fold speaker-disjoint CV (the only honest protocol for this dataset): 
> **humor AUC 0.69 ± 0.04, gold AUC 0.55 ± 0.13** (n=639 standup clips, 12 gold videos).
> A +0.163 per-language normalization claim from a single 5-fold CV was falsified by 
> 5×3 repeated CV (true effect: −0.127, p=0.0004).
> See `COUNCIL_FINAL_DECISION.md` on GitHub for the full audit.

# §4 Verified Benchmarks (5-fold speaker-disjoint CV, honest)
| Model | Humor AUC | Gold AUC | Notes |
|------|-----------|----------|-------|
| v7 (single-fold) | 0.860* | — | *leakage artifact |
| v8 (single-fold) | 0.802* | — | *leakage artifact |
| **v10 (5-fold)** | **0.6924 ± 0.0359** | **0.5453 ± 0.1313** | **honest** |
| v10 + Reddit (5f) | 0.7016 ± 0.0404 | 0.5000 ± 0.0000 | Reddit pretrain: +0.009 humor, **−0.045 gold (active harm)** |

# §5 Pipeline code
```python
from transformers import WavLMModel, AutoTokenizer
import torch, librosa, numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

ort_sess = ort.InferenceSession(hf_hub_download(
    "Hayasuki/hahascore-cascade", "v10_cascade_int8.onnx"))
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()

def predict_humor(wav_path):
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    n_segments = 20
    seg = len(y) // n_segments
    feats = []
    for i in range(n_segments):
        chunk = y[i*seg:(i+1)*seg]
        chunk = np.pad(chunk, (0, max(0, 8000-len(chunk))))[:8000]
        chunk_4k = librosa.resample(chunk, orig_sr=16000, target_sr=4000)
        with torch.no_grad():
            f = wavlm(torch.tensor(chunk_4k).unsqueeze(0)).last_hidden_state.mean(1).numpy().squeeze()
        feats.append(f)
    audio = np.stack(feats)[None].astype(np.float32)
    audio = np.pad(audio, ((0,0),(0,0),(0, audio.shape[-1])))  # 768→791
    text = np.zeros((1, 20, 768), dtype=np.float32)
    cross = np.zeros((1, 20, 16), dtype=np.float32)
    scores, conf, gate = ort_sess.run(None, {'text':text,'audio':audio,'cross':cross})
    return scores.squeeze().tolist()
```

# §6 Limitations & Failure Cases
- Gold AUC drops 0.12-0.15 under speaker-disjoint CV (identity leakage)
- Per-language: Spanish gold AUC 0.72, French gold AUC 0.26 (language confounds)
- Per-language normalization HURTS gold AUC by -0.127 (5×3 CV falsification)
- Reddit pretrain helps humor +0.009 but hurts gold -0.045 (net-negative)
- Not production-deployed; demo currently shows mock scores
```

## 🔢 Honest Numbers (from `5x3_cv_results.json` + `RETHINK_2026_v2.md`)

| Metric | Value | CV | Source |
|---|---|---|---|
| v10 humor AUC | **0.6924 ± 0.0359** | 5-fold speaker-disjoint | kaggle `hahascore-v10-5fold-cv` |
| v10 gold AUC | **0.5453 ± 0.1313** | 5-fold | kaggle `hahascore-v10-5fold-cv` |
| v10 + Reddit humor | 0.7016 ± 0.0404 | 5-fold | kaggle `hahascore-scaleup-100k` |
| v10 + Reddit gold | 0.5000 ± 0.0000 | 5-fold | (Reddit harm: -0.045) |
| Spanish gold (LOGO) | 0.72 | n=1, n_pos=~12 | `speaker_disjoint_cv_results.json` |
| French gold (LOGO) | 0.26 | n=1, n_pos=~5 | (language confounds) |
| Per-lang norm true effect | **-0.127 (p=0.0004)** | 5×3 repeated CV | `5x3_cv_results.json` (FALSIFIED +0.163) |
| Architecture ablation v7→v10 | σ_arch ≈ 0 ≪ σ_fold = 0.13 | D'Amour 2020 underspecification | `RETHINK_2026_v2.md` |

## 📋 Council Decision

**All 3 agents unanimously recommend**:

1. **Day 1 first**: Rewrite model card with honest 5-fold numbers + fix `pipeline_tag: audio-classification` (15 min) — without this, nothing else compounds because no one can find or trust the model.
2. **Day 2 next**: Fix `app.py` SHA-1 placeholder + upload v10 ONNX to HF Hub — these are 1-hour fixes that remove silent failure modes.
3. **Then**: 7-day plan as outlined.

**What NOT to do**:
- ❌ Don't keep citing inflated single-fold numbers (0.860, 0.823, 0.613) — they're all leakage artifacts per `RETHINK_2026_v2.md` audit.
- ❌ Don't use the stub `v8_cascade_fp32.onnx` (0.28 MB) — it's smaller than the INT8 sibling and a misleading twin.
- ❌ Don't pursue per-language normalization — falsified by 5×3 CV (-0.127, p=0.0004).
- ❌ Don't iterate architecture (v7→v8→v9→v10 are statistically indistinguishable, σ_arch ≈ 0 ≪ σ_fold = 0.13).

**The honest version is the only version that survives peer review.** The honest numbers are publishable; the inflated ones are not.
