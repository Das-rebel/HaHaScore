---
language:
  - en
license: apache-2.0
library_name: pytorch
pipeline_tag: text-classification
tags:
  - humor-detection
  - comedy
  - text-only
  - deberta
  - text-classification
---

# HaHaScore Text v7 — Text-Only Humor Baseline

**Hugging Face Model ID:** `Hayasuki/hahascore-text-v7`

⚠️ **DEPRECATED for fusion use. Use `Hayasuki/hahascore-fusion-v5` instead.**

This model is the **text-only baseline** for HaHaScore research. It was used to establish that text features alone carry near-zero humor signal at the sentence level.

---

## Key Research Finding

| Model | AUC | vs. Random |
|-------|-----|------------|
| Random | 0.500 | — |
| **Text-only (this model)** | **~0.50** | **≈ Random** |

> **Text alone cannot distinguish funny from not-funny sentences.** Comedy is about delivery — pitch, timing, prosodic emphasis. The words carry the content; the voice carries the humor.

This model exists as a **baseline ablation** to prove that audio prosodic features are the primary discriminative signal in humor detection.

---

## Architecture

Full RoBERTa-large fine-tuned for binary humor classification.

| Component | Value |
|-----------|-------|
| Base model | roberta-large |
| Input | Sentence text |
| Output | Binary (funny/not-funny) |
| Trained on | 3,774 sentence clips |

---

## Performance

| Metric | Score |
|--------|-------|
| Held-Out AUC | ~0.499 |

Essentially random — confirming that sentence-level humor detection requires audio.

---

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Hayasuki/hahascore-text-v7")
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

inputs = tokenizer("I told my wife she was drawing her eyebrows too high.",
                   return_tensors="pt")
outputs = model(**inputs)
# logits → probability of being "funny"
```

---

## Superseded

This text-only model has been superseded by the multimodal fusion model:

**[Hayasuki/hahascore-fusion-v5](https://huggingface.co/Hayasuki/hahascore-fusion-v5)** — combines DeBERTa-v3-base text + WavLM-base-plus audio for AUC 0.613.

---

## License

Apache 2.0

*Updated 2026-09-16*
