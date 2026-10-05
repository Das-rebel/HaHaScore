# Strategic Rethink — HaHaScore v1 (Oct 5 2026)

**Date**: 2026-10-05
**Status**: V1 plan validated; conflict between v5 "text-only random" and v1 "text-only Jester" resolved
**Verdict**: v1 is honest and worth 4 hours. v2 blocked by data. v3 blocked by data.

---

## The conflict that almost broke the v1 plan

I had two findings in tension:

| Finding | Source | Implication |
|---|---|---|
| Text-only is random (AUC ~0.50) for humor DETECTION | `hahascore_v5_findings` (Sep 15) | Text is not informative for binary classification |
| v1 = text-only RoBERTa regression on Jester | `rethink_v1` (Sep 8) | Continuous regression should work with text |

**Resolution**: The v5 null applies to **detection** (binary classification of standup clips). Jester is a different task — 1.76M jokes with continuous 0-100 ratings where the discriminator is *which text* gets votes, not *which delivery*. M3 already confirmed ρ=0.313, 8 points above TF-IDF baseline 0.277.

But the v5 ceiling IS real: **text-only will plateau**. Realistic ceiling: **ρ ≈ 0.35-0.40**, reachable with roberta-large or LoRA fine-tuning. The pre-registered gate (ρ ≥ 0.35 AND CI excludes 0.30) is correctly calibrated to detect fold-luck.

---

## The full path forward (with realistic ρ targets)

| Phase | Target metric | Lower bound | Upper bound | Status |
|---|---|---|---|---|
| **v1** text-only Jester | Spearman ρ | 0.313 (M3) | 0.40 (LoRA+large) | **Shippable today** |
| **v2** text + audio (AST) | ρ | 0.40 | 0.55 (v10 humor 0.69 mapped) | Blocked: no audio+Jester |
| **v3** humor-conditioned generation | n/a | n/a | n/a | Blocked: zero (joke, audio, rating) triplets |

**v2 / v3 are blocked by data, not by algorithm.** No escape until we have:
- Audio + Jester alignment (no public dataset exists)
- Labeled triplets for generation (no public dataset exists)

**This means: v1 is the only honest v currently.**

---

## Cross-project synthesis (research + commercial + jobs)

**Time budget** (8 hrs/week research window, job pipeline parallel):
- Research: 4 hrs/wk (preserve only honest, falsified-by-5×3-CV work)
- Commercial: 3 hrs/wk (mandate §10-12: 0/20 discovery conversations is binding)
- Admin: 1 hr/wk

**Pivot triggers:**
| v1 result | Action |
|---|---|
| ρ ≥ 0.35 | Write methodology paper, arXiv cs.CL |
| ρ < 0.35 | Methodology-only paper still publishable (5×3 repeated CV on continuous regression is novel for humor) |
| 95% CI spans 0.30 | Gate fails; DO NOT chase AUC; publish protocol contribution only |

**Exit criterion**: "Research is done" = (a) v1 paper on arXiv with honest 5×3 numbers, OR (b) 3 commercial discovery conversations closed. Do NOT exit on AUC claims alone — every prior high-AUC number in this repo (0.860, 0.823, +0.163) has been falsified.

---

## 4-week concrete plan

| Week | Action | Cost | Deliverable |
|---|---|---|---|
| **1** | Ship honest v1 + commercial seed | 12h | Result JSON + 5 LinkedIn DMs |
| **2** | Decide v1 vs commercial | 8h | Gate check OR pivot to commercial |
| **3** | v2 prep OR paper lock | 10h | arXiv skeleton OR 15 commercial calls |
| **4** | arXiv OR commercial close | 8h | Submitted paper OR 3 pitch meetings |

**Week 1 specific (start of v1 launch):**

```
Mon:  - Fix app.py real v10 load (replace SHA-1 fallback)
      - Delete v8 stub (hf_release/v8_cascade_int8.onnx)
      - Delete stale .pt (~155 MB freed)
Tue:  - Upload v1_jester_train.ipynb to Colab T4 (4-6h run)
      - 5 LinkedIn discovery DMs (parallel)
Wed-Thu: v1 result.json committed
        + 3 commercial follow-ups
Fri:  - If ρ ≥ 0.35: draft v1 paper §2-4
      - Else: methodology-only paper
```

---

## 7 failures to avoid (from memory)

1. ❌ **Do NOT cite AUC 0.860 or 0.823** — both are single-fold leakage artifacts (per `COUNCIL_FINAL_DECISION.md:1`). Already rewritten in README; demo badge still says 0.823 — needs update.
2. ❌ **Do NOT trust single-fold AUC on imbalanced data.** +0.163 was −0.127 in 5×3 CV (p=0.0004). v1 pre-registers this exact trap.
3. ❌ **Do NOT add per-language normalization.** Falsified Oct 2.
4. ❌ **Do NOT chase v3 generation.** Zero (joke, audio, rating) triplets exist; 340 audio_final files overlap zero Jester IDs.
5. ❌ **Do NOT ignore commercial-line gap.** Mandate §10-12 binding; 0/20 discovery conversations is a structural risk.
6. ❌ **Do NOT confuse "humor detection AUC 0.69" with "humor strength ρ 0.69."** Different scales, different tasks.
7. ❌ **Do NOT cite PR-AUC for v1** (continuous regression uses ρ; PR-AUC is for sparse binary).

---

## What v2 needs to become unblocked

| Need | Effort |
|---|---|
| Audio + Jester alignment (no public dataset) | Would need YouTube clip matching + AST inference |
| StandUp4AI + audio for ρ-style regression | Extract WavLM embeddings from 620v audio, train regression head on AST laugh-label column |

**Realistic v2 path**: Use AST to label StandUp4AI 620v with continuous laughter-strength scores (not binary). Train `text+AST-projected_voice_v2` model. Validate with 5×3 joke-disjoint CV. **Estimate: 6-8 weeks** after v1 ships, requires GPU.

---

## Single line summary

**v1 (RoBERTa-on-Jester) is honest and shippable. Realistic target ρ ≈ 0.30-0.40. v2/v3 blocked by data. Commercial line gap is the binding constraint per mandate §10-12. The 4-week plan above sequences research + commercial so neither starves the other.**

The +0.163 lesson — *single-fold metrics on imbalanced data are systematically misleading* — is the rule that should govern every future AUC claim in this repo. v1's pre-registered gate is the operationalization of that lesson.

---

**Commit**: this rethink doc will be the new anchor in `IMPROVEMENT_PLAN.md`. The 4-week schedule supersedes the original 4-week plan from the council realignment, because it adds the commercial-line work that was missing.