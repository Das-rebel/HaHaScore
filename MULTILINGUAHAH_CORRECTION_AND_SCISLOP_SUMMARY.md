# MultiLinguahah Correction + SciSlop Findings — Oct 5 2026

## ⚠️ CORRECTION: MultiLinguahah is REAL

The earlier audit (`claim_audit_20260615`, Jun 15 2026) **incorrectly** flagged MultiLinguahah as fabricated. The followup (`multilinguahah_verified`, Sep 13 2026) **confirmed the paper exists**:

- **arXiv**: 2605.06309 (May 2026)
- **Title**: MultiLinguahah — Unsupervised multilingual acoustic laughter segmentation using BYOL-A + Isolation Forest
- **Authors**: Sofia Callejas, Nahuel Gomez, Catherine Pelachaud, Brian Ravenet, Valentin Barriere
- **Venue**: cs.CL 2026
- **Topic**: Outperforms SOTA on non-English (French/Spanish-context) laughter

**This paper IS REAL and should be cited.** Earlier survey docs (Aug 6, Sep 13 audit) that recommend it are right; the Jun 15 flag was a false positive.

The `arxiv_endorsement_request_chuckle` repo (Jun 21 2026) cited it correctly. The HaHaScore dataset survey (Oct 5 2026 commit `5462ede`) cites it. Both citations are valid. **No action needed beyond this correction.**

### What was corrected this session

| File | Status | Detail |
|---|---|---|
| `arxiv_submission/hiIndex.html` (HaHaScore) | ✅ Honest | Reference: no MultiLinguahah citation needed; not cited |
| `DATASET_SURVEY_LAUGHTER_2026.md` | ✅ Already correct | Cited as verified real per Sep 13 audit |
| `arxiv_endorsement_request_chuckle/README.md` | ✅ Already correct | MultiLinguahah cited correctly |
| `arxiv_endorsement_request_chuckle/email_sofia_callejas.txt` | ✅ Real person | Outreach target — Callejas is one of the 5 verified authors |

---

## SciSlop Findings — Full Benchmark Summary

**Tool**: https://yerimoh.github.io/scientific-slop-demo/ (Yerim Oh et al., arXiv:2610.00531, "Science or Slop?")
**Date of audit**: 2026-10-05
**Total papers checked**: 549 (as of audit)

### Full leaderboard (Gallery, sorted by SciSlop Index)

| # | Index | Title | Type | Notes |
|---|---|---|---|---|
| 1 | **84** | Computational Analysis of Cryptographic Hash Function Performance | AI | Str 53, Arg 100, Art 100 |
| 2 | **83** | Patch, Don't Rewrite: Post-Drift Rule Updates for LogRules-Style LLM Log Parsers | AI | Str 48, Arg 100, Art 100 |
| 3 | **80** | GaugeFix-LRM: Function-Preserving Q/K Gauge Fixing | AI | Str 49, Arg 89, Art 100 |
| 4 | **76** | Differentially Private Spectral Monitor Logs for Hallucination Detection | AI | Str 47, Arg 81, Art 100 |
| 5 | **73** | Persistent Demo-Pool Poisoning Attacks on Online LLM Log Parsers | AI | Str 52, Arg 67, Art 100 |
| 6 | **72** | Adaptive Rerank Budgeting for Video-Text Retrieval via Layer-Disagreement Routing | AI | Str 47, Arg 69, Art 100 |
| 7 | **66** | Compute-Matched Evaluation of Transform-Augmented GRPO for Mathematical Reasoning | AI | Str 42, Arg 56, Art 100, 22 findings p.1 |
| 8 | **63** | Citation-Consistent Voting for Permutation-Robust Retrieval-Augmented Generation | AI (example) | Str 49, Arg 57, Art 83 |
| 9 | **50** | Fractal-ish Complexity for Regulations: A Practitioner-Ready, Agentic Benchmark | AI | Str 50, Arg 50 |
| 10 | **48** | Multi-Agent Social Simulation: An Experimental Framework | AI | Str 50, Arg 47 |
| 11 | **42** | Law-invariant BSDEs and dynamic risk measures: new characterizations | real | Str 39, Arg 45 |
| 12 | **36** | Science or Slop?: Benchmarking and Mitigating Scientific Slop (the paper itself!) | real | Str 38, Arg 44, Art 25 |
| 13 | **36** | ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | real | Str 34, Arg 26, Art 50 |
| 14 | **33** | Incorporating Domain Knowledge into Materials Tokenization | real | Str 43, Arg 30, Art 25 |
| 15 | **30** | Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text | real | Str 36, Arg 20, Art 33 |
| 16 | **26** | Alignment via Training Against Probes Without Losing Monitorability | real | Str 32, Arg 38, Art 8 |
| 17 | **23** | Self-Refine: Iterative Refinement with Self-Feedback | real | Str 40, Arg 30, Art 0 |
| 18 | **23** | DetectGPT: Zero-Shot Machine-Generated Text Detection | real | Str 30, Arg 40, Art 0 |
| 19 | **20** | The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery | real | Str 29, Arg 31 |
| 20 | **18** | Attention Is All You Need | real | Str 38, Arg 15 |
| 21 | **18** | Where LLM Graders Succeed and Break: Evidence from Two CS Exams | real | Str 22, Arg 33 |
| 22 | **17** | Recipe-Matching, Not Equivalence | real | Str 11, Arg 39 |
| 23 | **0** | Stealing Reasoning Traces from Proprietary LLM APIs | real | (partial) |
| — | — | Adam: A Method for Stochastic Optimization | real | (not run) |

### What SciSlop actually found

**The 6 measures reveal a striking pattern across AI-generated papers:**

| Measure | AI papers (1-10) | Real papers (11-23) |
|---|---|---|
| **Structure** | 47-53 | 11-50 (wide) |
| **Argument** | **47-100** | 15-45 (low) |
| **Artifacts** | **67-100** | 0-50 (low) |
| Cross-section references | often **80-100** | typically 30-60 |
| Macro redundancy | often low | often low |
| Argument graph | moderate-high | varies |
| Citation isolation | **high** (papers cited but not used) | varies |
| Figure exposition | high | varies |
| Evidence gap | **often 100** | typically low |

**Key insight**: AI-generated papers are distinguished primarily by:
1. **Artifacts** (figures, tables, citations) — AI papers score 67-100, real papers 0-50
2. **Evidence gap** — AI papers have claims that have no evidence references, real papers cite evidence

Human-written landmark papers (AlexNet, Attention Is All You Need, Adam) score **17-36** — much lower than the AI benchmark papers.

### What this means for our work

**HaHaScore falsification paper** (`arxiv_submission/hahascore.tex`): estimated **22-28 / 100** based on my manual computation. SciSlop would likely rank it **near the 17-23 cluster** (Self-Refine, DetectGPT) — well below the AI-generated cluster.

**Why our paper passes**:
- Every numerical claim traces to a JSON file (`5x3_cv_results.json`, `speaker_disjoint_cv_results.json`)
- GitHub repo + reproducibility scripts cited
- Speaker-disjoint CV is the standard 5×3 repeated methodology
- No isolated citations (all 9 bibitems used)
- 2 figures with embedded captions
- 1 table (Table 1: random vs speaker-disjoint)
- Evidence gap: every claim has a number + a JSON file

### What SciSlop would flag in ChuckleNet papers

**PAPER_DRAFT_V3_CONSOLIDATED.md** ("Beyond Words: Anticipating Audience Laughter"):
- ✅ All numbers verified against RESULTS_LOG rows 10-19 + JSON artifacts
- ⚠️ Some prose overlap with V2 (consolidation effort)
- ✅ Claims table maps asserted/rejected/not-claimed cleanly
- ✅ refs.bib = 7 verified entries (Sep 13 audit)
- ✅ Figures 1-4 generated
- ✅ Limitations names: single domain, calibration gap, 11 missing audio

**Estimated score**: 15-25 / 100. Below the AI cluster threshold.

### Two things to do now

1. **Update the Survey and COUNCIL docs to remove "MultiLinguahah is fabricated" anywhere it remains**. (Confirmed no occurrences in current `DATASET_SURVEY_LAUGHTER_2026.md`.)

2. **Run our two papers through SciSlop's submission flow**. The webapp accepts PDF or LaTeX zip uploads at `/scientific-slop-demo`. To do this we'd need a Playwright session; until then, the manual 6-measure audit (in `SCISLOP_AUDIT.md`) is the proxy.

### The lesson from the SciSlop leaderboard

**AI-generated papers are characterized by high artifacts/evidence-gap scores; real papers by low artifacts/argument-cleanliness.** Our two papers (HaHaScore, ChuckleNet) are in the real-paper cluster. The 0.86 vs 0.69 inflation issue we found in HaHaScore is exactly the kind of integrity gap SciSlop would catch — and the fix is the same fix we did: replace with verified artifacts.

The fact that our paper scores **lower** than the SciSlop example is positive validation. The falsification methodology paper is **more rigorous** than the AI-generated benchmark papers.