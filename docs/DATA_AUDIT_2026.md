# Complete Data Audit — GDrive + Kaggle + Local (2026)

## Verdict
No Kaggle dataset exists for humor strength. **Google Drive is authoritative.**
The "lost" training data is FOUND. Fusion bridge now exists via `aligned_segments.jsonl`.

---

## 1. Humor Strength 0–100 (text-only) — v1 [IN USE]
| Source | Count | Location |
|--------|-------|----------|
| Jester | 140 | Kaggle `chucklenet-predictor-v1-data` |
| rjokes | 4,844 | " |
| dadjokes | 3,000 | " |
| **Total** | **7,984** | `~/funny-strength-predictor/data/` |

## 2. Laughter-Labeled Audio — v2 fusion [NEWLY CONFIRMED]

### Primary: `gdrive:laughter_prediction/aligned/aligned_segments.jsonl` (141MB)
- **481,333 word-level samples**, 52 videos (+10 in batch1/batch2)
- **85,725 positive** (17.8% laughter rate)
- Fields: `video_id, comedian, word, start, end, duration, label(0/1), context_words, context_labels`
- Comedians: john_mulaney, ali_wong, dave_chappelle, downloaded, batch1, batch2
- **48/52 videos have audio on GDrive** (m4a/mp3/wav)

### Audio pool: **654 unique videos**
| Folder | Count | Format |
|--------|-------|--------|
| `gdrive:chuckle_net_1000/audio/` | 621 | m4a |
| `gdrive:chuckle_audio_all/{audio,audio_new,audio_all,audio_final}` | 645 | mp3/wav |
| `gdrive:laughter_prediction/audio/{en,hi-latn,zh}` | 49 | mp3 |
| `gdrive:multimodal_audio/hindi` | 4 | mp3 |
| `gdrive:chuckle_audio` | 71 | mp3 |
| Kaggle `chuckle-audio-620-videos` | 620 | m4a (mirror of cn1000) |

### StandUp4AI (EMNLP 2025) — multilingual gold
- `gdrive:standup4ai/seq-Standup4AI/dataset/`: **7,599 CSVs, 11 languages**
  (cs, en_uk, en_us, es, es_ch, es_latam, fr, fr_ca, hu, it, multilingual)
- Word-level **BIO-LU** labels: O=outside, B=begin, I=inside, L=last, U=unit laughter
- `wordlevel_wavlm_features/`: **103 videos pre-extracted WavLM features** (.npy per video)
- Checkpoints: `wordlevel_221_model.pt`, `top200_prosody_model.pt`, `experiments/best_fusion_model.pt`
- Kaggle mirror: `subhajitdas/standup4ai-en-uk-labels`

### VTT transcripts
- 1,094 unique (Kaggle `chuckle-vtt-labels` 627 + `chuckle_net_1000/vtt/` 627 same + lp 50)
- **185 VTTs contain timestamped `[Laughter]` markers → 2,972 laugh events**
- Laugh duration = continuous humor-strength proxy for those events

### Whisper transcripts (raw, no laugh labels)
- `gdrive:laughter_prediction/transcripts/`: 64 JSONs (segments+timestamps)
- incl. Shane Gillis Austin, Ali Wong, Dave Chappelle, John Mulaney

### Precomputed training sets (binary, audio+text)
| Asset | Size | Location |
|-------|------|----------|
| july16 WavLM expanded | 21,468 samples (WavLM768+prosody23) | local npz + Kaggle `chucklenet-wavlm-training-data` |
| prosody_fusion | 10,207/1,993/2,799 train/val/test | local `training/prosody_fusion_*` |
| scale221 word features | 40 videos (N,791) | local + GDrive |
| wavlm_555 | npz | Kaggle `chuckle-wavlm-555-videos` |
| top200 | npz | Kaggle `top200-youtube-comedy-prosody` |

## 3. Kaggle datasets (mine, 20 total)
Text/labels: `chucklenet-predictor-v1-data`, `chuckle-vtt-labels`, `standup4ai-en-uk-labels`, `standup4ai-eval`, `chuckle-vtt-frames*`, `chucklenet-batch3-candidates`
Audio: `chuckle-audio-620-videos`, `chuckle-audio-001`, `chucklenet-vtt-audio-tar`
Features: `chuckle-wavlm-555-videos`, `chucklenet-wavlm-training-data`, `chucklenet-scale221`, `scale221`, `chucklenet-221-wordlevel`, `chucklenet-48v-wordlevel`, `gillick272-prosody`, `chuckle-net-prosody-fusion`, `chuckle-net-phase-a-prosody`, `top200-youtube-comedy-prosody`

## 4. The Fusion Bridge (NEW)
**48 videos = word-level laughter labels (aligned) + actual audio (GDrive)**

v2 pipeline now unblocked:
1. Download 48 videos' audio from GDrive (~2GB)
2. Extract WavLM features per aligned word-window
3. Multitask: text encoder (RoBERTa v7) + audio (WavLM) → fused laughter prediction
4. Strength proxy: laugh-event duration from 2,972 VTT events → continuous 0–100 (calibrate against Jester gold)

## 5. Still does NOT exist
- (joke_text, audio, human-rated 0–100) triplets — nowhere
- True fix = human ratings OR laughter-duration proxy (scientifically defensible: laugh length ≈ funniness)
