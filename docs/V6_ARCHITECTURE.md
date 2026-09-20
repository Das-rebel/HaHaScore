# HaHaScore v6 Architecture

## Goal
Improve Bridge 4's AUC 0.842 by adding text features via Whisper transcription + cross-attention fusion.

## Data Pipeline

### Transcription (in progress)
- Whisper-base, auto-detect language, CPU
- Output: JSON {video_id → {text, segments: [{start, end, text}]}}
- Progress: 37/639 files done (~55s/file, ETA ~1:40 AM)
- Checkpoint: every 50 files

### Text Feature Extraction
- Use Whisper word-level timestamps to align text with Bridge 4 segments
- For each of 20 segments per file: extract text words within segment time bounds
- Pool word embeddings (RoBERTa-base or DistilBERT) → 768d text vector per segment
- If segment has no words: use zero vector

### Audio Features (already cached)
- Bridge 4 features: 768d WavLM + 23d prosody per segment = 791d
- Cached at: ~/tmp/bridge4_features.npz
- Shape: (639, 20, 791)

## v6 Model: TriModal BiGRU Fusion

```
Text features (768d) ──→ concat with audio (791d) ──→ BiGRU(128d×2) ──→ MLP ──→ score
                         ↓
Audio features (791d) ──┘
```

### Architecture Details
- Input per segment: 768d text + 791d audio = 1559d
- Position embedding: 4d (same as Bridge 4)
- BiGRU: hidden=128, num_layers=2, bidirectional, dropout=0.3
- Head: Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 1) → Sigmoid
- Trainable params: ~1.2M (vs Bridge 4's ~790K)

### Training
- Same 5-fold CV protocol as Bridge 4
- Same PROSODY_MEAN/PROSODY_STD normalization
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealing, T_max=30
- Loss: BCEWithLogitsLoss
- Batch size: 32

## Why This Should Help

### Bridge 4 Limitation
Bridge 4 has NO text signal. It can learn "this acoustic pattern sounds like comedy" but cannot learn "this SETUP has a PUNCHLINE structure." Text provides the semantic content that distinguishes:
- Setup: "My therapist told me I have trouble with boundaries..."
- Punchline: "...so I installed some!"

### Expected Improvement
- v5 (bilinear, text + audio, per-segment): AUC 0.632
- Bridge 4 (audio-only, BiGRU): AUC 0.842
- v6 (text + audio, BiGRU): Expected AUC 0.85-0.88

## Alternative: Cascade Gate

If text+audio concatenation doesn't help, try cascade:

```
text_confidence = sigmoid(text_proj → Linear(1))
gated_audio = text_confidence * audio_features
BiGRU(gated_audio + text_features)
```

If text is confident → trust audio. If text is uncertain → rely more on prosody.

## Alignment Strategy

Bridge 4 segments: 20 equal-duration segments per file
Whisper segments: word-level with start/end timestamps

For segment i (i from 0 to 19):
1. Compute segment time bounds: [i * dur/20, (i+1) * dur/20]
2. Find all Whisper words within this time range
3. Concatenate word texts → segment text
4. Encode with RoBERTa → 768d text feature

If no words in segment: text_feature = zeros(768d)
If segment is mostly silence: text_feature = zeros(768d)

## Risk: Text Might Not Help

Given that text-only achieves AUC 0.50, text features might add noise rather than signal.
Mitigation:
1. Start with frozen RoBERTa (don't fine-tune text)
2. Use high dropout on text branch (0.5)
3. Monitor text branch contribution via ablation

## Next Steps
1. Wait for transcription (ETA ~1:40 AM)
2. Align Whisper segments with Bridge 4 segments
3. Extract RoBERTa features per segment
4. Train v6 with same 5-fold CV protocol
5. Compare: text+audio vs audio-only BiGRU
