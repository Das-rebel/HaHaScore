#!/usr/bin/env python3
"""
v6 Feature Preparation: Align Whisper transcriptions with Bridge 4 segments
and extract RoBERTa text features per segment.

Bridge 4 segments: 20 equal-duration segments per file
Whisper segments: word/phrase-level with start/end timestamps
"""
import json, time, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pydub import AudioSegment
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
TRANSCRIPTIONS_FILE = Path("/Users/Subho/funny-strength-predictor/data/transcriptions.json")
AUDIO_DIR = Path("/Users/Subho/data/standup4ai_full")
BRIDGE4_FEATURES = Path("/Users/Subho/tmp/bridge4_features.npz")
LABELS_FILE = Path("/Users/Subho/funny-strength-predictor/data/pseudo_labels/pseudo_labels_641_v4.json")
OUT_FILE = Path("/Users/Subho/tmp/v6_features.npz")
N_SEGMENTS = 20
TEXT_MODEL = "roberta-base"
DEVICE = "cpu"
MAX_LENGTH = 128

print(f"Device: {DEVICE}")

# ── Load Transcription Data ─────────────────────────────────────────────────
print("Loading transcriptions...")
with open(TRANSCRIPTIONS_FILE) as f:
    data = json.load(f)
results = data.get("results", data)
print(f"  {len(results)} files transcribed")

# ── Load Labels (for video_id order) ──────────────────────────────────────
with open(LABELS_FILE) as f:
    labels_data = json.load(f)
vid_to_label_idx = {item["video_id"]: i for i, item in enumerate(labels_data)}
print(f"  {len(vid_to_label_idx)} video IDs in labels")

# ── Load Bridge 4 Features (cached) ─────────────────────────────────────
print("Loading Bridge 4 features...")
cached = np.load(BRIDGE4_FEATURES, allow_pickle=True)
all_wavlm = [np.asarray(f, dtype=np.float32) for f in cached["features"]]
all_labels = [np.asarray(l, dtype=np.float32) for l in cached["labels"]]
all_lengths = [int(x) for x in cached["lengths"]]
print(f"  {len(all_wavlm)} files, shape={all_wavlm[0].shape}")

# Verify order matches
labels_vids = [item["video_id"] for item in labels_data]
cached_vids = list(results.keys())
if labels_vids != cached_vids:
    print(f"WARNING: Order mismatch!")
    print(f"  Labels first 3: {labels_vids[:3]}")
    print(f"  Transcripts first 3: {cached_vids[:3]}")
    # Try to match by video ID
    print("  Will match by video_id...")
else:
    print("  ✓ Order matches labels")

# ── Load RoBERTa ──────────────────────────────────────────────────────────
print(f"Loading {TEXT_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
text_model = AutoModel.from_pretrained(TEXT_MODEL).eval().to(DEVICE)
text_dim = text_model.config.hidden_size  # 768
print(f"  Model loaded. Hidden dim: {text_dim}")

# ── Alignment Function ─────────────────────────────────────────────────────

def get_file_duration(audio_path):
    """Get audio duration using pydub."""
    try:
        audio = AudioSegment.from_file(str(audio_path), format='m4a')
        return len(audio) / 1000.0  # milliseconds to seconds
    except:
        return 0

def align_segments(transcription_segments, file_duration, n_segments=20):
    """
    Map Whisper segments to Bridge 4 equal-duration segments.
    Returns: list of n_segments texts (each text is a string, possibly empty)
    """
    segment_texts = [""] * n_segments
    
    if not transcription_segments or file_duration <= 0:
        return segment_texts
    
    seg_duration = file_duration / n_segments
    
    for seg in transcription_segments:
        start = seg.get("start", 0)
        end = seg.get("end", start)
        text = seg.get("text", "").strip()
        
        if not text:
            continue
        
        # Find which Bridge 4 segments this whisper segment overlaps with
        start_seg = min(n_segments - 1, max(0, int(start / seg_duration)))
        end_seg = min(n_segments - 1, max(0, int((end - 0.001) / seg_duration)))
        
        for i in range(start_seg, end_seg + 1):
            if segment_texts[i]:
                segment_texts[i] += " " + text
            else:
                segment_texts[i] = text
    
    return segment_texts


@torch.no_grad()
def extract_text_features_batch(texts, model, tokenizer, device, max_length=128):
    """
    Extract RoBERTa CLS embeddings for a batch of texts.
    Returns: (len(texts), 768) tensor
    """
    if not texts or all(t == "" for t in texts):
        return torch.zeros(len(texts), 768, dtype=torch.float32)
    
    # Replace empty strings with placeholder
    texts = [t if t.strip() else "[empty]" for t in texts]
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    outputs = model(**inputs)
    # Use CLS token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    return cls_embeddings


# ── Process All Files ─────────────────────────────────────────────────────
print(f"\nProcessing {len(results)} files...")

v6_text_features = []
aligned_texts_log = []
missing_audio = 0

start_time = time.time()

for vid in tqdm(labels_vids, desc="Aligning & extracting"):
    # Get transcription
    trans_data = results.get(vid, {"segments": [], "text": ""})
    whisper_segs = trans_data.get("segments", [])
    
    # Get file duration from audio
    # Try both lang-suffixed and non-suffixed versions
    audio_candidates = [
        AUDIO_DIR / f"{vid}.m4a",
        AUDIO_DIR / f"{vid.replace(',', '_', 1)}.m4a",  # e.g., "vid,fr" -> "vid_fr.m4a"
    ]
    audio_path = None
    for ap in audio_candidates:
        if ap.exists():
            audio_path = ap
            break
    
    if audio_path:
        file_duration = get_file_duration(audio_path)
    else:
        # Fallback: estimate from Whisper segment end times
        if whisper_segs:
            file_duration = max(seg.get("end", 0) for seg in whisper_segs)
        else:
            file_duration = 0
            missing_audio += 1
    
    # Align to Bridge 4 segments
    seg_texts = align_segments(whisper_segs, file_duration, N_SEGMENTS)
    
    # Extract text features (batch all 20 segments for this file)
    text_feats = extract_text_features_batch(seg_texts, text_model, tokenizer, DEVICE, MAX_LENGTH)
    
    v6_text_features.append(text_feats.numpy())
    aligned_texts_log.append(seg_texts)

elapsed = time.time() - start_time
n_files = len(v6_text_features)
print(f"\nExtracted {n_files} files × {N_SEGMENTS} segments")
print(f"Time: {elapsed/60:.1f} min ({elapsed/n_files:.1f}s per file)")
print(f"Missing audio files: {missing_audio}")

# Save text features
v6_text_features_arr = np.stack(v6_text_features)  # (N, 20, 768)
print(f"\nText features shape: {v6_text_features_arr.shape}")
print(f"  Dtype: {v6_text_features_arr.dtype}")

np.savez_compressed(
    OUT_FILE,
    text_features=v6_text_features_arr,
    video_ids=labels_vids,
    aligned_texts=aligned_texts_log
)
print(f"\nSaved to {OUT_FILE}")

# ── Verify ────────────────────────────────────────────────────────────────
print("\n=== Alignment Sample ===")
for i in range(min(3, len(aligned_texts_log))):
    vid = labels_vids[i]
    print(f"\n{vid}:")
    for j in range(min(5, N_SEGMENTS)):
        text = aligned_texts_log[i][j][:60] if aligned_texts_log[i][j] else "(no text)"
        print(f"  Seg {j:2d}: {text!r}")