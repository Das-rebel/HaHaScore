#!/usr/bin/env python3
"""
Transcribe all 639 StandUp4AI audio files using Whisper tiny.
Output: JSON with video_id → {text, segments: [{start, end, text}]}
"""
import json, time, os, sys
from pathlib import Path
from tqdm import tqdm
import whisper

AUDIO_DIR = Path("/Users/Subho/data/standup4ai_full")
OUT_FILE = Path("/Users/Subho/funny-strength-predictor/data/transcriptions.json")
MODEL_NAME = "base"
LANGUAGE = None  # Auto-detect per file
BATCH_SIZE = 32  # Process in small batches for memory

print(f"Loading Whisper {MODEL_NAME} model...")
model = whisper.load_model(MODEL_NAME, device="cpu")
print("Model loaded.")

# Find all m4a files
audio_files = sorted([f for f in AUDIO_DIR.glob("*.m4a")])
print(f"Found {len(audio_files)} audio files")

results = {}
errors = []
total_duration = 0

start_time = time.time()

for audio_path in tqdm(audio_files, desc="Transcribing"):
    vid = audio_path.stem  # "id,lang"
    try:
        result = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            task="transcribe",
            fp16=False,
            verbose=False
        )
        text = result["text"].strip()
        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result.get("segments", [])
        ]
        duration = result.get("duration", 0)
        total_duration += duration

        results[vid] = {
            "text": text,
            "segments": segments,
            "duration": duration,
            "language": LANGUAGE
        }

    except Exception as e:
        errors.append({"vid": vid, "error": str(e)})
        results[vid] = {"text": "", "segments": [], "duration": 0, "language": LANGUAGE, "error": str(e)}

    # Checkpoint every 50 files
    if len(results) % 50 == 0:
        with open(OUT_FILE, "w") as f:
            json.dump({"results": results, "errors": errors, "checkpoint": len(results)}, f)
        elapsed = time.time() - start_time
        files_done = len(results)
        per_file = elapsed / files_done
        eta_min = per_file * (len(audio_files) - files_done) / 60
        print(f"\n  Checkpoint: {files_done}/{len(audio_files)}, ETA {eta_min:.1f}min, {len(errors)} errors")

elapsed = time.time() - start_time
print(f"\nTranscription complete in {elapsed/60:.1f} min")
print(f"Total duration: {total_duration/60:.1f} min audio")
print(f"Average: {elapsed/len(results):.1f}s per file")
print(f"Errors: {len(errors)}")

# Save final
with open(OUT_FILE, "w") as f:
    json.dump({"results": results, "errors": errors}, f, indent=2)
print(f"Saved to {OUT_FILE}")