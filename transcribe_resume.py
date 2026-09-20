#!/usr/bin/env python3
"""
Resume transcription: skip already-transcribed files.
"""
import json, time, os, sys
from pathlib import Path
import whisper

AUDIO_DIR = Path("/Users/Subho/data/standup4ai_full")
OUT_FILE = Path("/Users/Subho/funny-strength-predictor/data/transcriptions.json")
MODEL_NAME = "base"

print(f"Loading Whisper {MODEL_NAME} model...")
model = whisper.load_model(MODEL_NAME, device="cpu")
print("Model loaded.")

# Load existing transcriptions
with open(OUT_FILE) as f:
    data = json.load(f)
results = data.get("results", data)
done_vids = set(results.keys())
print(f"Already transcribed: {len(done_vids)} files")

# Find all m4a files
all_files = sorted([f for f in AUDIO_DIR.glob("*.m4a")])
remaining = [f for f in all_files if f.stem not in done_vids]
print(f"Remaining: {len(remaining)} files")
print(f"First few remaining: {[f.stem for f in remaining[:5]]}")

errors = data.get("errors", [])
start_time = time.time()
last_checkpoint = len(results)

for i, audio_path in enumerate(remaining):
    vid = audio_path.stem
    try:
        result = model.transcribe(
            str(audio_path),
            language=None,  # auto-detect
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

        results[vid] = {
            "text": text,
            "segments": segments,
            "duration": duration,
            "language": None
        }

    except Exception as e:
        errors.append({"vid": vid, "error": str(e)})
        results[vid] = {"text": "", "segments": [], "duration": 0, "language": None, "error": str(e)}

    # Checkpoint every 10 files
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start_time
        per_file = elapsed / (i + 1)
        eta_s = per_file * (len(remaining) - i - 1)
        print(f"\n  [{i+1}/{len(remaining)}] {vid[:20]}... ETA {eta_s/60:.1f}min, {len(errors)} errors")
        with open(OUT_FILE, "w") as f:
            json.dump({"results": results, "errors": errors}, f)

    # Also checkpoint every file in case of crash
    if (i + 1) > last_checkpoint and (i + 1) % 1 == 0:
        with open(OUT_FILE, "w") as f:
            json.dump({"results": results, "errors": errors}, f)

elapsed = time.time() - start_time
print(f"\nDone! {len(remaining)} files in {elapsed/60:.1f} min")

# Final save
with open(OUT_FILE, "w") as f:
    json.dump({"results": results, "errors": errors}, f)
print(f"Saved {len(results)} files to {OUT_FILE}")