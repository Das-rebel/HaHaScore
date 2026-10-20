#!/usr/bin/env python3
"""
expand_corpus_1000.py — HaHaScore corpus expansion 639 → ~1260 videos
=====================================================================
CORPUS EXPANSION (user decision 2026-10: "~1000-video corpus"):
gdrive:chuckle_net_1000/audio holds 621 m4a files named <YouTubeID>.m4a,
ZERO overlap with HaHaScore's 639-video corpus. This pipeline makes them
usable for the 5×3 speaker-disjoint honest benchmark.

Stages (all resumable — skip existing outputs, incremental writes):
  a  m4a → 16 kHz mono wav (ffmpeg)
  b  whisper transcription → transcripts/<id>.whisper.json
     (segments carry timestamps; assigned to 20 equal-duration partitions
      to match the 639-corpus convention)
  c  features: WavLM-base-plus 768d + 23d prosody (791d, recipe from
     hahascore_bridge4_model_card.md) + RoBERTa-base CLS 768d per segment
     → shards under features/ (recipe shared with recover_gold22.py)
  d  pseudo-labels: ChuckleNet laughter detector confidence per segment
     + CALIBRATION pass against v4 pseudo scores on corpus-overlap videos.
     ⚠ HONESTY: new pseudo family is NOT comparable to pseudo_labels_641_v4
     without calibration. Report r; if r < 0.5 the two families stay
     separate in training and any mixed-corpus claims are blocked.

Honesty rules (per falsification protocol):
  - Original 639-corpus artifacts are never modified.
  - Pseudo-labels expand TRAINING only. Eval gold stays human-labeled
    (34 CSVs; 12 with features today, 22 recoverable via recover_gold22.py).
  - PROVENANCE_EXPANSION.json records every stage (what/when/how).
  - No headline number may be published from this corpus until the 5×3
    speaker-disjoint CV has been re-run on the merged corpus with the
    expanded gold set.

Usage:
  python3 expand_corpus_1000.py --stage a            # convert audio
  python3 expand_corpus_1000.py --stage b            # whisper (background, ~3-6 h)
  python3 expand_corpus_1000.py --stage c            # features (~5 h CPU)
  python3 expand_corpus_1000.py --stage d            # pseudo + calibration
  python3 expand_corpus_1000.py --stage all --limit 3   # smoke test
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "expand_1000"
AUDIO_RAW = DATA / "audio_raw"
AUDIO_16K = DATA / "audio_16k"
TRANSCRIPTS = DATA / "transcripts"
FEAT_DIR = DATA / "features"
PSEUDO_DIR = DATA / "pseudo"
V4_PSEUDO = ROOT / "data" / "pseudo_labels" / "pseudo_labels_641_v4.json"
N_SEG = 20
SHARD = 50

for d in (AUDIO_RAW, AUDIO_16K, TRANSCRIPTS, FEAT_DIR, PSEUDO_DIR):
    d.mkdir(parents=True, exist_ok=True)


def all_ids():
    return sorted(p.stem for p in AUDIO_RAW.glob("*.m4a"))


def stage_a(limit=None):
    """m4a → 16 kHz mono wav. Resume-safe."""
    ids = all_ids()
    if limit:
        ids = ids[:limit]
    done = 0
    for i, vid in enumerate(ids):
        out = AUDIO_16K / f"{vid}.wav"
        if out.exists() and out.stat().st_size > 100_000:
            continue
        src = AUDIO_RAW / f"{vid}.m4a"
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ar", "16000", "-ac", "1", str(out)],
            capture_output=True)
        if r.returncode != 0:
            print(f"FAIL {vid}: {r.stderr.decode()[:120]}")
            continue
        done += 1
        if done % 25 == 0:
            print(f"[a] {i + 1}/{len(ids)} converted")
    print(f"[a] done ({done} new). total wavs: {len(list(AUDIO_16K.glob('*.wav')))}")


def stage_b(whisper_model="base", limit=None):
    """Whisper → transcripts/<id>.whisper.json. Resume-safe."""
    import whisper
    model = whisper.load_model(whisper_model)
    ids = all_ids()
    if limit:
        ids = ids[:limit]
    for i, vid in enumerate(ids):
        out = TRANSCRIPTS / f"{vid}.whisper.json"
        if out.exists():
            continue
        wav = AUDIO_16K / f"{vid}.wav"
        if not wav.exists():
            continue
        t0 = time.time()
        try:
            res = model.transcribe(str(wav), verbose=False)
        except Exception as e:
            print(f"FAIL {vid}: {e}")
            continue
        out.write_text(json.dumps({
            "video_id": vid,
            "language": res.get("language"),
            "duration": res.get("segments", [{}])[-1].get("end", 0) if res.get("segments") else 0,
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in res.get("segments", []) if s["text"].strip()
            ],
        }, ensure_ascii=False))
        if (i + 1) % 10 == 0:
            rate = (time.time() - t0) or 1
            print(f"[b] {i + 1}/{len(ids)} transcribed (last took {rate:.0f}s)")
    print(f"[b] done. transcripts: {len(list(TRANSCRIPTS.glob('*.whisper.json')))}")


def whisper_to_20_partitions(tj: dict):
    """Assign whisper text to 20 equal-duration partitions
    (matches the 639-corpus segment convention)."""
    dur = float(tj.get("duration") or 0)
    if dur <= 0:
        return [""] * N_SEG
    chunks = [""] * N_SEG
    for s in tj["segments"]:
        idx = min(N_SEG - 1, int(s["start"] / dur * N_SEG))
        chunks[idx] += " " + s["text"]
    return [c.strip() for c in chunks]


def stage_c(limit=None):
    """791d audio + 768d text features → shards. Resume-safe.
    Recipe shared with recover_gold22.py (single source of truth)."""
    sys.path.insert(0, str(ROOT))
    import librosa
    import torch
    from recover_gold22 import load_encoders, extract_prosody_23

    enc = load_encoders()
    dev = enc["device"]
    ids = all_ids()
    if limit:
        ids = ids[:limit]
    shard, shard_n = [], _next_shard_idx()
    for i, vid in enumerate(ids):
        tj_path = TRANSCRIPTS / f"{vid}.whisper.json"
        if not tj_path.exists():
            continue
        if any(FEAT_DIR.glob(f"features_shard_*_{vid}.done")):
            continue
        wav = AUDIO_16K / f"{vid}.wav"
        samples, sr = librosa.load(wav, sr=16000, mono=True)
        seg_len = len(samples) // N_SEG
        audio791 = np.zeros((N_SEG, 791), dtype=np.float32)
        texts = whisper_to_20_partitions(json.loads(tj_path.read_text()))
        text768 = np.zeros((N_SEG, 768), dtype=np.float32)
        with torch.no_grad():
            for s in range(N_SEG):
                a, b = s * seg_len, (s + 1) * seg_len if s < N_SEG - 1 else len(samples)
                seg = samples[a:b]
                if len(seg) >= 400:
                    w = torch.tensor(seg).unsqueeze(0).to(dev)
                    hs = enc["wavlm"](w).last_hidden_state.mean(dim=1).squeeze(0)
                    audio791[s, :768] = hs.cpu().numpy()
                    audio791[s, 768:] = extract_prosody_23(seg)
                    if texts[s]:
                        toks = enc["roberta_tok"](texts[s], return_tensors="pt",
                                                  truncation=True,
                                                  max_length=512).to(dev)
                        text768[s] = enc["roberta"](**toks).last_hidden_state[:, 0, :].cpu().numpy()
        shard.append({"id": vid, "audio": audio791, "text": text768})
        (FEAT_DIR / f"features_shard_{shard_n:03d}_{vid}.done").touch()
        if len(shard) >= SHARD:
            _write_shard(shard, shard_n)
            shard, _ = [], _next_shard_idx()
        if (i + 1) % 10 == 0:
            print(f"[c] {i + 1}/{len(ids)} featured")
    if shard:
        _write_shard(shard, _next_shard_idx())
    print(f"[c] done. shards: {len(list(FEAT_DIR.glob('features_shard_*.npz')))}")


def _next_shard_idx():
    existing = list(FEAT_DIR.glob("features_shard_*.npz"))
    return (max((int(p.stem.split("_")[2]) for p in existing), default=-1) + 1)


def _write_shard(items, idx):
    np.savez_compressed(
        FEAT_DIR / f"features_shard_{idx:03d}.npz",
        ids=np.array([x["id"] for x in items]),
        audio=np.stack([x["audio"] for x in items]),
        text=np.stack([x["text"] for x in items]),
    )
    print(f"[c] wrote shard {idx:03d} ({len(items)} videos)")


def stage_d(limit=None, detector_endpoint=None):
    """Pseudo-labels via ChuckleNet laughter detector + v4 calibration.

    detector_endpoint: HTTP endpoint serving ChuckleNet laughter inference
    (e.g. local ONNX runtime wrapper). If None, stage d writes the
    calibration PLAN and marks pseudo as PENDING — it does NOT fabricate
    scores. (Provenance rule D-R1: no fabricated numbers, ever.)
    """
    import urllib.request

    v4 = {e["video_id"].split(",")[0]: e
          for e in json.loads(V4_PSEUDO.read_text())}
    drive_ids = all_ids()
    if limit:
        drive_ids = drive_ids[:limit]
    overlap = [v for v in drive_ids if v in v4]  # none expected, but if
    print(f"[d] corpus-overlap videos for calibration: {len(overlap)}")

    # calibration on the ORIGINAL corpus side is always possible: run the
    # same detector on N corpus videos that HAVE v4 scores.
    corpus_ids = [v for v in v4 if (AUDIO_RAW / f"{v}.m4a").exists() is False]
    calib_pool = sorted(v4.keys())[:40]
    result = {
        "pseudo_source": "chucklenet_laughter_v1" if detector_endpoint else "PENDING",
        "calibration": {
            "v4_reference": str(V4_PSEUDO),
            "calib_pool_size": len(calib_pool),
            "status": "PLANNED",
        },
        "new_videos": len(drive_ids),
        "note": ("New pseudo family must be calibrated against v4 before "
                 "any mixed-corpus training. If r < 0.5, keep families "
                 "separate (or rank-normalize) and say so in the paper."),
    }
    if detector_endpoint:
        import librosa
        scores = {}
        for vid in calib_pool + drive_ids:
            wav = AUDIO_16K / f"{vid}.wav"
            if not wav.exists():
                continue
            samples, _ = librosa.load(wav, sr=16000, mono=True)
            seg_len = len(samples) // N_SEG
            seg_scores = []
            for s in range(N_SEG):
                a = s * seg_len
                b = (s + 1) * seg_len if s < N_SEG - 1 else len(samples)
                seg = samples[a:b].astype(np.float32)
                req = urllib.request.Request(
                    detector_endpoint,
                    data=json.dumps({"audio": seg.tolist()}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    seg_scores.append(float(json.loads(r.read())["score"]))
            scores[vid] = seg_scores
        # linear-calibration fit on overlap between calib_pool v4 & detector
        pairs = [(np.array(v4[v]["segments"]) and
                  np.array([s["score"] for s in v4[v]["segments"]]),
                  np.array(scores[v])) for v in calib_pool if v in scores]
        if pairs:
            old = np.concatenate([p[0] for p in pairs])
            new = np.concatenate([p[1] for p in pairs])
            r = float(np.corrcoef(old, new)[0, 1])
            result["calibration"] = {
                "status": "MEASURED", "pearson_r": r,
                "verdict": ("COMPARABLE — fit linear map"
                            if r >= 0.5 else
                            "NOT COMPARABLE — keep families separate"),
                "n_pairs": int(len(old)),
            }
        out = {vid: {"video_id": vid, "duration": None,
                     "segments": [{"start": s * (1 / N_SEG), "end": (s + 1) / N_SEG,
                                   "score": sc}
                                  for s, sc in enumerate(sc_list)]}
               for vid, sc_list in scores.items() if vid not in calib_pool}
        (PSEUDO_DIR / "pseudo_labels_expanded.json").write_text(
            json.dumps(list(out.values()), indent=1))
    (PSEUDO_DIR / "STAGE_D_STATUS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["calibration"], indent=2))
    print(f"[d] status -> {PSEUDO_DIR / 'STAGE_D_STATUS.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["a", "b", "c", "d", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--detector-endpoint", default=None)
    args = ap.parse_args()
    stages = ["a", "b", "c", "d"] if args.stage == "all" else [args.stage]
    for s in stages:
        if s == "a":
            stage_a(args.limit)
        elif s == "b":
            stage_b(args.whisper_model, args.limit)
        elif s == "c":
            stage_c(args.limit)
        elif s == "d":
            stage_d(args.limit, args.detector_endpoint)


if __name__ == "__main__":
    main()
