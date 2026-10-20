#!/usr/bin/env python3
"""
recover_gold22.py — Recover the 22 gold-labeled videos missing features
=======================================================================
DISCOVERY (funnyb session, Oct 2026): data/gold_labels/ contains 34 CSVs,
but v6_features.npz only contains 639 videos and only 12 of the 34 match.
22 gold-labeled videos were excluded from every evaluation purely because
their features were never extracted:

    gold-set 12 -> 34 is possible WITHOUT any new human labeling.

This script closes that gap using the DOCUMENTED feature recipe:

  bridge4 audio (791d/segment, 20 segments/video):
      768d = WavLM-base-plus last_hidden_state mean-pooled per segment
       23d = prosody (PROSODY_MEAN/PROSODY_STD from
             hahascore_bridge4_model_card.md; the exact 23-dim semantics
             are NOT fully documented — hence mandatory validation below)
  v6 text (768d/segment):
      RoBERTa-base CLS embedding of the segment transcript

  ⚠ HONESTY RULES (non-negotiable, per falsification protocol):
    1. RUN --mode validate FIRST. It recomputes features for a video that
       IS in bridge4_features.npz and reports per-dim correlation against
       the stored features. Do NOT trust recovery output unless WavLM dims
       correlate (Pearson r >= 0.95 per segment-0) — if prosody dims
       (769..791) diverge, the original extract_prosody is not recoverable
       from the constants alone, and Plan B (below) applies.
    2. Pseudo-labels for recovered videos are UNKNOWN. The script writes
       0.5 (neutral) and marks pseudo_source='neutral_placeholder'.
       Neutral pseudo must NOT be used for TRAINING-set claims; recovered
       videos expand the GOLD-EVAL set. For training inclusion, recompute
       pseudo-labels with the laughter-detector pipeline first.
    3. Originals are never overwritten: outputs go to recovered_features/.

  Plan B (if validation fails on prosody dims): re-extract ALL videos
  (639 + 22) with a single consistent pipeline (WavLM 768d + your best
  23d prosody), then retrain + rerun 5x3 speaker-disjoint CV. Costlier,
  but removes the undocumented-recipe problem entirely and feeds E3.

Usage:
  # 0. fetch audio+captions for the missing IDs (yt-dlp, 16 kHz mono):
  #    yt-dlp -x --audio-format wav --postprocessor-args "-ar 16000 -ac 1" \
  #        --write-auto-subs --sub-lang en,fr,es -o "%(id)s" -- <id(s)>
  # 1. validate recipe on a known video (audio must be present):
  python3 recover_gold22.py --mode validate --audio-dir data/audio_raw \
      --video-id -1FrUOEswOk
  # 2. recover the 22:
  python3 recover_gold22.py --mode recover --audio-dir data/audio_raw \
      --vtt-dir data/vtt
  # outputs: recovered_features/recovered_bridge4.npz
  #          recovered_features/recovered_v6.npz
  #          recovered_features/RECOVERY_REPORT.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_SEG = 20
AUDIO_SR = 16000
PROSODY_MEAN = np.array([0.05, 0.02, 0.10, 0.05, 150.0, 30.0, 200.0, 80.0,
                         10.0, 3.2, 2.3, 0.0, 20.0, 0.0, 15.0, 0.0, 10.0,
                         0.0, 8.0, 0.0, 6.0, 0.0, 5.0], dtype=np.float32)
PROSODY_STD = np.array([0.03, 0.01, 0.05, 0.02, 50.0, 15.0, 80.0, 50.0,
                        5.0, 1.5, 0.2, 20.0, 10.0, 15.0, 8.0, 10.0, 5.0,
                        8.0, 4.0, 6.0, 3.0, 5.0, 2.5], dtype=np.float32)

# The 22 gold-labeled videos with NO features in v6_features.npz
MISSING_IDS = [
    "0g7nezWZyfY", "1tO9MWWOgHk", "6JQzl2LlXbQ", "G-BPYtfjpS0",
    "J9HLFJgUCW0", "KVMbGry8AgM", "MMpTINFg9dQ", "N7Z7Xc3sJuE",
    "_QdTi-N_Pgk", "cOTolCho3r8", "iiyLmS-0GFQ", "j8xbbnDIdRE",
    "mVoqH-XvlkI", "oiRyNnyG698", "pbYlQ-fZHWk", "sXn99zQfges",
    "tvn8EJwYl7Q", "xTjVsInWdjs", "xcBog4qtnv0", "xvNoe0HZnVc",
    "y-9nbIdUpbA", "yGgqulbbxgA",
]

_encoders = {}


def load_encoders():
    """Lazy-load WavLM-base-plus + RoBERTa-base (frozen, eval mode)."""
    import torch
    from transformers import WavLMModel, RobertaModel, RobertaTokenizer
    if not _encoders:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _encoders["wavlm"] = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus").eval().to(dev)
        _encoders["roberta"] = RobertaModel.from_pretrained(
            "roberta-base").eval().to(dev)
        _encoders["roberta_tok"] = RobertaTokenizer.from_pretrained(
            "roberta-base")
        _encoders["device"] = dev
    return _encoders


def extract_prosody_23(seg: np.ndarray, sr: int = AUDIO_SR) -> np.ndarray:
    """23-dim prosody vector. Semantics inferred from PROSODY_MEAN/STD in
    hahascore_bridge4_model_card.md — MUST be validated (mode validate)
    before trusting recovered features. Documented guess:
      [0:2]  rms mean/std            [2:4] short-time energy mean/std
      [4:8]  f0 mean/std/max/range   [8:10] zcr mean/std
      [10:23] spectral stats (centroid/bandwidth/rolloff/flatness means
              and deltas) — exact composition undocumented.
    All dims are (x - PROSODY_MEAN) / PROSODY_STD normalized, matching the
    model card's normalization convention."""
    import librosa

    seg = seg.astype(np.float32)
    if len(seg) < 400:  # <25 ms — return normalized zeros
        return ((np.zeros(23, np.float32) - PROSODY_MEAN) / PROSODY_STD)

    frame = max(len(seg) // 8, 400)
    rms = librosa.feature.rms(y=seg, frame_length=frame)[0]
    energy = (seg ** 2).reshape(-1)[: len(rms) * 1]  # coarse proxy
    f0 = librosa.yin(seg, fmin=65, fmax=400, sr=sr, frame_length=frame)
    zcr = librosa.feature.zero_crossing_rate(seg, frame_length=frame)[0]
    cent = librosa.feature.spectral_centroid(y=seg, sr=sr)[0]
    band = librosa.feature.spectral_bandwidth(y=seg, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=seg, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=seg)[0]

    raw = np.array([
        float(rms.mean()), float(rms.std()),
        float(energy.mean()), float(energy.std()),
        float(np.nanmean(f0)), float(np.nanstd(f0)),
        float(np.nanmax(f0)), float(np.nanmax(f0) - np.nanmin(f0)),
        float(zcr.mean()), float(zcr.std()),
        float(flat.mean()), float(cent.mean()), float(cent.std()),
        float(band.mean()), float(band.std()),
        float(roll.mean()), float(roll.std()),
        float(flat.std()), float(np.median(cent)),
        float(np.median(band)), float(np.median(roll)),
        float(np.median(flat)), float(np.median(f0)) if np.isfinite(f0).any() else 0.0,
    ], dtype=np.float32)
    return (raw - PROSODY_MEAN) / PROSODY_STD


@np.errstate(all="ignore")
def audio_791(video_id: str, audio_dir: Path) -> np.ndarray | None:
    """(20, 791) bridge4-style features for one video."""
    import librosa
    import torch

    wav_path = next((p for p in audio_dir.glob(f"{video_id}.*")
                     if p.suffix.lower() in {".wav", ".m4a", ".mp3",
                                             ".flac", ".ogg"}), None)
    if wav_path is None:
        return None
    enc = load_encoders()
    samples, sr = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
    seg_len = len(samples) // N_SEG
    out = np.zeros((N_SEG, 791), dtype=np.float32)
    with torch.no_grad():
        for i in range(N_SEG):
            s, e = i * seg_len, (i + 1) * seg_len if i < N_SEG - 1 else len(samples)
            seg = samples[s:e]
            if len(seg) < 400:
                continue
            wav = torch.tensor(seg).unsqueeze(0).to(enc["device"])
            hs = enc["wavlm"](wav).last_hidden_state  # (1, T, 768)
            emb = hs.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)
            out[i, :768] = emb
            out[i, 768:] = extract_prosody_23(seg)
    return out


def text_768(video_id: str, vtt_dir: Path) -> np.ndarray | None:
    """(20, 768) v6-style RoBERTa-base CLS features from a VTT transcript
    split into 20 equal-duration chunks (v6 'aligned_texts' convention:
    transcript cut into N_SEG parts; segments without text -> zeros)."""
    import torch

    vtt = next((p for p in vtt_dir.glob(f"{video_id}*.vtt")), None)
    if vtt is None:
        return None
    enc = load_encoders()
    cues = parse_vtt(vtt.read_text(encoding="utf-8", errors="replace"))
    if not cues:
        return None
    # dedup (YouTube auto-captions duplicate every cue — v20.2 lesson)
    seen, uniq = set(), []
    for t0, t1, txt in cues:
        key = (round(t0, 2), round(t1, 2))
        if key not in seen:
            seen.add(key)
            uniq.append((t0, t1, txt))
    dur = uniq[-1][1]
    chunks = [""] * N_SEG
    for t0, t1, txt in uniq:
        idx = min(N_SEG - 1, int(t0 / max(dur, 1e-6) * N_SEG))
        chunks[idx] += " " + txt
    out = np.zeros((N_SEG, 768), dtype=np.float32)
    with torch.no_grad():
        for i, txt in enumerate(chunks):
            txt = txt.strip()
            if not txt:
                continue
            toks = enc["roberta_tok"](txt, return_tensors="pt",
                                      truncation=True, max_length=512).to(enc["device"])
            cls = enc["roberta"](**toks).last_hidden_state[:, 0, :]
            out[i] = cls.squeeze(0).cpu().numpy()
    return out


def parse_vtt(text: str):
    """Minimal VTT parser -> [(t0, t1, text)] in seconds."""
    import re
    ts = re.compile(r"(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*(\d+):(\d+):(\d+)\.(\d+)")
    cues = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = ts.search(ln)
        if m:
            g = [int(x) for x in m.groups()]
            t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            body = []
            for j in range(i + 1, min(i + 6, len(lines))):
                if not lines[j].strip() or "-->" in lines[j]:
                    break
                body.append(lines[j].strip())
            txt = " ".join(body).strip()
            if txt:
                cues.append((t0, t1, txt))
    return cues


def mode_validate(args):
    """Recompute features for a KNOWN video; compare with stored bridge4."""
    b4 = np.load(args.bridge4, allow_pickle=True)
    vids = [str(v).split(",")[0] for v in b4["features"] * 0 +
            np.load(args.bridge4, allow_pickle=True)["labels"].astype(object) * 0]
    # stored video ids live in v6 npz; bridge4 shares ordering with v6
    v6 = np.load(args.v6, allow_pickle=True)
    vids = [str(v).split(",")[0] for v in v6["video_ids"]]
    if args.video_id not in vids:
        sys.exit(f"{args.video_id} not among stored video ids")
    pos = vids.index(args.video_id)
    stored = np.asarray(b4["features"][pos], dtype=np.float32)  # (20, 791)

    recomputed = audio_791(args.video_id, Path(args.audio_dir))
    if recomputed is None:
        sys.exit(f"no audio for {args.video_id} in {args.audio_dir} — "
                 f"download it first (see docstring step 0)")

    # per-block Pearson r between stored and recomputed (segment 0 and mean)
    def block_r(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        a, b = a - a.mean(), b - b.mean()
        d = (np.linalg.norm(a) * np.linalg.norm(b))
        return float((a * b).sum() / d) if d > 0 else 0.0

    r_wavlm = [block_r(stored[i, :768], recomputed[i, :768]) for i in range(N_SEG)]
    r_pros = [block_r(stored[i, 768:], recomputed[i, 768:]) for i in range(N_SEG)]
    report = {
        "video_id": args.video_id,
        "wavlm_block_r_mean": float(np.mean(r_wavlm)),
        "wavlm_block_r_min": float(np.min(r_wavlm)),
        "prosody_block_r_mean": float(np.mean(r_pros)),
        "prosody_block_r_min": float(np.min(r_pros)),
        "verdict_wavlm": "PASS" if np.mean(r_wavlm) >= 0.95 else "FAIL",
        "verdict_prosody": "PASS" if np.mean(r_pros) >= 0.90 else "FAIL",
    }
    print(json.dumps(report, indent=2))
    if report["verdict_wavlm"] == "FAIL":
        print("\n❌ WavLM block fails — check audio resampling/segmentation "
              "before any recovery run.")
    elif report["verdict_prosody"] == "FAIL":
        print("\n⚠ WavLM matches but prosody diverges — original "
              "extract_prosody is not recoverable from the constants alone. "
              "Plan B: full re-extraction of all 661 videos with one "
              "consistent pipeline, then retrain + rerun 5x3 CV.")
    else:
        print("\n✅ Recipe validated — safe to run --mode recover.")
    Path("recovered_features").mkdir(exist_ok=True)
    Path("recovered_features/VALIDATION.json").write_text(json.dumps(report, indent=2))


def mode_recover(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = out_dir / "VALIDATION.json"
    if not val.exists():
        sys.exit("Refusing to recover without a prior --mode validate run "
                 "(honesty rule #1).")
    vres = json.loads(val.read_text())
    if vres["verdict_wavlm"] == "FAIL":
        sys.exit("Validation FAILED — recovery blocked. See "
                 "recovered_features/VALIDATION.json.")

    feats, texts, ids, lengths = [], [], [], []
    skipped = []
    for vid in MISSING_IDS:
        a = audio_791(vid, Path(args.audio_dir))
        t = text_768(vid, Path(args.vtt_dir))
        if a is None:
            skipped.append({"id": vid, "reason": "no audio"})
            print(f"skip {vid}: no audio"); continue
        if t is None:
            t = np.zeros((N_SEG, 768), dtype=np.float32)
            print(f"warn {vid}: no VTT -> zero text features (audio-only)")
        feats.append(a); texts.append(t); ids.append(vid); lengths.append(N_SEG)
        print(f"ok  {vid}: audio (20,791), text (20,768)")

    np.savez_compressed(
        out_dir / "recovered_bridge4.npz",
        features=np.array(feats, dtype=object),
        labels=np.full((len(feats), N_SEG), 0.5, dtype=np.float32),  # neutral
        lengths=np.array(lengths),
        pseudo_source=np.array(["neutral_placeholder"] * len(feats)),
    )
    np.savez_compressed(
        out_dir / "recovered_v6.npz",
        text_features=np.array(texts, dtype=object),
        video_ids=np.array(ids),
        aligned_texts=np.array([None] * len(ids), dtype=object),
    )
    report = {
        "recovered": ids,
        "skipped": skipped,
        "n_gold_after_merge": 12 + len(ids),
        "validation": vres,
        "honesty_notes": [
            "pseudo labels are neutral 0.5 placeholders — NOT for training-set claims",
            "originals untouched; merge must be explicit",
            "gold CSVs unchanged; eval-set expansion only",
        ],
    }
    (out_dir / "RECOVERY_REPORT.json").write_text(json.dumps(report, indent=2))
    print(f"\nRecovered {len(ids)}/22 -> {out_dir}/recovered_*.npz")
    print(f"Gold eval set: 12 -> {12 + len(ids)} (pending merge + re-run of "
          f"speaker_disjoint_cv.py with recovered npz)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "recover"], required=True)
    ap.add_argument("--audio-dir", default="data/audio_raw")
    ap.add_argument("--vtt-dir", default="data/vtt")
    ap.add_argument("--bridge4", default="/Users/Subho/tmp/bridge4_features.npz")
    ap.add_argument("--v6", default="/Users/Subho/tmp/v6_features.npz")
    ap.add_argument("--video-id", default="-1FrUOEswOk")
    ap.add_argument("--out", default="recovered_features")
    args = ap.parse_args()
    if args.mode == "validate":
        mode_validate(args)
    else:
        mode_recover(args)


if __name__ == "__main__":
    main()
