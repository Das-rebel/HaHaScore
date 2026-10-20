#!/usr/bin/env python3
"""
e3_encoder_benchmark.py — Low-Cost Deep-Encoder Ablation (E3 arm)
=================================================================
Per RESEARCH_CONTINUATION.md (funnyb session) + council decision:
  "best ML architecture at lowest cost"

Goal: beat the honest multimodal floor (v5 bilinear fusion, 5-fold CV
AUC 0.632±0.007) and approach the honest 5×3 speaker-disjoint ceiling
(v10 cascade-gate, humor AUC 0.6924±0.0359) using a CHEAPER audio
encoder than WavLM-base-plus.

Candidates (frozen encoder + logistic-regression head — cheapest
honest protocol, matches the audio-only 0.538 baseline):
  1. DistilHuBERT   (~29M,  3x faster than HuBERT)
  2. Wav2Vec2-Base  (~95M,  strong SSL baseline)
  3. HuBERT-Base    (~95M,  masked-prediction SSL)
  4. WavLM-Base+    (~94M,  current v5 audio encoder — reference arm)

All evaluated with laugho_cv.repeated_speaker_disjoint_cv (5×3,
bootstrap CI) — the paper's own minimum reporting standard.

Usage (local CPU or Kaggle T4):
    python3 e3_encoder_benchmark.py --audio-dir data/audio \
        --labels data/gold_labels.csv --encoders distilhubert,wavlm-base-plus
    # results -> e3_encoder_results.json

Kaggle notebook cell:
    !python e3_encoder_benchmark.py --encoders all --out /kaggle/working/e3_encoder_results.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from laugho_cv import repeated_speaker_disjoint_cv

# ---------------------------------------------------------------- #
# Encoder registry — frozen SSL encoders, mean-pooled to 1 vector   #
# per utterance, LR head. params = backbone size (head ~1K params). #
# ---------------------------------------------------------------- #
ENCODERS = {
    "distilhubert": {
        "hf_id": "ntu-spml/distilhubert",
        "params_m": 29,
        "sample_rate": 16000,
        "layer": -2,          # penultimate transformer layer (best probing)
    },
    "wav2vec2-base": {
        "hf_id": "facebook/wav2vec2-base",
        "params_m": 95,
        "sample_rate": 16000,
        "layer": -2,
    },
    "hubert-base": {
        "hf_id": "facebook/hubert-base-ls960",
        "params_m": 95,
        "sample_rate": 16000,
        "layer": -2,
    },
    "wavlm-base-plus": {
        "hf_id": "microsoft/wavlm-base-plus",
        "params_m": 94,
        "sample_rate": 16000,
        "layer": -2,
    },
}


def load_audio_encoder(name: str):
    """Load frozen encoder + feature extractor. Returns (extract_fn, meta)."""
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    meta = ENCODERS[name]
    fe = Wav2Vec2FeatureExtractor.from_pretrained(meta["hf_id"])
    model = AutoModel.from_pretrained(meta["hf_id"]).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    layer = meta["layer"]

    @torch.no_grad()
    def extract(waveforms: list[np.ndarray]) -> np.ndarray:
        """waveforms: list of 16 kHz float arrays (variable length).
        Returns (n, d) mean-pooled penultimate-layer embeddings."""
        inputs = fe(waveforms, sampling_rate=meta["sample_rate"],
                    return_tensors="pt", padding=True).to(device)
        out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[layer]              # (B, T, D)
        emb = hidden.mean(dim=1)                       # (B, D)
        return emb.cpu().numpy()

    return extract, meta


def make_fold_fn(extract, cache: dict, seed: int):
    """fold_fn for laugho_cv: LR on frozen embeddings.
    Embeddings are cached per utterance id so the encoder runs once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def fold_fn(tr_X, tr_y, va_X, va_y, seed):
        # X entries are utterance ids; resolve to cached embeddings
        tr_emb = np.stack([cache[u] for u in tr_X])
        va_emb = np.stack([cache[u] for u in va_X])
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        )
        clf.fit(tr_emb, tr_y)
        scores = clf.predict_proba(va_emb)[:, 1]
        return roc_auc_score(va_y, scores)

    return fold_fn


def benchmark_encoder(name: str, utterances, labels, groups, waves):
    """Run 5×3 speaker-disjoint CV for one encoder.
    utterances: ids | labels: 0/1 humor | groups: video/comedian ids
    waves: dict id -> 16 kHz waveform"""
    extract, meta = load_audio_encoder(name)

    t0 = time.time()
    cache = {}
    ids = list(utterances)
    B = 16
    for i in range(0, len(ids), B):
        batch_ids = ids[i:i + B]
        embs = extract([waves[u] for u in batch_ids])
        for u, e in zip(batch_ids, embs):
            cache[u] = e
    embed_secs = time.time() - t0

    results = repeated_speaker_disjoint_cv(
        np.array(ids), np.array(labels), np.array(groups),
        fold_fn=make_fold_fn(extract, cache, seed=42),
        n_seeds=3, n_folds=5,
    )
    results["encoder"] = name
    results["hf_id"] = meta["hf_id"]
    results["params_m"] = meta["params_m"]
    results["embed_seconds"] = round(embed_secs, 1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/gold_manifest.json",
                    help="JSON: [{id, label, group, wav_path}, ...]")
    ap.add_argument("--encoders", default="distilhubert,wav2vec2-base,hubert-base,wavlm-base-plus")
    ap.add_argument("--out", default="e3_encoder_results.json")
    args = ap.parse_args()

    import soundfile as sf
    man = json.loads(Path(args.manifest).read_text())
    waves, ids, labels, groups = {}, [], [], []
    for row in man:
        wav, sr = sf.read(row["wav_path"], dtype="float32")
        if sr != 16000:
            raise ValueError(f"{row['wav_path']}: expected 16 kHz, got {sr}")
        waves[row["id"]] = wav
        ids.append(row["id"]); labels.append(row["label"]); groups.append(row["group"])

    chosen = list(ENCODERS) if args.encoders == "all" else args.encoders.split(",")
    all_results = {}
    for name in chosen:
        if name not in ENCODERS:
            raise SystemExit(f"unknown encoder {name!r}; choose from {list(ENCODERS)}")
        print(f"[E3] benchmarking {name} ({ENCODERS[name]['params_m']}M) ...")
        r = benchmark_encoder(name, ids, labels, groups, waves)
        all_results[name] = r
        print(f"     AUC {r['mean']:.4f} ± {r['std']:.4f} "
              f"(CI {r['ci_low']:.4f}–{r['ci_high']:.4f}, {r['embed_seconds']}s embed)")
        print(f"     verdict: {r['verdict']}")

    # Compare against honest anchors (NEVER 0.860 / 0.823 / 0.590)
    v5_floor, v10_ceiling = 0.632, 0.6924
    summary = {
        "protocol": "5x3 repeated speaker-disjoint CV, bootstrap 95% CI",
        "anchors": {"v5_floor_5fold": v5_floor, "v10_ceiling_5x3": v10_ceiling},
        "best_encoder": max(all_results, key=lambda k: all_results[k]["mean"]),
        "beats_v5_floor": {
            k: bool(all_results[k]["mean"] > v5_floor) for k in all_results
        },
        "results": all_results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n[E3] saved -> {args.out}")
    print(f"[E3] best: {summary['best_encoder']} "
          f"({all_results[summary['best_encoder']]['mean']:.4f})")


if __name__ == "__main__":
    main()
