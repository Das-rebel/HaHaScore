#!/usr/bin/env python3
"""
Enhanced Inference Pipeline for Cascade Gate v7
================================================
Provides:
- Single-utterance inference
- Batch inference
- Confidence estimation
- Robustness utilities
- CLI entry point
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from enhanced_cascade import create_enhanced_model
from enhanced_features import build_enhanced_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "enhanced"
DEFAULT_MODEL = MODEL_DIR / "enhanced_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: Path = None,
               text_dim: int = 780,
               audio_dim: int = 823,
               cross_dim: int = 16,
               hidden: int = 128) -> torch.nn.Module:
    """Load trained enhanced model from disk."""
    if model_path is None:
        model_path = DEFAULT_MODEL
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = create_enhanced_model(text_dim=text_dim, audio_dim=audio_dim,
                                  cross_dim=cross_dim, hidden=hidden)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


def infer_utterance(model,
                    text_features: np.ndarray,
                    audio_features: np.ndarray,
                    audio_samples: np.ndarray = None,
                    sr: int = 16000) -> dict:
    """
    Run inference for a single utterance.

    Args:
        model: Trained enhanced model.
        text_features: np.ndarray of shape (n_segments, 768).
        audio_features: np.ndarray of shape (n_segments, 791).
        audio_samples: np.ndarray of raw audio samples (optional).
        sr: sample rate of audio_samples.

    Returns:
        dict with keys: scores, confidence, gate_weight,
                        text_confidence, average_score, inference_ms.
    """
    # Build enhanced features
    et, ea, ec = build_enhanced_features(text_features, audio_features,
                                         audio_samples, sr=sr)

    # Convert to torch tensors
    t = torch.from_numpy(et).unsqueeze(0).to(DEVICE)
    a = torch.from_numpy(ea).unsqueeze(0).to(DEVICE)
    c = torch.from_numpy(ec).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        scores, text_conf, gate_weight = model(t, a, c)
    inference_ms = (time.perf_counter() - t0) * 1000

    scores_np = scores.squeeze(0).squeeze(-1).cpu().numpy()
    text_conf_np = text_conf.squeeze(0).squeeze(-1).cpu().numpy()
    gate_np = gate_weight.squeeze(0).squeeze(-1).cpu().numpy()

    return {
        "scores": scores_np.tolist(),
        "text_confidence": text_conf_np.tolist(),
        "gate_weight": gate_np.tolist(),
        "average_score": float(scores_np.mean()),
        "median_score": float(np.median(scores_np)),
        "max_score": float(scores_np.max()),
        "min_score": float(scores_np.min()),
        "inference_ms": round(inference_ms, 3),
    }


def infer_batch(model,
                text_features_batch: np.ndarray,
                audio_features_batch: np.ndarray,
                audio_samples_batch: list = None,
                sr: int = 16000) -> list:
    """
    Run inference for a batch of utterances.

    Args:
        model: Trained enhanced model.
        text_features_batch: np.ndarray of shape (B, n_segments, 768).
        audio_features_batch: np.ndarray of shape (B, n_segments, 791).
        audio_samples_batch: list of raw audio arrays (optional).
        sr: sample rate.

    Returns:
        list of dicts (one per utterance).
    """
    results = []
    for i in range(len(text_features_batch)):
        audio_samples = audio_samples_batch[i] if audio_samples_batch else None
        result = infer_utterance(
            model,
            text_features_batch[i],
            audio_features_batch[i],
            audio_samples,
            sr=sr
        )
        result["index"] = i
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced Cascade Gate v7 Inference")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path to model checkpoint")
    parser.add_argument("--text-features", type=str, required=True,
                        help="Path to .npy file with text features (n_segments, 768)")
    parser.add_argument("--audio-features", type=str, required=True,
                        help="Path to .npy file with audio features (n_segments, 791)")
    parser.add_argument("--audio-samples", type=str, default=None,
                        help="Path to raw audio .npy file (optional)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--text-dim", type=int, default=780,
                        help="Enhanced text feature dimension")
    parser.add_argument("--audio-dim", type=int, default=823,
                        help="Enhanced audio feature dimension")
    parser.add_argument("--cross-dim", type=int, default=16,
                        help="Cross-modal feature dimension")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dim")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    model = load_model(
        model_path=Path(args.model),
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        cross_dim=args.cross_dim,
        hidden=args.hidden,
    )

    text_features = np.load(args.text_features)
    audio_features = np.load(args.audio_features)
    audio_samples = np.load(args.audio_samples) if args.audio_samples else None

    print(f"Text features: {text_features.shape}")
    print(f"Audio features: {audio_features.shape}")

    result = infer_utterance(model, text_features, audio_features,
                             audio_samples, sr=args.sr)
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
