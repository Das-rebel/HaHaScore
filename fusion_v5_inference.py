#!/usr/bin/env python3
"""
Fusion v5 — Per-Segment Bilinear Humor Fusion
============================================
Lightweight per-segment humor scorer using DeBERTa + WavLM bilinear fusion.

Usage:
    python fusion_v5_inference.py "I told my wife she was drawing her eyebrows too high." audio.m4a
    python fusion_v5_inference.py --interactive

Output:
    Single humor score (0-1) with interpretation
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

DEVICE = "cpu"


class BilinearFusionMLP(nn.Module):
    """Bilinear fusion of text (DeBERTa) + audio (WavLM)."""

    def __init__(self, txt_dim=768, aud_dim=512, hidden=128):
        super().__init__()
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.aud_proj = nn.Linear(aud_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.3),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, txt_feat, aud_feat):
        t = self.txt_proj(txt_feat)
        a = self.aud_proj(aud_feat)
        h = t * a  # Hadamard
        x = torch.cat([h, t, a], dim=1)
        return self.net(x)


@torch.no_grad()
def score_sentence(text, audio_samples, sr, model, text_model, text_tokenizer, audio_model):
    """
    Score a single sentence + audio clip.

    Args:
        text: Sentence string
        audio_samples: numpy array of audio samples at sr Hz
        sr: sample rate
        model: BilinearFusionMLP
        text_model: DeBERTa-v3-base
        text_tokenizer: DeBERTa tokenizer
        audio_model: WavLM-base-plus

    Returns:
        float score (0-1)
    """
    # Text features — DeBERTa [CLS] pooler
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    txt_out = text_model(**inputs)
    txt_feat = txt_out.last_hidden_state[:, 0, :]  # (1, 768)

    # Audio features — 4× downsample + WavLM mean-pool
    if len(audio_samples) < sr:  # Pad short audio
        audio_samples = np.pad(audio_samples, (0, sr - len(audio_samples)))
    audio_4k = librosa.resample(audio_samples.astype(np.float32), orig_sr=sr, target_sr=4000)
    aud_inp = torch.tensor(audio_4k).unsqueeze(0).to(DEVICE)
    aud_out = audio_model(aud_inp)
    aud_feat = aud_out.last_hidden_state.mean(dim=1)  # (1, 512)

    # Bilinear fusion
    score = model(txt_feat.float(), aud_feat.float()).item()
    return score


def load_models():
    """Load all required models."""
    from transformers import AutoModel, AutoTokenizer, WavLMModel

    # Load fusion
    model_path = "fusion_v5_model.bin"
    if not Path(model_path).exists():
        model_path = "pytorch_model.bin"  # fallback for bridge4 repo
    model = BilinearFusionMLP()
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load encoders
    text_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").eval()
    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    audio_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()

    return model, text_model, text_tokenizer, audio_model


def interpret(score):
    """Convert score to human-readable interpretation."""
    if score >= 0.8: return "🔥 Strong comedy — likely a punchline"
    elif score >= 0.6: return "😄 Moderate humor — setup or funny bit"
    elif score >= 0.4: return "😐 Ambiguous — neutral delivery"
    elif score >= 0.2: return "😑 Low humor — deadpan or transition"
    else: return "😶 Not funny — serious content"


if __name__ == "__main__":
    import argparse, librosa
    from pydub import AudioSegment

    parser = argparse.ArgumentParser(description="Fusion v5 Humor Scorer")
    parser.add_argument("text", nargs="?", help="Sentence text")
    parser.add_argument("audio", nargs="?", help="Audio file (m4a/wav/mp3)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    print("Loading models (first run downloads ~500MB)...")
    model, text_model, text_tokenizer, audio_model = load_models()
    print("Models ready.\n")

    if args.interactive:
        print("HaHaScore Fusion v5 Interactive Mode")
        print("Enter text and audio path (or 'quit' to exit)\n")
        while True:
            text = input("Sentence: ").strip()
            if text.lower() == 'quit': break
            audio_path = input("Audio file: ").strip()
            if not Path(audio_path).exists():
                print(f"File not found: {audio_path}")
                continue
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            score = score_sentence(text, samples, 16000, model, text_model, text_tokenizer, audio_model)
            print(f"Score: {score:.3f} — {interpret(score)}\n")

    elif args.text and args.audio:
        audio = AudioSegment.from_file(args.audio)
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        score = score_sentence(args.text, samples, 16000, model, text_model, text_tokenizer, audio_model)
        if args.json:
            print(json.dumps({"score": score, "interpretation": interpret(score)}))
        else:
            print(f"Humor Score: {score:.3f}")
            print(f"Interpretation: {interpret(score)}")

    else:
        parser.print_help()
