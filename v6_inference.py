#!/usr/bin/env python3
"""
v6 TriModal Cross-Attention Inference
=====================================
Text (RoBERTa) + Audio (WavLM+prosody) -> CrossAttention -> BiGRU -> Score

Usage:
    from v6_inference import score_segments
    scores = score_segments(text_features, audio_features)
    # text_features: (20, 768) numpy array
    # audio_features: (20, 791) numpy array
    # Returns: (20,) array of scores 0-1
"""
import json, numpy as np, torch, torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEGMENTS = 20

class TriModalFusion(nn.Module):
    def __init__(self, text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True
        )
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True
        )
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)
        self.gru = nn.GRU(
            hidden * 4 + 4,  # text_proj + audio_proj + text_attn + audio_attn + pos
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, text, audio, lengths=None):
        batch_size, seq_len, _ = text.shape
        text_h = self.text_proj(text)
        audio_h = self.audio_proj(audio)
        text_attn_out, _ = self.text_to_audio_attn(text_h, audio_h, audio_h)
        audio_attn_out, _ = self.audio_to_text_attn(audio_h, text_h, text_h)
        fused = torch.cat([text_h, audio_h, text_attn_out, audio_attn_out], dim=-1)
        positions = torch.arange(seq_len, device=text.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        fused = torch.cat([fused, pos_emb], dim=-1)
        out, _ = self.gru(fused)
        return self.head(out).squeeze(-1)


def load_model(model_path=None):
    """Load v6 model."""
    model_path = model_path or "v6_trimodal.pt"
    model = TriModalFusion().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def score_segments(text_features, audio_features, model=None):
    """Score humor strength for segments.
    text_features: (20, 768) numpy array or list
    audio_features: (20, 791) numpy array or list
    Returns: (20,) numpy array of scores 0-1
    """
    if model is None:
        model = load_model()

    text_t = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    audio_t = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        scores = model(text_t, audio_t).squeeze(0).cpu().numpy()

    return scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v6 TriModal Humor Scoring")
    parser.add_argument("--text", required=True, help="Path to text features (.npy, shape 20x768)")
    parser.add_argument("--audio", required=True, help="Path to audio features (.npy, shape 20x791)")
    parser.add_argument("--model", default="v6_trimodal.pt", help="Model path")
    args = parser.parse_args()

    text_feats = np.load(args.text)
    audio_feats = np.load(args.audio)
    scores = score_segments(text_feats, audio_feats, load_model(args.model))
    print("Scores:", scores)
    print("Mean:", scores.mean())
