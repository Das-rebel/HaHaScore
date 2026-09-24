#!/usr/bin/env python3
"""
Bridge 7 Cascade Gate — Gradio Demo
====================================
Interactive humor strength scoring using pre-computed features.

Usage:
    python bridge7_demo.py
    Then open http://127.0.0.1:7860
"""
import json, gradio as gr, numpy as np, torch, torch.nn as nn
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "models/bridge7_cascade.pt"
BRIDGE4_FEATURES = Path("/tmp/bridge4_features.npz")
V6_TEXT_FEATURES = Path("/tmp/v6_features.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEGMENTS = 20

# ── Model ──────────────────────────────────────────────────────────────────

class CascadeGateFusion(nn.Module):
    def __init__(self, text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.text_confidence = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.pos_embedding = nn.Embedding(N_SEGMENTS + 1, 4)
        self.gru = nn.GRU(
            hidden * 3 + 4, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, text, audio, lengths=None):
        batch_size, seq_len, _ = text.shape
        text_h = self.text_proj(text)
        text_conf = torch.sigmoid(self.text_confidence(text_h))
        audio_h = self.audio_proj(audio)
        gated_audio = audio_h * text_conf
        text_attn_out, _ = self.audio_to_text_attn(audio_h, text_h, text_h)
        fused = torch.cat([text_h, gated_audio, text_attn_out], dim=-1)
        positions = torch.arange(seq_len, device=text.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        fused = torch.cat([fused, pos_emb], dim=-1)
        out, _ = self.gru(fused)
        return self.head(out).squeeze(-1), text_conf.squeeze(-1)


# ── Load Data ────────────────────────────────────────────────────────────────

def load_data():
    """Load pre-computed features."""
    cached = np.load(BRIDGE4_FEATURES, allow_pickle=True)
    all_audio = np.stack([np.asarray(f, dtype=np.float32) for f in cached["features"]])
    all_lengths = list(cached["lengths"])

    v6 = np.load(V6_TEXT_FEATURES, allow_pickle=True)
    all_text = np.stack([np.asarray(f, dtype=np.float32) for f in v6["text_features"]])
    video_ids = list(v6["video_ids"])

    return all_text, all_audio, video_ids, all_lengths


# ── Load Model ──────────────────────────────────────────────────────────────

def load_model():
    model = CascadeGateFusion().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ── Score Function ──────────────────────────────────────────────────────────

@torch.no_grad()
def score_video(video_idx: int, model=None):
    """Score all 20 segments for a given video index."""
    if model is None:
        model = load_model()

    text_feat = all_text[video_idx]
    audio_feat = all_audio[video_idx]

    text_t = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    audio_t = torch.tensor(audio_feat, dtype=np.float32).unsqueeze(0).to(DEVICE)

    scores, confs = model(text_t, audio_t)
    scores = scores.squeeze(0).cpu().numpy()
    confs = confs.squeeze(0).cpu().numpy()

    return scores, confs


# ── Main ────────────────────────────────────────────────────────────────────

print("Loading data...")
all_text, all_audio, video_ids, all_lengths = load_data()
print(f"Loaded {len(video_ids)} videos")

model = load_model()
print(f"Model loaded on {DEVICE}")


# ── Gradio UI ───────────────────────────────────────────────────────────────

def make_chart(scores, confs):
    """Create a bar chart of humor scores per segment."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x = list(range(1, N_SEGMENTS + 1))
    colors = plt.cm.RdYlGn(scores)  # green=funny, red=not funny
    bars = ax.bar(x, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Segment (1=start, 20=end)")
    ax.set_ylabel("Humor Score")
    ax.set_title("Humor Strength per Segment (Bridge 7 Cascade Gate)")
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(scores), color='blue', linestyle='--', label=f'Mean={np.mean(scores):.3f}')
    ax.legend()
    plt.tight_layout()
    return fig


def make_dual_chart(scores, confs):
    """Create dual bar chart: scores + text confidence."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    x = list(range(1, N_SEGMENTS + 1))

    colors1 = plt.cm.RdYlGn(scores)
    ax1.bar(x, scores, color=colors1, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel("Humor Score")
    ax1.set_title("Bridge 7 Humor Scores (top) + Text Confidence (bottom)")
    ax1.set_ylim(0, 1)
    ax1.axhline(y=np.mean(scores), color='blue', linestyle='--', label=f'Mean={np.mean(scores):.3f}')
    ax1.legend()

    colors2 = plt.cm.Blues(confs)
    ax2.bar(x, confs, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel("Segment (1=start, 20=end)")
    ax2.set_ylabel("Text Confidence")
    ax2.set_ylim(0, 1)
    ax2.axhline(y=np.mean(confs), color='navy', linestyle='--', label=f'Mean={np.mean(confs):.3f}')
    ax2.legend()

    plt.tight_layout()
    return fig


def demo_fn(video_choice):
    """Main Gradio function."""
    idx = int(video_choice.split(" [")[0])
    scores, confs = score_video(idx, model)

    chart = make_dual_chart(scores, confs)

    # Summary stats
    summary = f"""
### Video: `{video_ids[idx]}`
- **Mean Humor Score:** {np.mean(scores):.3f} (0=not funny, 1=very funny)
- **Max Score:** {np.max(scores):.3f} (segment {np.argmax(scores)+1})
- **Min Score:** {np.min(scores):.3f} (segment {np.argmin(scores)+1})
- **Mean Text Confidence:** {np.mean(confs):.3f}
- **End Score (seg 20):** {scores[-1]:.3f}
- **High Funny (≥0.7):** {(scores >= 0.7).sum()}/20 segments
- **Low Funny (<0.3):** {(scores < 0.3).sum()}/20 segments

### Segment Scores:
| Seg | Score | Conf |
|-----|-------|------|
"""
    for i in range(N_SEGMENTS):
        emoji = "😄" if scores[i] >= 0.7 else "😐" if scores[i] >= 0.4 else "😶"
        summary += f"| {i+1:2d} | {scores[i]:.3f} {emoji} | {confs[i]:.3f} |\n"

    return summary.strip(), chart


# Build video choices
video_choices = []
for i, vid in enumerate(video_ids):
    lang = vid.split(',')[1] if ',' in str(vid) else 'en'
    video_choices.append(f"{i} [{lang}] — {vid[:20]}")

# Stats for header
header = f"""
# HaHaScore 😄 — Bridge 7 Cascade Gate Demo

**Model:** Bridge 7 Cascade Gate (AUC 0.860 ± 0.018 on pseudo-labels)

### Performance Summary
| Model | AUC (pseudo) | AUC (gold laughs) |
|-------|-------------|-----------------|
| Bridge 4 (audio-only) | 0.842 | 0.576 |
| v6 (cross-attention) | 0.858 | -- |
| **Bridge 7 (cascade gate)** | **0.860** | **0.590** |

### How to Read Scores
- **Score ≥ 0.7:** High humor strength (funny delivery)
- **Score 0.4-0.7:** Moderate humor
- **Score < 0.4:** Low humor / deadpan / setup

### Model Architecture
Bridge 7 uses a cascade gate: text confidence modulates how much the model trusts audio.
When text shows clear semantic structure (setup/punchline), audio contribution is amplified.
This mirrors human perception: we listen more carefully when we expect a punchline.

**Dataset:** {len(video_ids)} StandUp4AI comedy videos, 20 segments each
**Note:** Scores are on pseudo-labels (perceived funniness), NOT audience laughter.
"""

demo = gr.Blocks(title="HaHaScore — Bridge 7 Demo")

with demo:
    gr.Markdown(header)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Select a Video")
            dropdown = gr.Dropdown(video_choices, value=video_choices[0], label="Video")
            btn = gr.Button("Score Video", variant="primary")
            gr.Markdown("*Scores are computed on CPU — may take 10-30s*")

        with gr.Column(scale=2):
            output_text = gr.Markdown()
            output_plot = gr.Plot()

    btn.click(fn=demo_fn, inputs=dropdown, outputs=[output_text, output_plot])
    dropdown.change(fn=demo_fn, inputs=dropdown, outputs=[output_text, output_plot])

print(f"Launching demo on http://127.0.0.1:7860")
demo.launch(share=False, server_name="127.0.0.1")
