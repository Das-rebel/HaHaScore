"""
HaHaScore Fusion v5 — Gradio Demo App
======================================
Humor strength predictor for spoken comedy sentences.
Launch: python app.py

Requirements: pip install gradio torch transformers librosa pydub
"""

import os
import tempfile

import gradio as gr
import numpy as np
import torch
import librosa
from pydub import AudioSegment
from transformers import AutoModel, AutoTokenizer, WavLMModel
from huggingface_hub import hf_hub_download

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Load Fusion v5 ─────────────────────────────────────────────────────
print("Loading fusion model...")
model_path = hf_hub_download("Hayasuki/hahascore-fusion-v5", "pytorch_model.bin")
checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)


class BilinearFusionMLP(torch.nn.Module):
    """Fusion MLP from HaHaScore Fusion v5."""

    def __init__(self, txt_dim=768, aud_dim=512, hidden=128):
        super().__init__()
        self.txt_proj = torch.nn.Linear(txt_dim, hidden)
        self.aud_proj = torch.nn.Linear(aud_dim, hidden)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden * 3, hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, txt_feat, aud_feat):
        t = self.txt_proj(txt_feat)  # (B, 128)
        a = self.aud_proj(aud_feat)  # (B, 128)
        h = t * a  # Hadamard product
        x = torch.cat([h, t, a], dim=1)  # (B, 384)
        return self.net(x)


fusion = BilinearFusionMLP()
fusion.load_state_dict(checkpoint["model_state_dict"], strict=False)
fusion.eval().to(DEVICE)
print("Fusion model loaded.")

# ── Load Encoders ───────────────────────────────────────────────────────
print("Loading DeBERTa-v3-base...")
text_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").eval()
text_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

print("Loading WavLM-base-plus...")
audio_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
print("All models loaded.")


# ── Feature Extraction ───────────────────────────────────────────────────

def extract_text_features(sentence: str) -> np.ndarray:
    """DeBERTa [CLS] pooler → (1, 768)."""
    inputs = text_tokenizer(
        sentence, return_tensors="pt", truncation=True, max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = text_model(**inputs)
        feat = out.last_hidden_state[:, 0, :].cpu().numpy()
    return feat


def extract_audio_features_from_samples(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    WavLM mean-pooled → (1, 512).
    Accepts raw PCM samples at sr Hz. Audio is 4× downsampled (16k→4kHz).
    """
    # Pad to 6 seconds if short
    target_len = sr * 6
    if len(samples) < target_len:
        samples = np.pad(samples, (0, target_len - len(samples)))
    else:
        samples = samples[:target_len]

    # 4× downsample for WavLM
    samples_4k = librosa.resample(samples, orig_sr=sr, target_sr=4000)

    with torch.no_grad():
        x = torch.tensor(samples_4k, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        out = audio_model(x)
        feat = out.last_hidden_state.mean(dim=1).cpu().numpy()
    return feat


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load any pydub-supported audio → (samples, sr)."""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples, 16000


# ── Main Prediction ─────────────────────────────────────────────────────

def predict_humor(text: str, audio) -> dict:
    """
    Predict humor strength of a spoken sentence.

    Args:
        text: Transcribed sentence (English).
        audio: Audio file uploaded via Gradio.

    Returns:
        dict with score, interpretation, and confidence bar.
    """
    if not text or not text.strip():
        return {"score": "—", "interpretation": "Please enter a sentence.", "confidence": 0}

    if audio is None:
        return {"score": "—", "interpretation": "Please upload an audio clip.", "confidence": 0}

    try:
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(audio.name)[1], delete=False
        ) as tmp:
            tmp.write(audio.read())
            tmp_path = tmp.name

        # Load and extract audio features
        samples, sr = load_audio(tmp_path)
        aud_feat = extract_audio_features_from_samples(samples, sr)  # (1, 512)

        # Extract text features
        txt_feat = extract_text_features(text.strip())  # (1, 768)

        # Forward through fusion
        with torch.no_grad():
            score = fusion(
                torch.tensor(txt_feat).float().to(DEVICE),
                torch.tensor(aud_feat).float().to(DEVICE),
            ).item()

        os.unlink(tmp_path)

        # Interpret
        if score >= 0.7:
            interpretation = "🔥 Strong comedy signal — likely a punchline or funny bit"
        elif score >= 0.5:
            interpretation = "😄 Moderate humor — setup or mildly funny segment"
        elif score >= 0.3:
            interpretation = "😐 Ambiguous — neutral delivery or transition"
        else:
            interpretation = "😑 Not funny — deadpan, serious, or background"

        return {
            "score": f"{score:.3f}",
            "interpretation": interpretation,
            "confidence": float(score),
        }

    except Exception as e:
        return {
            "score": "—",
            "interpretation": f"Error: {str(e)[:100]}",
            "confidence": 0,
        }


# ── Gradio Interface ────────────────────────────────────────────────────

examples = [
    [
        "I told my wife she was drawing her eyebrows too high.",
        "example_comedy.wav",
    ],
    [
        "So I said, that's not a knife, this is a knife.",
        "example_dialogue.wav",
    ],
]

css = """
#title { text-align: center; font-size: 2.5em; font-weight: bold; margin-bottom: 0.3em; }
#subtitle { text-align: center; color: #666; margin-bottom: 1em; }
.score-box { border-radius: 12px; padding: 20px; text-align: center; }
"""

with gr.Blocks(css=css, title="HaHaScore Fusion v5") as demo:
    gr.Markdown(
        '<div id="title">HaHaScore Fusion v5</div>'
        '<div id="subtitle">Multimodal Humor Strength Predictor — sentence-level comedy scoring</div>'
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Sentence (transcribed)",
                placeholder="Enter the sentence as transcribed...",
                lines=3,
                info="English text of the spoken sentence",
            )
            audio_input = gr.Audio(
                label="Audio clip (WAV/MP3/M4A)",
                type="file",
                info="3–6 second clip of the spoken sentence",
            )
            submit_btn = gr.Button("Predict Humor Score", variant="primary")

        with gr.Column(scale=1):
            score_output = gr.Textbox(label="Humor Score", lines=1)
            interp_output = gr.Textbox(label="Interpretation", lines=2)
            confidence_bar = gr.Number(
                label="Confidence", interactive=False, visible=False
            )
            confidence = gr.Label(
                label="Score Visualization",
                num_classes=3,
                visible=True,
            )

    gr.Examples(
        examples=[[e[0], None] for e in examples],
        inputs=text_input,
    )

    gr.Markdown(
        "---"
        "\n**Model:** HaHaScore Fusion v5 — Bilinear Fusion of DeBERTa-v3-base + WavLM-base-plus"
        "\n**Performance:** 5-Fold CV AUC 0.632 ± 0.007 · Held-Out AUC 0.613"
        "\n**⚠️** Trained on English stand-up comedy only. Text alone ≈ random; audio prosody carries the primary signal."
        "\n[GitHub](https://github.com/Das-rebel/HaHaScore) · [Model Card](https://huggingface.co/Hayasuki/hahascore-fusion-v5)"
    )

    submit_btn.click(
        fn=predict_humor,
        inputs=[text_input, audio_input],
        outputs=[score_output, interp_output, confidence_bar],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
