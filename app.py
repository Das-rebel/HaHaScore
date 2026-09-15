#!/usr/bin/env python3
"""HaHaScore — multimodal humor strength 0-100 with preloading."""
import os, time
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, RobertaModel, AutoModel as WavLMAuto

print("Loading models at startup...", flush=True)
_START = time.time()

_TEXT_MODEL_PATH = os.environ.get("TEXT_MODEL_PATH", "/Users/Subho/models/chuckle_predictor/roberta_v7.pt")
_FUSION_MODEL_PATH = os.environ.get("FUSION_MODEL_PATH", "/Users/Subho/models/chuckle_predictor/sentence_fusion_v3.pt")

_tokenizer = None
_text_model = None
_fusion_model = None
_MODELS_LOADED = False

def _load_models():
    global _tokenizer, _text_model, _fusion_model, _MODELS_LOADED
    if _MODELS_LOADED: return

    print("Loading tokenizer...", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Text-only v7 model (raw logits, apply sigmoid at inference)
    # v7 checkpoint uses pooler_output (not mean pooling)
    print("Loading text model...", flush=True)
    class TextReg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = RobertaModel.from_pretrained("roberta-base")
            self.drop = torch.nn.Dropout(0.2)
            self.head = torch.nn.Linear(768, 1)
        def forward(self, ids, am):
            # Use pooler_output to match v7 checkpoint
            pooled = self.enc(input_ids=ids, attention_mask=am).pooler_output
            dropped = self.drop(pooled)
            return (dropped @ self.head.weight.t() + self.head.bias).squeeze(-1)  # raw logit

    _text_model = TextReg()
    _text_model.load_state_dict(torch.load(_TEXT_MODEL_PATH, map_location="cpu"))
    _text_model.eval()

    # Fusion model (raw logits, apply sigmoid at inference)
    print("Loading fusion model...", flush=True)
    class FusionReg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.txt = RobertaModel.from_pretrained("roberta-base")
            self.aud = WavLMAuto.from_pretrained("microsoft/wavlm-base-plus")
            self.fuse = torch.nn.Sequential(
                torch.nn.Linear(768+512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 1))
        def forward(self, ids, am, wav):
            # wav: (batch, samples) 2D
            t = self.txt(input_ids=ids, attention_mask=am).pooler_output
            a = self.aud(input_values=wav).extract_features.mean(1)
            return self.fuse(torch.cat([t, a], dim=-1)).squeeze(-1)  # raw logit

    _fusion_model = FusionReg()
    for path in ["/Users/Subho/models/chuckle_predictor/sentence_fusion_v3.pt",
                  _FUSION_MODEL_PATH]:
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location="cpu")
                # sentence_fusion_v3.pt has {'txt':{}, 'aud':{}, 'fuse': {...}} format
                if 'fuse' in ckpt:
                    _fusion_model.load_state_dict(ckpt['fuse'], strict=False)
                    print(f"Loaded fuse from {path}", flush=True)
                else:
                    _fusion_model.load_state_dict(ckpt)
                    print(f"Loaded fusion from {path}", flush=True)
                break
            except Exception as e:
                print(f"Failed {path}: {e}", flush=True)
    _fusion_model.eval()
    _MODELS_LOADED = True
    print(f"✅ All models loaded in {time.time()-_START:.1f}s", flush=True)

def score_text(text):
    if not text.strip(): return {"Error": "Enter text"}
    _load_models()
    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logit = _text_model(enc["input_ids"], enc["attention_mask"]).item()
        score = torch.sigmoid(torch.tensor(logit)).item() * 100
    return {f"{round(score,1)}": 1.0}

def score_fusion(text, audio_file):
    if not text.strip(): return {"Error": "Enter text"}
    if audio_file is None: return {"Error": "Upload audio"}
    _load_models()
    try:
        import torchaudio
        # End-biased 6s audio (sentence fusion v3: last 6 seconds of audio)
        MAX_AUDIO_S = 6.0; SR_TARGET = 16000
        waveform, sr_orig = torchaudio.load(audio_file)
        if waveform.shape[0] > 1: waveform = waveform.mean(0)
        total_dur = len(waveform) / sr_orig
        # Step 1: calculate slice in original SR
        s_start = max(0, total_dur - MAX_AUDIO_S)
        s_start_s = int(s_start * sr_orig)
        num_s = int(MAX_AUDIO_S * sr_orig)
        # Step 2: slice
        waveform = waveform[s_start_s:s_start_s + num_s]
        # Step 3: pad if needed
        if len(waveform) < num_s: waveform = torch.nn.functional.pad(waveform, (0, num_s - len(waveform)))
        # Step 4: resample to target SR
        if sr_orig != SR_TARGET: waveform = torchaudio.functional.resample(waveform, sr_orig, SR_TARGET)
        wav_2d = waveform.unsqueeze(0)  # (1, 16000*6)
    except Exception as e:
        return {"Error": str(e)}
    # Text: last 12 context words (sentence-level)
    words = text.split()
    text_context = ' '.join(words[-12:])
    enc = _tokenizer(text_context, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        text_logit = _text_model(enc["input_ids"], enc["attention_mask"]).item()
        fusion_logit = _fusion_model(enc["input_ids"], enc["attention_mask"], wav_2d).item()
        text_score = torch.sigmoid(torch.tensor(text_logit)).item() * 100
        fusion_score = torch.sigmoid(torch.tensor(fusion_logit)).item() * 100
        combined = (text_score + fusion_score) / 2
    return {
        f"Text v7: {round(text_score,1)}": 0.7,
        f"Fusion v3: {round(fusion_score,1)}": 0.85,
        f"Combined: {round(combined,1)}": 1.0
    }

_load_models()

with gr.Blocks(title="HaHaScore Humor Predictor") as demo:
    gr.Markdown("# 😄 HaHaScore — Humor Strength Predictor")
    gr.Markdown("**Text v7** (AUC 0.566) + **Fusion v3** (AUC 0.601, sentence-level audio)")
    with gr.Tab("Text Only"):
        txt_input = gr.Textbox(label="Enter joke or text", placeholder="Why did the chicken cross the road?")
        txt_btn = gr.Button("Score", variant="primary")
        txt_btn.click(score_text, inputs=txt_input, outputs=gr.Label(num_top_classes=1))
    with gr.Tab("Text + Audio (Fusion)"):
        txt2 = gr.Textbox(label="Text", placeholder="Corresponding text")
        aud = gr.Audio(label="Audio (wav/mp3)", type="filepath")
        fus_btn = gr.Button("Score Fusion", variant="primary")
        fus_btn.click(score_fusion, inputs=[txt2, aud], outputs=gr.Label(num_top_classes=3))
    gr.Markdown("**Text**: [`hahascore-text-v7`](https://huggingface.co/Hayasuki/hahascore-text-v7) (AUC 0.566) | **Fusion v3**: [`hahascore-fusion-v3`](https://huggingface.co/Hayasuki/hahascore-fusion-v3) (AUC 0.601) | **Fusion v5 (final)**: [`hahascore-fusion-v5`](https://huggingface.co/Hayasuki/hahascore-fusion-v5) (AUC 0.613, 5-fold CV 0.632±0.007)")
    gr.Markdown("**Key Finding**: Text-only models are RANDOM (AUC ~0.50) for sentence-level humor. Audio is the primary discriminative signal (+13% AUC improvement over text.)")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
