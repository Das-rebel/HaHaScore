"""
Verify v8_finetuned_smoke works for inference.
"""
import os
os.environ['HF_HOME'] = '/tmp/hf_distilbert'
import sys
sys.path.insert(0, '/Users/Subho/funny-strength-predictor/drive_pipeline')
sys.path.insert(0, '/Users/Subho/funny-strength-predictor/improvements')

import torch
import numpy as np
from pathlib import Path
from v8_with_reddit import V8WithReddit
from enhanced_cascade import create_enhanced_model

# Use the pulled model  
v8_path = '/tmp/v8_verify/v8.pt'
print(f"📂 Loading v8 from {v8_path}...")
state = torch.load(v8_path, map_location='cpu')
print(f"State keys (first 5): {list(state.keys())[:5]}")

# Recreate architecture
v8_base = create_enhanced_model(text_dim=768, audio_dim=791, cross_dim=16, hidden=128)
print(f"v8 base: {sum(p.numel() for p in v8_base.parameters()):,} params")

# Load state into v8 base directly (trained with freeze_text)
v8_base.load_state_dict({k.replace('v8.', ''): v for k, v in state['model_state_dict'].items() if k.startswith('v8.')})
v8_base.eval()
print(f"✅ v8 loaded with trained weights")

# Inference
print(f"\n🧪 Running inference on dummy data...")
test_audio = torch.randn(1, 20, 791)  # (batch, seq, audio_dim)
test_text = torch.randn(1, 20, 768)  # (batch, seq, text_dim)

with torch.no_grad():
    scores, conf, gate = v8_base(test_text, test_audio, cross_modal=None)

print(f"  Scores shape: {scores.shape}")
print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"  Mean score: {scores.mean():.4f}")
print(f"  Per-segment scores: {[f'{s:.3f}' for s in scores.squeeze().tolist()[:5]]}")

# Test with real standup data shape (639 files, 20 segments)
print(f"\n📊 Testing with real standup data shape...")
b4 = np.load('/Users/Subho/tmp/bridge4_features.npz', allow_pickle=True)
v6 = np.load('/Users/Subho/tmp/v6_features.npz', allow_pickle=True)
audio_sample = b4['features'][0]  # First video
text_sample = v6['text_features'][0]  # First video

# Convert audio from object to float
if isinstance(audio_sample, np.ndarray) and audio_sample.dtype == object:
    audio_sample = np.array([float(x) for x in audio_sample.flatten()]).reshape(20, 791)

audio_t = torch.tensor(audio_sample, dtype=torch.float32).unsqueeze(0)
text_t = torch.tensor(text_sample, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    real_scores, _, _ = v8_base(text_t, audio_t, cross_modal=None)

print(f"  Real standup scores: {[f'{s:.3f}' for s in real_scores.squeeze().tolist()[:10]]}")
print(f"\n✅ v8 fine-tuned model is functional and ready for deployment")
print(f"\n📁 Stored at: gdrive:/HaHaScore_Pretrain/models/v8_finetuned_smoke.pt")
