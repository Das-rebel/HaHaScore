#!/usr/bin/env python3
"""
v8 Inference: Reddit DistilBERT + Cascade Gate v8
====================================================

Pipeline:
  Text input → Reddit DistilBERT → 768-dim → v8 text_proj → 128-dim → Cascade Gate
  Audio input → WavLM/prosody → 791-dim → v8 audio_proj → 128-dim → Cascade Gate
  Cross-modal features → direct → Cascade Gate
  Cascade Gate → Score (0-1)
"""
import os
os.environ['HF_HOME'] = '/tmp/hf_distilbert'
import torch
import torch.nn as nn
from transformers import DistilBertModel


class RedditTextEncoder(nn.Module):
    """DistilBERT pretrained on Reddit upvotes for continuous funniness."""

    def __init__(self, reddit_state_path=None):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)

        if reddit_state_path and os.path.exists(reddit_state_path):
            state = torch.load(reddit_state_path, map_location='cpu')
            # Map our save format 'bb.X' → HF 'distilbert.X'
            new_state = {}
            for k, v in state.items():
                if k.startswith('bb.'):
                    new_key = k.replace('bb.', 'distilbert.')
                    new_state[new_key] = v
            missing, unexpected = self.load_state_dict(new_state, strict=False)
            loaded = sum(1 for n in new_state if n in self.state_dict())
            print(f"Loaded {loaded}/{len(new_state)} Reddit tensors. Missing={len(missing)}, Unexpected={len(unexpected)}")

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.dropout(pooled)  # (batch, 768)


class V8WithReddit(nn.Module):
    """Cascade Gate v8 + Reddit-pretrained text tower.

    Inputs:
      - input_ids: (batch, seq_len) tokenized text
      - attention_mask: (batch, seq_len)
      - audio: (batch, seq_len, 791) audio features
      - cross: (batch, seq_len, 16) cross-modal features (optional)
    """
    def __init__(self, v8_model, reddit_state_path=None, freeze_text=True):
        super().__init__()
        self.text_encoder = RedditTextEncoder(reddit_state_path)
        self.v8 = v8_model  # Expects text features of dim 768

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, audio, cross=None, lengths=None):
        # Encode text → 768-dim
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch, 768)
        # Broadcast across segments
        batch_size, seq_len, _ = audio.shape
        text_per_seg = text_features.unsqueeze(1).expand(-1, seq_len, -1)
        return self.v8(text_per_seg, audio, cross, lengths)


# Smoke test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/Subho/funny-strength-predictor/improvements')
    from enhanced_cascade import create_enhanced_model
    
    v8 = create_enhanced_model(text_dim=768, audio_dim=791, cross_dim=16, hidden=128)
    model = V8WithReddit(v8, reddit_state_path=None)
    
    batch, seq = 2, 20
    input_ids = torch.randint(0, 30000, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
    audio = torch.randn(batch, seq, 791)
    cross = torch.randn(batch, seq, 16)
    
    scores, conf, gate = model(input_ids, attn, audio, cross)
    print(f"Scores: {scores.shape}, Conf: {conf.shape}, Gate: {gate.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ v8 inference module works")
