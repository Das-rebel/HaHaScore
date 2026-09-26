#!/usr/bin/env python3
"""
Build Cascade Gate v8 with Reddit-pretrained text tower
========================================================

Architecture:
  Standup text segments → RedditDistilBERT (768-dim) → v8 text_proj (128-dim)
                                                              ↓
                                                          Cascade Gate ← Audio (WavLM 791-dim)
                                                              ↓
                                                              ↓
                                                          Cross-modal attention
                                                              ↓
                                                          Score (0-1)

Strategy:
1. Pull Reddit-pretrained DistilBERT from Drive
2. Build v8 architecture (enhanced_cascade.EnhancedCascadeGateFusion)
3. Add Reddit DistilBERT as upstream text feature extractor
4. Map Reddit's hidden states → v8's text_proj input
5. Save v8_with_reddit.py that works end-to-end
6. Generate inference example for v8
"""
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

# Drive paths
DRIVE_BASE = "gdrive:/HaHaScore_Pretrain"

def pull_reddit_model():
    """Pull Reddit DistilBERT state dict from Drive."""
    temp_dir = Path(tempfile.mkdtemp(prefix="v8_init_"))
    print("📥 Pulling Reddit DistilBERT from Drive...")

    # Find latest Reddit model
    result = subprocess.run(
        ['rclone', 'lsf', f'{DRIVE_BASE}/models/'],
        capture_output=True, text=True
    )
    models = [m for m in result.stdout.strip().split('\n') if m.endswith('.pt')]
    if not models:
        print("❌ No Reddit models on Drive yet")
        return None, None

    # Prefer the v1 (larger) over smoke
    preferred = 'reddit_distilbert_v1.pt' if 'reddit_distilbert_v1.pt' in models else models[0]
    print(f"Using: {preferred}")

    result = subprocess.run(
        ['rclone', 'copyto', f'{DRIVE_BASE}/models/{preferred}',
         str(temp_dir / 'reddit_distilbert.pt'),
         '--retries=3', '--retries-sleep=10s'],
        capture_output=True, text=True
    )

    model_path = temp_dir / 'reddit_distilbert.pt'
    if not model_path.exists():
        print(f"❌ Pull failed: {result.stderr}")
        return None, None
    print(f"✅ Pulled ({model_path.stat().st_size/1e6:.1f} MB)")
    return model_path, temp_dir


def build_text_extractor(model_path):
    """Build a DistilBERT-based text feature extractor from Reddit checkpoint."""
    import torch
    import torch.nn as nn
    from transformers import DistilBertModel, DistilBertConfig

    print(f"Loading Reddit state dict from {model_path}...")
    state = torch.load(model_path, map_location='cpu')
    print(f"State has {len(state)} tensors, sample shapes:")
    for k in list(state.keys())[:5]:
        print(f"  {k}: {state[k].shape}")

    # DistilBERT structure
    class RedditTextEncoder(nn.Module):
        """DistilBERT text encoder pretrained on Reddit upvotes."""
        def __init__(self):
            super().__init__()
            self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.pool = nn.AdaptiveAvgPool1d(1)  # Will use mean pool instead
            self.dropout = nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask):
            # DistilBERT forward
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Mean pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            return self.dropout(pooled)  # (batch, 768)

        def get_input_embeddings(self):
            return self.distilbert.embeddings.word_embeddings

    encoder = RedditTextEncoder()
    # Try to load — weights may have different naming
    try:
        # Map state dict to our encoder
        new_state = {}
        for k, v in state.items():
            # Different naming conventions between our script and HF
            if k.startswith('bb.'):
                # Our torch.save(model.state_dict()) saves 'bb' prefix
                new_key = k.replace('bb.', 'distilbert.')
                new_state[new_key] = v
            elif k.startswith('distilbert.'):
                new_state[k] = v
            else:
                # Skip other keys (head)
                continue
        missing, unexpected = encoder.load_state_dict(new_state, strict=False)
        print(f"Loaded Reddit encoder. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys (first 3): {missing[:3]}")
        if unexpected:
            print(f"  Unexpected keys (first 3): {unexpected[:3]}")
    except Exception as e:
        print(f"⚠️  Load error: {e}")
        print("Will use fresh DistilBERT weights")

    encoder.eval()
    return encoder


def integrate_with_v8():
    """Combine Reddit encoder + Cascade Gate v8."""
    import sys
    # Add parent dir to path
    sys.path.insert(0, '/Users/Subho/funny-strength-predictor')

    # Import v8 architecture
    sys.path.insert(0, '/Users/Subho/funny-strength-predictor/improvements')
    from enhanced_cascade import create_enhanced_model, count_parameters

    # Create v8 base model (text_proj expects 768-dim DistilBERT output)
    v8 = create_enhanced_model(
        text_dim=768,
        audio_dim=791,
        cross_dim=16,
        hidden=128
    )
    print(f"v8 base: {count_parameters(v8):,} params")

    return v8


def create_v8_inference_module():
    """Create the integrated v8 inference module — saved to drive_pipeline/."""
    code = '''#!/usr/bin/env python3
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
from transformers import DistilBertModel, AutoTokenizer


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
            print(f"Loaded Reddit pretrain. Missing={len(missing)}, Unexpected={len(unexpected)}")

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.dropout(pooled)  # (batch, 768)


class V8WithReddit(nn.Module):
    """Cascade Gate v8 + Reddit-pretrained text tower.

    Inputs:
      - text: (batch, seq_len) tokenized text
      - audio: (batch, seq_len, 791) audio features
      - cross: (batch, seq_len, 16) cross-modal features (optional)

    Returns:
      - scores: (batch, seq_len, 1)
      - text_conf: (batch, seq_len, 1) confidence
      - gate_weight: (batch, seq_len, 1)
    """
    def __init__(self, v8_model, reddit_state_path=None, freeze_text=True):
        super().__init__()
        self.text_encoder = RedditTextEncoder(reddit_state_path)
        self.v8 = v8_model  # Expects text features of dim 768

        # Freeze text encoder initially (only fine-tune v8's gate/audio)
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, audio, cross=None, lengths=None):
        # Encode text through Reddit-trained DistilBERT
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch, 768)

        # Expand text to per-segment (broadcast same encoding across segments)
        # Or use a more sophisticated per-segment text encoding
        batch_size, seq_len, _ = audio.shape
        # For now, broadcast same text encoding across all segments
        text_per_seg = text_features.unsqueeze(1).expand(-1, seq_len, -1)
        # TODO: replace with actual per-segment text encoding

        return self.v8(text_per_seg, audio, cross, lengths)


# Quick smoke test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/Subho/funny-strength-predictor/improvements')
    from enhanced_cascade import create_enhanced_model

    v8 = create_enhanced_model(text_dim=768, audio_dim=791, cross_dim=16, hidden=128)
    model = V8WithReddit(v8, reddit_state_path=None)  # Will use DistilBERT defaults

    # Test forward
    batch, seq = 2, 20
    input_ids = torch.randint(0, 30000, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
    audio = torch.randn(batch, seq, 791)
    cross = torch.randn(batch, seq, 16)

    scores, conf, gate = model(input_ids, attn, audio, cross)
    print(f"Scores: {scores.shape}, Conf: {conf.shape}, Gate: {gate.shape}")
    print("✅ v8 inference module works")
'''
    out_path = Path('/Users/Subho/funny-strength-predictor/drive_pipeline/v8_with_reddit.py')
    with open(out_path, 'w') as f:
        f.write(code)
    print(f"✅ Created: {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("🔨 Building Cascade Gate v8 with Reddit Pretrain")
    print("=" * 60)

    # Step 1: Pull Reddit model
    reddit_path, temp_dir = pull_reddit_model()
    if not reddit_path:
        return

    # Step 2: Build text encoder
    encoder = build_text_extractor(reddit_path)
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")

    # Step 3: Integrate with v8
    v8_model = integrate_with_v8()

    # Step 4: Create integrated inference module
    v8_module = create_v8_inference_module()

    # Push v8 module to Drive
    print(f"\n📤 Pushing v8 module to Drive...")
    subprocess.run(
        ['rclone', 'copyto', str(v8_module),
         f'{DRIVE_BASE}/models/v8_with_reddit.py',
         '--retries=3', '--retries-sleep=10s'],
        capture_output=False
    )

    # Save integration config
    config = {
        'reddit_model': str(reddit_path.name),
        'reddit_state_keys': 'bb.distilbert.* (DistilBERT backbone)',
        'v8_params': 1189462,
        'integration_approach': '''
1. Reddit-pretrained DistilBERT outputs 768-dim text features
2. v8.text_proj projects 768→128 (existing in v8 architecture)
3. Cascade Gate fuses text (128) + audio (128) + cross-attended
4. Fine-tune end-to-end on 641 standup files

Key benefit: Reddit pretrain transfers humor-style knowledge
(upvotes as continuous funniness proxy).
''',
        'next_steps': [
            'Wait for full Reddit training (30K samples) to finish',
            'Re-pull updated model',
            'Fine-tune v8 on 641 standup files',
            'Evaluate on gold laughter + pseudo-labels',
            'Update v7 cascade gate to v8 in HaHaScore repo',
            'Push v8 to HuggingFace',
        ]
    }
    config_path = temp_dir / "v8_integration.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(
        ['rclone', 'copyto', str(config_path),
         f'{DRIVE_BASE}/results/v8_integration.json',
         '--retries=3', '--retries-sleep=10s'],
        capture_output=False
    )

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n✅ v8 integration module created and pushed to Drive")
    print("📁 Location: gdrive:/HaHaScore_Pretrain/models/v8_with_reddit.py")


if __name__ == "__main__":
    main()
