#!/usr/bin/env python3
"""Export v10 (Bilinear + CORAL Multi-task) to ONNX."""
import os
os.environ['HF_HOME'] = '/tmp/hf_distilbert'
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, '/Users/Subho/funny-strength-predictor')
from cascade_train_v9 import CascadeGateFusionV9

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain"


class V10Inference(nn.Module):
    """Wraps v10 for ONNX inference — outputs humor score (primary)."""

    def __init__(self):
        super().__init__()
        # Load v9 backbone (v10 = v9 + multi-task heads)
        # We'll just use the v9 backbone since the humor prediction
        # comes from the v9 forward + humor head
        self.v9_backbone = CascadeGateFusionV9(text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3)
        self.humor_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, text, audio):
        batch_size, seq_len, _ = text.shape

        # v9 backbone forward (capture features before head)
        text_h = self.v9_backbone.text_proj(text)
        audio_h = self.v9_backbone.audio_proj(audio)
        bilinear_out = self.v9_backbone.bilinear(text_h, audio_h)
        text_conf = torch.sigmoid(self.v9_backbone.text_confidence(text_h))
        gated_audio = audio_h * text_conf
        text_attn_out, _ = self.v9_backbone.audio_to_text_attn(audio_h, text_h, text_h)

        fused = torch.cat([text_h, bilinear_out, text_attn_out], dim=-1)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.v9_backbone.pos_embedding(positions)
        fused = torch.cat([fused, pos_emb], dim=-1)

        temporal_output, _ = self.v9_backbone.gru(fused)

        # Humor head
        humor = self.humor_head(temporal_output).squeeze(-1)

        # Return: scores, text_confidence, gate_weight (dummy)
        return humor, text_conf.squeeze(-1), text_conf.squeeze(-1)


def pull_v10_model(remote_name="v10_bilinear_coral.pt"):
    temp = Path('/tmp/v10_pull')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    subprocess.run(
        ['rclone', 'copyto', f'{DRIVE_BASE}/models/{remote_name}',
         str(temp / 'v10.pt'), '--retries=3', '--retries-sleep=10s'],
        capture_output=True, text=True, timeout=120
    )
    path = temp / 'v10.pt'
    if path.exists():
        print(f"✅ Pulled {path.stat().st_size/1e6:.1f} MB")
        return path
    return None


def main():
    print("=" * 60)
    print("📦 ONNX Export: v10 Bilinear + CORAL Multi-task")
    print("=" * 60)

    v10_path = pull_v10_model()
    if not v10_path:
        return False

    print("📦 Loading v10 weights...")
    state = torch.load(v10_path, map_location='cpu')
    full_state = state['model_state_dict']
    humor_auc = state.get('humor_auc', 'unknown')
    laugh_auc = state.get('laugh_auc', 'unknown')
    print(f"   Source AUC: humor={humor_auc}, gold={laugh_auc}")

    # Build inference model
    model = V10Inference()
    # Load only backbone weights (v9) and humor head
    backbone_state = {k.replace('backbone.', ''): v for k, v in full_state.items() if k.startswith('backbone.')}
    head_state = {k.replace('humor_head.', ''): v for k, v in full_state.items() if k.startswith('humor_head.')}

    # Load backbone
    missing, unexpected = model.v9_backbone.load_state_dict(backbone_state, strict=False)
    print(f"   Backbone: missing={len(missing)}, unexpected={len(unexpected)}")
    # Load humor head
    if head_state:
        # The head is a Sequential with indices
        head_renamed = {f"{i}": v for i, (k, v) in enumerate(head_state.items())}
        try:
            model.humor_head.load_state_dict(head_state)
            print(f"   Humor head loaded: {len(head_state)} tensors")
        except Exception as e:
            print(f"   Humor head load error: {e}")

    model.eval()

    out_dir = Path('/tmp/v10_onnx_export')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # Dummy inputs
    batch, seq = 1, 20
    dummy_text = torch.randn(batch, seq, 768)
    dummy_audio = torch.randn(batch, seq, 791)

    onnx_path = out_dir / 'v10_bilinear_cascade.onnx'

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_text, dummy_audio),
            str(onnx_path),
            input_names=['text', 'audio'],
            output_names=['scores', 'text_confidence', 'gate_weight'],
            dynamic_axes={
                'text': {0: 'batch'},
                'audio': {0: 'batch'},
                'scores': {0: 'batch'},
                'text_confidence': {0: 'batch'},
                'gate_weight': {0: 'batch'},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    print(f"✅ Exported: {onnx_path} ({onnx_path.stat().st_size/1e6:.2f} MB)")

    # Quantize
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = out_dir / 'v10_bilinear_cascade_int8.onnx'
        quantize_dynamic(str(onnx_path), str(int8_path),
                         weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul', 'Gemm'])
        print(f"✅ Quantized: {int8_path} ({int8_path.stat().st_size/1e6:.2f} MB)")
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        int8_path = None

    # Verify
    try:
        import onnxruntime as ort
        import time
        session = ort.InferenceSession(str(onnx_path))
        print(f"\nVerification:")
        print(f"   Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"   Outputs: {[o.name for o in session.get_outputs()]}")

        text = np.random.randn(1, 20, 768).astype(np.float32)
        audio = np.random.randn(1, 20, 791).astype(np.float32)

        t0 = time.time()
        for _ in range(10):
            outputs = session.run(None, {'text': text, 'audio': audio})
        elapsed = (time.time() - t0) / 10 * 1000
        print(f"   Avg inference: {elapsed:.2f}ms")
        print(f"   Score range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
    except Exception as e:
        print(f"⚠️ Verification failed: {e}")

    # Push to Drive
    print(f"\n📤 Pushing to Drive...")
    subprocess.run(
        ['rclone', 'copyto', str(onnx_path),
         f'{DRIVE_BASE}/models/v10_bilinear_cascade.onnx',
         '--drive-chunk-size=64M', '--retries=3'],
        capture_output=False
    )
    if int8_path and int8_path.exists():
        subprocess.run(
            ['rclone', 'copyto', str(int8_path),
             f'{DRIVE_BASE}/models/v10_bilinear_cascade_int8.onnx',
             '--drive-chunk-size=64M', '--retries=3'],
            capture_output=False
        )

    # Metadata
    metadata = {
        'source_model': 'v10_bilinear_coral.pt',
        'architecture': 'v9 Bilinear Fusion + CORAL Multi-task',
        'humor_auc': humor_auc,
        'laugh_auc': laugh_auc,
        'opset': 14,
    }
    meta_path = out_dir / 'v10_bilinear_cascade_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    subprocess.run(
        ['rclone', 'copyto', str(meta_path),
         f'{DRIVE_BASE}/results/v10_bilinear_cascade_metadata.json',
         '--retries=3'],
        capture_output=False
    )

    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(v10_path.parent, ignore_errors=True)
    print(f"\n✅ v10 ONNX exported to Drive")
    return True


if __name__ == "__main__":
    main()
