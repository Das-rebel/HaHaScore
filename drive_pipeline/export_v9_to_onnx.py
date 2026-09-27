#!/usr/bin/env python3
"""Export v9 (CascadeGateFusionV9 with Bilinear) to ONNX."""
import os
os.environ['HF_HOME'] = '/tmp/hf_distilbert'
import argparse
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


def pull_v9_model(remote_name="v9_bilinear_v1.pt"):
    temp = Path('/tmp/v9_pull')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    subprocess.run(
        ['rclone', 'copyto', f'{DRIVE_BASE}/models/{remote_name}',
         str(temp / 'v9.pt'), '--retries=3', '--retries-sleep=10s'],
        capture_output=True, text=True, timeout=120
    )
    path = temp / 'v9.pt'
    if path.exists():
        print(f"✅ Pulled {path.stat().st_size/1e6:.1f} MB")
        return path
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='v9_bilinear_v1.pt')
    p.add_argument('--output-name', default='v9_bilinear_cascade')
    args = p.parse_args()

    print("=" * 60)
    print(f"📦 ONNX Export: {args.model}")
    print("=" * 60)

    v9_path = pull_v9_model(args.model)
    if not v9_path:
        return False

    print("📦 Extracting v9 state...")
    state = torch.load(v9_path, map_location='cpu')
    full_state = state.get('model_state_dict', state)

    # Load v9 model
    v9 = CascadeGateFusionV9(text_dim=768, audio_dim=791, hidden=128, num_layers=2, dropout=0.3)
    v9.load_state_dict(full_state)
    v9.eval()

    # Dummy inputs (v9 doesn't use cross_modal)
    batch, seq = 1, 20
    dummy_text = torch.randn(batch, seq, 768)
    dummy_audio = torch.randn(batch, seq, 791)

    out_dir = Path('/tmp/v9_onnx_export')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # Export v9 (without cross modal - we'll add zeros)
    onnx_path = out_dir / f'{args.output_name}.onnx'

    class V9Wrapper(nn.Module):
        """Wrap v9 to add cross_modal zero input for ONNX compatibility."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, text, audio):
            cross = torch.zeros(text.shape[0], text.shape[1], 16, device=text.device)
            scores, conf = self.model(text, audio)
            # Return scores, conf, dummy gate
            return scores, conf, conf

    wrapper = V9Wrapper(v9)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
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
    print(f"✅ Exported: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    # Quantize
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = out_dir / f'{args.output_name}_int8.onnx'
        quantize_dynamic(str(onnx_path), str(int8_path),
                         weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul', 'Gemm'])
        print(f"✅ Quantized: {int8_path} ({int8_path.stat().st_size/1e6:.1f} MB)")
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
         f'{DRIVE_BASE}/models/{args.output_name}.onnx',
         '--drive-chunk-size=64M', '--retries=3'],
        capture_output=False
    )
    if int8_path and int8_path.exists():
        subprocess.run(
            ['rclone', 'copyto', str(int8_path),
             f'{DRIVE_BASE}/models/{args.output_name}_int8.onnx',
             '--drive-chunk-size=64M', '--retries=3'],
            capture_output=False
        )

    # Metadata
    metadata = {
        'source_model': args.model,
        'architecture': 'CascadeGateFusionV9 with Bilinear Fusion',
        'val_auc': state.get('val_auc', 'unknown'),
        'opset': 14,
    }
    meta_path = out_dir / f'{args.output_name}_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    subprocess.run(
        ['rclone', 'copyto', str(meta_path),
         f'{DRIVE_BASE}/results/{args.output_name}_metadata.json',
         '--retries=3'],
        capture_output=False
    )

    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(v9_path.parent, ignore_errors=True)
    print(f"\n✅ v9 ONNX exported to Drive: {args.output_name}.onnx")


if __name__ == "__main__":
    main()
