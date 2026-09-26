#!/usr/bin/env python3
"""
Export Cascade Gate v8 to ONNX for Fast HF Deployment
======================================================

Pipeline:
1. Pull v8_finetuned_full.pt from Drive
2. Extract just the v8 component (skip Reddit DistilBERT for speed)
3. Export to ONNX with dynamic batch/sequence dimensions
4. Apply INT8 dynamic quantization
5. Push to Drive for HF Space deployment

Target: <10ms inference on CPU, ~10MB quantized model
"""
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
sys.path.insert(0, '/Users/Subho/funny-strength-predictor/drive_pipeline')
sys.path.insert(0, '/Users/Subho/funny-strength-predictor/improvements')

from v8_with_reddit import V8WithReddit
from enhanced_cascade import create_enhanced_model

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain"


def pull_v8_model(remote_name="v8_finetuned_full.pt"):
    """Pull v8 model from Drive."""
    temp = Path('/tmp/v8_onnx_pull')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    print(f"📥 Pulling {remote_name}...")
    result = subprocess.run(
        ['rclone', 'copyto', f'{DRIVE_BASE}/models/{remote_name}',
         str(temp / 'v8.pt'), '--retries=3', '--retries-sleep=10s'],
        capture_output=True, text=True, timeout=120
    )
    path = temp / 'v8.pt'
    if path.exists():
        print(f"✅ Pulled {path.stat().st_size/1e6:.1f} MB")
        return path
    return None


def extract_v8_state(v8_path):
    """Extract just the v8 Cascade Gate component (skip Reddit DistilBERT)."""
    print(f"📦 Extracting v8 state...")
    state = torch.load(v8_path, map_location='cpu')
    full_state = state['model_state_dict']

    # Strip 'v8.' prefix to get just the cascade gate
    v8_state = {}
    for k, v in full_state.items():
        if k.startswith('v8.'):
            v8_state[k.replace('v8.', '')] = v

    print(f"   Full state: {len(full_state)} tensors")
    print(f"   v8 state: {len(v8_state)} tensors")
    return v8_state


def export_onnx(v8_state, output_path, hidden=128, audio_dim=791):
    """Export v8 to ONNX."""
    print(f"🔄 Exporting to ONNX...")

    # Recreate v8 architecture
    v8 = create_enhanced_model(
        text_dim=768,
        audio_dim=audio_dim,
        cross_dim=16,
        hidden=hidden
    )
    v8.load_state_dict(v8_state)
    v8.eval()

    # Dummy inputs
    batch_size, seq_len = 1, 20
    dummy_text = torch.randn(batch_size, seq_len, 768)
    dummy_audio = torch.randn(batch_size, seq_len, audio_dim)
    dummy_cross = torch.randn(batch_size, seq_len, 16)

    # Export
    torch.onnx.export(
        v8,
        (dummy_text, dummy_audio, dummy_cross),
        str(output_path),
        input_names=['text', 'audio', 'cross'],
        output_names=['scores', 'text_confidence', 'gate_weight'],
        dynamic_axes={
            'text': {0: 'batch', 1: 'sequence'},
            'audio': {0: 'batch', 1: 'sequence'},
            'cross': {0: 'batch', 1: 'sequence'},
            'scores': {0: 'batch', 1: 'sequence'},
            'text_confidence': {0: 'batch', 1: 'sequence'},
            'gate_weight': {0: 'batch', 1: 'sequence'},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✅ Exported: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
    return output_path


def quantize_int8(onnx_path, output_path):
    """Apply INT8 dynamic quantization."""
    print(f"⚙️  Applying INT8 dynamic quantization...")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=QuantType.QInt8,
            # Preserve sigmoid/dynamic ops
            op_types_to_quantize=['MatMul', 'Gemm', 'Conv']
        )
        print(f"✅ Quantized: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
        return output_path
    except ImportError:
        print("⚠️  onnxruntime not available, skipping quantization")
        return None


def verify_onnx(onnx_path):
    """Verify ONNX model works correctly."""
    print(f"🧪 Verifying ONNX model...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        print(f"   Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"   Outputs: {[o.name for o in session.get_outputs()]}")

        # Test inference
        batch, seq = 1, 20
        text = np.random.randn(batch, seq, 768).astype(np.float32)
        audio = np.random.randn(batch, seq, 791).astype(np.float32)
        cross = np.random.randn(batch, seq, 16).astype(np.float32)

        import time
        t0 = time.time()
        outputs = session.run(None, {'text': text, 'audio': audio, 'cross': cross})
        t1 = time.time()

        print(f"   Inference time: {(t1-t0)*1000:.2f}ms")
        print(f"   Output shapes: {[o.shape for o in outputs]}")
        print(f"   Score range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
        return True
    except ImportError:
        print("⚠️  onnxruntime not available for verification")
        return False


def benchmark_pytorch(v8_state, n_iterations=10):
    """Benchmark PyTorch model for comparison."""
    print(f"⏱️  Benchmarking PyTorch model...")
    v8 = create_enhanced_model(text_dim=768, audio_dim=791, cross_dim=16, hidden=128)
    v8.load_state_dict(v8_state)
    v8.eval()

    batch, seq = 1, 20
    text = torch.randn(batch, seq, 768)
    audio = torch.randn(batch, seq, 791)
    cross = torch.randn(batch, seq, 16)

    import time
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = v8(text, audio, cross)
        # Benchmark
        t0 = time.time()
        for _ in range(n_iterations):
            _ = v8(text, audio, cross)
        t1 = time.time()

    avg_ms = (t1 - t0) * 1000 / n_iterations
    print(f"   PyTorch avg: {avg_ms:.2f}ms ({n_iterations} iterations)")
    return avg_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='v8_finetuned_full.pt')
    p.add_argument('--output-name', default='v8_cascade_int8')
    p.add_argument('--no-quantize', action='store_true')
    args = p.parse_args()

    print("=" * 60)
    print(f"📦 ONNX Export: {args.model}")
    print("=" * 60)

    # Step 1: Pull model
    v8_path = pull_v8_model(args.model)
    if not v8_path:
        print("❌ Failed to pull model")
        return False

    # Step 2: Extract v8 state
    v8_state = extract_v8_state(v8_path)

    # Step 3: Benchmark PyTorch baseline
    pt_ms = benchmark_pytorch(v8_state)

    # Step 4: Export ONNX
    out_dir = Path('/tmp/v8_onnx_export')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    onnx_path = out_dir / f'{args.output_name}.onnx'
    export_onnx(v8_state, onnx_path)

    # Step 5: Quantize
    int8_path = None
    if not args.no_quantize:
        int8_path = out_dir / f'{args.output_name}_int8.onnx'
        int8_path = quantize_int8(onnx_path, int8_path)

    # Step 6: Verify
    verify_onnx(onnx_path)
    if int8_path and int8_path.exists():
        print(f"\n   INT8 model:")
        verify_onnx(int8_path)

    # Step 7: Push to Drive
    print(f"\n📤 Pushing to Drive...")

    subprocess.run(
        ['rclone', 'copyto', str(onnx_path),
         f'{DRIVE_BASE}/models/{args.output_name}.onnx',
         '--drive-chunk-size=64M', '--transfers=1', '--retries=3'],
        capture_output=False
    )

    if int8_path and int8_path.exists():
        subprocess.run(
            ['rclone', 'copyto', str(int8_path),
             f'{DRIVE_BASE}/models/{args.output_name}_int8.onnx',
             '--drive-chunk-size=64M', '--transfers=1', '--retries=3'],
            capture_output=False
        )

    # Save deployment metadata
    metadata = {
        'pytorch_avg_ms': pt_ms,
        'onnx_path': f'{args.output_name}.onnx',
        'int8_path': f'{args.output_name}_int8.onnx' if int8_path else None,
        'source_model': args.model,
        'architecture': 'EnhancedCascadeGateFusion (v8)',
        'inputs': ['text: float32 (batch, seq, 768)',
                  'audio: float32 (batch, seq, 791)',
                  'cross: float32 (batch, seq, 16)'],
        'outputs': ['scores: float32 (batch, seq, 1)',
                   'text_confidence: float32 (batch, seq, 1)',
                   'gate_weight: float32 (batch, seq, 1)'],
        'dynamic_axes': True,
        'opset': 17,
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

    # Cleanup
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(v8_path.parent, ignore_errors=True)
    print(f"\n✅ ONNX export complete:")
    print(f"   Drive: {DRIVE_BASE}/models/{args.output_name}.onnx")
    if int8_path:
        print(f"   INT8: {DRIVE_BASE}/models/{args.output_name}_int8.onnx")
    return True


if __name__ == "__main__":
    main()
