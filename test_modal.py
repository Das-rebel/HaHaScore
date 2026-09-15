"""Modal GPU smoke test - uses cached hahascore-v6 image"""
import modal

# Reuse the hahascore-v6 app image (already has torch + transformers)
app = modal.App("hahascore-test")

@app.function(image=modal.Image.debian_slim().pip_install("torch", "numpy"), gpu="T4")
def test_gpu():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    return {"ok": True, "torch": torch.__version__, "cuda": torch.cuda.is_available()}

@app.local_entrypoint()
def main():
    result = test_gpu.remote()
    print(result)
