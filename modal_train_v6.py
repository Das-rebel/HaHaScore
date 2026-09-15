"""
HaHaScore v6 Modal Training
============================
GPU: Tesla T4 (16GB) — confirmed working on Modal
Run: modal run modal_train_v6.py

Week 1 Goals:
1. HuBERT vs WavLM head-to-head comparison (D-G2)
2. Kinetic energy extraction from videos (D-G3)
"""

import modal
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# APP & IMAGE
# ══════════════════════════════════════════════════════════════════════════════
app = modal.App(
    "hahascore-v6",
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "transformers>=4.36",
        "datasets>=2.14",
        "accelerate>=0.25",
        "scikit-learn>=1.3",
        "librosa>=0.10",
        "torchaudio>=2.0",
        "pandas>=2.0",
        "numpy>=1.24",
        "huggingface_hub>=0.19",
        "tqdm",
        "ultralytics",
    ),
)

GPU_CONFIG = "T4"  # 16GB, confirmed working

# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE HELPERS — run locally (no GPU needed)
# ══════════════════════════════════════════════════════════════════════════════
@app.function(timeout=300)
def extract_kinetic_from_videos(video_dir: str, output_path: str):
    """
    Extract kinetic energy features using YOLOv8s-pose.
    TIC-TALK finding: kinetic energy r=-0.75 with laughter (stillness = more laughter).
    """
    from ultralytics import YOLO
    import numpy as np
    from pathlib import Path
    from tqdm.auto import tqdm

    print("Loading YOLOv8s-pose...")
    model = YOLO("yolov8s-pose.pt")

    video_dir = Path(video_dir)
    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.avi", "*.mov"]:
        video_files.extend(video_dir.rglob(ext))

    print(f"Found {len(video_files)} videos")
    all_ids, all_feats = [], []

    for vp in tqdm(video_files, desc="Kinetic energy"):
        try:
            results = model(str(vp), verbose=False, device=0)
            keypoints_list = []
            for r in results:
                if r.keypoints is not None and r.keypoints.data is not None:
                    kp = r.keypoints.data.cpu().numpy()  # [N, 17, 3]
                    if len(kp) > 0:
                        keypoints_list.append(kp[0])

            if len(keypoints_list) < 2:
                continue

            kp_arr = np.array(keypoints_list)           # [T, 17, 3]
            kp_xy  = kp_arr[:, :, :2]                   # [T, 17, 2]
            velocity = np.diff(kp_xy, axis=0)             # [T-1, 17, 2]
            rms_vel = np.sqrt((velocity**2).sum(axis=(1,2)))  # [T-1]
            kinetic = 0.5 * rms_vel**2                    # mass=1

            feat = np.array([
                kinetic.mean(), kinetic.std(), kinetic.max(),
                np.percentile(kinetic, 25), np.percentile(kinetic, 75),
                np.median(kinetic), kinetic.sum(),
            ], dtype=np.float32)

            all_ids.append(vp.stem)
            all_feats.append(feat)
        except Exception as e:
            print(f"Error on {vp}: {e}")
            continue

    if all_feats:
        feat_tensor = torch.tensor(np.array(all_feats), dtype=torch.float32)
        torch.save({"features": feat_tensor, "ids": all_ids}, output_path)
        print(f"\nSaved {feat_tensor.shape[0]} kinetic features → {output_path}")
    else:
        print("No features extracted!")

    return {"n_videos": len(video_files), "n_features": len(all_feats)}


@app.function(timeout=600)
def extract_audio_features_modal(
    video_dir: str,
    output_path: str,
    model_name: str = "microsoft/wavlm-base-plus",
):
    """
    Extract audio SSL features from videos.
    Supports: microsoft/wavlm-base-plus, facebook/hubert-base
    """
    import numpy as np
    from pathlib import Path
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoFeatureExtractor
    import subprocess

    print(f"Loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.eval()

    video_dir = Path(video_dir)
    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.avi", "*.mov"]:
        video_files.extend(video_dir.rglob(ext))

    print(f"Found {len(video_files)} videos")
    all_ids, all_feats = [], []

    for vp in tqdm(video_files, desc=f"Audio ({model_name})"):
        try:
            audio_path = f"/tmp/{vp.stem}.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(vp),
                "-ac", "1", "-ar", "16000", "-q:a", "0", audio_path
            ], capture_output=True, timeout=60)

            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            inputs = feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state   # [B, T, D]
                feat = hidden.mean(dim=1).squeeze(0).cpu()  # [D]

            all_ids.append(vp.stem)
            all_feats.append(feat)
            os.remove(audio_path)
        except Exception as e:
            print(f"Error on {vp}: {e}")
            continue

    if all_feats:
        feat_tensor = torch.stack(all_feats)
        torch.save({"features": feat_tensor, "ids": all_ids, "model": model_name}, output_path)
        print(f"\nSaved {feat_tensor.shape} → {output_path}")

    return {"n_videos": len(video_files), "n_features": len(all_feats), "model": model_name}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
class BilinearFusion(nn.Module):
    """v5 bilinear fusion architecture."""
    def __init__(self, text_dim=768, audio_dim=512, fusion_dim=256):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.bilinear   = nn.Bilinear(fusion_dim, fusion_dim, 1)
        self.dropout    = nn.Dropout(0.3)
        self.text_norm  = nn.LayerNorm(fusion_dim)
        self.audio_norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_feat, audio_feat):
        t = self.dropout(self.text_norm(F.gelu(self.text_proj(text_feat))))
        a = self.dropout(self.audio_norm(F.gelu(self.audio_proj(audio_feat))))
        return self.bilinear(t, a)


class CrossAttentionFusion(nn.Module):
    """DARC-CLIP inspired cross-attention fusion."""
    def __init__(self, text_dim=768, audio_dim=512, fusion_dim=256, num_heads=4):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 4, fusion_dim),
        )
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, 1), nn.Sigmoid())
        self.head = nn.Linear(fusion_dim, 1)

    def forward(self, text_feat, audio_feat):
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        ca_out, _ = self.cross_attn(t.unsqueeze(1), a.unsqueeze(1), a.unsqueeze(1))
        ca_out = ca_out.squeeze(1)
        t = self.norm1(t + ca_out)
        t = self.norm2(t + self.ffn(t))
        gate_val = self.gate(torch.cat([t, a], dim=-1))
        fused = gate_val * t + (1 - gate_val) * a
        return self.head(fused)


class TriModalFusion(nn.Module):
    """Text + Audio + Kinetic energy fusion."""
    def __init__(self, text_dim=768, audio_dim=512, kinetic_dim=7, fusion_dim=256):
        super().__init__()
        self.text_proj    = nn.Linear(text_dim, fusion_dim)
        self.audio_proj   = nn.Linear(audio_dim, fusion_dim)
        self.kinetic_proj = nn.Linear(kinetic_dim, fusion_dim)
        self.bilinear     = nn.Bilinear(fusion_dim, fusion_dim, 1)
        self.kinetic_gate = nn.Sequential(
            nn.Linear(fusion_dim + kinetic_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.head = nn.Linear(fusion_dim, 1)

    def forward(self, text_feat, audio_feat, kinetic_feat):
        t = F.gelu(self.text_proj(text_feat))
        a = F.gelu(self.audio_proj(audio_feat))
        k = F.gelu(self.kinetic_proj(kinetic_feat))
        bi = self.bilinear(t, a)
        gate_in = torch.cat([bi, k], dim=-1)
        gate = self.kinetic_gate(gate_in)
        modulated = gate * t + (1 - gate) * a
        out = self.ffn(modulated)
        return self.head(out)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
@app.function(gpu=GPU_CONFIG, timeout=3600)
def train_hahascore_v6(
    text_features_path: str,
    audio_features_path: str,
    kinetic_features_path: str = None,
    output_dir: str = "/models",
    n_epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-4,
    k_folds: int = 5,
):
    """
    Full v6 training with 5-fold CV.
    Compares: Bilinear vs Cross-Attention vs TriModal
    """
    import json
    import numpy as np
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"HaHaScore v6 Training — {GPU_CONFIG}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print(f"{'='*60}")

    # Load features
    print(f"\nLoading features...")
    text_data  = torch.load(text_features_path, weights_only=False)
    audio_data = torch.load(audio_features_path, weights_only=False)

    text_feats  = text_data["features"].float()
    audio_feats = audio_data["features"].float()
    audio_model = audio_data.get("model", "unknown")
    print(f"Text:  {text_feats.shape}")
    print(f"Audio: {audio_feats.shape} ({audio_model})")

    # Load labels
    labels_path = "/tmp/sentence_labels.pt"
    try:
        labels_data = torch.load(labels_path, weights_only=False)
        labels = labels_data["labels"].float()
        print(f"Labels: {len(labels)}")
    except:
        print("WARNING: No labels. Using scaffold mode.")
        labels = torch.rand(min(len(text_feats), len(audio_feats))) > 0.5

    n = min(len(text_feats), len(audio_feats), len(labels))
    text_feats  = text_feats[:n]
    audio_feats = audio_feats[:n]
    labels      = labels[:n].float()

    audio_dim = 768 if audio_feats.shape[1] == 768 else 512
    text_dim  = text_feats.shape[1]

    # Cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []

    print(f"\n{k_folds}-fold CV: Bilinear vs Cross-Attention vs TriModal\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(text_feats, labels)):
        print(f"--- Fold {fold+1}/{k_folds} ---")
        train_t  = text_feats[train_idx].to(DEVICE)
        train_a  = audio_feats[train_idx].to(DEVICE)
        train_l  = labels[train_idx].to(DEVICE)
        val_t    = text_feats[val_idx].to(DEVICE)
        val_a    = audio_feats[val_idx].to(DEVICE)
        val_l    = labels[val_idx].to(DEVICE)

        fold_results = {}

        # Bilinear (v5 baseline)
        model_bi = BilinearFusion(text_dim=text_dim, audio_dim=audio_dim).to(DEVICE)
        bi_auc = _train_and_eval(model_bi, train_t, train_a, None, val_t, val_a, None, val_l, n_epochs, lr, batch_size)
        fold_results["bilinear"] = bi_auc
        print(f"  Bilinear AUC: {bi_auc:.4f}")

        # Cross-Attention
        model_ca = CrossAttentionFusion(text_dim=text_dim, audio_dim=audio_dim).to(DEVICE)
        ca_auc = _train_and_eval(model_ca, train_t, train_a, None, val_t, val_a, None, val_l, n_epochs, lr, batch_size)
        fold_results["crossattn"] = ca_auc
        print(f"  CrossAttn AUC: {ca_auc:.4f}")

        # TriModal (if kinetic)
        if kinetic_features_path:
            kdata = torch.load(kinetic_features_path, weights_only=False)
            kfeats = kdata["features"].float()[:n].to(DEVICE)
            model_tm = TriModalFusion(text_dim=text_dim, audio_dim=audio_dim, kinetic_dim=7).to(DEVICE)
            tm_auc = _train_and_eval(model_tm, train_t, train_a, kfeats[train_idx], val_t, val_a, kfeats[val_idx], val_l, n_epochs, lr, batch_size)
            fold_results["trimodal"] = tm_auc
            print(f"  TriModal AUC: {tm_auc:.4f}")

        results.append(fold_results)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name in list(results[0].keys()):
        aucs = [r[name] for r in results]
        print(f"{name:>12s}: AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    best = max(results[0].keys(), key=lambda n: np.mean([r[n] for r in results]))
    print(f"\nBest: {best} ({np.mean([r[best] for r in results]):.4f})")

    summary = {"results": results, "best": best, "audio_model": audio_model, "n_samples": n}
    with open(f"{output_dir}/v6_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _train_and_eval(model, train_t, train_a, train_k, val_t, val_a, val_k, val_l, n_epochs, lr, bs):
    """Train one model for one fold. Returns val AUC."""
    DEVICE = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.BCEWithLogitsLoss()

    # Pad kinetic to zeros if None
    n_train = len(train_t)
    if train_k is None:
        train_k = torch.zeros(n_train, 7, device=DEVICE)
    if val_k is None:
        val_k = torch.zeros(len(val_t), 7, device=DEVICE)

    dataset = torch.utils.data.TensorDataset(train_t, train_a, train_k)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    best_auc = 0
    best_state = None
    patience, no_improve = 5, 0

    for epoch in range(n_epochs):
        model.train()
        for bt, ba, bk in loader:
            bt, ba, bk = bt.to(DEVICE), ba.to(DEVICE), bk.to(DEVICE)
            optimizer.zero_grad()
            logits = (model(bt, ba, bk) if bk.sum() > 0 else model(bt, ba)).squeeze(-1)
            loss = criterion(logits, torch.ones_like(logits) * 0.5)  # Placeholder — real labels needed
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        model.eval()
        with torch.no_grad():
            preds = (model(val_t, val_a, val_k) if val_k.sum() > 0 else model(val_t, val_a)).squeeze(-1)
            probs = torch.sigmoid(preds)
            try:
                auc = roc_auc_score(val_l.cpu().numpy(), probs.cpu().numpy())
            except:
                auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}: auc={auc:.4f}")

        if no_improve >= patience:
            print(f"    Early stop @ epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_auc


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           HaHaScore v6 — Modal Training Pipeline             ║
╠══════════════════════════════════════════════════════════════╣
║  WEEK 1:                                                  ║
║  1. Kinetic energy extraction (YOLOv8s-pose)  → r=-0.75   ║
║  2. HuBERT vs WavLM head-to-head                          ║
║  3. Bilinear vs Cross-Attention 5-fold CV                ║
║  TARGET: AUC ≥ 0.70 (from 0.632 ± 0.007)                ║
╚══════════════════════════════════════════════════════════════╝
    Scaffold ready. Sync feature files to Modal volume, then run.
    """)
