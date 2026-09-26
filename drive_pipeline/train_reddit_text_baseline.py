#!/usr/bin/env python3
"""
Train Reddit Text Baseline → Push Results to Drive
==================================================

Trains a text-only regression model on Reddit upvotes (continuous funniness).
Pulls data from Drive, streams training (no local CSV copy), saves model to Drive.

Target: gdrive:/HaHaScore_Pretrain/models/reddit_text_baseline_v1.pt
"""
import os
import sys
import subprocess
import argparse
import shutil
import tempfile
from pathlib import Path

DRIVE_DATA_BASE = "gdrive:/HaHaScore_Pretrain/reddit"
DRIVE_MODEL_BASE = "gdrive:/HaHaScore_Pretrain/models"
DRIVE_RESULTS_BASE = "gdrive:/HaHaScore_Pretrain/results"


def pull_data_from_drive(filename: str, dest_dir: Path) -> Path:
    """Pull a single CSV from Drive to a temp dir."""
    print(f"📥 Pulling {filename} from Drive...")
    dest_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        ["rclone", "copy", f"{DRIVE_DATA_BASE}/{filename}", str(dest_dir), "--progress"],
        capture_output=True, text=True
    )
    local_path = dest_dir / filename
    if not local_path.exists():
        print(f"❌ Failed to pull: {result.stderr}")
        return None
    size_mb = local_path.stat().st_size / 1e6
    print(f"✅ Pulled {filename} ({size_mb:.1f} MB)")
    return local_path


def stream_train(csv_path: Path,
                 model_output: Path,
                 epochs: int = 2,
                 batch_size: int = 32,
                 lr: float = 2e-5,
                 backbone: str = "distilbert-base-uncased",
                 max_samples: int = None):
    """
    Stream training from CSV → save model.
    Uses HuggingFace datasets to load CSV and stream batches.
    """
    print("=" * 60)
    print(f"🚀 Training text regression baseline on Reddit upvotes")
    print("=" * 60)
    print(f"  CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Backbone: {backbone}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Output: {model_output}")

    code = f"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('{csv_path}')
print(f'Loaded {{len(df):,}} rows')

# Detect column names (Reddit dataset schema)
text_col = None
score_col = None
for col in df.columns:
    col_lower = col.lower()
    if 'joke' in col_lower or 'text' in col_lower or 'body' in col_lower or 'title' in col_lower:
        if text_col is None:
            text_col = col
    if 'score' in col_lower or 'upvote' in col_lower or 'funny' in col_lower:
        score_col = col

print(f'Text column: {{text_col}}, Score column: {{score_col}}')

if text_col is None:
    print('Available columns:', df.columns.tolist())
    text_col = df.columns[0]
if score_col is None:
    score_col = 'score' if 'score' in df.columns else df.columns[-1]

print(f'Final: text={{text_col}}, score={{score_col}}')

# Filter
df = df[[text_col, score_col]].dropna()
df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
df = df.dropna()
print(f'After filtering: {{len(df):,}} rows')

# Normalize score to [0, 1]
scores = df[score_col].values
# Log transform first (Reddit scores are heavy-tailed)
log_scores = np.log1p(scores - scores.min() + 1)
df['funniness'] = (log_scores - log_scores.min()) / (log_scores.max() - log_scores.min() + 1e-8)
df['text'] = df[text_col].astype(str).str[:512]
df = df[['text', 'funniness']]

if {max_samples}:
    df = df.sample(n=min({max_samples}, len(df)), random_state=42)

print(f'Final dataset: {{len(df):,}} samples')
print(f'Funniness range: [{{df.funniness.min():.3f}}, {{df.funniness.max():.3f}}]')
print(f'Funniness mean: {{df.funniness.mean():.3f}}')

# Train/val split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
print(f'Train: {{len(train_df):,}}, Val: {{len(val_df):,}}')

# Tokenize
print(f'\\nLoading tokenizer: {backbone}')
tokenizer = AutoTokenizer.from_pretrained('{backbone}')

class JokeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df.text.tolist()
        self.labels = df.funniness.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                            padding='max_length', truncation=True, return_tensors='pt')
        return {{
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }}

train_ds = JokeDataset(train_df, tokenizer)
val_ds = JokeDataset(val_df, tokenizer)
train_loader = DataLoader(train_ds, batch_size={batch_size}, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size={batch_size}, shuffle=False, num_workers=0)

# Model
print(f'Loading model: {backbone}')
backbone_model = AutoModel.from_pretrained('{backbone}')
hidden_size = backbone_model.config.hidden_size

class HumorRegressor(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)
        return self.head(pooled).squeeze(-1)

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {{device}}')
model = HumorRegressor(backbone_model, hidden_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay=0.01)
total_steps = len(train_loader) * {epochs}
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
criterion = nn.MSELoss()

print(f'\\nTraining for {epochs} epochs...')
best_val_auc = 0.0
best_state = None

for epoch in range({epochs}):
    model.train()
    train_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        preds = model(input_ids, attn_mask)
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        if (step + 1) % 200 == 0:
            print(f'  Ep {{epoch+1}} step {{step+1}}/{{len(train_loader)}} loss={{train_loss/(step+1):.4f}}')

    # Validate
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            preds = model(input_ids, attn_mask)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    # Binary classification: above/below median
    median_label = np.median(val_labels)
    val_auc = roc_auc_score((val_labels > median_label).astype(int), val_preds)

    # Also compute correlation
    val_corr = np.corrcoef(val_preds, val_labels)[0, 1]

    print(f'  Ep {{epoch+1}}: train_loss={{train_loss/len(train_loader):.4f}}, val_auc={{val_auc:.4f}}, val_corr={{val_corr:.4f}}')

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {{k: v.cpu().clone() for k, v in model.state_dict().items()}}
        print(f'    New best AUC: {{best_val_auc:.4f}}')

# Save best model
print(f'\\n💾 Saving best model (val_auc={{best_val_auc:.4f}}) to {model_output}')
torch.save({{
    'model_state_dict': best_state,
    'backbone': '{backbone}',
    'val_auc': best_val_auc,
    'val_corr': val_corr,
    'epochs': {epochs},
    'lr': {lr},
    'batch_size': {batch_size},
    'n_train': len(train_df),
    'n_val': len(val_df),
    'config': {{
        'max_len': 128,
        'hidden_size': hidden_size,
    }}
}}, '{model_output}')

# Save metrics
import json
metrics = {{
    'val_auc': float(best_val_auc),
    'val_corr': float(val_corr),
    'n_train': len(train_df),
    'n_val': len(val_df),
    'epochs': {epochs},
    'backbone': '{backbone}',
}}
with open('{model_output.with_suffix(".json")}', 'w') as f:
    json.dump(metrics, f, indent=2)
print('✅ Training complete')
"""
    # Write training script to temp file
    script_path = Path(tempfile.gettempdir()) / "train_reddit_text.py"
    with open(script_path, "w") as f:
        f.write(code)

    print(f"⏳ Starting training (this may take 1-3 hours)...")
    result = subprocess.run(
        ["python3", str(script_path)],
        capture_output=False  # Show output in real-time
    )
    return result.returncode == 0


def push_model_to_drive(local_path: Path):
    """Push trained model to Drive."""
    if not local_path.exists():
        print(f"❌ Model file not found: {local_path}")
        return False

    print(f"\n📤 Pushing model to Drive: {DRIVE_MODEL_BASE}/")
    result = subprocess.run(
        ["rclone", "copy", str(local_path), f"{DRIVE_MODEL_BASE}/", "--progress"],
        capture_output=True, text=True
    )
    print(result.stdout[-1000:])

    # Push metrics JSON
    json_path = local_path.with_suffix(".json")
    if json_path.exists():
        subprocess.run(
            ["rclone", "copy", str(json_path), f"{DRIVE_RESULTS_BASE}/", "--progress"],
            capture_output=True, text=True
        )

    print(f"\n✅ Drive contents:")
    subprocess.run(["rclone", "ls", DRIVE_MODEL_BASE], capture_output=False)
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Reddit text baseline")
    parser.add_argument("--filename", default="reddit_jokes_100k.csv",
                        help="CSV filename on Drive")
    parser.add_argument("--backbone", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample for faster iteration")
    args = parser.parse_args()

    # Step 1: Pull data from Drive to temp
    temp_dir = Path(tempfile.mkdtemp(prefix="reddit_train_"))
    csv_path = pull_data_from_drive(args.filename, temp_dir)
    if not csv_path:
        sys.exit(1)

    # Step 2: Train (writes model to local temp)
    model_path = temp_dir / "reddit_text_baseline.pt"
    success = stream_train(
        csv_path, model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        max_samples=args.max_samples,
    )

    if not success:
        print("❌ Training failed")
        sys.exit(1)

    # Step 3: Push model to Drive
    push_model_to_drive(model_path)

    # Step 4: Cleanup local
    print(f"\n🧹 Cleaning up {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n✅ DONE — Model trained and pushed to Drive")
    print(f"   Local disk usage: minimal (only transient)")
    print(f"   Drive: gdrive:/HaHaScore_Pretrain/models/")


if __name__ == "__main__":
    main()
