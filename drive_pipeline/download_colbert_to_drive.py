#!/usr/bin/env python3
"""
Download ColBERT 200K humor detection dataset → Google Drive
============================================================

200K balanced humorous/non-humorous short texts from ColBERT paper.
CC-BY-2.0 license. Designed to defeat length/format shortcuts in humor detection.

Output: gdrive:/HaHaScore_Pretrain/colbert/
"""
import os
import subprocess
from pathlib import Path

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain/colbert"


def download_colbert():
    """Download ColBERT 200K from HuggingFace."""
    print("=" * 60)
    print("📥 Downloading CreativeLang/ColBERT_Humor_Detection")
    print("=" * 60)
    print(f"Source: HF dataset (CC-BY-2.0)")
    print(f"Target: {DRIVE_BASE}/")

    temp_dir = Path("/tmp/colbert_download")
    temp_dir.mkdir(exist_ok=True)

    code = """
import os
os.environ['HF_HOME'] = '/tmp/hf_cache'
from datasets import load_dataset

print('Loading CreativeLang/ColBERT_Humor_Detection...')
ds = load_dataset('CreativeLang/ColBERT_Humor_Detection')
print(f'Dataset splits: {list(ds.keys())}')

for split_name, split_data in ds.items():
    print(f'  {split_name}: {len(split_data):,} samples')
    df = split_data.to_pandas()
    out_path = f'/tmp/colbert_download/colbert_{split_name}.csv'
    df.to_csv(out_path, index=False)
    print(f'  Saved to {out_path}')

# Print column schema
print(f'\\nColumns: {list(ds[list(ds.keys())[0]].column_names)}')
print('Sample row:')
sample = ds[list(ds.keys())[0]][0]
for k, v in sample.items():
    print(f'  {k}: {str(v)[:100]}')
"""
    script = temp_dir / "download.py"
    with open(script, "w") as f:
        f.write(code)

    result = subprocess.run(
        ["python3", str(script)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False

    # Push to Drive
    print(f"\n📤 Pushing to Drive: {DRIVE_BASE}/")
    for csv_file in temp_dir.glob("*.csv"):
        print(f"  Pushing {csv_file.name}...")
        result = subprocess.run(
            ["rclone", "copy", str(csv_file), f"{DRIVE_BASE}/", "--progress"],
            capture_output=True, text=True
        )
        print(result.stdout[-500:] if result.stdout else "")

    # Cleanup
    subprocess.run(["rm", "-rf", str(temp_dir)], capture_output=True)

    # Verify
    print(f"\n✅ Drive contents:")
    subprocess.run(["rclone", "ls", DRIVE_BASE], capture_output=False)
    return True


if __name__ == "__main__":
    download_colbert()
