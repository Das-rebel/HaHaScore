#!/usr/bin/env python3
"""
Download Reddit Jokes → Google Drive (streaming, no local storage)
====================================================================

Routes the 1M Reddit jokes CSV directly to Google Drive to avoid
filling the local disk (currently 98% full).

Output: gdrive:/HaHaScore_Pretrain/reddit/reddit_jokes_*.csv
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain/reddit"


def download_reddit_full():
    """Download full 1M Reddit jokes via HF, stream to Drive."""
    print("=" * 60)
    print("📥 Downloading SocialGrep/one-million-reddit-jokes (HF)")
    print("=" * 60)
    print(f"Source: HF dataset (CC-BY-4.0, 2.6GB CSV)")
    print(f"Target: {DRIVE_BASE}/reddit_jokes_full.csv")

    # Step 1: Download to a temp file
    temp_dir = Path("/tmp/reddit_jokes_download")
    temp_dir.mkdir(exist_ok=True)
    temp_csv = temp_dir / "reddit_jokes_full.csv"

    if temp_csv.exists():
        print(f"✅ Already downloaded: {temp_csv} ({temp_csv.stat().st_size / 1e6:.1f} MB)")
    else:
        print("⏳ Downloading from HuggingFace (this may take 5-10 min)...")
        code = """
import os
os.environ['HF_HOME'] = '/tmp/hf_cache'
from datasets import load_dataset
print('Loading dataset...')
ds = load_dataset('SocialGrep/one-million-reddit-jokes')
df = ds['train'].to_pandas()
print(f'Loaded {len(df):,} rows')
df.to_csv('/tmp/reddit_jokes_download/reddit_jokes_full.csv', index=False)
print(f'Saved to /tmp/reddit_jokes_download/reddit_jokes_full.csv')
"""
        with open(temp_dir / "download.py", "w") as f:
            f.write(code)
        result = subprocess.run(
            ["python3", str(temp_dir / "download.py")],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False

    # Step 2: Stratified 100K sample (avoid copying 1M to Drive if not needed)
    sample_csv = temp_dir / "reddit_jokes_100k.csv"
    if not sample_csv.exists():
        print("⏳ Creating stratified 100K sample (top by score)...")
        code = f"""
import pandas as pd
df = pd.read_csv('{temp_csv}')
print(f'Full dataset: {{len(df):,}} rows')
print(f'Score range: [{{df.score.min()}}, {{df.score.max()}}]')

# Stratified sample: top 100K by score with diversity
df_top = df.nlargest(100_000, 'score')
df_top.to_csv('{sample_csv}', index=False)
print(f'Saved {{len(df_top):,}} stratified rows to {sample_csv}')
"""
        with open(temp_dir / "sample.py", "w") as f:
            f.write(code)
        result = subprocess.run(
            ["python3", str(temp_dir / "sample.py")],
            capture_output=True, text=True
        )
        print(result.stdout)

    # Step 3: Push to Google Drive
    print(f"\n📤 Pushing to Google Drive: {DRIVE_BASE}/")
    print("⏳ Streaming files to Drive (no local copy)...")

    # Push full CSV
    result = subprocess.run(
        ["rclone", "copy", str(temp_csv), f"{DRIVE_BASE}/", "--progress"],
        capture_output=True, text=True
    )
    print(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print(f"❌ rclone error: {result.stderr}")
        return False

    # Push stratified sample
    result = subprocess.run(
        ["rclone", "copy", str(sample_csv), f"{DRIVE_BASE}/", "--progress"],
        capture_output=True, text=True
    )
    print(result.stdout[-2000:] if result.stdout else "")

    # Step 4: Cleanup temp to free local disk
    print("\n🧹 Cleaning up local temp files...")
    subprocess.run(["rm", "-rf", str(temp_dir)], capture_output=True)

    # Verify on Drive
    print(f"\n✅ Drive contents:")
    subprocess.run(["rclone", "lsf", DRIVE_BASE], capture_output=False)
    subprocess.run(["rclone", "ls", DRIVE_BASE], capture_output=False)

    return True


def check_already_downloaded():
    """Check if files already exist on Drive."""
    result = subprocess.run(
        ["rclone", "ls", DRIVE_BASE],
        capture_output=True, text=True
    )
    files = result.stdout.strip()
    if "reddit_jokes_full.csv" in files:
        print(f"✅ reddit_jokes_full.csv already on Drive")
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Reddit jokes to Google Drive")
    parser.add_argument("--check-only", action="store_true",
                        help="Just check if files exist on Drive")
    args = parser.parse_args()

    if args.check_only:
        check_already_downloaded()
    else:
        download_reddit_full()
