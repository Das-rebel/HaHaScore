#!/usr/bin/env python3
"""
Download UR-FUNNY multimodal dataset → Google Drive
====================================================

UR-FUNNY is the only public multimodal humor dataset (text+audio+video).
Licensed MIT. Contains 1,866 TED punchline samples.

Routes features directly to Google Drive to preserve local disk space.

Output: gdrive:/HaHaScore_Pretrain/ur_funny/
"""
import os
import subprocess
import argparse
from pathlib import Path

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain/ur_funny"
GITHUB_URL = "https://github.com/ROC-HCI/UR-FUNNY.git"


def check_git():
    """Check git availability."""
    result = subprocess.run(["which", "git"], capture_output=True, text=True)
    return result.returncode == 0


def clone_metadata():
    """Clone UR-FUNNY repo (metadata + text only — videos on YouTube)."""
    temp_dir = Path("/tmp/ur_funny_clone")
    if temp_dir.exists():
        print(f"✅ Already cloned: {temp_dir}")
        return temp_dir

    print(f"⏳ Cloning UR-FUNNY repo (metadata + text labels)...")
    result = subprocess.run(
        ["git", "clone", GITHUB_URL, str(temp_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Clone error: {result.stderr}")
        return None

    # Explore structure
    print(f"\n📂 UR-FUNNY repo structure:")
    subprocess.run(["find", str(temp_dir), "-maxdepth", "2", "-type", "f"],
                   capture_output=False)
    return temp_dir


def extract_features_to_drive(repo_dir: Path):
    """
    Extract features from UR-FUNNY (text + audio where available)
    and stream to Drive.

    UR-FUNNY only provides video IDs and YouTube URLs.
    Audio extraction requires YouTube downloads which are blocked from India IP.
    We extract text features (punchline + context) and push to Drive.
    """
    out_csv = repo_dir / "ur_funny_text_metadata.csv"

    # Find the data file
    data_files = list(repo_dir.glob("**/*.csv")) + list(repo_dir.glob("**/*.json"))
    if not data_files:
        print(f"⚠️ No data files found in {repo_dir}")
        print(f"   Will need to extract from README + scripts")
        return None

    print(f"📄 Found data files:")
    for f in data_files:
        print(f"   - {f.relative_to(repo_dir)}")

    # Try to extract punchline + context info
    print(f"\n⏳ Extracting text metadata to {out_csv}")

    code = f"""
import pandas as pd
import json
from pathlib import Path

repo = Path('{repo_dir}')
all_records = []

# Look for csv/json files with punchline data
for f in repo.rglob('*'):
    if f.suffix in ['.csv', '.json']:
        try:
            if f.suffix == '.csv':
                df = pd.read_csv(f)
                print(f'{{f.name}}: {{len(df)}} rows, columns: {{list(df.columns)}}')
                df['_source_file'] = f.relative_to(repo).as_posix()
                all_records.append(df)
            elif f.suffix == '.json':
                with open(f) as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                    print(f'{{f.name}}: {{len(df)}} rows')
                    df['_source_file'] = f.relative_to(repo).as_posix()
                    all_records.append(df)
        except Exception as e:
            print(f'⚠️ Skipping {{f.name}}: {{e}}')

if all_records:
    combined = pd.concat(all_records, ignore_index=True)
    combined.to_csv('{out_csv}', index=False)
    print(f'Saved {{len(combined):,}} records to {{out_csv}}')
else:
    print('No records extracted')
"""
    script = repo_dir / "extract.py"
    with open(script, "w") as f:
        f.write(code)
    result = subprocess.run(
        ["python3", str(script)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return None

    return out_csv


def push_metadata_to_drive(csv_file: Path):
    """Push the metadata CSV to Drive."""
    if not csv_file or not csv_file.exists():
        print("⚠️ No CSV to push")
        return False

    print(f"\n📤 Pushing to Drive: {DRIVE_BASE}/")
    result = subprocess.run(
        ["rclone", "copy", str(csv_file), f"{DRIVE_BASE}/", "--progress"],
        capture_output=True, text=True
    )
    print(result.stdout[-1500:] if result.stdout else "")

    # Also push README + scripts for reference
    repo = csv_file.parent
    for ext in ["*.md", "*.txt", "*.py"]:
        for f in repo.glob(ext):
            subprocess.run(
                ["rclone", "copy", str(f), f"{DRIVE_BASE}/", "--progress"],
                capture_output=True, text=True
            )

    print(f"\n✅ Drive contents:")
    subprocess.run(["rclone", "ls", DRIVE_BASE], capture_output=False)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download UR-FUNNY to Drive")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        subprocess.run(["rclone", "ls", DRIVE_BASE])
        return

    if not check_git():
        print("❌ git not available")
        return

    repo = clone_metadata()
    if repo:
        csv = extract_features_to_drive(repo)
        push_metadata_to_drive(csv)


if __name__ == "__main__":
    main()
