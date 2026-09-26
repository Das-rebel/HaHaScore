#!/usr/bin/env python3
"""
Drive Pipeline Status Check
============================
Quick check of what's on Google Drive for HaHaScore pretraining.
"""
import subprocess
import sys

DRIVE_BASE = "gdrive:/HaHaScore_Pretrain"


def check_folder(subfolder):
    """Check contents of a subfolder."""
    path = f"{DRIVE_BASE}/{subfolder}"
    result = subprocess.run(
        ["rclone", "ls", path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"  ❌ Error: {result.stderr}"
    files = result.stdout.strip()
    if not files:
        return f"  (empty)"
    return files


def main():
    print("=" * 60)
    print(f"📂 HaHaScore Drive Pipeline Status")
    print(f"   Base: {DRIVE_BASE}")
    print("=" * 60)

    # Top level
    print("\n📁 Top-level folders:")
    subprocess.run(["rclone", "lsd", DRIVE_BASE], capture_output=False)

    # Check each subfolder
    for subfolder in ["reddit", "ur_funny", "colbert", "models", "results"]:
        print(f"\n📁 {DRIVE_BASE}/{subfolder}/:")
        print(check_folder(subfolder))

    # Local disk usage
    print("\n💾 Local disk status:")
    result = subprocess.run(["df", "-h", "/Users/Subho"], capture_output=True, text=True)
    print(result.stdout)

    print("\n✅ Status check complete")


if __name__ == "__main__":
    main()
