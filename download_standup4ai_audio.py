"""
Download StandUp4AI audio from GDrive for HaHaScore v6 feature extraction.
Uses rclone to pull .m4a files from gdrive:standup4ai/audio_1000/

Usage:
    python3 download_standup4ai_audio.py --max-files 50 --output-dir /tmp/standup4ai_audio
    python3 download_standup4ai_audio.py --youtube-id q112mLKiUCw --output-dir /tmp/standup4ai_audio
"""
import argparse
import os
import subprocess
import sys


def get_audio_files():
    """List all audio files available on GDrive."""
    result = subprocess.run(
        ['rclone', 'lsf', 'gdrive:standup4ai/audio_1000'],
        capture_output=True, text=True, check=True
    )
    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
    audio_files = {}
    for f in files:
        if f.endswith('.m4a'):
            # Format: YOUTUBE_ID[.lang].m4a or just YOUTUBE_ID.m4a
            parts = f.rsplit('.', 1)
            yt_id = parts[0].split(',')[0]  # remove language suffix
            lang = parts[0].split(',')[1] if ',' in parts[0] else 'en_uk'
            audio_files[yt_id] = {'file': f, 'lang': lang}
    return audio_files


def download_files(youtube_ids, output_dir, max_workers=4):
    """Download specified audio files from GDrive."""
    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0
    errors = 0
    
    # Build file list for rclone
    files_to_download = []
    for yt_id in youtube_ids:
        # Try audio_1000 first, then audio
        for gdrive_dir in ['standup4ai/audio_1000', 'standup4ai/audio']:
            result = subprocess.run(
                ['rclone', 'lsf', f'gdrive:{gdrive_dir}'],
                capture_output=True, text=True
            )
            matching = [line.strip() for line in result.stdout.strip().split('\n') 
                       if line.strip().startswith(yt_id) and line.strip().endswith('.m4a')]
            if matching:
                files_to_download.append(f"{gdrive_dir}/{matching[0]}")
                break
    
    print(f"Downloading {len(files_to_download)} files...")
    
    # Use rclone copy with file list
    for src in files_to_download:
        yt_id = src.split('/')[-1].rsplit('.', 1)[0].split(',')[0]
        dest = os.path.join(output_dir, f"{yt_id}.m4a")
        
        if os.path.exists(dest):
            print(f"  Already exists: {yt_id}")
            downloaded += 1
            continue
        
        result = subprocess.run(
            ['rclone', 'copy', f'gdrive:{src}', output_dir, '--transfers', '1', '-v'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ {yt_id}")
            downloaded += 1
        else:
            print(f"  ❌ {yt_id}: {result.stderr[:100]}")
            errors += 1
    
    print(f"\nDone: {downloaded} downloaded, {errors} errors")
    return downloaded, errors


def download_sample(n=10, output_dir='/tmp/standup4ai_audio'):
    """Download first N audio files as sample."""
    audio_files = get_audio_files()
    yt_ids = list(audio_files.keys())[:n]
    print(f"Sample: downloading {n} files → {output_dir}")
    return download_files(yt_ids, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download StandUp4AI audio from GDrive')
    parser.add_argument('--max-files', type=int, default=0, help='Max files to download (0=all)')
    parser.add_argument('--output-dir', default='/tmp/standup4ai_audio', help='Output directory')
    parser.add_argument('--youtube-id', help='Download specific YouTube ID')
    parser.add_argument('--youtube-ids', help='File with list of YouTube IDs (one per line)')
    parser.add_argument('--list-only', action='store_true', help='Just list available files')
    args = parser.parse_args()

    if args.list_only:
        af = get_audio_files()
        print(f"Total audio files available: {len(af)}")
        for yt_id, info in list(af.items())[:10]:
            print(f"  {yt_id} ({info['lang']})")
        print(f"  ... and {len(af)-10} more")
        sys.exit(0)

    if args.youtube_ids:
        with open(args.youtube_ids) as f:
            yt_ids = [line.strip() for line in f if line.strip()]
        print(f"Downloading {len(yt_ids)} YouTube IDs from file")
        download_files(yt_ids, args.output_dir)
    elif args.youtube_id:
        download_files([args.youtube_id], args.output_dir)
    elif args.max_files > 0:
        download_sample(n=args.max_files, output_dir=args.output_dir)
    else:
        print("Specify --max-files N, --youtube-id ID, or --youtube-ids FILE")
        print("\nAvailable files:")
        af = get_audio_files()
        print(f"Total: {len(af)} audio files")
        for yt_id, info in list(af.items())[:5]:
            print(f"  {yt_id} ({info['lang']})")
