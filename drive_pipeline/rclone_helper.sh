#!/bin/bash
# rclone helpers for HaHaScore Drive pipeline
# Usage: source this file or call as functions

DRIVE_BASE="gdrive:/HaHaScore_Pretrain"

# List top-level structure
drive_tree() {
    echo "📂 HaHaScore Drive Structure:"
    rclone lsf "${DRIVE_BASE}/" --dirs-only 2>&1
}

# List contents of a subfolder
drive_ls() {
    local subfolder="${1:-}"
    rclone lsf "${DRIVE_BASE}/${subfolder}" 2>&1
}

# Get file size on Drive
drive_size() {
    local path="$1"
    rclone ls "${DRIVE_BASE}/${path}" 2>&1
}

# Sync local file to Drive (single file)
drive_push() {
    local local_file="$1"
    local subfolder="$2"
    rclone copy "$local_file" "${DRIVE_BASE}/${subfolder}/" --progress 2>&1
}

# Pull Drive file to local (single file)
drive_pull() {
    local subfolder="$1"
    local filename="$2"
    local dest="${3:-.}"
    rclone copy "${DRIVE_BASE}/${subfolder}/${filename}" "$dest" --progress 2>&1
}

# Check Drive usage
drive_du() {
    rclone du "${DRIVE_BASE}/" 2>&1
}

echo "✅ Drive helpers loaded. Use: drive_tree, drive_ls <sub>, drive_push <file> <sub>, drive_pull <sub> <file> <dest>"
