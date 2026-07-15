#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# QT Robot Dementia Speech System — Launcher
#
# This script is meant to be double-clicked from the desktop.
# It handles:
#   1. Activating the Python virtual environment
#   2. Installing/updating dependencies only when requirements.txt changes
#   3. Launching the application
# ─────────────────────────────────────────────────────────────────

# ── Configuration ────────────────────────────────────────────────
# Path to this script's directory (the project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the virtual environment inside the project
VENV_DIR="$SCRIPT_DIR/dss_venv"

# Hash cache file — gitignored, lives in the project root
HASH_CACHE="$SCRIPT_DIR/.requirements_hash"

# ── Check the virtual environment exists ─────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[Launcher] Virtual environment not found at $VENV_DIR"
    echo "[Launcher] Creating it now..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[Launcher] ERROR: Failed to create virtual environment."
        read -p "Press Enter to close..."
        exit 1
    fi
fi

# ── Activate virtual environment ────────────────────────
source "$VENV_DIR/bin/activate"

# ── dependency check ──────────────────────────
# Compute the hash of requirements.txt
CURRENT_HASH=$(md5sum "$SCRIPT_DIR/requirements.txt" | awk '{print $1}')

# Read the previously saved hash (empty string if file doesn't exist)
SAVED_HASH=""
if [ -f "$HASH_CACHE" ]; then
    SAVED_HASH=$(cat "$HASH_CACHE")
fi

# Only run pip install if the hash has changed or cache doesn't exist
if [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
    echo "[Launcher] requirements.txt has changed (or first run). Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r "$SCRIPT_DIR/requirements.txt"
    if [ $? -ne 0 ]; then
        echo "[Launcher] ERROR: Dependency installation failed."
        read -p "Press Enter to close..."
        exit 1
    fi
    # Save the new hash so we don't reinstall next time
    echo "$CURRENT_HASH" > "$HASH_CACHE"
    echo "[Launcher] Dependencies installed successfully."
else
    echo "[Launcher] Dependencies up to date. Skipping install."
fi

# ── Launch the application ──────────────────────────────
echo "[Launcher] Starting QT Robot Speech System..."
cd "$SCRIPT_DIR"
python3 main.py