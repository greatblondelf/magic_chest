#!/usr/bin/env bash
#
# setup.sh — environment setup for the hand tracker on Raspberry Pi.
#
# Unlike the desktop version, this one is meant to be RUN, not sourced:
#
#     ./setup.sh            # USB webcam (default)
#     ./setup.sh picam      # Pi Camera Module (CSI ribbon) via picamera2
#
# It creates a virtualenv in ./.venv here in the pi/ folder, installs the
# Python dependencies, installs the system libraries OpenCV/MediaPipe need,
# and downloads the hand-landmark model. Re-running it is safe.
#
# After it finishes, activate the venv and run the tracker:
#     source .venv/bin/activate
#     python track_hands.py
#
set -euo pipefail

CAMERA_MODE="${1:-usb}"          # "usb" (default) or "picam"
VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Raspberry Pi hand-tracker setup (camera mode: $CAMERA_MODE) ==="

# --- 1. Sanity checks -------------------------------------------------------
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
    echo ""
    echo "WARNING: architecture is '$ARCH', not 'aarch64'."
    echo "MediaPipe only ships wheels for 64-bit (aarch64) Raspberry Pi OS."
    echo "On 32-bit Pi OS 'pip install mediapipe' will fail. If you're on a"
    echo "Pi, re-flash with the 64-bit image. Continuing anyway..."
    echo ""
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Could not find '$PYTHON_BIN'. Install it with:"
    echo "    sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi

# --- 2. System libraries ----------------------------------------------------
# OpenCV's prebuilt wheel needs a few shared libs that aren't guaranteed to be
# present on a fresh Pi OS, plus the GUI stack for the preview window.
if command -v apt-get >/dev/null 2>&1; then
    APT_PACKAGES=(libgl1 libglib2.0-0 libatlas-base-dev)
    if [ "$CAMERA_MODE" = "picam" ]; then
        # The Pi Camera Module is driven by libcamera via picamera2, which is
        # a system package (not a pip wheel).
        APT_PACKAGES+=(python3-picamera2)
    fi
    echo "Installing system packages: ${APT_PACKAGES[*]}"
    echo "(this needs sudo; you may be prompted for your password)"
    sudo apt-get update -qq || true
    sudo apt-get install -y "${APT_PACKAGES[@]}" || {
        echo "WARNING: apt install failed. If the preview window or camera"
        echo "doesn't work, install these manually: ${APT_PACKAGES[*]}"
    }
else
    echo "apt-get not found; skipping system-library step."
    echo "Make sure libGL / libglib2.0 are available for OpenCV's GUI."
fi

# --- 3. Create the virtualenv ----------------------------------------------
# When using the Pi Camera we build the venv with --system-site-packages so
# the apt-installed picamera2 is importable inside it. USB mode uses an
# isolated venv, which is cleaner.
VENV_ARGS=()
if [ "$CAMERA_MODE" = "picam" ]; then
    VENV_ARGS+=(--system-site-packages)
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv in ./$VENV_DIR ${VENV_ARGS[*]:-} ..."
    "$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$VENV_DIR"
else
    echo "Virtualenv already exists at ./$VENV_DIR — reusing it."
    if [ "$CAMERA_MODE" = "picam" ]; then
        echo "Note: if picamera2 isn't importable, delete ./$VENV_DIR and"
        echo "re-run './setup.sh picam' so it's built with --system-site-packages."
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- 4. Python dependencies -------------------------------------------------
echo "Upgrading pip ..."
pip install --upgrade pip --quiet

echo "Installing Python packages (first run can take several minutes on a Pi) ..."
pip install -r requirements.txt --quiet

# --- 5. Hand-landmark model -------------------------------------------------
MODEL_FILE="hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading hand-landmark model (a few MB, one-time) ..."
    if ! curl -fsSL -o "$MODEL_FILE" "$MODEL_URL"; then
        echo "Failed to download $MODEL_FILE. Check your connection and re-run."
        rm -f "$MODEL_FILE"
        exit 1
    fi
else
    echo "Hand-landmark model already present — skipping download."
fi

# --- 6. Done ----------------------------------------------------------------
cat <<EOF

Setup complete.

To run the tracker:
    source .venv/bin/activate
    python track_hands.py

EOF
if [ "$CAMERA_MODE" = "picam" ]; then
    cat <<EOF
You set up for the Pi Camera Module. If auto-detection doesn't pick it up, force it:
    CAMERA_BACKEND=picamera2 python track_hands.py
EOF
else
    cat <<EOF
Using a USB webcam. If you have more than one camera, pick it with:
    CAMERA_INDEX=1 python track_hands.py
EOF
fi
