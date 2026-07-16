#!/usr/bin/env bash
#
# setup.sh — idempotent environment setup for the hand tracker.
#
# IMPORTANT: This script must be *sourced*, not executed, so the virtualenv
# activation persists in your current shell session afterward.
#
#   source setup.sh
#
# Running it again later is safe: it reuses the existing virtualenv and
# only installs packages that aren't already installed.

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Make sure this was sourced, not executed -----------------------------
( return 0 2>/dev/null )
if [ "$?" -ne 0 ]; then
    echo "This script needs to be *sourced* so the virtualenv stays active"
    echo "in your shell. Run it like this instead:"
    echo ""
    echo "    source setup.sh"
    echo ""
    exit 1
fi

# --- Check for python3 ------------------------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Could not find '$PYTHON_BIN'. Install Python 3 first, e.g.:"
    echo "    brew install python"
    return 1
fi

# --- Create the virtualenv if it doesn't already exist ---------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv in ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtualenv already exists at ./$VENV_DIR — reusing it."
fi

# --- Activate it -------------------------------------------------------------
source "$VENV_DIR/bin/activate"

# --- Install / update dependencies -------------------------------------------
echo "Upgrading pip ..."
pip install --upgrade pip --quiet

echo "Installing required packages (this may take a minute the first time) ..."
pip install -r requirements.txt --quiet

# --- Download the hand landmark model file, if we don't already have it ----
MODEL_FILE="hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading hand landmark model (a few MB, one-time) ..."
    if ! curl -fsSL -o "$MODEL_FILE" "$MODEL_URL"; then
        echo "Failed to download $MODEL_FILE. Check your internet connection"
        echo "and re-run 'source setup.sh'."
        rm -f "$MODEL_FILE"
        return 1
    fi
else
    echo "Hand landmark model already present — skipping download."
fi

echo ""
echo "Setup complete. Virtualenv is active:"
echo "  python: $(which python)"
echo ""
echo "Run the tracker with:"
echo "  python track_hands.py"
echo ""
echo "When you're done, leave the virtualenv with: deactivate"
