#!/usr/bin/env bash
#
# run.sh — convenience launcher for the Raspberry Pi hand tracker.
#
# Activates the ./.venv created by setup.sh and starts the tracker. Any
# arguments/env vars you'd normally pass to python still work, e.g.:
#
#     ./run.sh
#     CAMERA_BACKEND=picamera2 ./run.sh
#     CAPTURE_WIDTH=320 CAPTURE_HEIGHT=240 ./run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "No virtualenv found at ./$VENV_DIR."
    echo "Run the setup first:"
    echo "    ./setup.sh          # USB webcam"
    echo "    ./setup.sh picam    # Pi Camera Module"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

exec python track_hands.py "$@"
