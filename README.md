# Hand Tracker

Watches your webcam, detects your hand(s) using MediaPipe, and draws a fading
trail showing where each hand has moved over the last 3 seconds.

## 1. One-time / every-time setup

In Terminal, unzip this package, `cd` into the folder, then run:

```bash
source setup.sh
```

**Important:** use `source setup.sh`, not `./setup.sh` or `bash setup.sh`.
Sourcing it is what lets the virtualenv stay active in your terminal
afterward.

This script is safe to run repeatedly:
- If the virtualenv (`.venv/`) doesn't exist yet, it creates it.
- If it already exists, it reuses it.
- It installs/updates the required packages (`opencv-python`, `mediapipe`,
  `numpy`) each time, which is fast if they're already installed.

You'll know it worked when you see `Setup complete.` and your terminal
prompt shows `(.venv)` at the start.

## 2. Run the tracker

With the virtualenv active:

```bash
python track_hands.py
```

A window will open showing your webcam feed with your hand(s) tracked and a
colored trail behind each one (blue-ish for your left hand, orange-ish for
your right, from the camera's mirrored point of view).

**To quit:** press the **spacebar**, or just close the video window.

## 3. macOS camera permissions

The first time you run this, macOS will likely prompt you to allow camera
access for your terminal app (Terminal, iTerm, etc.). If you accidentally
deny it, fix it at:

System Settings → Privacy & Security → Camera → enable it for your terminal
app, then re-run the script.

## 4. Leaving the virtualenv

When you're done:

```bash
deactivate
```

## Files

- `setup.sh` — idempotent script that creates a `.venv` virtualenv, installs
  dependencies into it, and downloads the hand-tracking model file.
- `requirements.txt` — the Python packages installed by `setup.sh`.
- `track_hands.py` — the hand-tracking script itself.
- `hand_landmarker.task` — MediaPipe's pretrained hand model. Downloaded
  automatically by `setup.sh` the first time you run it (a few MB); not
  included in this zip.

## Note on MediaPipe versions

This uses MediaPipe's newer "Tasks" API (`HandLandmarker`). An older API
(`mp.solutions.hands`) used to be the standard way to do this, but recent
MediaPipe releases on PyPI no longer reliably expose it, so this package
uses the currently-supported approach instead.
