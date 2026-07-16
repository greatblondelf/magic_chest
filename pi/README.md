# Hand Tracker — Raspberry Pi edition

Watches a camera, detects your hand(s) with MediaPipe, draws a fading trail
behind each one, and celebrates (red flash + sparkles + a sound) when you
trace a target shape — a **star** by default.

This is the Raspberry Pi port of the desktop version. The gesture logic is
identical; the camera handling, sound, and setup are adapted for the Pi.

## Requirements

- **A 64-bit Raspberry Pi OS (aarch64).** MediaPipe ships no 32-bit
  (`armv7l`) wheels, so `pip install mediapipe` fails on 32-bit Pi OS. Check
  with `uname -m` — it must print `aarch64`.
- **A Raspberry Pi 4 or 5.** Pi 3 and earlier are too slow / RAM-limited for
  smooth landmark detection.
- **A desktop / monitor** (or `ssh -X`). The preview window uses `cv2.imshow`,
  which needs a display — a fully headless Pi has nothing to draw to.
- **A camera**: either a USB webcam or the Pi Camera Module (CSI ribbon).

## Setup

From inside this `pi/` folder:

```bash
# USB webcam (the default):
./setup.sh

# ...or the Pi Camera Module (CSI ribbon cable):
./setup.sh picam
```

Unlike the desktop version, **run** this script (`./setup.sh`), don't source
it. It will:

1. Warn if you're not on 64-bit OS.
2. `apt install` the system libraries OpenCV needs (`libgl1`, `libglib2.0-0`,
   `libatlas-base-dev`), plus `python3-picamera2` in `picam` mode. This step
   uses `sudo` and may prompt for your password.
3. Create a virtualenv in `./.venv` (built with `--system-site-packages` in
   `picam` mode so the system `picamera2` is visible).
4. Install the Python packages from `requirements.txt`.
5. Download the `hand_landmarker.task` model (a few MB, one time).

## Run

The quick way:

```bash
./run.sh
```

`run.sh` activates the venv and launches the tracker for you. Any environment
variables still work, e.g. `CAMERA_BACKEND=picamera2 ./run.sh`.

Or do it by hand:

```bash
source .venv/bin/activate
python track_hands.py
```

A window opens showing the camera feed with your hand(s) tracked and a colored
trail behind each. Trace a **star** in the air with one hand to trigger the
celebration.

**To quit:** press **SPACE**, or close the window.

## Camera selection

The script auto-detects: if `picamera2` is installed it uses the CSI camera,
otherwise it falls back to a USB webcam via OpenCV/V4L2. Override with
environment variables (no code edits needed):

```bash
CAMERA_BACKEND=picamera2 python track_hands.py   # force the Pi Camera Module
CAMERA_BACKEND=opencv     python track_hands.py   # force USB / V4L2
CAMERA_INDEX=1            python track_hands.py    # pick /dev/video1 (USB)
CAPTURE_WIDTH=320 CAPTURE_HEIGHT=240 python track_hands.py   # smaller = faster
```

If a USB webcam won't open, confirm it appears in `ls /dev/video*` and that
your user is in the `video` group (`sudo usermod -aG video "$USER"`, then log
out and back in).

## Sound

On the Pi the celebration sound plays through `paplay` (PulseAudio/PipeWire)
or `aplay` (ALSA), using a system sound file if one is found. Point it at your
own file with:

```bash
SOUND_PATH=/path/to/celebrate.wav python track_hands.py
```

If no player or sound file is available it falls back to the terminal bell —
sound is cosmetic and never blocks tracking.

## Performance notes

Landmark detection is CPU-bound. Expect roughly single-digit-to-low-teens FPS
on a Pi 4, better on a Pi 5. The trail and shape recognition are time-based
(not frame-based), so it still works when the frame rate is low — the trail is
just sparser. If it feels sluggish, drop the resolution with
`CAPTURE_WIDTH`/`CAPTURE_HEIGHT` as shown above.

## Tuning the target shape

Open `track_hands.py` and edit the config block near the top:

- `TARGET_SHAPE` — `"star"`, `"square"`, `"triangle"`, or `"circle"`.
- `MATCH_THRESHOLD` — 0..1; lower is easier to trigger.
- `TRAIL_SECONDS` — how much drawing history is considered a "stroke".

## Files

- `setup.sh` — Pi setup: system libs, venv, Python deps, model download.
- `run.sh` — convenience launcher: activates the venv and starts the tracker.
- `requirements.txt` — Python packages (installed into `./.venv`).
- `track_hands.py` — the Pi-adapted tracker.
- `hand_landmarker.task` — MediaPipe's pretrained model; downloaded by
  `setup.sh`, not committed here.

## How this differs from the desktop version

- **Camera:** adds a `picamera2`/libcamera path for the CSI camera and forces
  the V4L2 backend for USB webcams; `VideoCapture(0)` alone does not work with
  the Pi Camera Module on modern Pi OS.
- **Sound:** uses `paplay`/`aplay` instead of macOS's `afplay`.
- **Resolution:** defaults to 640×480 to stay responsive on Pi CPUs.
- **Setup:** installs system `apt` libraries and is run, not sourced.
