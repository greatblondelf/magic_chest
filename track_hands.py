#!/usr/bin/env python3
"""
track_hands.py

Opens your webcam, detects one or two hands using MediaPipe's Hand Landmarker
task, and draws a trailing line showing where each hand has moved over the
last 3 seconds.

Controls:
    - Press SPACE to quit
    - Or just close the video window (click the X)

On macOS you may need to grant your terminal app camera access the first
time you run this: System Settings > Privacy & Security > Camera.

Note: this uses MediaPipe's newer "Tasks" API (mp.tasks.vision.HandLandmarker)
rather than the older mp.solutions.hands API. As of late 2025, recent
mediapipe releases on PyPI no longer expose mp.solutions reliably, so the
Tasks API is the current supported way to do this.
"""

import os
import sys
import time
import math
import random
import subprocess
from collections import deque

import cv2
import mediapipe as mp

# ---------------------------------------------------------------- Config --
TRAIL_SECONDS = 4.0     # how much movement history to draw, in seconds
MAX_HANDS = 1
TRAIL_LANDMARK = 9      # 9 = middle-finger MCP joint, a stable "palm center"
WINDOW_NAME = "Hand Tracker  —  SPACE or close window to quit"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# BGR colors per hand label, used for the trail and skeleton
TRAIL_COLORS = {
    "Left": (255, 120, 0),     # blue-ish
    "Right": (0, 200, 255),    # orange-ish
}
DEFAULT_COLOR = (0, 255, 0)
JOINT_COLOR = (255, 255, 255)

# Distinct BGR colors cycled across tracked hands by stable track id. We track
# hands by position continuity (see main loop) rather than handedness, so a
# single hand keeps one color instead of oscillating between Left/Right colors.
TRACK_PALETTE = [
    (255, 120, 0),     # blue
    (0, 200, 255),     # orange
    (0, 255, 0),       # green
    (255, 0, 255),     # magenta
]

# --- Shape-activation (sliding-window) config ---
# Every frame we feed the recent trail to the $1 recognizer; if it matches
# TARGET_SHAPE with a score >= MATCH_THRESHOLD, we "activate" by turning the
# trails red for ACTIVATION_SECONDS. Tune the threshold / TRAIL_SECONDS window
# to taste once you've settled on the shape.
TARGET_SHAPE = "star" # "square"   # one of TEMPLATES below: square/triangle/circle/star
MATCH_THRESHOLD = 0.8    # $1 score in 0..1; lower = easier to trigger
MIN_MATCH_POINTS = 16     # don't try to match trails shorter than this
ACTIVATION_SECONDS = 2.0  # how long the red "activated" state lasts
RED = (0, 0, 255)         # BGR, trail color while activated

# --- Celebration effects config ---
SPARKLE_BURST = 110       # particles spawned the instant the shape is matched
SPARKLE_PER_FRAME = 7     # extra twinkles spawned each frame during the flash
SOUND_ENABLED = True
# A triumphant macOS system sound; falls back to a terminal bell elsewhere.
SOUND_PATH = "/System/Library/Sounds/Hero.aiff"

# The 21-point hand topology MediaPipe's hand model uses (wrist=0,
# thumb=1-4, index=5-8, middle=9-12, ring=13-16, pinky=17-20). Defined here
# directly so this script doesn't depend on the legacy mp.solutions module.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # wrist to pinky base
]
# ---------------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_hand(frame, points_px, color):
    """Draw the hand skeleton (joints + connecting lines) on frame."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points_px[a], points_px[b], color, 2, cv2.LINE_AA)
    for x, y in points_px:
        cv2.circle(frame, (x, y), 4, JOINT_COLOR, -1, cv2.LINE_AA)


# --------------------------------------------------- $1 shape recognizer --
# Compact implementation of the $1 Unistroke Recognizer (Wobbrock, Wilson &
# Li, 2007). Given a stroke (list of (x, y)), it resamples to a fixed number
# of points, then normalizes for rotation, scale and position so the same
# shape matches at any angle/size/location. recognize() compares against the
# precomputed TEMPLATES and returns (best_name, score) where score is 0..1.
N_POINTS = 64
SQUARE_SIZE = 250.0
ANGLE_RANGE = math.radians(45.0)
ANGLE_PRECISION = math.radians(2.0)
PHI = 0.5 * (-1.0 + math.sqrt(5.0))              # golden ratio
HALF_DIAGONAL = 0.5 * math.sqrt(2 * SQUARE_SIZE * SQUARE_SIZE)


def _path_length(pts):
    return sum(math.dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))


def _resample(pts, n):
    """Re-space a stroke into n points evenly spread along its arc length."""
    pts = [list(p) for p in pts]
    total = _path_length(pts)
    if total == 0:
        return None
    interval = total / (n - 1)
    D = 0.0
    new = [pts[0][:]]
    i = 1
    while i < len(pts):
        d = math.dist(pts[i - 1], pts[i])
        if D + d >= interval:
            t = (interval - D) / d
            q = [
                pts[i - 1][0] + t * (pts[i][0] - pts[i - 1][0]),
                pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]),
            ]
            new.append(q)
            pts.insert(i, q)  # continue measuring from the new point
            D = 0.0
        else:
            D += d
        i += 1
    while len(new) < n:  # guard against float rounding losing the last point
        new.append(pts[-1][:])
    return new[:n]


def _centroid(pts):
    n = len(pts)
    return [sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n]


def _indicative_angle(pts):
    c = _centroid(pts)
    return math.atan2(c[1] - pts[0][1], c[0] - pts[0][0])


def _rotate_by(pts, theta):
    c = _centroid(pts)
    cos, sin = math.cos(theta), math.sin(theta)
    return [
        [
            (x - c[0]) * cos - (y - c[1]) * sin + c[0],
            (x - c[0]) * sin + (y - c[1]) * cos + c[1],
        ]
        for x, y in pts
    ]


def _scale_to(pts, size):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return [
        [x * (size / w) if w else x, y * (size / h) if h else y]
        for x, y in pts
    ]


def _translate_to_origin(pts):
    c = _centroid(pts)
    return [[x - c[0], y - c[1]] for x, y in pts]


def _path_distance(a, b):
    return sum(math.dist(a[i], b[i]) for i in range(len(a))) / len(a)


def _distance_at_angle(pts, template, theta):
    return _path_distance(_rotate_by(pts, theta), template)


def _distance_at_best_angle(pts, template):
    """Golden-section search for the rotation that best aligns pts/template."""
    a, b = -ANGLE_RANGE, ANGLE_RANGE
    x1 = PHI * a + (1 - PHI) * b
    f1 = _distance_at_angle(pts, template, x1)
    x2 = (1 - PHI) * a + PHI * b
    f2 = _distance_at_angle(pts, template, x2)
    while abs(b - a) > ANGLE_PRECISION:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = PHI * a + (1 - PHI) * b
            f1 = _distance_at_angle(pts, template, x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = (1 - PHI) * a + PHI * b
            f2 = _distance_at_angle(pts, template, x2)
    return min(f1, f2)


def _normalize(pts):
    pts = _resample(pts, N_POINTS)
    if pts is None:
        return None
    pts = _rotate_by(pts, -_indicative_angle(pts))
    pts = _scale_to(pts, SQUARE_SIZE)
    return _translate_to_origin(pts)


def recognize(pts):
    """Return (best_template_name, score) for a stroke; (None, 0.0) if empty."""
    cand = _normalize(pts)
    if cand is None:
        return None, 0.0
    best_name, best_dist = None, float("inf")
    for name, template in TEMPLATES.items():
        d = _distance_at_best_angle(cand, template)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, 1.0 - best_dist / HALF_DIAGONAL


def _make_polygon(n, rot=-math.pi / 2):
    """n-gon vertices (closed) on the unit circle, first vertex at the top."""
    pts = [
        [math.cos(rot + 2 * math.pi * i / n), math.sin(rot + 2 * math.pi * i / n)]
        for i in range(n)
    ]
    pts.append(pts[0][:])  # close the loop
    return pts


def _make_star(points=5):
    """A pentagram: visit every 2nd vertex of a regular polygon, then close."""
    verts = _make_polygon(points, rot=-math.pi / 2)[:-1]
    order = [(i * 2) % points for i in range(points)]
    pts = [verts[i][:] for i in order]
    pts.append(pts[0][:])
    return pts


# Raw ideal shapes; normalized once at import into matchable templates.
_RAW_TEMPLATES = {
    "square": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
    "triangle": _make_polygon(3),
    "circle": _make_polygon(48),
    "star": _make_star(5),
}
TEMPLATES = {name: _normalize(raw) for name, raw in _RAW_TEMPLATES.items()}


# --------------------------------------------------- Celebration effects --
def play_sound():
    """Best-effort, non-blocking triumphant sound (afplay on macOS)."""
    if not SOUND_ENABLED:
        return
    try:
        if sys.platform == "darwin" and os.path.exists(SOUND_PATH):
            subprocess.Popen(
                ["afplay", SOUND_PATH],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print("\a", end="", flush=True)  # terminal bell fallback
    except Exception:
        pass  # effects are cosmetic; never let them crash the loop


def fit_template(name, box):
    """Map a raw ideal shape into the pixel bounding box the user drew in,
    returning a list of (x, y) int points forming the closed outline."""
    raw = _RAW_TEMPLATES[name]
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    rw = (max(xs) - min(xs)) or 1.0
    rh = (max(ys) - min(ys)) or 1.0
    minx, miny, maxx, maxy = box
    bw, bh = maxx - minx, maxy - miny
    return [
        (
            int(minx + (x - min(xs)) / rw * bw),
            int(miny + (y - min(ys)) / rh * bh),
        )
        for x, y in raw
    ]


def spawn_sparkles(sparkles, polyline, n, now):
    """Add n sparkle particles scattered evenly along a polyline outline."""
    if len(polyline) < 2:
        return
    palette = [(255, 255, 255), (0, 215, 255), (200, 255, 255)]  # white/gold/pale
    for _ in range(n):
        i = random.randint(1, len(polyline) - 1)
        t = random.random()
        ax, ay = polyline[i - 1]
        bx, by = polyline[i]
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(25, 170)
        sparkles.append({
            "x": ax + (bx - ax) * t,
            "y": ay + (by - ay) * t,
            "vx": math.cos(ang) * spd,
            "vy": math.sin(ang) * spd,
            "born": now,
            "life": random.uniform(0.4, 1.1),
            "size": random.uniform(2.0, 4.5),
            "color": random.choice(palette),
        })


def update_and_draw_sparkles(frame, sparkles, now, dt):
    """Advance every particle, draw it as a fading twinkle, drop the dead."""
    alive = []
    for s in sparkles:
        age = now - s["born"]
        if age >= s["life"]:
            continue
        s["x"] += s["vx"] * dt
        s["y"] += s["vy"] * dt
        s["vy"] += 90 * dt  # gentle gravity
        fade = 1.0 - age / s["life"]
        col = tuple(int(c * fade) for c in s["color"])
        x, y = int(s["x"]), int(s["y"])
        r = max(1, int(s["size"] * fade))
        cv2.line(frame, (x - r * 2, y), (x + r * 2, y), col, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y - r * 2), (x, y + r * 2), col, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, col, -1, cv2.LINE_AA)
        alive.append(s)
    sparkles[:] = alive


def draw_ideal(frame, pts, thickness):
    """Draw the 'perfect' ideal figure as a glowing white outline."""
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):  # soft outer glow
        cv2.line(frame, pts[i - 1], pts[i], (130, 130, 130), thickness + 6, cv2.LINE_AA)
    for i in range(1, len(pts)):  # crisp bright core
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), thickness, cv2.LINE_AA)
    for x, y in pts:              # bright nodes at the vertices
        cv2.circle(frame, (x, y), max(2, thickness // 2), (255, 255, 255), -1, cv2.LINE_AA)


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            "Could not find hand_landmarker.task next to this script.\n"
            "Run 'source setup.sh' again to download it, then try once more."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. On macOS, check System Settings > "
            "Privacy & Security > Camera and make sure your terminal app "
            "is allowed to use the camera, then try again."
        )

    # one trail (deque of (timestamp, x, y)) per *tracked hand*, keyed by a
    # stable integer id we assign by position continuity (NOT by MediaPipe's
    # handedness, which flips frame-to-frame on a mirrored image).
    trails = {}
    next_id = 0

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    start_time = time.time()
    prev_now = start_time
    activated_until = 0.0   # wall-clock time the red "activated" state ends
    flash_start = 0.0       # when the current celebration began
    flash_stroke = []       # frozen red copy of the stroke that triggered it
    flash_overlay = []       # the "perfect" ideal figure, fit to that stroke
    sparkles = []           # live particle list

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror image, feels more natural
            h, w = frame.shape[:2]
            now = time.time()
            dt = min(0.1, now - prev_now)  # frame delta, clamped for stability
            prev_now = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Gather this frame's detections as pixel-space landmark lists.
            detections = [
                [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for hand_landmarks in (result.hand_landmarks or [])
            ]

            # --- position-continuity tracking ---
            # Match each detected hand to the nearest existing trail head, so a
            # single hand keeps one stable id (and color) even when MediaPipe's
            # handedness flips. The allowed match distance GROWS with how long
            # since that track was last seen: a continuously-tracked hand uses a
            # tight radius (keeps two hands distinct), but a hand returning after
            # a detection dropout gets a generous radius and re-attaches instead
            # of spawning a new track/color. Any detection still unmatched starts
            # a new track.
            base_radius = 0.25 * w          # tight radius for a fresh head
            reacquire_speed = w             # extra px allowed per second of gap
            heads = {tid: t[-1] for tid, t in trails.items() if t}  # (ts, x, y)
            candidates = []
            for di, pts in enumerate(detections):
                cx, cy = pts[TRAIL_LANDMARK]
                for tid, (ts, hx, hy) in heads.items():
                    dist = math.hypot(cx - hx, cy - hy)
                    radius = base_radius + reacquire_speed * (now - ts)
                    if dist <= radius:
                        candidates.append((dist, di, tid))
            candidates.sort()
            assigned, used = {}, set()
            for dist, di, tid in candidates:
                if di in assigned or tid in used:
                    continue
                assigned[di] = tid
                used.add(tid)
            for di in range(len(detections)):
                if di not in assigned:
                    assigned[di] = next_id
                    trails[next_id] = deque()
                    next_id += 1

            # draw each hand's skeleton and extend its trail
            for di, pts in enumerate(detections):
                tid = assigned[di]
                color = TRACK_PALETTE[tid % len(TRACK_PALETTE)]
                draw_hand(frame, pts, color)
                cx, cy = pts[TRAIL_LANDMARK]
                trails[tid].append((now, cx, cy))

            # prune old points; drop tracks that have fully aged out so a
            # returning hand gets a fresh id rather than a stale head to snap to
            for tid in list(trails):
                trail = trails[tid]
                while trail and now - trail[0][0] > TRAIL_SECONDS:
                    trail.popleft()
                if not trail:
                    del trails[tid]

            # --- sliding-window shape recognition ---
            # Run the $1 recognizer on each hand's recent trail every frame.
            best_name, best_score, best_pts = None, 0.0, None
            for trail in trails.values():
                if len(trail) < MIN_MATCH_POINTS:
                    continue
                pts = [(x, y) for _, x, y in trail]
                name, score = recognize(pts)
                if score > best_score:
                    best_name, best_score, best_pts = name, score, pts
            active = now < activated_until

            # Rising-edge trigger: fire the celebration only when we're not
            # already mid-flash, so the sound plays once and the burst is clean.
            if (
                not active
                and best_name == TARGET_SHAPE
                and best_score >= MATCH_THRESHOLD
            ):
                activated_until = now + ACTIVATION_SECONDS
                flash_start = now
                active = True
                xs = [p[0] for p in best_pts]
                ys = [p[1] for p in best_pts]
                flash_stroke = list(best_pts)  # freeze the triggering stroke
                flash_overlay = fit_template(
                    best_name, (min(xs), min(ys), max(xs), max(ys))
                )
                spawn_sparkles(sparkles, flash_overlay, SPARKLE_BURST, now)
                play_sound()
                trails.clear()  # discharge so it won't instantly re-trigger

            if active:
                # Celebration: frozen red trigger-stroke, the glowing "perfect"
                # ideal figure pulsing on top, plus an ongoing twinkle of sparks.
                for i in range(1, len(flash_stroke)):
                    cv2.line(frame, flash_stroke[i - 1], flash_stroke[i],
                             RED, 8, cv2.LINE_AA)
                pulse = 0.5 + 0.5 * math.sin((now - flash_start) * 12.0)
                draw_ideal(frame, flash_overlay, 6 + int(7 * pulse))
                spawn_sparkles(sparkles, flash_overlay, SPARKLE_PER_FRAME, now)
            else:
                # normal live trails, one fading color per tracked hand
                for tid, trail in trails.items():
                    color = TRACK_PALETTE[tid % len(TRACK_PALETTE)]
                    points = list(trail)
                    for i in range(1, len(points)):
                        _, x1, y1 = points[i - 1]
                        t2, x2, y2 = points[i]
                        fade = max(0.0, 1.0 - (now - t2) / TRAIL_SECONDS)
                        thickness = max(1, int(6 * fade))
                        faded_color = tuple(
                            int(c * fade + 30 * (1 - fade)) for c in color
                        )
                        cv2.line(frame, (x1, y1), (x2, y2),
                                 faded_color, thickness, cv2.LINE_AA)

            # sparkles always advance/draw so they finish their life gracefully
            update_and_draw_sparkles(frame, sparkles, now, dt)

            # live readout: best-matching shape + score (handy for tuning)
            readout = f"{best_name or '--'}: {best_score:.2f}  (target: {TARGET_SHAPE})"
            cv2.putText(
                frame, readout, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                RED if active else (255, 255, 255), 2, cv2.LINE_AA,
            )
            if active:
                cv2.putText(
                    frame, "ACTIVATED", (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, RED, 2, cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "SPACE or close window to quit",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # spacebar
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break  # user closed the window

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
