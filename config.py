# =============================================================
# config.py — Central configuration (v2 — non-blocking)
# =============================================================

import os

# ─── Camera Sources ───────────────────────────────────────────
CAMERA_SOURCES = [
    0,          # USB Camera 0
    # 1,        # USB Camera 1 (optional)
    # "rtsp://username:password@192.168.1.127:554/stream",
    # "http://192.168.1.100:8080/video",  # DroidCam
]

# ─── Frame Settings ───────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 360
JPEG_QUALITY = 80         # lower = faster encode, smaller payload

# ─── Streaming ────────────────────────────────────────────────
TARGET_FPS   = 30         # Flask MJPEG stream target FPS
                          # Camera capture runs as fast as it can
                          # (usually 30 fps for USB cams)

# ─── AI Processing Throttle ───────────────────────────────────
# The AI worker runs the full model stack once every N frames it grabs.
# Higher = smoother stream, lower latency detections (but more CPU/GPU).
#
#   AI_PROCESS_EVERY = 1   → every frame  (max accuracy, high CPU)
#   AI_PROCESS_EVERY = 2   → every 3rd    (good balance — RECOMMENDED)
#   AI_PROCESS_EVERY = 6   → every 6th    (smooth on low-end hardware)
#   AI_PROCESS_EVERY = 10  → every 10th   (ultra-light, detections lag)
#
AI_PROCESS_EVERY = 2

# ─── AI Toggle ────────────────────────────────────────────────
AI_ENABLED_DEFAULT = True   # Can be toggled at runtime via API

# ─── YOLO ─────────────────────────────────────────────────────
YOLO_MODEL       = "yolov8n.pt"   # nano=fast, s/m=accurate
YOLO_CONFIDENCE  = 0.40
YOLO_PERSON_CLS  = 0
YOLO_VEHICLE_CLS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ─── DeepSORT ─────────────────────────────────────────────────
DEEPSORT_MAX_AGE    = 30
DEEPSORT_N_INIT     = 3
DEEPSORT_MAX_IOU    = 0.7

# ─── MediaPipe Pose ───────────────────────────────────────────
POSE_DETECTION_CONFIDENCE  = 0.5
POSE_TRACKING_CONFIDENCE   = 0.5

# ─── Violence Detection ───────────────────────────────────────
VIOLENCE_SPEED_THRESHOLD   = 0.08
VIOLENCE_KICK_THRESHOLD    = 0.10
VIOLENCE_WINDOW            = 10
VIOLENCE_HIT_COUNT         = 4

# ─── Weapon Detection (Roboflow) ──────────────────────────────
ROBOFLOW_API_KEY = "t6T0pTV4rzmFKX0siajY"
ROBOFLOW_MODEL_ID   = "weapons-kjy5m/1"
ROBOFLOW_CONFIDENCE = 0.50
# How many AI-worker cycles between weapon API calls
# (weapon detection is async, so this just limits API rate)
WEAPON_EVERY_N      = 10

# ─── Logging ──────────────────────────────────────────────────
LOG_FILE       = os.path.join(os.path.dirname(__file__), "logs", "events.log")
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "logs", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ─── Flask ────────────────────────────────────────────────────
SECRET_KEY = "change-me-in-production"
DEBUG      = False          # Keep False — reloader breaks threading
