# =============================================================
# streams/stream_handler.py — Non-Blocking AI Processing Pipeline v2
#
# Architecture (Producer / Consumer, fully decoupled):
#
#   ┌─────────────────────────────────────────────────────────┐
#   │  CameraStream threads  (one per camera)                 │
#   │  → grab() + retrieve() at full camera FPS               │
#   │  → stores latest frame in CameraStream._frame (locked) │
#   └────────────────────┬────────────────────────────────────┘
#                        │ read() — instant copy
#   ┌────────────────────▼────────────────────────────────────┐
#   │  AIWorker threads  (one per camera)                     │
#   │  → runs every AI_PROCESS_EVERY frames                   │
#   │  → YOLO → DeepSORT → MediaPipe → Violence → Weapon      │
#   │  → stores results in PerCameraState (locked)            │
#   └────────────────────┬────────────────────────────────────┘
#                        │ get_results() — instant
#   ┌────────────────────▼────────────────────────────────────┐
#   │  generate() — Flask MJPEG loop                          │
#   │  → reads latest frame (instant)                        │
#   │  → reads latest AI results (instant, no wait)          │
#   │  → draws annotations on frame copy                     │
#   │  → encodes JPEG → yields bytes                         │
#   └─────────────────────────────────────────────────────────┘
#
# Key guarantee: generate() NEVER waits for AI models.
#   It always has a fresh camera frame AND the most recent
#   detection results (even if they're a few frames old).
# =============================================================

import cv2
import time
import datetime
import threading
import logging
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from utils.camera  import CameraManager
from utils.drawing import (draw_box, draw_cam_label,
                            draw_violence_overlay, draw_hud,
                            draw_fps, OBJECT_COLORS)
from models.yolo_model import YOLODetector
from models.tracker    import MultiCameraTracker
from models.violence   import ViolenceDetector
from models.weapons    import WeaponDetector
from config import (AI_PROCESS_EVERY, JPEG_QUALITY,
                    SCREENSHOT_DIR, LOG_FILE,
                    AI_ENABLED_DEFAULT, TARGET_FPS)

log = logging.getLogger("stream")


# ── Priority Engine ───────────────────────────────────────────

def compute_violence_level(is_violent: bool, weapons: list) -> str:
    labels   = {w["label"].lower() for w in weapons}
    has_gun  = bool(labels & {"pistol", "gun", "handgun", "weapon"})
    has_knife = bool(labels & {"knife", "blade"})
    if is_violent and has_gun:   return "level3"
    if is_violent and has_knife: return "level2"
    if is_violent:               return "level1"
    return "safe"


# ── Per-camera AI result container ────────────────────────────

class PerCameraState:
    """
    Thread-safe container for the most-recent AI results for one camera.
    The AIWorker writes; the stream generator reads — never blocking each other.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "tracks":   [],     # [{global_id, bbox, centroid}]
            "vehicles": [],     # [{label, bbox}]
            "weapons":  [],     # [{label, confidence, bbox}]
            "violence": {
                "is_violent": False,
                "score":      0.0,
                "reasons":    [],
                "annotated_frame": None,
            },
            "level":    "safe",
            "fps_ai":   0.0,
        }

    def write(self, data: dict):
        with self._lock:
            self._data.update(data)

    def read(self) -> dict:
        with self._lock:
            return dict(self._data)   # shallow copy is fine


# ── Event Logger ─────────────────────────────────────────────

class EventLogger:
    def __init__(self):
        import os
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        self._lock = threading.Lock()

    def log(self, cam_id, level, persons, vehicles, weapons,
            frame: np.ndarray = None):
        ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wlabels = [w["label"] for w in weapons]
        entry   = (f"[{ts}] CAM={cam_id} LEVEL={level.upper()} "
                   f"persons={persons} vehicles={vehicles} "
                   f"weapons={wlabels}\n")
        with self._lock:
            with open(LOG_FILE, "a") as fh:
                fh.write(entry)
        if level in ("level2", "level3") and frame is not None:
            fname = (f"{SCREENSHOT_DIR}/CAM{cam_id}_"
                     f"{ts.replace(':','-').replace(' ','_')}_{level}.jpg")
            cv2.imwrite(fname, frame)


# ── AI Worker — one per camera ────────────────────────────────

class AIWorker:
    """
    Runs in its own daemon thread.
    Pulls the latest frame from CameraStream, runs the AI stack,
    and pushes results into PerCameraState — completely independent
    of the Flask streaming loop.
    """

    def __init__(self, cam_id: int, cam_stream,
                 yolo: YOLODetector,
                 tracker: MultiCameraTracker,
                 violence_det: ViolenceDetector,
                 weapon_det: WeaponDetector,
                 state: PerCameraState,
                 logger: EventLogger,
                 alert_queue: deque):

        self.cam_id       = cam_id
        self.cam          = cam_stream
        self.yolo         = yolo
        self.tracker      = tracker
        self.violence_det = violence_det
        self.weapon_det   = weapon_det
        self.state        = state
        self.logger       = logger
        self.alert_queue  = alert_queue

        self._stop_evt    = threading.Event()
        self._enabled     = True          # toggled from outside
        self._frame_count = 0

        # Weapon detection runs in a separate thread-pool executor
        # so its (potentially slow) API call never stalls MediaPipe/YOLO
        self._weapon_executor = ThreadPoolExecutor(max_workers=1,
                                                   thread_name_prefix=f"wpn-{cam_id}")
        self._pending_weapons  = []       # last known weapon results
        self._weapon_future    = None     # future for async weapon call

        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"ai-worker-{cam_id}"
        )
        self._thread.start()

    def _run(self):
        target_interval = 1.0 / max(1, TARGET_FPS)  # AI budget per frame

        while not self._stop_evt.is_set():
            t0 = time.monotonic()

            if not self._enabled or not self.cam.online:
                time.sleep(0.05)
                continue

            self._frame_count += 1

            # ── Throttle: only process every N frames ──────────
            # We sleep between runs so the thread does not spin.
            # The camera thread is still grabbing at full speed.
            if self._frame_count % AI_PROCESS_EVERY != 0:
                time.sleep(0.001)
                continue

            # ── Grab latest frame ──────────────────────────────
            frame = self.cam.read()

            try:
                # ── YOLO ───────────────────────────────────────
                detections   = self.yolo.detect(frame)
                persons_det  = detections["persons"]
                vehicles_det = detections["vehicles"]

                # ── DeepSORT ───────────────────────────────────
                tracks = self.tracker.update(self.cam_id, persons_det, frame)

                # ── MediaPipe Pose + Violence ──────────────────
                v_result = self.violence_det.process(frame)

                # ── Async Weapon Detection ─────────────────────
                # Check if previous async call finished
                if self._weapon_future is not None and self._weapon_future.done():
                    try:
                        self._pending_weapons = self._weapon_future.result()
                    except Exception:
                        self._pending_weapons = []
                    self._weapon_future = None

                # Submit new weapon detection job if none running
                if self._weapon_future is None:
                    frame_copy = frame.copy()
                    self._weapon_future = self._weapon_executor.submit(
                        self.weapon_det.detect, frame_copy
                    )

                weapons = self._pending_weapons

                # ── Priority Level ─────────────────────────────
                level = compute_violence_level(v_result["is_violent"], weapons)

                # ── Push results to shared state ───────────────
                self.state.write({
                    "tracks":   tracks,
                    "vehicles": vehicles_det,
                    "weapons":  weapons,
                    "violence": v_result,
                    "level":    level,
                    "fps_ai":   round(1.0 / max(0.001,
                                      time.monotonic() - t0), 1),
                })

                # ── Logging / Alerts ───────────────────────────
                if level in ("level1", "level2", "level3"):
                    self.logger.log(
                        self.cam_id, level, len(tracks),
                        len(vehicles_det), weapons,
                        frame if level in ("level2","level3") else None,
                    )
                    alert = {
                        "time":    datetime.datetime.now().strftime("%H:%M:%S"),
                        "cam":     self.cam_id,
                        "level":   level,
                        "reasons": v_result.get("reasons", []),
                        "weapons": [w["label"] for w in weapons],
                    }
                    self.alert_queue.appendleft(alert)

            except Exception as e:
                log.error(f"[AIWorker {self.cam_id}] Error: {e}", exc_info=False)

            # ── Sleep to maintain AI frame budget ──────────────
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, target_interval * AI_PROCESS_EVERY - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def stop(self):
        self._stop_evt.set()
        self._weapon_executor.shutdown(wait=False)


# ── Stream Handler ────────────────────────────────────────────

class StreamHandler:
    """
    Ties everything together.

    Public API:
      generate()           → MJPEG generator for Flask (never blocks on AI)
      get_status()         → JSON dict
      get_alerts(n)        → list of alert dicts
      add_camera(src)      → int (new cam_id)
      remove_camera(id)    → None
      set_ai_enabled(bool) → None  (toggle all AI workers)
    """

    def __init__(self):
        # ── Models (shared across all cameras) ─────────────────
        log.info("Loading AI models…")
        self.yolo    = YOLODetector()
        self.weapons = WeaponDetector()
        self.logger  = EventLogger()
        log.info("Models loaded")

        # ── Cameras ────────────────────────────────────────────
        self.cam_mgr = CameraManager()
        n            = self.cam_mgr.count()

        # ── Per-camera objects ─────────────────────────────────
        self.tracker  = MultiCameraTracker(n)
        self._states: list[PerCameraState] = []
        self._workers: list[AIWorker]      = []
        self._alert_queue: deque           = deque(maxlen=200)

        # Global toggle
        self._ai_enabled = AI_ENABLED_DEFAULT
        self._state_lock = threading.Lock()
        self._global: dict = {
            "overall_level": "safe",
            "total_persons":  0,
            "total_vehicles": 0,
            "total_weapons":  0,
        }

        # ── Stream-level FPS tracking ──────────────────────────
        self._stream_fps_times: deque = deque(maxlen=60)

        # Start AI workers
        for i, cam in enumerate(self.cam_mgr.cameras):
            state = PerCameraState()
            worker = AIWorker(
                cam_id       = i,
                cam_stream   = cam,
                yolo         = self.yolo,
                tracker      = self.tracker,
                violence_det = ViolenceDetector(),
                weapon_det   = self.weapons,
                state        = state,
                logger       = self.logger,
                alert_queue  = self._alert_queue,
            )
            self._states.append(state)
            self._workers.append(worker)

    # ── MJPEG Generator ───────────────────────────────────────

    def generate(self):
        """
        Flask MJPEG generator.

        This loop:
          1. Reads the latest raw frame from each camera  (instant)
          2. Reads the latest AI results from each camera (instant)
          3. Draws annotations (fast, pure NumPy/OpenCV — no ML)
          4. Encodes to JPEG
          5. Yields bytes

        It NEVER waits for YOLO / MediaPipe / Roboflow.
        If the AI workers haven't finished yet, it uses the previous result.
        """
        frame_interval = 1.0 / max(1, TARGET_FPS)

        while True:
            t_start = time.monotonic()

            raw_frames  = self.cam_mgr.get_frames()
            annotated   = []
            cam_summaries = []

            for cam_id, frame, online in raw_frames:
                state_data = (self._states[cam_id].read()
                              if cam_id < len(self._states) else {})
                ann = self._annotate(cam_id, frame, online, state_data)
                annotated.append(ann)
                cam_summaries.append({
                    "level":    state_data.get("level", "safe"),
                    "persons":  len(state_data.get("tracks", [])),
                    "vehicles": len(state_data.get("vehicles", [])),
                    "weapons":  state_data.get("weapons", []),
                    "tracks":   [t["global_id"]
                                 for t in state_data.get("tracks", [])],
                    "fps_ai":   state_data.get("fps_ai", 0.0),
                })

            # Update global state
            overall = self._worst_level(cam_summaries)
            with self._state_lock:
                self._global.update({
                    "overall_level":  overall,
                    "total_persons":  sum(c["persons"]  for c in cam_summaries),
                    "total_vehicles": sum(c["vehicles"] for c in cam_summaries),
                    "total_weapons":  sum(len(c["weapons"]) for c in cam_summaries),
                    "cameras":        {i: c for i, c in enumerate(cam_summaries)},
                })

            # Compose multi-camera grid
            composite = self.cam_mgr.compose(annotated)

            # Stream FPS overlay on composite
            now = time.monotonic()
            self._stream_fps_times.append(now)
            if len(self._stream_fps_times) >= 2:
                elapsed = self._stream_fps_times[-1] - self._stream_fps_times[0]
                stream_fps = (len(self._stream_fps_times) - 1) / max(elapsed, 0.001)
            else:
                stream_fps = 0.0
            draw_fps(composite, stream_fps)

            # Encode to JPEG
            ok, buf = cv2.imencode(
                ".jpg", composite,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )

            # Pace the generator to TARGET_FPS
            elapsed = time.monotonic() - t_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Annotation (pure OpenCV, no ML) ──────────────────────

    def _annotate(self, cam_id: int, frame: np.ndarray,
                  online: bool, state: dict) -> np.ndarray:
        """
        Draw bounding boxes, labels, and overlays on a COPY of the frame.
        This is fast and never blocks — it only uses the cached AI results.
        """
        out = frame.copy()

        if not online:
            draw_cam_label(out, cam_id, online)
            return out

        level    = state.get("level", "safe")
        tracks   = state.get("tracks",   [])
        vehicles = state.get("vehicles", [])
        weapons  = state.get("weapons",  [])
        v_data   = state.get("violence", {})

        # Person boxes
        for trk in tracks:
            x1, y1, x2, y2 = trk["bbox"]
            draw_box(out, x1, y1, x2, y2, "person",
                     OBJECT_COLORS["person"], trk["global_id"])

        # Vehicle boxes
        for veh in vehicles:
            x1, y1, x2, y2 = veh["bbox"]
            draw_box(out, x1, y1, x2, y2, veh.get("label","vehicle"),
                     OBJECT_COLORS["vehicle"])

        # Weapon boxes
        for wpn in weapons:
            x1, y1, x2, y2 = wpn["bbox"]
            clr = OBJECT_COLORS.get(wpn["label"].lower(), OBJECT_COLORS["pistol"])
            draw_box(out, x1, y1, x2, y2,
                     f"{wpn['label']} {wpn.get('confidence',0):.0%}", clr)

        # Skeleton overlay — copy annotated pose frame if available
        pose_frame = v_data.get("annotated_frame")
        if pose_frame is not None and pose_frame.shape == out.shape:
            cv2.addWeighted(pose_frame, 0.35, out, 0.65, 0, out)

        # Violence border
        draw_violence_overlay(out, level)

        # HUD
        draw_hud(out, {
            "persons":  len(tracks),
            "vehicles": len(vehicles),
            "weapons":  len(weapons),
        })

        # Camera label
        draw_cam_label(out, cam_id, online)

        return out

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _worst_level(cam_summaries: list) -> str:
        priority = {"level3": 3, "level2": 2, "level1": 1, "safe": 0}
        worst = "safe"
        for cs in cam_summaries:
            if priority.get(cs["level"], 0) > priority[worst]:
                worst = cs["level"]
        return worst

    # ── Public API ────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._state_lock:
            data = dict(self._global)
        data["camera_count"]  = self.cam_mgr.count()
        data["ai_enabled"]    = self._ai_enabled
        data["stream_fps"]    = round(
            (len(self._stream_fps_times) - 1) /
            max(0.001, (self._stream_fps_times[-1] - self._stream_fps_times[0])
                if len(self._stream_fps_times) >= 2 else 0.001), 1
        )
        data["camera_fps"]    = self.cam_mgr.get_fps()
        return data

    def get_alerts(self, n: int = 50) -> list:
        return list(self._alert_queue)[:n]

    def set_ai_enabled(self, enabled: bool):
        self._ai_enabled = enabled
        for w in self._workers:
            w.set_enabled(enabled)

    def add_camera(self, src) -> int:
        cam_id = self.cam_mgr.add_camera(src)
        state  = PerCameraState()
        worker = AIWorker(
            cam_id       = cam_id,
            cam_stream   = self.cam_mgr.cameras[cam_id],
            yolo         = self.yolo,
            tracker      = self.tracker,
            violence_det = ViolenceDetector(),
            weapon_det   = self.weapons,
            state        = state,
            logger       = self.logger,
            alert_queue  = self._alert_queue,
        )
        self.tracker.add_camera()
        self._states.append(state)
        self._workers.append(worker)
        return cam_id

    def remove_camera(self, cam_id: int):
        if 0 <= cam_id < len(self._workers):
            self._workers[cam_id].stop()
            self._workers.pop(cam_id)
            self._states.pop(cam_id)
        self.cam_mgr.remove_camera(cam_id)
