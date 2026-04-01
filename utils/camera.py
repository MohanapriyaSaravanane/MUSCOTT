# =============================================================
# utils/camera.py — Non-Blocking Multi-Camera Manager v2
#
# Architecture:
#   Each camera runs in its OWN daemon thread that does nothing
#   but cap.grab() as fast as possible.  cap.retrieve() is only
#   called when the consumer asks for a frame via read().
#   This prevents the reader from ever blocking the stream.
#
#   get_frame() / read() always return INSTANTLY — the latest
#   decoded frame is cached and returned under a lock.
# =============================================================

import cv2
import threading
import numpy as np
import time
import logging
from collections import deque
from config import CAMERA_SOURCES, FRAME_WIDTH, FRAME_HEIGHT

log = logging.getLogger("camera")


class CameraStream:
    """
    Non-blocking threaded camera reader.

    The background thread calls cap.grab() in a tight loop (very cheap —
    just DMA-copies the frame into the driver buffer without decoding).
    When the consumer calls read(), we call cap.retrieve() which decodes
    the *most recent* grabbed frame — so we never accumulate a backlog.

    On disconnect the thread retries every RECONNECT_INTERVAL seconds.
    """

    RECONNECT_INTERVAL = 5   # seconds between reconnect attempts
    GRAB_SLEEP         = 0  # tiny yield to avoid 100% CPU on fast cams

    def __init__(self, src, cam_id: int):
        self.cam_id = cam_id
        self.src    = src

        # Public state (read from outside under _lock)
        self.online     = False
        self.fps_actual = 0.0

        # Internal
        self._lock      = threading.Lock()
        self._frame     = self._blank_frame()
        self._cap       = None
        self._stop_evt  = threading.Event()

        # FPS tracking
        self._fps_times: deque = deque(maxlen=30)

        # Start capture + grab thread
        self._connect()
        self._thread = threading.Thread(
            target=self._grab_loop, daemon=True, name=f"cam-{cam_id}"
        )
        self._thread.start()

    # ── Helpers ───────────────────────────────────────────────

    def _blank_frame(self) -> np.ndarray:
        blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            blank, f"CAM {self.cam_id}  OFFLINE",
            (FRAME_WIDTH // 2 - 130, FRAME_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA,
        )
        return blank

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT),
                              interpolation=cv2.INTER_LINEAR)
        return frame

    def _connect(self) -> bool:
        """Try to open the camera. Returns True on success."""
        try:
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                log.error(f"[CAM {self.cam_id}] Cannot open {self.src}")
                self.online = False
                return False

            # Reduce internal OpenCV buffer to 1 frame — critical!
            # Without this, OpenCV queues up to 10+ frames and you get stale images.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm-up: grab + retrieve once to confirm stream is alive
            if not cap.grab():
                cap.release()
                self.online = False
                return False

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                cap.release()
                self.online = False
                return False

            self._cap = cap
            with self._lock:
                self._frame = self._resize(frame)
            self.online = True
            log.info(f"[CAM {self.cam_id}] Connected: {self.src}")
            return True

        except Exception as e:
            log.error(f"[CAM {self.cam_id}] Connection error: {e}")
            self.online = False
            return False

    # ── Background grab loop ──────────────────────────────────

    def _grab_loop(self):
        """
        Main capture loop — runs forever in a daemon thread.

        Strategy:
          1. Call cap.grab()  ← just signals hardware, no decode, very fast
          2. Call cap.retrieve() ← decode latest grabbed frame
          3. Store decoded frame under lock
          4. On failure → mark offline, wait, reconnect

        Because we immediately call retrieve() after grab(), the internal
        OpenCV queue never grows — we always have the *newest* frame.
        """
        while not self._stop_evt.is_set():
            if not self.online or self._cap is None:
                time.sleep(self.RECONNECT_INTERVAL)
                self._connect()
                continue

            try:
                grabbed = self._cap.grab()
                if not grabbed:
                    self._handle_disconnect("grab() failed")
                    continue

                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    self._handle_disconnect("retrieve() failed")
                    continue

                resized = self._resize(frame)
                with self._lock:
                    self._frame = resized

                # FPS tracking
                now = time.monotonic()
                self._fps_times.append(now)
                if len(self._fps_times) >= 2:
                    elapsed = self._fps_times[-1] - self._fps_times[0]
                    if elapsed > 0:
                        self.fps_actual = round(
                            (len(self._fps_times) - 1) / elapsed, 1
                        )

                # Tiny sleep prevents spinning at >200 fps on fast USB cams
                time.sleep(self.GRAB_SLEEP)

            except Exception as e:
                self._handle_disconnect(f"exception: {e}")

    def _handle_disconnect(self, reason: str):
        log.warning(f"[CAM {self.cam_id}] Disconnected — {reason}")
        self.online = False
        self.fps_actual = 0.0
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        with self._lock:
            self._frame = self._blank_frame()
        time.sleep(self.RECONNECT_INTERVAL)

    # ── Public API ────────────────────────────────────────────

    def read(self) -> np.ndarray:
        """Return latest frame (always instant, never blocks)."""
        with self._lock:
            return self._frame.copy()

    def get_frame(self) -> np.ndarray:
        """Alias for read() — matches CameraStream interface in prompt."""
        return self.read()

    def update(self):
        """No-op: updating is done by the background thread automatically."""
        pass

    def stop(self):
        """Signal the grab thread to exit and release the capture device."""
        self._stop_evt.set()
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass


# ── Camera Manager ────────────────────────────────────────────

class CameraManager:
    """Manages a pool of CameraStream instances."""

    def __init__(self):
        self.cameras: list[CameraStream] = []
        self._init_cameras()

    def _init_cameras(self):
        for i, src in enumerate(CAMERA_SOURCES):
            self.cameras.append(CameraStream(src, i))

    def add_camera(self, src) -> int:
        cam_id = len(self.cameras)
        self.cameras.append(CameraStream(src, cam_id))
        return cam_id

    def remove_camera(self, cam_id: int):
        if 0 <= cam_id < len(self.cameras):
            self.cameras[cam_id].stop()
            self.cameras.pop(cam_id)

    def get_frames(self) -> list[tuple[int, np.ndarray, bool]]:
        """Return (cam_id, frame, is_online) — always instant."""
        return [(i, cam.read(), cam.online) for i, cam in enumerate(self.cameras)]

    def get_fps(self) -> list[float]:
        return [cam.fps_actual for cam in self.cameras]

    def compose(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        1 cam  → full view
        2 cams → side-by-side
        3+ cams → 2-column grid
        """
        n = len(frames)
        if n == 0:
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        if n == 1:
            return frames[0]
        if n == 2:
            return np.hstack(frames)
        rows = []
        for i in range(0, n, 2):
            pair = frames[i:i + 2]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            rows.append(np.hstack(pair))
        return np.vstack(rows)

    def count(self) -> int:
        return len(self.cameras)

    def stop_all(self):
        for cam in self.cameras:
            cam.stop()
