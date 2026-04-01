# =============================================================
# models/violence.py — Motion-Based Violence Detection
# Uses MediaPipe Pose keypoints to detect punching, kicking,
# aggressive speed, and fight patterns (close interaction).
# =============================================================

import mediapipe as mp
import numpy as np
from collections import deque
from config import (
    POSE_DETECTION_CONFIDENCE, POSE_TRACKING_CONFIDENCE,
    VIOLENCE_SPEED_THRESHOLD, VIOLENCE_KICK_THRESHOLD,
    VIOLENCE_WINDOW, VIOLENCE_HIT_COUNT
)


class ViolenceDetector:
    """
    Per-camera violence detector.  Call `process(frame)` each frame.
    Returns a ViolenceResult with:
      - is_violent: bool
      - score:      0.0 – 1.0  (confidence)
      - reasons:    list[str]  (e.g. ["punch", "kick"])
      - pose_drawn: annotated frame (BGR)
    """

    # Landmark indices (MediaPipe Pose)
    _LEFT_WRIST  = 15
    _RIGHT_WRIST = 16
    _LEFT_ANKLE  = 27
    _RIGHT_ANKLE = 28
    _LEFT_SHOULDER  = 11
    _RIGHT_SHOULDER = 12
    _LEFT_HIP    = 23
    _RIGHT_HIP   = 24

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose    = self._mp_pose.Pose(
            min_detection_confidence=POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
            model_complexity=0,   # lightweight (0=lite, 1=full, 2=heavy)
        )
        self._mp_draw = mp.solutions.drawing_utils

        # Sliding windows of normalised landmark positions per joint
        self._prev: dict[int, np.ndarray] = {}
        self._motion_window: deque[bool]  = deque(maxlen=VIOLENCE_WINDOW)

    def _landmark_xy(self, landmarks, idx: int) -> np.ndarray | None:
        lm = landmarks.landmark[idx]
        if lm.visibility < 0.5:
            return None
        return np.array([lm.x, lm.y])

    def _speed(self, idx: int, landmarks) -> float:
        """Pixel-normalised velocity for a single landmark."""
        cur = self._landmark_xy(landmarks, idx)
        if cur is None:
            return 0.0
        prev = self._prev.get(idx)
        self._prev[idx] = cur
        if prev is None:
            return 0.0
        return float(np.linalg.norm(cur - prev))

    def _detect_close_interaction(self, landmarks_list) -> bool:
        """Return True if two tracked people are abnormally close (fight)."""
        if len(landmarks_list) < 2:
            return False
        def hip_center(lm):
            l = lm.landmark[self._LEFT_HIP]
            r = lm.landmark[self._RIGHT_HIP]
            return np.array([(l.x + r.x) / 2, (l.y + r.y) / 2])
        for i in range(len(landmarks_list)):
            for j in range(i+1, len(landmarks_list)):
                d = np.linalg.norm(hip_center(landmarks_list[i]) -
                                   hip_center(landmarks_list[j]))
                if d < 0.18:   # normalised screen distance
                    return True
        return False

    def process(self, frame: np.ndarray) -> dict:
        """
        Process a single BGR frame.
        Returns dict: {is_violent, score, reasons, annotated_frame}
        """
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True
        annotated = frame.copy()

        reasons:  list[str] = []
        is_frame_aggressive = False

        # Gather all detected pose instances (MediaPipe single-person model
        # returns one; for multi-person you'd need multi-pose or crop per bbox).
        all_landmarks = []

        if results.pose_landmarks:
            lm = results.pose_landmarks
            all_landmarks.append(lm)

            # ── Draw skeleton (subtle) ─────────────────────────────
            self._mp_draw.draw_landmarks(
                annotated, lm, self._mp_pose.POSE_CONNECTIONS,
                self._mp_draw.DrawingSpec(color=(100,100,255), thickness=1, circle_radius=2),
                self._mp_draw.DrawingSpec(color=(200,200,200), thickness=1),
            )

            # ── Speed-based punch detection ────────────────────────
            lw = self._speed(self._LEFT_WRIST,  lm)
            rw = self._speed(self._RIGHT_WRIST, lm)
            if max(lw, rw) > VIOLENCE_SPEED_THRESHOLD:
                reasons.append("punch")
                is_frame_aggressive = True

            # ── Speed-based kick detection ─────────────────────────
            la = self._speed(self._LEFT_ANKLE,  lm)
            ra = self._speed(self._RIGHT_ANKLE, lm)
            if max(la, ra) > VIOLENCE_KICK_THRESHOLD:
                reasons.append("kick")
                is_frame_aggressive = True

            # ── Arm elevation (raised fist / overhead strike) ──────
            lsh = self._landmark_xy(lm, self._LEFT_SHOULDER)
            rsh = self._landmark_xy(lm, self._RIGHT_SHOULDER)
            lw_pos = self._landmark_xy(lm, self._LEFT_WRIST)
            rw_pos = self._landmark_xy(lm, self._RIGHT_WRIST)
            if lsh is not None and lw_pos is not None and lw_pos[1] < lsh[1] - 0.05:
                reasons.append("raised_fist")
                is_frame_aggressive = True
            if rsh is not None and rw_pos is not None and rw_pos[1] < rsh[1] - 0.05:
                if "raised_fist" not in reasons:
                    reasons.append("raised_fist")
                is_frame_aggressive = True

        # ── Close-interaction check ────────────────────────────────
        if self._detect_close_interaction(all_landmarks):
            reasons.append("close_contact")
            is_frame_aggressive = True

        # ── Sliding window pattern check ──────────────────────────
        self._motion_window.append(is_frame_aggressive)
        hit_count = sum(self._motion_window)
        is_violent = hit_count >= VIOLENCE_HIT_COUNT

        score = min(1.0, hit_count / VIOLENCE_WINDOW)

        return {
            "is_violent":       is_violent,
            "score":            score,
            "reasons":          list(set(reasons)),
            "annotated_frame":  annotated,
        }

    def reset(self):
        """Reset state between cameras or after scene changes."""
        self._prev.clear()
        self._motion_window.clear()
