# =============================================================
# utils/drawing.py — Frame annotation utilities (v2)
# =============================================================

import cv2
import numpy as np

LEVEL_COLORS = {
    "safe":   (34,  197, 94),
    "level1": (234, 179, 8),
    "level2": (249, 115, 22),
    "level3": (239, 68,  68),
}

OBJECT_COLORS = {
    "person":  (99,  102, 241),
    "vehicle": (20,  184, 166),
    "pistol":  (239, 68,  68),
    "gun":     (239, 68,  68),
    "handgun": (239, 68,  68),
    "knife":   (249, 115, 22),
}


def draw_box(frame, x1, y1, x2, y2, label, color, track_id=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"#{track_id} {label}" if track_id is not None else label
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def draw_cam_label(frame, cam_id: int, online: bool):
    status = "LIVE" if online else "OFFLINE"
    color  = (34, 197, 94) if online else (239, 68, 68)
    label  = f" CAM {cam_id}  {status} "
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (6, 6), (6 + tw + 4, 6 + th + 6), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 6 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def draw_violence_overlay(frame, level: str):
    if level in ("level2", "level3"):
        color     = LEVEL_COLORS[level]
        thickness = 14 if level == "level3" else 7
        overlay   = frame.copy()
        h, w      = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), color, thickness * 2)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def draw_hud(frame, stats: dict):
    h, w   = frame.shape[:2]
    text   = (f"  P:{stats.get('persons',0)}"
              f"  V:{stats.get('vehicles',0)}"
              f"  W:{stats.get('weapons',0)}  ")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    y0 = h - th - 10
    cv2.rectangle(frame, (4, y0 - 4), (4 + tw + 4, y0 + th + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, y0 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    """Draw stream FPS counter in the top-right corner of the composite."""
    h, w   = frame.shape[:2]
    label  = f" {fps:.1f} fps "
    color  = (34, 197, 94) if fps >= 15 else (249, 115, 22)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    x0 = w - tw - 10
    cv2.rectangle(frame, (x0 - 2, 6), (x0 + tw + 2, 6 + th + 8), (0, 0, 0), -1)
    cv2.putText(frame, label, (x0, 6 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
