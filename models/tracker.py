# =============================================================
# models/tracker.py — DeepSORT Multi-Camera Tracker
# Cross-camera tracking: maintains the same track ID when a
# person moves between cameras using centroid proximity.
# =============================================================

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from config import DEEPSORT_MAX_AGE, DEEPSORT_N_INIT, DEEPSORT_MAX_IOU


class MultiCameraTracker:
    """
    Manages one DeepSort instance per camera and provides cross-camera
    identity resolution via a shared global-ID registry.

    object_camera_map:  global_id → {"cam_id": int, "local_id": int, "centroid": (cx,cy)}
    """

    def __init__(self, num_cameras: int):
        self.trackers: list[DeepSort] = [
            DeepSort(
                max_age=DEEPSORT_MAX_AGE,
                n_init=DEEPSORT_N_INIT,
                max_iou_distance=DEEPSORT_MAX_IOU,
            )
            for _ in range(num_cameras)
        ]
        # global_id → last known camera & centroid
        self.object_camera_map: dict[int, dict] = {}
        self._global_id_counter = 0
        # local (cam_id, local_track_id) → global_id
        self._local_to_global: dict[tuple, int] = {}

    # ── Internal helpers ──────────────────────────────────────

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _closest_global_id(self, centroid, max_dist=120):
        """Find the nearest global ID whose last known centroid is within max_dist pixels."""
        best_gid, best_d = None, max_dist
        for gid, info in self.object_camera_map.items():
            cx, cy = info["centroid"]
            d = ((centroid[0]-cx)**2 + (centroid[1]-cy)**2) ** 0.5
            if d < best_d:
                best_d, best_gid = d, gid
        return best_gid

    def _get_or_create_global_id(self, cam_id: int, local_id: int, centroid: tuple) -> int:
        key = (cam_id, local_id)
        if key in self._local_to_global:
            gid = self._local_to_global[key]
        else:
            # Try to match to an existing global ID from a different camera
            gid = self._closest_global_id(centroid)
            if gid is None:
                self._global_id_counter += 1
                gid = self._global_id_counter
            self._local_to_global[key] = gid

        self.object_camera_map[gid] = {
            "cam_id":   cam_id,
            "local_id": local_id,
            "centroid": centroid,
        }
        return gid

    # ── Public API ────────────────────────────────────────────

    def update(self, cam_id: int, detections: list, frame: np.ndarray) -> list:
        """
        Update the tracker for `cam_id` with raw YOLO person detections.

        detections: list of {"bbox": [x1,y1,x2,y2], "conf": float}

        Returns list of:
          {"global_id": int, "local_id": int, "bbox": [x1,y1,x2,y2]}
        """
        if cam_id >= len(self.trackers):
            return []

        # DeepSORT input: list of ([left,top,w,h], conf, class_label)
        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            ds_input.append(([x1, y1, x2-x1, y2-y1], det["conf"], "person"))

        tracks = self.trackers[cam_id].update_tracks(ds_input, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            lid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            centroid = self._centroid([x1, y1, x2, y2])
            gid = self._get_or_create_global_id(cam_id, lid, centroid)
            results.append({
                "global_id": gid,
                "local_id":  lid,
                "bbox":      [x1, y1, x2, y2],
                "centroid":  centroid,
            })
        return results

    def add_camera(self):
        """Add a tracker for a newly added camera."""
        self.trackers.append(
            DeepSort(
                max_age=DEEPSORT_MAX_AGE,
                n_init=DEEPSORT_N_INIT,
                max_iou_distance=DEEPSORT_MAX_IOU,
            )
        )
