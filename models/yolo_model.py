# =============================================================
# models/yolo_model.py — YOLOv8 Detection Wrapper
# =============================================================

from ultralytics import YOLO
import numpy as np
from config import (
    YOLO_MODEL, YOLO_CONFIDENCE,
    YOLO_PERSON_CLS, YOLO_VEHICLE_CLS
)


class YOLODetector:
    """
    Wraps YOLOv8 for person and vehicle detection.
    Returns structured detection dicts compatible with DeepSORT.
    """

    def __init__(self):
        print("[INFO] Loading YOLO model...")
        self.model = YOLO(YOLO_MODEL)
        self.model.overrides["verbose"] = False
        print("[INFO] YOLO model loaded")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run inference on a single frame.

        Returns:
          {
            "persons":  [ {"bbox": [x1,y1,x2,y2], "conf": float} ],
            "vehicles": [ {"bbox": [x1,y1,x2,y2], "conf": float, "label": str} ],
            "raw":      YOLO Results object
          }
        """
        results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)[0]
        persons  = []
        vehicles = []

        for box in results.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox  = [x1, y1, x2, y2]

            if cls == YOLO_PERSON_CLS:
                persons.append({"bbox": bbox, "conf": conf})
            elif cls in YOLO_VEHICLE_CLS:
                label = results.names[cls]
                vehicles.append({"bbox": bbox, "conf": conf, "label": label})

        return {"persons": persons, "vehicles": vehicles, "raw": results}
