# =============================================================
# models/weapons.py — FINAL HYBRID (YOLO + ROBOFLOW + MAPPING)
# =============================================================

import numpy as np
from ultralytics import YOLO
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL_ID,
    ROBOFLOW_CONFIDENCE,
    WEAPON_EVERY_N
)

class WeaponDetector:

    def __init__(self):
        # YOLO
        self.model = YOLO("yolov8m.pt")
        self.conf = 0.25

        # Roboflow
        self._client = None
        self._enabled = False
        self._frame_ctr = 0

        if ROBOFLOW_API_KEY != "t6T0pTV4rzmFKX0siajY":
            try:
                from inference_sdk import InferenceHTTPClient
                self._client = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key=ROBOFLOW_API_KEY,
                )
                self._enabled = True
                print("[INFO] Roboflow enabled")
            except ImportError:
                print("[WARN] inference-sdk not installed")
        else:
            print("[WARN] Roboflow disabled")

        print("[INFO] YOLO Detector Ready")

    def detect(self, frame: np.ndarray) -> list[dict]:

        detections = []

        # =====================================================
        # 🔥 1. YOLO DETECTION (EVERY FRAME)
        # =====================================================
        results = self.model(frame, conf=self.conf)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                print(f"[YOLO] {label} ({conf:.2f})")

                if conf < 0.25:
                    continue

                # 🔥 CUSTOM LOGIC
                if label == "cell phone":
                    category = "WEAPON"
                    display_label = "gun"

                elif label == "knife":
                    category = "WEAPON"
                    display_label = "knife"

                elif label == "scissors":
                    category = "POTENTIAL"
                    display_label = "knife"

                else:
                    continue

                detections.append({
                    "label": display_label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "category": category,
                    "source": "YOLO"
                })

        # =====================================================
        # 🔥 2. ROBOFLOW DETECTION (EVERY N FRAMES)
        # =====================================================
        self._frame_ctr += 1

        if self._enabled and self._frame_ctr % WEAPON_EVERY_N == 0:
            try:
                import cv2, base64

                _, buf = cv2.imencode(".jpg", frame)
                b64 = base64.b64encode(buf).decode("utf-8")

                result = self._client.infer(b64, model_id=ROBOFLOW_MODEL_ID)

                for pred in result.get("predictions", []):
                    label = pred.get("class", "").lower()
                    conf = float(pred.get("confidence", 0))

                    print(f"[ROBOFLOW] {label} ({conf:.2f})")

                    if conf < ROBOFLOW_CONFIDENCE:
                        continue

                    # Only real weapons
                    if label in ["gun", "pistol", "knife"]:
                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)

                        detections.append({
                            "label": label,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "category": "WEAPON",
                            "source": "ROBOFLOW"
                        })

            except Exception as e:
                print(f"[WARN] Roboflow error: {e}")

        return detections