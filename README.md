# MUSCOTT — Multi-Camera AI Surveillance System

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Roboflow API key for weapon detection
```bash
export ROBOFLOW_API_KEY="your_key_here"
```
Or edit `config.py` directly.

### 3. Configure cameras
Edit `config.py` → `CAMERA_SOURCES`:
```python
CAMERA_SOURCES = [
    0,                    # USB Camera 0
    # 1,                  # USB Camera 1
    # "rtsp://user:pass@192.168.1.127:554/stream",
    # "http://192.168.1.100:8080/video",  # DroidCam
]
```

### 4. Run
```bash
python app.py
```
Open **http://localhost:5000**

---

## Architecture

```
Camera Frames (threaded)
    ↓
YOLOv8 Detection (persons + vehicles)
    ↓
DeepSORT Tracking (cross-camera global IDs)
    ↓
MediaPipe Pose (keypoints)
    ↓
Motion Violence Detection (punch/kick/speed)
    ↓
Roboflow Weapon Detection (pistol / knife)
    ↓
Priority Engine → Level 1/2/3/Safe
    ↓
Flask MJPEG Stream + JSON API → Dashboard UI
```

## API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Dashboard |
| `/video_feed` | GET | MJPEG stream |
| `/status` | GET | JSON threat status |
| `/alerts` | GET | Recent alert log |
| `/log_file` | GET | Raw event log |
| `/add_camera` | POST | Add camera dynamically |
| `/remove_camera/<id>` | DELETE | Remove camera |

## Violence Levels

| Level | Colour | Trigger |
|---|---|---|
| CRITICAL (3) | 🔴 Red | Violence + Firearm |
| HIGH (2) | 🟠 Orange | Violence + Knife |
| ELEVATED (1) | 🟡 Yellow | Motion-only violence |
| SECURE | 🟢 Green | No threat |

## Performance Tips
- Reduce `FRAME_SKIP` in `config.py` for faster detection (uses more CPU)
- Use `yolov8n.pt` (nano) for speed, `yolov8s.pt` for accuracy
- Increase `WEAPON_EVERY_N` to reduce Roboflow API calls
