# =============================================================
# app.py — Flask Application Entry Point (v2)
# Run:  python app.py
# Open: http://localhost:5000
# =============================================================

import os
import logging
from flask import (Flask, render_template, Response,
                   jsonify, request, abort)

# Configure logging before imports that use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from config import SECRET_KEY, LOG_FILE
from streams.stream_handler import StreamHandler

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Single global handler — initialised once at startup
handler = StreamHandler()


# ─── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           cam_count=handler.cam_mgr.count())


@app.route("/video_feed")
def video_feed():
    """Non-blocking MJPEG stream."""
    return Response(
        handler.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return jsonify(handler.get_status())


@app.route("/alerts")
def alerts():
    n = request.args.get("n", 50, type=int)
    return jsonify(handler.get_alerts(n))


@app.route("/log_file")
def log_file():
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()[-200:]
        return Response("".join(lines), mimetype="text/plain")
    except FileNotFoundError:
        return Response("No events logged yet.", mimetype="text/plain")


@app.route("/toggle_ai", methods=["POST"])
def toggle_ai():
    """Enable or disable all AI workers at runtime."""
    data    = request.get_json(force=True)
    enabled = bool(data.get("enabled", True))
    handler.set_ai_enabled(enabled)
    return jsonify({"ai_enabled": enabled})


@app.route("/add_camera", methods=["POST"])
def add_camera():
    data = request.get_json(force=True)
    src  = data.get("source")
    if src is None:
        abort(400, "Missing 'source' field")
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass
    cam_id = handler.add_camera(src)
    return jsonify({"cam_id": cam_id, "message": f"Camera {cam_id} added"})


@app.route("/remove_camera/<int:cam_id>", methods=["DELETE"])
def remove_camera(cam_id):
    handler.remove_camera(cam_id)
    return jsonify({"message": f"Camera {cam_id} removed"})


# ─── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    # use_reloader=False is CRITICAL — the reloader forks the process
    # which breaks all our daemon threads.
    app.run(host="0.0.0.0", port=5000, debug=False,
            threaded=True, use_reloader=False)
