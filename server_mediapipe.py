# server_mediapipe.py
import io
import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import cv2                               

# MediaPipe imports ---------------------------------------------------------
import mediapipe as mp

# choose a detector; you can swap in object_detection, etc.
mp_object = mp.solutions.objectron       # or mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils
# ---------------------------------------------------------------------------

app = Flask(__name__)


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.route("/upload", methods=["POST"])
def upload():
    """
    identical interface to server.py:
      - expects form-data 'file'
      - returns list of objects metadata
    but calling MediaPipe instead of SAM
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["file"]
    fname = file.filename or "unnamed"
    if fname == "":
        return jsonify({"error": "empty filename"}), 400

    # --- check cache first ---
    cached = app.config.get("mp_objects", {}).get(fname)
    if cached is not None:
        return jsonify({"objects": cached})

    # Ensure uploads dir exists and save the original image
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", fname)
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"could not open image: {e}"}), 400
    img.save(filepath)

    np_img = np.array(img)
    # MediaPipe expects BGR
    np_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    objects = []
    session_masks = []

    with mp_object.Objectron(static_image_mode=True) as detector:
        results = detector.process(np_bgr)

    for i, detection in enumerate(results.detections or []):
        rel = detection.location_data.relative_bounding_box
        h, w, _ = np_img.shape
        x1 = int(rel.xmin * w)
        y1 = int(rel.ymin * h)
        x2 = int((rel.xmin + rel.width) * w)
        y2 = int((rel.ymin + rel.height) * h)
        bbox = [x1, y1, x2, y2]

        score = float(detection.score[0])
        label = mp_object.CLASS_NAME[detection.label_id]

        # simple mask: fill the bbox region
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True

        objects.append({
            "id": i,
            "bbox": bbox,
            "score": score,
            "label": label,
        })
        session_masks.append(mask)

    app.config.setdefault("mp_objects", {})[fname] = objects
    app.config.setdefault("mp_masks", {})[fname] = session_masks
    return jsonify({"objects": objects})


@app.route("/paint", methods=["POST"])
def paint():
    """
    JSON body: { "filename": "...", "object_id": 3, "color": "#ff0000" }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "invalid json"}), 400

    fname = data.get("filename")
    obj_id = data.get("object_id")
    color = data.get("color")
    if fname is None or obj_id is None or color is None:
        return jsonify({"error": "missing fields"}), 400

    masks = app.config.get("mp_masks", {}).get(fname)
    if masks is None:
        return jsonify({"error": "unknown filename"}), 404

    try:
        mask = masks[int(obj_id)]
    except (IndexError, ValueError):
        return jsonify({"error": "invalid object_id"}), 400

    try:
        img = Image.open(os.path.join("uploads", fname)).convert("RGB")
    except FileNotFoundError:
        return jsonify({"error": "original image not found"}), 404

    np_img = np.array(img)

    color = color.lstrip("#")
    if len(color) != 6:
        return jsonify({"error": "invalid color"}), 400
    rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    np_img[mask] = rgb

    out = Image.fromarray(np_img)
    out_fname = f"modified_{fname}"
    out.save(os.path.join("uploads", out_fname))

    return send_file(io.BytesIO(image_to_bytes(out)), mimetype="image/png")


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(port=5101, debug=True)   # different port than the SAM server