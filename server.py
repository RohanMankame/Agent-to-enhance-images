# server.py
import io
import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import torch

# Import SamAutomaticMaskGenerator instead of SamPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

app = Flask(__name__)

print("Available SAM models:", sam_model_registry.keys())

MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"

# load once at startup
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(DEVICE)

# Use the automatic generator with more reasonable parameters to avoid OOM
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=48,            # Modest increase in density
    pred_iou_thresh=0.8,           # Lower threshold to keep more masks
    stability_score_thresh=0.8,    # Lower threshold
    min_mask_region_area=100       # Filter out tiny noise
)


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.route("/upload", methods=["POST"])
def upload():
    """
    Expects form-data with 'file' field.
    Returns a list of object masks with simple metadata.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["file"]
    fname = file.filename or "unnamed"
    if fname == "":
        return jsonify({"error": "empty filename"}), 400

    # --- check cache first ---
    cached = app.config.get("objects", {}).get(fname)
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

    # Downscale to limit memory use
    max_dim = 1024
    if max(np_img.shape[:2]) > max_dim:
        scale = max_dim / max(np_img.shape[:2])
        img = img.resize((int(np_img.shape[1]*scale), int(np_img.shape[0]*scale)), Image.Resampling.LANCZOS)
        np_img = np.array(img)

    try:
        masks = mask_generator.generate(np_img)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"mask generation failed: {exc}"}), 500

    objects = []
    session_masks = []
    for i, mask_data in enumerate(masks):
        seg = mask_data.get("segmentation")
        if seg is None:
            continue
        session_masks.append(seg)
        x, y, w, h = mask_data.get("bbox", (0, 0, 0, 0))
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        area = int(mask_data.get("area", 0))
        objects.append({"id": i, "bbox": bbox, "area": area})

    # cache both metadata and masks
    app.config.setdefault("objects", {})[fname] = objects
    app.config.setdefault("masks", {})[fname] = session_masks

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

    masks = app.config.get("masks", {}).get(fname)
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
    app.run(port=5100, debug=True)