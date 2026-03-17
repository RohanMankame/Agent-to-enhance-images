import io
import os
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# Sam2 Checkpoints are avalable at 
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: sam2 library not found. Please install it from the official Meta SAM 2 repository.")

app = Flask(__name__)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
# Update these paths to where your weights are stored
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml" 

# --- MODEL INITIALIZATION ---
print(f"Loading Grounding DINO ({DINO_MODEL_ID})...")
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)

print("Loading SAM 2...")
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["file"]
    fname = file.filename or "unnamed"
    
    # Simple in-memory cache
    if fname in app.config.get("objects", {}):
        return jsonify({"objects": app.config["objects"][fname]})

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", fname)
    img = Image.open(file.stream).convert("RGB")
    img.save(filepath)
    np_img = np.array(img)

    # 1. Grounding DINO - Detection & Labeling
    # This prompt tells DINO what classes to look for
    prompt = "all objects. person. chair. table. bottle. cup. computer. plant."
    inputs = dino_processor(images=img, text=prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    results = dino_processor.post_process_grounded_object_detection(
        outputs, 
        inputs.input_ids, 
        threshold=0.35, 
        target_sizes=[img.size[::-1]]
    )[0]

    if len(results["boxes"]) == 0:
        return jsonify({"objects": [], "message": "No objects found"})

    # 2. SAM 2 - Precise Segmentation
    predictor.set_image(np_img)
    boxes = results["boxes"].cpu().numpy()
    
    # SAM 2 can predict masks from multiple boxes at once
    masks, scores, _ = predictor.predict(
        box=boxes,
        multimask_output=False
    )

    objects = []
    session_masks = []
    
    # 3. Format Response
    for i, (mask, box, label, score) in enumerate(zip(masks, boxes, results["labels"], results["scores"])):
        # mask is [1, H, W], we want [H, W]
        binary_mask = mask[0] > 0.5 
        session_masks.append(binary_mask)
        
        objects.append({
            "id": i,
            "label": label,
            "bbox": [int(x) for x in box],
            "area": int(binary_mask.sum()),
            "score": round(float(score), 3)
        })

    app.config.setdefault("objects", {})[fname] = objects
    app.config.setdefault("masks", {})[fname] = session_masks

    return jsonify({"objects": objects})

@app.route("/paint", methods=["POST"])
def paint():
    data = request.get_json(force=True)
    fname = data.get("filename")
    obj_id = data.get("object_id")
    color_hex = data.get("color", "#FF0000")

    masks = app.config.get("masks", {}).get(fname)
    if not masks or obj_id >= len(masks):
        return jsonify({"error": "Unknown object or filename"}), 400

    img = Image.open(os.path.join("uploads", fname)).convert("RGB")
    np_img = np.array(img)
    mask = masks[int(obj_id)]

    # Convert hex to RGB
    rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    np_img[mask] = rgb

    out = Image.fromarray(np_img)
    return send_file(io.BytesIO(image_to_bytes(out)), mimetype="image/png")

if __name__ == "__main__":
    app.run(port=5100, debug=True)