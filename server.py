import io
import os
import torch
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
from openai import OpenAI
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from dotenv import load_dotenv

# Load key from .env
load_dotenv()

# SAM 2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: sam2 library not found. Install from Meta's SAM 2 repository.")

app = Flask(__name__)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml"

# Initialize OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- MODEL INITIALIZATION ---
print(f"Loading Grounding DINO ({DINO_MODEL_ID})...")
dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)

print("Loading SAM 2...")
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)






def get_dynamic_labels(image_path):
    """ Get object names from GPT4o-mini vision. Returns a comma-separated string of nouns. """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "List every distinct object in this image as a simple comma-separated string of nouns. No descriptions, just nouns. If there are multiple of the same type of object keep listing the object as many times as it appears in the image. make sure view the entire image before responding."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            ]
        }],
        max_tokens=100
    )
    return response.choices[0].message.content






def image_to_bytes(img: Image.Image) -> bytes:
    """
    convert PIL Image to bytes.
    """
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
    
    # Store dynamic metadata and masks in memory
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", fname)
    img = Image.open(file.stream).convert("RGB")
    img.save(filepath)
    np_img = np.array(img)

    # 1. GPT Dynamic Prompt
    try:
        raw_labels = get_dynamic_labels(filepath)
        dino_prompt = raw_labels.replace(",", ".")
        print(f"Vision Labels: {dino_prompt}")
    except Exception as e:
        return jsonify({"error": f"GPT vision failed: {e}"}), 500

    # 2. Grounding DINO Detection
    inputs = dino_processor(images=img, text=dino_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.25, target_sizes=[img.size[::-1]]
    )[0]

    if len(results["boxes"]) == 0:
        return jsonify({"objects": [], "message": "No objects detected by DINO."})

    # 3. SAM 2 Segmentation
    predictor.set_image(np_img)
    boxes = results["boxes"].cpu().numpy()
    masks, _, _ = predictor.predict(box=boxes, multimask_output=False)

    objects = []
    session_masks = []
    for i, (mask, box, label, score) in enumerate(zip(masks, boxes, results["labels"], results["scores"])):
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
    """
    Expects JSON with: filename, object_id, color (hex string), current_image (base64).
    paints the specified object with the given color and returns the modified image.
    """
    data = request.get_json(force=True)
    fname = data.get("filename")
    obj_id = int(data.get("object_id"))
    color_hex = data.get("color")
    current_image_b64 = data.get("current_image")

    masks = app.config.get("masks", {}).get(fname)
    if not masks or obj_id >= len(masks):
        return jsonify({"error": f"Object {obj_id} not found in cache for {fname}"}), 404

    # Use current_image from frontend if provided, otherwise fallback to disk
    if current_image_b64:
        img = Image.open(io.BytesIO(base64.b64decode(current_image_b64))).convert("RGB")
    else:
        img = Image.open(os.path.join("uploads", fname)).convert("RGB")
        
    np_img = np.array(img)
    mask = masks[obj_id]

    rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    np_img[mask] = rgb

    out = Image.fromarray(np_img)
    return send_file(io.BytesIO(image_to_bytes(out)), mimetype="image/png")

if __name__ == "__main__":
    app.run(port=5100, debug=True)