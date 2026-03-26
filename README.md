# AI Image Object Detection & Enhancement Tool

This project is a sophisticated image manipulation tool that uses a three-stage AI orchestration pipeline to detect, label, and selectively recolor objects in any image.

## Advanced AI Orchestration
Unlike traditional tools with fixed labels, this project uses a dynamic "Chain of Thought" pipeline:

1.  **GPT-4o-mini:** Analyzes the image visually via OpenAI's API to generate a custom list of every visible object (e.g., "vintage clock", "mahogany desk").
2.  **Grounding DINO:** Processes the GPT-generated list as a text prompt to find precise bounding boxes for those specific objects.
3.  **SAM 2:** Takes the bounding boxes and generates high-resolution, pixel-perfect segmentation masks using Segment Anything Model 2 by Meta.

## Features
- **Zero-Shot Dynamic Labeling:** No pre-defined classes. If GPT can see it, the tool Grounding DINO can find and label it by name.
- **Interactive Object Selection:** Click directly on objects in the image preview to select them (powered by `streamlit-image-coordinates`).
- **Visual Highlighting:** The currently selected object is highlighted in yellow for immediate feedback.
- **High-Precision Painting:** Pixel-level color replacement that preserves edges and background textures.
- **High-Resolution Support:** Optimized to handle large uploads while maintaining pixel clarity.

## Project Structure
- `ui.py`: Streamlit frontend with an interactive clickable canvas.
- `server.py`: Flask backend orchestrating GPT, DINO, and SAM 2.
- `checkpoints/`: Storage for model weights (e.g., `sam2_hiera_tiny.pt`).
- `uploads/`: Temporary storage for original and modified high-resolution images.
- `.env`: Secure storage for `OPENAI_API_KEY`.
- `requirements.txt`: Project dependencies.

## Prerequisites & Installation

1. **Setup Environment**:
   ```bash
   python -m venv venv
   venv/Scripts/Activate  # Mac: venv\bin\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up SAM2**:
You must configure an downloaded SAM2 checkpoint to have SAM2 work. Via the officaial SAM2 github(https://github.com/facebookresearch/sam2) choose and download a checkpoint. Add it to the checkpoints folder in the root of this app. Make sure ```SAM2_CHECKPOINT=...``` in server.py is set to your downloded checkpoint e.g.```SAM2_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt
"``` If you have a gpu you can go with a higher end checkpoint. If you only have a cpu make sure to use ```sam2_hiera_tiny.pt```.

## Usage
1. Setup project following the Prerequisites & Installation section.
2. Configure OpenAI key in .env
3. Start Backend ```python server.py```
4. Start Frontend ```streamlit run ui.py```

