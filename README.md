# Image Object Detection & Enhancement Tool

This project provides a web-based tool to automatically detect objects in an uploaded image and allows the user to selectively "paint" or highlight a specific object with a chosen color.

It consists of a **Streamlit frontend** and a **Flask backend** that uses Meta's **Segment Anything Model (SAM)** for high-quality, pixel-perfect object segmentation masks.

## Features

- **Upload Images**: Upload `.jpg` or `.png` images via the Streamlit UI.
- **Automatic Object Detection**: The backend automatically detects distinct objects in the image.
- **Visual Bounding Boxes**: The UI overlays bounding boxes and IDs for detected objects.
- **Selective Painting**: Choose a specific object by its ID and apply a custom color mask over it.

## Project Structure

- `ui.py`: The Streamlit frontend application.
- `server.py`: Flask backend utilizing the SAM (Segment Anything Model) pipeline. Runs on port `5100`.
- `checkpoints/`: Directory intended for storing downloaded model weights (e.g., SAM model checkpoints like `sam_vit_b_01ec64.pth`).
- `uploads/`: Directory used by the backend to temporarily store original and modified images.

## Prerequisites & Installation

1. **Clone the repository** (or navigate to the project directory).
2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies**:
   Ensure you have the following installed (you can install them manually or create a `requirements.txt`):
   ```bash
   pip install flask pillow numpy torch torchvision segment-anything opencv-python streamlit requests
   ```

4. **Download SAM Checkpoint**:
   You must download the SAM weights (e.g., `sam_vit_b_01ec64.pth`) and place them in the `checkpoints/` folder.

## Usage

### 1. Start the SAM Backend Server

```bash
python server.py
```
*(Runs on port 5100)*

### 2. Start the Streamlit UI

In a new terminal window (with your virtual environment activated):

```bash
streamlit run ui.py
```

### 3. Enhance Images

1. Open the provided Streamlit URL in your browser (usually `http://localhost:8501`).
2. Upload an image.
3. Review the preview image with bounding boxes mapping out detected objects.
4. Select an object ID from the dropdown menu to see an isolated preview.
5. Pick an overlay color.
6. Click **Paint** to apply the color to the object and view the final result!
