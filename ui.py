import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

API_URL = "http://localhost:5100"

st.set_page_config(page_title="AI Image Enhancer", layout="wide")
st.title("AI Object Labeler & Painter")

uploaded = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded:
    col1, col2 = st.columns(2)
    
    data = uploaded.getvalue()
    orig = Image.open(io.BytesIO(data))
    
    with col1:
        st.image(orig, caption="Original Image", use_container_width=True)

    # 1. Send to Backend
    files = {"file": (uploaded.name, io.BytesIO(data), uploaded.type)}
    with st.spinner("Analyzing image with DINO + SAM 2..."):
        r = requests.post(f"{API_URL}/upload", files=files)

    if r.ok:
        res_data = r.json()
        objects = res_data.get("objects", [])
        
        # 2. Draw Labels and Boxes for the Preview
        preview_img = orig.copy()
        draw = ImageDraw.Draw(preview_img)
        
        for o in objects:
            box = o["bbox"] # [xmin, ymin, xmax, ymax]
            label_text = f"{o['id']}: {o['label']}"
            
            # Draw Box
            draw.rectangle(box, outline="lime", width=3)
            # Draw Label Background
            draw.rectangle([box[0], box[1] - 20, box[0] + 80, box[1]], fill="lime")
            # Draw Label Text
            draw.text((box[0] + 5, box[1] - 18), label_text, fill="black")
            
        with col2:
            st.image(preview_img, caption="AI Detections", use_container_width=True)

        # 3. User Controls
        st.divider()
        # Dropdown now uses the LABELS!
        choices = [f"{o['id']}: {o['label']} (Score: {o.get('score', 0):.2f})" for o in objects]
        selection = st.selectbox("Pick an object to recolor:", choices)
        
        color = st.color_picker("Pick a brand new color:", "#FF0000")

        if st.button("✨ Paint Object"):
            sel_id = int(selection.split(":")[0])
            payload = {
                "filename": uploaded.name, 
                "object_id": sel_id, 
                "color": color
            }
            
            with st.spinner("Applying digital paint..."):
                r2 = requests.post(f"{API_URL}/paint", json=payload)
                
            if r2.ok:
                st.image(io.BytesIO(r2.content), caption="Final Result", width=1000)
            else:
                st.error("Painting failed. Check backend logs.")
    else:
        st.error(f"Detection failed: {r.text}")