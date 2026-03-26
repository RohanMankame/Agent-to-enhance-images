import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
from streamlit_image_coordinates import streamlit_image_coordinates

API_URL = "http://localhost:5100"

st.set_page_config(page_title="Click-to-Paint AI", layout="wide")

# Custom CSS to limit image height to 50% of the viewport height
st.markdown(
    """
    <style>
    [data-testid="stImage"] img {
        max-height: 50vh;
        width: auto !important;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI enhance images")

# Initialize Session States
if "selected_id" not in st.session_state:
    st.session_state.selected_id = 0

if "objects_list" not in st.session_state:
    st.session_state.objects_list = []

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# Support jpg and png
uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

# Clear cache if a new file is uploaded
if uploaded and uploaded.name != st.session_state.last_uploaded:
    st.session_state.objects_list = []
    st.session_state.selected_id = 0
    st.session_state.last_uploaded = uploaded.name

if uploaded:
    file_bytes = uploaded.getvalue()
    orig = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    # Upload to backend only once per image
    if not st.session_state.objects_list:
        with st.spinner("Analyzing with GPT + DINO + SAM 2..."):
            files = {"file": (uploaded.name, io.BytesIO(file_bytes), uploaded.type)}
            r = requests.post(f"{API_URL}/upload", files=files)
            if r.ok:
                st.session_state.objects_list = r.json().get("objects", [])
                st.rerun()
            else:
                st.error("Server analysis failed.")

    objects = st.session_state.objects_list

    # --- SECTION 1: PREVIEW ---
    st.subheader("1. Identify Objects")
    
    # Prepare Preview
    preview = orig.copy()
    draw = ImageDraw.Draw(preview)
    
    for o in objects:
        is_active = (o["id"] == st.session_state.selected_id)
        color = "yellow" if is_active else "lime"
        draw.rectangle(o["bbox"], outline=color, width=(4 if is_active else 1))
        if not is_active:
            draw.text((o["bbox"][0], o["bbox"][1]-12), f"{o['id']}", fill="lime")

    # Static Preview (Interaction removed for better reliability)
    st.image(preview, caption="Object Preview", use_container_width=True, width=None)

    st.divider()

    # --- SECTION 2: PAINT ---
    st.subheader("2. Paint Selection")
    if objects:
        # Layout for controls
        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        
        with ctrl_col1:
            # Sync Dropdown with Click State
            choices = [f"{o['id']}: {o['label']}" for o in objects]
            choice_ids = [o["id"] for o in objects]
            
            # Find current index for dropdown
            try:
                curr_idx = choice_ids.index(st.session_state.selected_id)
            except:
                curr_idx = 0

            selection = st.selectbox("Current Selection", choices, index=curr_idx)
            # Update state immediately if dropdown changes
            selected_id = int(selection.split(":")[0])
            if st.session_state.selected_id != selected_id:
                st.session_state.selected_id = selected_id
                st.rerun()

        with ctrl_col2:
            color = st.color_picker("Pick Paint Color", "#FF0000")

        if st.button("Apply Paint", use_container_width=True):
            payload = {
                "filename": uploaded.name,
                "object_id": st.session_state.selected_id,
                "color": color
            }
            with st.spinner("Modifying pixels..."):
                r_paint = requests.post(f"{API_URL}/paint", json=payload)
                if r_paint.ok:
                    res_bytes = r_paint.content
                    st.image(io.BytesIO(res_bytes), caption="Painted Result", use_container_width=True)
                    
                    # Download Button
                    st.download_button(
                        label="Download Image",
                        data=res_bytes,
                        file_name=f"enhanced_{uploaded.name}",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    st.error(f"Paint request failed: {r_paint.text}")