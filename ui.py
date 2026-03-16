import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

API_URL = "http://localhost:5100"

st.title("Image Object Detecition tool")

uploaded = st.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded:
    # grab the raw bytes one time
    data = uploaded.getvalue()
    orig = Image.open(io.BytesIO(data))
    st.image(orig, caption="original", width=600)

    # now send a fresh BytesIO built from the bytes
    files = {"file": (uploaded.name, io.BytesIO(data), uploaded.type)}
    r = requests.post(f"{API_URL}/upload", files=files)

    st.write("upload status", r.status_code)
    st.write("upload response", r.text)
    if not r.ok:
        st.error("upload request failed")
    else:
        data = r.json()
        objects = data.get("objects", [])
        st.write(f"{len(objects)} objects detected")

        # Draw all bounding boxes on a preview image so user can see their IDs
        all_preview = orig.copy()
        draw_all = ImageDraw.Draw(all_preview)
        for o in objects:
            xmin, ymin, xmax, ymax = o["bbox"]
            obj_id = str(o["id"])
            draw_all.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
            draw_all.text((xmin, ymin), obj_id, fill="red")
        
        st.image(all_preview, caption="All detected objects (with IDs)", width=600)

        choices = [
            f"object {o['id']} (area={o['area']})" for o in objects
        ]
        selection = st.selectbox("pick an object", choices)

        if selection:
            sel_id = int(selection.split()[1])
            info = next(o for o in objects if o["id"] == sel_id)
            xmin, ymin, xmax, ymax = info["bbox"]

            preview = orig.copy()
            draw = ImageDraw.Draw(preview)
            draw.rectangle([xmin, ymin, xmax, ymax],
                           outline="red", width=3)
            st.image(preview, caption="selection preview", width=600)

        color = st.color_picker("pick a colour", "#ff0000")
        if st.button("paint"):
            obj_id = int(selection.split()[1])
            payload = {
                "filename": uploaded.name,
                "object_id": obj_id,
                "color": color,
            }
            r2 = requests.post(f"{API_URL}/paint", json=payload)
            #st.write("paint status", r2.status_code)
            st.write("paint response", r2.text[:500])
            if r2.ok:
                img = Image.open(io.BytesIO(r2.content))
                st.image(img, caption="modified", width=600)
            else:
                st.error("paint request failed")