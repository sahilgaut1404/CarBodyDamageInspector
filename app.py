from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
from torchvision.transforms.functional import to_tensor
from inspector import CLASSES, load_model

st.title("Car Body Damage Inspector")
st.caption("Upload a car photo to review visible dents, scratches and cracks.")
file = st.file_uploader("Inspection photo", type=["jpg", "jpeg", "png"])
if file is None:
    st.stop()
try:
    photo = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    photo.thumbnail((1600, 1600))
except (OSError, Image.DecompressionBombError):
    st.error("Please upload a readable JPEG or PNG photo.")
    st.stop()
st.image(photo, caption="Original inspection photo")
weights = Path(__file__).with_name("damage.pt")
if not weights.exists():
    st.info("To enable predictions, complete the training notebook and place damage.pt beside app.py.")
    st.stop()


@st.cache_resource
def get_model(version):
    return load_model(weights, "cpu")


@st.cache_data(max_entries=4, show_spinner="Inspecting the photo…")
def detect(rgb, version):
    with torch.inference_mode():
        result = get_model(version)([to_tensor(rgb)])[0]
    return [(int(label), float(score), (mask[0].numpy() >= 0.5).astype("uint8"))
            for label, score, mask in zip(result["labels"], result["scores"], result["masks"])
            if score >= 0.1]


threshold = st.slider("Minimum confidence", 0.1, 0.95, 0.5, 0.05)
rgb = np.asarray(photo)
canvas = rgb.copy()
version = weights.stat().st_mtime_ns
photo_id = sha256(file.getvalue()).hexdigest()
rows = []
for index, (label, score, mask) in enumerate(detect(rgb, version), 1):
    if score < threshold or not mask.any():
        continue
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (245, 165, 35), 2)
    x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
    name = CLASSES[label]
    cv2.putText(canvas, f"{index} {name}", (x, max(18, y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (245, 165, 35), 2)
    rows.append(dict(id=index, defect=name, confidence=round(score, 4)))
if rows:
    st.image(canvas, caption="Model predictions")
else:
    st.info("No dent, scratch or crack detected at this confidence threshold.")
for row in rows:
    key = f"{photo_id}-{version}-{row['id']}"
    st.write(f"#{row['id']} · {row['defect'].title()} · {row['confidence']:.0%} confidence")
    row["decision"] = st.selectbox("Review decision", ["Pending", "Confirm", "Reject"], key=key)
    row["panel"] = st.text_input("Body panel", placeholder="e.g. left front door", key=key+"panel")
report = dict(photo=file.name, threshold=threshold, findings=rows)
st.download_button("Download review", json.dumps(report, indent=2), "inspection.json", "application/json")
