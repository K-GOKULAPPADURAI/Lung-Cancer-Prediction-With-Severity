import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import base64

def load_model():
    model = YOLO(r"runs\detect\train8\weights\best.pt")  # Replace with actual model path
    return model

def predict(image, model):
    results = model(image)
    bounding_boxes = []
    total_area = image.shape[0] * image.shape[1]
    affected_area = 0
    class_labels = {0: "Cancer", 1: "Nodule", 2: "Multi Nodule"}
    detected_classes = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            box_area = (x2 - x1) * (y2 - y1)
            affected_area += box_area
            bounding_boxes.append([int(x1), int(y1), int(x2), int(y2), class_id])
            detected_classes.append(class_labels[class_id])
    
    severity_percentage = (affected_area / total_area) * 100
    severity_label = categorize_severity(severity_percentage)
    
    return bounding_boxes, severity_percentage, severity_label, detected_classes

def categorize_severity(severity_percentage):
    if severity_percentage < 10:
        return "Mild"
    elif severity_percentage < 30:
        return "Moderate"
    else:
        return "Severe"

def draw_bounding_boxes(image, bounding_boxes):
    colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    for box in bounding_boxes:
        x1, y1, x2, y2, class_id = box
        color = colors[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

# Streamlit UI Customization
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")



def get_base64_of_gif(file_path):
    with open(file_path, "rb") as file:
        encoded_gif = base64.b64encode(file.read()).decode()
    return encoded_gif

bg_gif = get_base64_of_gif("bg.gif")

# background-image: url("data:image/gif;base64,{bg_gif}");
    
# Background styling with your GIF
page_bg_img = f'''
<style>

.block-container {{
            padding-top: 0rem;  /* Or any value you want */
            padding-bottom: 0rem; /* Optional: adjust bottom padding */
        }}

[data-testid="stAppViewContainer"] {{
    background-image: url("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExcTl2MjFzdG1iYjRqdjcxazdoOWEyNjdjbWM2YWhkYTFrMnZnamVvdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dt0KXLj7bzwZuRQBwY/giphy.gif");
    background-size: 45% auto;
    background-repeat: no-repeat;
    background-position: right center;
    background-attachment: fixed;
    padding-top: 0px;
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}

/* File uploader styling */
[data-testid="stFileUploader"] {{
        width: 50%; /* Or any desired width */
        float: left; /* Or any desired layout */
    }}

.st-emotion-cache-1gulkj5 {{
    width: 100% !important;
    max-width: 100% !important;
}}

.st-emotion-cache-7ym5gk {{
    width: 100% !important;
    max-width: 100% !important;
}}

/* Content container */
.st-emotion-cache-1v0mbdj {{
    margin: 0 auto;
    display: block;
}}

/* Column styling */
.st-emotion-cache-1kyxreq {{
    justify-content: center;
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🫁 Lung Cancer Detection & Severity Assessment")
st.subheader("Upload a Chest X-ray or CT Scan 🔬")
model = load_model()

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    bounding_boxes, severity_percentage, severity_label, detected_classes = predict(image, model)
    
    image_with_boxes = draw_bounding_boxes(image.copy(), bounding_boxes)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(image_with_boxes, caption="Detection Results")
    with col2:
        st.markdown(f"### 🏥 Severity Level: {severity_label}")
        st.markdown(f"**Affected Area:** {severity_percentage:.2f}% of Lungs")
        st.markdown(f"### 🔍 Detected Conditions:")
        for cls in set(detected_classes):
            st.markdown(f"- **{cls}**")