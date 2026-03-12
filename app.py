import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="wide"
)

# Load trained model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/lung_cancer_model.h5")

model = load_model()

IMG_SIZE = 128

# ---------------- HEADER ---------------- #

st.markdown(
"""
<h1 style='text-align: center;'>🫁 Lung Cancer Detection System</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<p style='text-align: center; font-size:18px; color:gray;'>
"Early detection saves lives — AI can assist doctors in identifying lung cancer faster."
</p>
""",
unsafe_allow_html=True
)

st.info(
"This AI system analyzes CT scan images using a **Convolutional Neural Network (CNN)** "
"to detect possible signs of lung cancer."
)

st.divider()

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Project Information")

st.sidebar.write(
"""
### Technology Used
- Python
- TensorFlow / Keras
- CNN Deep Learning
- Streamlit

### Purpose
Assist in early detection of lung cancer from CT scan images.

### Classes
- Cancer
- Normal
"""
)

# ---------------- FILE UPLOADER ---------------- #

uploaded_file = st.file_uploader(
"Upload CT Scan Image",
type=["jpg", "jpeg", "png"]
)

st.subheader("Try Example CT Scans")

col1, col2 = st.columns(2)

example_image = None

with col1:
    if st.button("Example 1"):
        example_image = "examples/example1.jpg"

with col2:
    if st.button("Example 2"):
        example_image = "examples/example2.jpg"

if example_image:
    uploaded_file = example_image

# ---------------- PREDICTION ---------------- #

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # Display uploaded image
    with col1:
        st.subheader("Uploaded CT Scan")
        st.image(image, use_container_width=True)

    # Preprocess image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    # Show result
    with col2:
        st.subheader("Prediction Result")

        if prediction > 0.5:
            st.success("Normal Lung")
            confidence = prediction
        else:
            st.error("Cancer Detected")
            confidence = 1 - prediction

        st.write(f"Confidence: **{confidence*100:.2f}%**")

        # Probability chart
        data = pd.DataFrame({
            "Class": ["Cancer", "Normal"],
            "Probability": [1-prediction, prediction]
        })

        st.bar_chart(data.set_index("Class"))