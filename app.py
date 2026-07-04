import streamlit as st
import numpy as np
from PIL import Image
import os
import onnxruntime as ort
import requests

st.set_page_config(
    page_title="🥦 Produce Freshness Predictor",
    page_icon="🥦",
    layout="centered"
)

st.title("🥦 AI Freshness & Shelf-Life Predictor")
st.markdown("### Built for Quick-Commerce Platforms like Zepto & Blinkit")
st.markdown("**Supports:** 🍌 Banana &nbsp;|&nbsp; 🍅 Tomato &nbsp;|&nbsp; 🥔 Potato")
st.write("Upload a produce image to get an instant freshness score and dispatch recommendation.")

@st.cache_resource
def load_model():
    if not os.path.exists('best_model_v2.onnx'):
        st.info("Downloading model... please wait (~30 seconds)")
        url = 'https://drive.google.com/uc?export=download&id=1ISydbI2W2oMrZlLShuiQHn6tmQVFCQKb'
        session = requests.Session()
        response = session.get(url, stream=True)

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = url + '&confirm=' + value
                response = session.get(url, stream=True)
                break

        with open('best_model_v2.onnx', 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

    return ort.InferenceSession('best_model_v2.onnx')

session = load_model()

st.markdown("---")

uploaded_file = st.file_uploader(
    "📸 Upload produce image (banana, tomato or potato)",
    type=["jpg","jpeg","png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: img_array})[0][0][0]
    freshness_score = (1 - pred) * 100

    st.markdown("---")
    st.metric(label="Freshness Score", value=f"{freshness_score:.1f} / 100")

    # Progress bar for visual impact
    st.progress(int(freshness_score))

    if freshness_score >= 60:
        st.success("✅ DISPATCH IMMEDIATELY — High freshness, prioritize for delivery")
        st.info("📦 Recommended Action: Standard dispatch within 24 hours")
    elif freshness_score >= 25:
        st.warning("⚠️ DISCOUNT & PRIORITIZE — Moderate freshness detected")
        st.info("💰 Recommended Action: Apply 20–30% discount, dispatch today")
    else:
        st.error("❌ REMOVE FROM INVENTORY — Spoilage risk detected")
        st.info("🗑️ Recommended Action: Flag for removal, do not dispatch")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "MobileNetV2")
    col2.metric("Test Accuracy", "99%")
    col3.metric("Test Images", "3,040")

    st.markdown("**Produce Supported:** Banana · Tomato · Potato")
    st.markdown("**Use Case:** Automated quality control at warehouse intake")
