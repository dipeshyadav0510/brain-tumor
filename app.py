"""
Streamlit app for Brain Tumor Detection.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from preprocessing import preprocess_image

MODEL_PATH = Path("models/best_mlp.h5")
SCALER_PATH = Path("models/hog_scaler.pkl")


@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_from_upload(upload, model, scaler):
    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as tmp:
        tmp.write(upload.getbuffer())
        tmp_path = Path(tmp.name)
    processed = preprocess_image(tmp_path)
    features = scaler.transform(processed.hog_features.reshape(1, -1))
    prob = float(model.predict(features)[0][0])
    label = "Tumor Detected" if prob >= 0.5 else "No Tumor Detected"
    return label, prob, processed


def main():
    st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
    st.title("Brain Tumor Detection from MRI")
    st.markdown(
        "Upload an MRI image to detect tumor presence. "
        "**This tool is for educational/research purposes only. Consult a medical professional for diagnosis.**"
    )

    uploaded = st.file_uploader("Upload MRI image (.jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"])
    if uploaded:
        model, scaler = load_model_and_scaler()
        label, prob, processed = predict_from_upload(uploaded, model, scaler)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(Image.open(uploaded), use_column_width=True)
        with col2:
            st.subheader("Preprocessed (cropped/resized)")
            st.image(processed.resized_rgb, use_column_width=True)

        st.markdown(f"### Prediction: **{label}**")
        st.progress(int(prob * 100))
        st.write(f"Confidence: {prob:.2%}")

        with st.expander("View preprocessing steps"):
            st.image(processed.mask, caption="Brain mask", use_column_width=True)
            hog_viz = np.clip(processed.hog_image, 0.0, 1.0).astype(np.float32)
            st.image(hog_viz, caption="HOG visualization", use_column_width=True, clamp=True)

        st.info("Prediction time includes preprocessing and HOG extraction.")
    else:
        st.write("Awaiting image upload.")


if __name__ == "__main__":
    main()

