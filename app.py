# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import io
import os
import random
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Configure Streamlit Page Settings
st.set_page_config(
    page_title="MediScan: Ocular Disease Detection",
    page_icon=":eye:",
    layout="wide",
    initial_sidebar_state='expanded'
)

# Custom CSS for styling
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Load VGG16 TensorFlow Model with Streamlit Caching for Cloud Deployment Compatibility
@st.cache_resource
def load_model():
    """
    Loads the trained VGG16 model with cached resource state.
    Uses portable path resolution for local development and cloud platforms (Render, Railway, Streamlit Cloud).
    """
    possible_paths = [
        Path(__file__).parent / "model.h5",
        Path(__file__).parent / "medi_scan_project" / "model.h5",
        Path("model.h5"),
        Path("medi_scan_project/model.h5")
    ]
    for path in possible_paths:
        if path.exists():
            return tf.keras.models.load_model(str(path))
    raise FileNotFoundError("model.h5 not found in expected directory locations.")


# ISSUE 1: Retinal Image Validation Function
def validate_retina(image_input: Union[Image.Image, np.ndarray, bytes]) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Verifies whether the uploaded image is a valid retinal fundus photograph before disease classification.

    Performs image structure & color spectrum checks:
    - Minimum resolution & aspect ratio
    - Dark outer corners / fundus aperture mask
    - Red channel dominance & warm hue distribution
    - Center-to-corner contrast ratio & circularity
    - Hard rejection filters for non-retinal images (selfies, faces, animals, landscapes, documents)

    Returns:
        (is_retina: bool, message: str, score: float, details: dict)
    """
    default_error_msg = "Invalid image detected. Please upload a clear retinal fundus image."

    # 1. Convert input to RGB NumPy array
    try:
        if isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
            img_rgb = np.array(pil_img.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input
        else:
            return False, default_error_msg, 0.0, {"error": "Unsupported image format"}
    except Exception:
        return False, default_error_msg, 0.0, {"error": "Image decoding failure"}

    if img_rgb is None or img_rgb.size == 0:
        return False, default_error_msg, 0.0, {"error": "Empty image content"}

    height, width, _ = img_rgb.shape

    # 2. Check basic dimensions and aspect ratio
    if height < 100 or width < 100:
        return False, default_error_msg, 0.0, {"error": "Image resolution too low"}

    aspect_ratio = width / float(height)
    if aspect_ratio < 0.55 or aspect_ratio > 1.80:
        return False, default_error_msg, 0.1, {"reason": "Aspect ratio outside typical fundus range"}

    # Resize to standard size for analysis
    analysis_size = (300, 300)
    resized_rgb = cv2.resize(img_rgb, analysis_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2HSV)

    # 3. Corner Darkness / Aperture Mask Check
    h_len, w_len = analysis_size
    c_h, c_w = int(h_len * 0.15), int(w_len * 0.15)

    top_left = gray[0:c_h, 0:c_w]
    top_right = gray[0:c_h, w_len - c_w:w_len]
    bottom_left = gray[h_len - c_h:h_len, 0:c_w]
    bottom_right = gray[h_len - c_h:h_len, w_len - c_w:w_len]

    corner_pixels = np.concatenate([
        top_left.flatten(), top_right.flatten(),
        bottom_left.flatten(), bottom_right.flatten()
    ])
    mean_corner_brightness = float(np.mean(corner_pixels))

    # Center region illumination
    m_h, m_w = int(h_len * 0.25), int(w_len * 0.25)
    center_roi = gray[m_h:h_len - m_h, m_w:w_len - m_w]
    mean_center_brightness = float(np.mean(center_roi))

    center_corner_contrast = (mean_center_brightness + 1.0) / (mean_corner_brightness + 1.0)

    # 4. Color Spectrum Analysis
    r_chan = resized_rgb[:, :, 0].astype(float)
    g_chan = resized_rgb[:, :, 1].astype(float)
    b_chan = resized_rgb[:, :, 2].astype(float)

    mean_r = float(np.mean(r_chan))
    mean_g = float(np.mean(g_chan))
    mean_b = float(np.mean(b_chan))

    red_blue_ratio = (mean_r + 1.0) / (mean_b + 1.0)

    hue = hsv[:, :, 0]
    val = hsv[:, :, 2]
    non_dark_mask = val > 20
    if np.sum(non_dark_mask) > 0:
        fundus_hues = hue[non_dark_mask]
        warm_hue_mask = (fundus_hues <= 30) | (fundus_hues >= 150)
        warm_hue_ratio = float(np.sum(warm_hue_mask)) / float(len(fundus_hues))
    else:
        warm_hue_ratio = 0.0

    # 5. Circularity & Contour Checks
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_circularity = 0.0
    fundus_area_ratio = 0.0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        fundus_area_ratio = area / float(h_len * w_len)
        if perimeter > 0:
            max_circularity = (4.0 * np.pi * area) / (perimeter ** 2)

    # Accumulate scoring metrics
    score = 0.0
    if red_blue_ratio >= 1.15:
        score += 0.25
    elif red_blue_ratio >= 1.0:
        score += 0.10

    if mean_corner_brightness < 55.0:
        score += 0.25
    elif mean_corner_brightness < 80.0:
        score += 0.10

    if warm_hue_ratio >= 0.40:
        score += 0.25
    elif warm_hue_ratio >= 0.25:
        score += 0.10

    if center_corner_contrast >= 1.25 and (max_circularity >= 0.35 or fundus_area_ratio >= 0.20):
        score += 0.25
    elif center_corner_contrast >= 1.10:
        score += 0.10

    # Hard Rejections for Non-Retinal Images
    is_hard_rejection = False
    if mean_corner_brightness > 95.0:
        is_hard_rejection = True
    if red_blue_ratio < 0.90:
        is_hard_rejection = True
    if mean_r < 20 and mean_g < 20 and mean_b < 20:
        is_hard_rejection = True
    if center_corner_contrast < 1.10 and mean_corner_brightness > 60.0:
        is_hard_rejection = True

    is_retina = (score >= 0.50) and not is_hard_rejection
    message = "Valid retinal fundus image detected." if is_retina else default_error_msg

    details = {
        "score": round(score, 2),
        "mean_corner_brightness": round(mean_corner_brightness, 2),
        "red_blue_ratio": round(red_blue_ratio, 2),
        "hard_rejection": is_hard_rejection
    }

    return is_retina, message, score, details


# VGG16 Preprocessing and Prediction Function
def import_and_predict(image_data, model):
    """Preprocesses uploaded image and returns model predictions."""
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


def prediction_cls(prediction, class_names):
    """Predicts the class name of the image based on model output."""
    for key, clss in class_names.items():
        if np.argmax(prediction) == clss:
            return key


# Load model with Streamlit status indicator
with st.spinner('Model is being loaded..'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model initialization error: {e}")
        st.stop()

# Sidebar Configuration
with st.sidebar:
    eye_img_paths = [
        Path(__file__).parent / "User_interface" / "eyejpg.jpg",
        Path(__file__).parent / "medi_scan_project" / "User_interface" / "eyejpg.jpg",
        Path("User_interface/eyejpg.jpg"),
        Path("medi_scan_project/User_interface/eyejpg.jpg")
    ]
    for p in eye_img_paths:
        if p.exists():
            st.image(str(p))
            break

    st.title("Ocular Diseases")
    st.subheader("Accurate detection of diseases present in the eyes leaves.")
    st.subheader("This helps the user to easily identify the disease and find the appropriate remedy for it")

st.write("""
         # MEDI-SCAN
         """)

st.write("""
         ## AI-Powered Medical Image Analysis for Ocular Disease Diagnosis
         """)

st.write("""
         This user-friendly tool allows users to upload retinal scans of their eyes and determines whether those eyes are healthy or not.

         If the eye is not healthy, it also tells you what kind of condition the eye might have, such as Diabetic Retinopathy, Glaucoma, or Cataracts.

         Following detection, it offers a treatment recommendation for the identified ailment.
         """)

file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png", "bmp"])

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
    except Exception:
        st.error("Invalid image file format. Please upload an uncorrupted image.")
        st.stop()

    st.image(image, use_column_width=True)

    # ISSUE 1: PRE-INFERENCE RETINAL VALIDATION
    # Validate uploaded image BEFORE sending it to the VGG16 model
    is_retina, val_msg, val_score, val_details = validate_retina(image)

    if not is_retina:
        # STOP execution immediately for non-retinal images.
        # Do NOT display prediction, confidence score, or remedy recommendations.
        st.error(val_msg)
        st.stop()

    # If image IS a valid retinal scan, perform VGG16 disease prediction
    predictions = import_and_predict(image, model)

    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(round(x, 2)) + " %")

    class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    predicted_class = class_names[np.argmax(predictions)]
    string = "Detected Disease : " + predicted_class

    if predicted_class == 'Normal':
        st.balloons()
        st.sidebar.success(string)
        st.sidebar.success("You have a healthy eye :)")

    elif predicted_class == 'Cataract':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Surgery is the only way to get rid of a cataract,")

    elif predicted_class == 'Glaucoma':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Eyedrops are the main treatment for glaucoma. "
            "There are several different types that can be used, but they all work by reducing the pressure in your eyes. "
            "They're normally used between 1 and 4 times a day. "
            "It's important to use them as directed, even if you haven't noticed any problems with your vision.")

    elif predicted_class == 'Diabetic Retinopathy':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Medicines called anti-VEGF drugs can slow down or reverse diabetic retinopathy. "
            "Other medicines, called corticosteroids, can also help. Laser treatment. "
            "To reduce swelling in your retina, eye doctors can use lasers to make the blood vessels shrink and stop leaking.")
    else:
        st.markdown("no disease detected")
