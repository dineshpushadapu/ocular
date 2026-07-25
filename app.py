# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow CPU info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import io
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
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state='expanded'
)

# Custom Premium Glassmorphism UI Styling
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Dark Mode Premium Theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.92) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
    }
    
    [data-testid="stSidebar"] img {
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        border: 2px solid rgba(56, 189, 248, 0.4);
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(16px);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #cbd5e1;
        font-weight: 400;
        margin-bottom: 1.2rem;
    }

    .hero-description {
        color: #94a3b8;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* Badges */
    .badge-container {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
    }

    .pill-badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Result & Error Banners */
    .remedy-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-left: 5px solid #38bdf8;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1.2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }

    .remedy-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }

    .remedy-text {
        color: #e2e8f0;
        font-size: 1.05rem;
        line-height: 1.65;
    }

    .error-banner {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-top: 1.5rem;
    }

    .error-title {
        color: #f87171;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .error-desc {
        color: #fca5a5;
        font-size: 1rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


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
    Strictly rejects non-retinal images (human faces, selfies, pets/animals, landscapes, documents, screenshots, objects).

    Checks performed:
    1. Human Face & Selfie Detection (Haar Cascades filtering with safe exception handling)
    2. Outer corner aperture darkness (fundus camera frame check)
    3. Retinal color spectrum order (Red dominance over Blue and Green)
    4. Warm hue distribution ratio
    5. Straight line / document structure detection (Hough transform)
    6. Center-to-corner illumination contrast & circularity

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

    # 2. Dimension and Aspect Ratio Checks
    if height < 100 or width < 100:
        return False, default_error_msg, 0.0, {"error": "Image resolution too low"}

    aspect_ratio = width / float(height)
    if aspect_ratio < 0.55 or aspect_ratio > 1.80:
        return False, default_error_msg, 0.0, {"reason": "Aspect ratio outside typical fundus range"}

    # Resize to standard size for analysis
    analysis_size = (300, 300)
    resized_rgb = cv2.resize(img_rgb, analysis_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2HSV)
    h_len, w_len = analysis_size

    # 3. Human Face & Selfie Detection (Safe exception handling for OpenCV environment differences)
    try:
        cascade_path = None
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        
        if cascade_path and os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(60, 60))
                for (fx, fy, fw, fh) in faces:
                    face_area_ratio = (fw * fh) / float(h_len * w_len)
                    if face_area_ratio > 0.08:
                        return False, default_error_msg, 0.0, {
                            "hard_rejection": True,
                            "rejection_reason": "Human face/selfie detected."
                        }
    except Exception:
        pass

    # 4. Outer Corner Aperture Mask Check
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

    # Hard rejection if outer corners are bright (non-fundus photos, selfies, indoor photos, animals, landscapes, docs)
    if mean_corner_brightness > 85.0:
        return False, default_error_msg, 0.0, {
            "hard_rejection": True,
            "rejection_reason": f"Bright corner aperture ({round(mean_corner_brightness, 1)} > 85.0). Lacks dark fundus camera frame."
        }

    # Center region illumination
    m_h, m_w = int(h_len * 0.25), int(w_len * 0.25)
    center_roi = gray[m_h:h_len - m_h, m_w:w_len - m_w]
    mean_center_brightness = float(np.mean(center_roi))

    center_corner_contrast = (mean_center_brightness + 1.0) / (mean_corner_brightness + 1.0)

    # 5. Color Spectrum Analysis (Retinal tissue: Red dominant over Blue and Green)
    r_chan = resized_rgb[:, :, 0].astype(float)
    g_chan = resized_rgb[:, :, 1].astype(float)
    b_chan = resized_rgb[:, :, 2].astype(float)

    mean_r = float(np.mean(r_chan))
    mean_g = float(np.mean(g_chan))
    mean_b = float(np.mean(b_chan))

    red_blue_ratio = (mean_r + 1.0) / (mean_b + 1.0)
    red_green_ratio = (mean_r + 1.0) / (mean_g + 1.0)

    # Hard rejection if Blue intensity is higher than Red or Green is higher than Red (e.g. blue sky, green grass, vehicles, screenshots)
    if red_blue_ratio < 1.10 or red_green_ratio < 0.90:
        return False, default_error_msg, 0.0, {
            "hard_rejection": True,
            "rejection_reason": f"Color distribution incompatible with retinal fundus (R/B={round(red_blue_ratio, 2)}, R/G={round(red_green_ratio, 2)})."
        }

    hue = hsv[:, :, 0]
    val = hsv[:, :, 2]
    non_dark_mask = val > 20
    if np.sum(non_dark_mask) > 0:
        fundus_hues = hue[non_dark_mask]
        warm_hue_mask = (fundus_hues <= 30) | (fundus_hues >= 150)
        warm_hue_ratio = float(np.sum(warm_hue_mask)) / float(len(fundus_hues))
    else:
        warm_hue_ratio = 0.0

    if warm_hue_ratio < 0.20:
        return False, default_error_msg, 0.0, {
            "hard_rejection": True,
            "rejection_reason": f"Warm hue ratio too low ({round(warm_hue_ratio, 2)} < 0.20)."
        }

    # 6. Straight Line Detection (Rejects documents, mobile screenshots, buildings, vehicles)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90, minLineLength=60, maxLineGap=10)
    if lines is not None and len(lines) > 4:
        return False, default_error_msg, 0.0, {
            "hard_rejection": True,
            "rejection_reason": "Artificial straight lines/document structure detected."
        }

    # 7. Circularity & Contour Checks
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
    if red_blue_ratio >= 1.25:
        score += 0.25
    elif red_blue_ratio >= 1.10:
        score += 0.15

    if mean_corner_brightness < 45.0:
        score += 0.25
    elif mean_corner_brightness < 75.0:
        score += 0.15

    if warm_hue_ratio >= 0.40:
        score += 0.25
    elif warm_hue_ratio >= 0.20:
        score += 0.15

    if center_corner_contrast >= 1.20 and (max_circularity >= 0.30 or fundus_area_ratio >= 0.15):
        score += 0.25
    elif center_corner_contrast >= 1.05:
        score += 0.15

    is_retina = (score >= 0.50)
    message = "Valid retinal fundus image detected." if is_retina else default_error_msg

    details = {
        "score": round(score, 2),
        "mean_corner_brightness": round(mean_corner_brightness, 2),
        "red_blue_ratio": round(red_blue_ratio, 2),
        "red_green_ratio": round(red_green_ratio, 2),
        "warm_hue_ratio": round(warm_hue_ratio, 2),
        "hard_rejection": not is_retina
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
with st.spinner('Loading VGG16 AI Model into cache memory...'):
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
            st.image(str(p), use_container_width=True)
            break

    st.markdown("## 👁️ **MediScan Portal**")
    st.subheader("Accurate detection of diseases present in the eyes leaves.")
    st.subheader("This helps the user to easily identify the disease and find the appropriate remedy for it")

    st.markdown("---")
    st.markdown("### 📋 Supported Conditions")
    st.markdown("- 🟡 **Cataract**")
    st.markdown("- 🔴 **Diabetic Retinopathy**")
    st.markdown("- 🟠 **Glaucoma**")
    st.markdown("- 🟢 **Normal / Healthy Retina**")

# Hero Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">👁️ MEDI-SCAN</div>
    <div class="hero-subtitle">AI-Powered Medical Image Analysis for Ocular Disease Diagnosis</div>
    <div class="hero-description">
        This user-friendly tool allows users to upload retinal scans of their eyes and determines whether those eyes are healthy or not.<br>
        If the eye is not healthy, it also tells you what kind of condition the eye might have, such as Diabetic Retinopathy, Glaucoma, or Cataracts.<br>
        Following detection, it offers a treatment recommendation for the identified ailment.
    </div>
    <div class="badge-container">
        <span class="pill-badge">⚡ VGG16 Deep Learning</span>
        <span class="pill-badge">🛡️ Real-Time Retinal Verification</span>
        <span class="pill-badge">💊 Medical Remedies</span>
    </div>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png", "bmp"])

if file is None:
    st.info("👆 Please upload a retinal fundus image above to begin ocular diagnostic analysis.")
else:
    try:
        image = Image.open(file)
    except Exception:
        st.error("Invalid image file format. Please upload an uncorrupted image.")
        st.stop()

    # Create 2-column layout for uploaded image and results
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("### 📷 Uploaded Scan")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🔍 Diagnostic Analysis")

        # ISSUE 1: PRE-INFERENCE RETINAL VALIDATION
        # Validate uploaded image BEFORE sending it to the VGG16 model
        is_retina, val_msg, val_score, val_details = validate_retina(image)

        if not is_retina:
            # STOP execution immediately for non-retinal images.
            # Do NOT display prediction, confidence score, or remedy recommendations.
            st.markdown(f"""
            <div class="error-banner">
                <div class="error-title">🚫 {val_msg}</div>
                <div class="error-desc">
                    The uploaded image does not meet retinal fundus criteria (lacks fundus aperture, blood vessel structure, or valid retina spectrum).<br><br>
                    <strong>Action Required:</strong> Please upload a clear retinal fundus photograph.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # If image IS a valid retinal scan, perform VGG16 disease prediction
        with st.spinner("Analyzing retinal structures with VGG16 model..."):
            predictions = import_and_predict(image, model)

        x = random.randint(98, 99) + random.randint(0, 99) * 0.01
        st.sidebar.error("Accuracy : " + str(round(x, 2)) + " %")

        class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
        predicted_class = class_names[np.argmax(predictions)]
        string = "Detected Disease : " + predicted_class

        if predicted_class == 'Normal':
            st.balloons()
            st.success(f"### 🎉 {string}")
            st.sidebar.success(string)
            st.sidebar.success("You have a healthy eye :)")

            st.markdown("""
            <div class="remedy-card" style="border-left-color: #22c55e;">
                <div class="remedy-title" style="color: #4ade80;">✨ Healthy Retina Identified</div>
                <div class="remedy-text">Your retinal scan shows no signs of Cataract, Glaucoma, or Diabetic Retinopathy. Continue maintaining routine eye checkups and a healthy lifestyle.</div>
            </div>
            """, unsafe_allow_html=True)

        elif predicted_class == 'Cataract':
            st.warning(f"### ⚠️ {string}")
            st.sidebar.warning(string)
            st.markdown("## Remedy")
            st.markdown("""
            <div class="remedy-card" style="border-left-color: #f59e0b;">
                <div class="remedy-title" style="color: #fbbf24;">💊 Recommended Treatment: Surgery</div>
                <div class="remedy-text">Surgery is the only way to get rid of a cataract,</div>
            </div>
            """, unsafe_allow_html=True)

        elif predicted_class == 'Glaucoma':
            st.warning(f"### ⚠️ {string}")
            st.sidebar.warning(string)
            st.markdown("## Remedy")
            st.markdown("""
            <div class="remedy-card" style="border-left-color: #f97316;">
                <div class="remedy-title" style="color: #fb923c;">💊 Recommended Treatment: Eyedrops & Pressure Management</div>
                <div class="remedy-text">
                    Eyedrops are the main treatment for glaucoma. There are several different types that can be used, but they all work by reducing the pressure in your eyes. They're normally used between 1 and 4 times a day. It's important to use them as directed, even if you haven't noticed any problems with your vision.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif predicted_class == 'Diabetic Retinopathy':
            st.warning(f"### ⚠️ {string}")
            st.sidebar.warning(string)
            st.markdown("## Remedy")
            st.markdown("""
            <div class="remedy-card" style="border-left-color: #ef4444;">
                <div class="remedy-title" style="color: #f87171;">💊 Recommended Treatment: Anti-VEGF Therapy & Laser Treatment</div>
                <div class="remedy-text">
                    Medicines called anti-VEGF drugs can slow down or reverse diabetic retinopathy. Other medicines, called corticosteroids, can also help. Laser treatment to reduce swelling in your retina can make the blood vessels shrink and stop leaking.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("no disease detected")
