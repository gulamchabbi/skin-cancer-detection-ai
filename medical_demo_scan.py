# Author: Gulam N Chabbi
# Project: Skin Cancer Detection using AI
# Created: January 2026
# GitHub: gulam89513
# NOTE: This repository was made public for learning/demo purposes.
# Unauthorized academic submission or reuse without permission is prohibited.

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import cv2
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Skin Disease Detection | Developed by Gulam N Chabbi",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SKIN DETECTION LOGIC ---
def is_skin_present(image):
    """Checks if the uploaded image contains human skin tones."""
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Range for human skin tones in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    skin_percentage = (np.count_nonzero(mask) / (img_cv.shape[0] * img_cv.shape[1])) * 100
    return skin_percentage > 15  # 15% threshold to filter out background objects

# --- 3. MEDICAL DATABASE ---
MEDICAL_DB = {
    "Actinic Keratoses": {
        "severity": "high",
        "risk_label": "PRE-CANCEROUS / HIGH RISK",
        "description": "A rough, scaly patch on the skin caused by years of sun exposure.",
        "features": "• Sandpaper-like texture\n• Red, pink, or brown scaly patch",
        "causes": "☀️ Cumulative UV damage from sunlight or tanning beds.",
        "treatment": "💊 Cryotherapy (freezing) or clinical creams.",
        "action": "⚠️ Consult Dermatologist soon."
    },
    "Basal Cell Carcinoma": {
        "severity": "high",
        "risk_label": "MALIGNANT / HIGH RISK",
        "description": "Most common skin cancer. Grows slowly and rarely spreads.",
        "features": "• Pearly bump\n• Visible blood vessels\n• Non-healing sore",
        "causes": "☀️ Intense, intermittent sun exposure causing DNA mutations.",
        "treatment": "💊 Mohs Surgery or Excision.",
        "action": "🚨 Schedule Biopsy immediately."
    },
    "Benign Keratosis": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "Non-cancerous 'stuck-on' growth common with aging.",
        "features": "• Waxy appearance\n• Well-defined borders",
        "causes": "🧬 Genetic aging process. Not sun-related.",
        "treatment": "✅ None needed unless irritated.",
        "action": "✅ Safe: No action required."
    },
    "Dermatofibroma": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "Firm bump often forming after minor injury (like a bug bite).",
        "features": "• Dimples when pinched\n• Hard nodule",
        "causes": "🐜 Reaction to small trauma or bites.",
        "treatment": "✅ Harmless. Usually left alone.",
        "action": "✅ Safe."
    },
    "Melanocytic Nevi": {
        "severity": "low",
        "risk_label": "BENIGN / MONITOR REQUIRED",
        "description": "A common mole. A benign cluster of pigment cells.",
        "features": "• Uniform color\n• Round/Oval shape",
        "causes": "🧬 Genetic clustering of melanocytes.",
        "treatment": "✅ No treatment needed.",
        "action": "🔍 Monitor for changes (ABCDE rule)."
    },
    "Melanoma": {
        "severity": "critical",
        "risk_label": "🔴 MALIGNANT / CRITICAL LIFE THREAT",
        "description": "Most dangerous skin cancer. Rapid uncontrolled pigment growth.",
        "features": "• Irregular borders\n• Multiple colors\n• Larger than 6mm",
        "causes": "⚠️ Severe UV damage triggering rapid cell growth.",
        "treatment": "🚨 Immediate wide excision and possible immunotherapy.",
        "action": "🚨 EMERGENCY: See a specialist TODAY."
    },
    "Vascular Lesions": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "Abnormal bunching of blood vessels near the surface.",
        "features": "• Bright red/purple\n• Soft touch",
        "causes": "🩸 Aging or hormonal changes.",
        "treatment": "✅ Laser therapy if desired for cosmetics.",
        "action": "✅ Safe."
    }
}

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ MediScan Controls")
    st.warning("🚨 **Disclaimer:** Educational tool only. Not for medical diagnosis.")
    st.divider()
    confidence_threshold = st.slider("Accuracy Threshold (%)", 0, 100, 45)
    st.caption(f"App Version: 2.1 (Skin Filter Enabled)")
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# --- 6. MAIN UI ---
st.title("🏥 Skin Disease Detection")
st.caption("Developed by Gulam N Chabbi")

tab_scan, tab_dict, tab_help = st.tabs(["🔍 Clinical Scanner", "📚 Encyclopedia", "🚑 Specialist Locator"])

with tab_scan:
    col1, col2 = st.columns([0.8, 1.2])
    
    with col1:
        st.subheader("1. Specimen Input")
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Preview", use_container_width=True)
            
            if st.button("🚀 Run Diagnostics", type="primary"):
                # --- NEW FILTER CHECK ---
                if not is_skin_present(img):
                    st.error("❌ INVALID IMAGE: No skin detected.")
                    st.warning("The AI detected a background object (table/furniture). Please upload a close-up of skin.")
                else:
                    with st.spinner("Analyzing..."):
                        model = load_model()
                        st.session_state['results'] = model(img)

    with col2:
        st.subheader("2. Diagnostic Results")
        if 'results' in st.session_state:
            top = st.session_state['results'][0]
            score = top['score'] * 100
            label = top['label'].replace('_', ' ').title()
            
            if score < confidence_threshold:
                st.error("⚠️ ANALYSIS INCONCLUSIVE: Low Confidence.")
            else:
                info = MEDICAL_DB.get(label, {"severity": "low", "risk_label": "Unknown", "description": "N/A", "features": "N/A", "causes": "N/A", "treatment": "N/A", "action": "Consult doctor"})
                
                # Dynamic Header
                if info['severity'] == "critical": st.error(f"🔴 {label.upper()}")
                elif info['severity'] == "high": st.warning(f"🟠 {label.upper()}")
                else: st.success(f"🟢 {label.upper()}")

                st.write(f"**Risk:** {info['risk_label']}")
                st.metric("Confidence", f"{score:.2f}%")
                
                with st.expander("👁️ Visual Breakdown", expanded=True):
                    st.write(info['description'])
                    st.markdown(info['features'])
                
                st.markdown(f"""<div style='background-color: #fce4e4; color: #000; padding: 10px; border-radius: 5px;'><strong>ACTION:</strong> {info['action']}</div>""", unsafe_allow_html=True)
                
                st.divider()
                st.bar_chart(pd.DataFrame([{"Condition": r['label'].title(), "Prob (%)": r['score']*100} for r in st.session_state['results'][:3]]).set_index("Condition"))

with tab_dict:
    st.header("📚 Encyclopedia")
    sel = st.selectbox("Search disease:", list(MEDICAL_DB.keys()))
    st.write(MEDICAL_DB[sel]['description'])

with tab_help:
    st.header("🚑 Emergency Help")
    st.link_button("🔍 Find Dermatologist Near Me", "https://www.google.com/maps/search/dermatologist+near+me")
