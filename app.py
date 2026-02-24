import streamlit as st
import pickle
import os
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# ---------------- CLEAN MEDICAL STYLE ----------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: #f5f7fb;
}

/* Remove header bar */
[data-testid="stHeader"] {
    background: transparent;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 10px;
}

/* Button */
.stButton>button {
    background: #ec4899;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 24px;
}

/* Inputs */
.stNumberInput input {
    background: white !important;
    border-radius: 8px;
}

/* Progress bar */
.stProgress > div > div > div {
    background: #ec4899;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR,"breast_cancer_model.pkl"),"rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR,"scaler.pkl"),"rb"))

# ---------------- HEADER ----------------
st.markdown('<div class="title">🧬 AI Breast Cancer Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered tumor classification using Random Forest</div>', unsafe_allow_html=True)
st.caption("🔬 Clinical Decision Support Tool")

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("🔬 Tumor Measurements")

col1, col2 = st.columns(2)

feature_names = [
    "radius_mean","texture_mean","smoothness_mean","compactness_mean",
    "concavity_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","smoothness_se","compactness_se",
    "concavity_se","concave_points_se","symmetry_se",
    "fractal_dimension_se","smoothness_worst","compactness_worst",
    "concavity_worst","symmetry_worst","fractal_dimension_worst"
]

features = []

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        value = col1.number_input(feature, value=0.0)
    else:
        value = col2.number_input(feature, value=0.0)
    features.append(value)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Tumor Data"):

    input_array = np.array(features).reshape(1,-1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    confidence = max(prob) * 100

    # Malignant result
    if prediction == 1:
        st.markdown(f"""
        <div style="
        background:#7f1d1d;
        color:white;
        padding:26px;
        border-radius:14px;
        text-align:center;
        font-size:24px;
        font-weight:bold;">
        ⚠️ Malignant Tumor Detected 🚨<br>
        <span style="font-size:18px;">Immediate medical consultation recommended</span><br><br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # Benign result
    else:
        st.markdown(f"""
        <div style="
        background:#065f46;
        color:white;
        padding:26px;
        border-radius:14px;
        text-align:center;
        font-size:24px;
        font-weight:bold;">
        🛡️ Benign Tumor (No Cancer) ✔<br>
        <span style="font-size:18px;">Low risk classification</span><br><br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(confidence))