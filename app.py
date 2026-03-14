import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
from datetime import datetime
from fpdf import FPDF

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="NAP Hospital AI Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# ---------------- Styling ----------------

st.markdown("""
<style>
.card {
padding: 15px;
border-radius: 10px;
background-color: #f0f2f6;
box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
text-align: center;
}
.stTextInput>div>div>input {
border-radius: 8px;
}
.stNumberInput>div>div>input {
border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------

st.image("https://img.icons8.com/color/96/hospital.png", width=90)

st.markdown("""
# 🏥 NAP Hospital  
### AI Breast Cancer Diagnostic System
""")

# ---------------- Dashboard Cards ----------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="card"><h3>🤖 AI Model</h3><p>Random Forest</p></div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card"><h3>📊 Dataset</h3><p>Breast Cancer Wisconsin</p></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card"><h3>🎯 Accuracy</h3><p>96%</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- Load Model ----------------

model = pickle.load(open("breast_cancer_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- Layout ----------------

col1, col2 = st.columns(2)

# ---------------- Patient Info ----------------

with col1:

    st.subheader("👩 Patient Information")

    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Female", "Male"])
    patient_id = st.text_input("Patient ID")
    doctor = st.text_input("Doctor Name")

# ---------------- Tumor Measurements ----------------

with col2:

    st.subheader("🔬 Tumor Measurements")

    radius = st.number_input("Mean Radius")
    texture = st.number_input("Mean Texture")
    perimeter = st.number_input("Mean Perimeter")
    area = st.number_input("Mean Area")
    concave_points = st.number_input("Mean Concave Points")

feature_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Concave Points"
]

st.markdown("---")

# ---------------- Buttons ----------------

b1, b2 = st.columns(2)

with b1:
    analyze = st.button("🔍 Analyze Tumor", use_container_width=True)

with b2:
    if st.button("🔄 New Patient", use_container_width=True):
        st.rerun()

# ---------------- Prediction ----------------

if analyze:

    input_data = np.array([[radius, texture, perimeter, area, concave_points]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    confidence = np.max(probability) * 100

    st.markdown("## 📊 Diagnosis Result")

    if prediction[0] == 0:

        st.error("⚠️ Malignant Tumor Detected")
        result = "Malignant"

        if confidence > 80:
            st.error("🔴 High Cancer Risk")
        else:
            st.warning("🟠 Moderate Cancer Risk")

    else:

        st.balloons()
        st.success("✅ Benign (Non‑Cancerous)")
        st.success("🟢 Low Cancer Risk")
        result = "Benign"

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.markdown("---")

    # ---------------- Table ----------------

    st.subheader("📋 Tumor Measurement Summary")

    table_data = {
        "Feature": feature_names,
        "Value": [radius, texture, perimeter, area, concave_points]
    }

    st.table(pd.DataFrame(table_data))

    st.markdown("---")

    # ---------------- Explainability ----------------

    st.subheader("🔍 Main Influencing Feature")

    try:

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values

        shap_array = np.array(shap_array).flatten()
        shap_array = shap_array[:len(feature_names)]

        importance = np.abs(shap_array)

        top_feature_index = np.argmax(importance)
        top_feature = feature_names[top_feature_index]

        st.info(f"Most Influential Feature: **{top_feature}**")

    except:

        top_feature = "Unavailable"
        st.warning("Feature importance could not be calculated.")

    st.markdown("---")

    # ---------------- Counterfactual ----------------

    st.subheader("🔄 Counterfactual Feature Adjustment")

    if result == "Malignant":

        suggestions_found = False

        for i in range(len(feature_names)):

            temp = input_data.copy()
            new_value = temp[0][i] * 0.8
            temp[0][i] = new_value

            temp_scaled = scaler.transform(temp)
            new_pred = model.predict(temp_scaled)

            if new_pred[0] == 1:

                st.write(
                    f"Adjust **{feature_names[i]}** to approximately **{new_value:.3f}** to change prediction to **Benign**."
                )

                suggestions_found = True

        if not suggestions_found:
            st.info("No small feature adjustment could change the prediction.")

    else:

        st.success("Tumor already predicted as **Benign**.")

    st.markdown("---")

    # ---------------- PDF Report ----------------

    st.subheader("🧾 Hospital Diagnosis Report")

    if st.button("📄 Generate PDF Report"):

        now = datetime.now().strftime("%d-%m-%Y %H:%M")

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 10, "NAP HOSPITAL", ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "AI Breast Cancer Diagnostic Report", ln=True, align="C")
        pdf.cell(0, 8, "Address: Hyderabad, India", ln=True, align="C")

        pdf.ln(10)

        pdf.cell(0, 8, f"Report Generated: {now}", ln=True)

        pdf.ln(5)

        pdf.cell(0, 8, f"Patient Name: {name}", ln=True)
        pdf.cell(0, 8, f"Age: {age}", ln=True)
        pdf.cell(0, 8, f"Gender: {gender}", ln=True)
        pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
        pdf.cell(0, 8, f"Doctor: {doctor}", ln=True)

        pdf.ln(10)

        pdf.cell(0, 8, "Tumor Measurements", ln=True)

        pdf.cell(0, 8, f"Mean Radius: {radius}", ln=True)
        pdf.cell(0, 8, f"Mean Texture: {texture}", ln=True)
        pdf.cell(0, 8, f"Mean Perimeter: {perimeter}", ln=True)
        pdf.cell(0, 8, f"Mean Area: {area}", ln=True)
        pdf.cell(0, 8, f"Mean Concave Points: {concave_points}", ln=True)

        pdf.ln(10)

        pdf.cell(0, 8, "Diagnosis Result", ln=True)

        pdf.cell(0, 8, f"Prediction: {result}", ln=True)
        pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)
        pdf.cell(0, 8, f"Main Influencing Feature: {top_feature}", ln=True)

        pdf.ln(15)

        pdf.cell(0, 8, "Generated by NAP Hospital AI Diagnostic System", ln=True)

        pdf_output = pdf.output(dest="S").encode("latin-1")

        st.download_button(
            "⬇ Download Hospital Report (PDF)",
            pdf_output,
            "NAP_Hospital_Report.pdf",
            "application/pdf"
        )

# ---------------- Footer ----------------

st.markdown("---")
st.markdown("Developed for **NAP Hospital AI Research Lab**")