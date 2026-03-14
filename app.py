import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
from datetime import datetime

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="NAP Hospital AI Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# ---------------- Header ----------------

st.image("https://img.icons8.com/color/96/hospital.png", width=90)

st.markdown("""
# 🏥 NAP Hospital  
### AI Breast Cancer Diagnostic System
""")

st.info("Model Accuracy: 96% (Breast Cancer Wisconsin Dataset)")

st.markdown("---")

# ---------------- Load Model ----------------

model = pickle.load(open("breast_cancer_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- Layout ----------------

col1, col2 = st.columns(2)

# ---------------- Patient Information ----------------

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

colA, colB = st.columns(2)

with colA:
    analyze = st.button("🔍 Analyze Tumor", use_container_width=True)

with colB:
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

        st.success("✅ Benign (Non‑Cancerous)")
        st.success("🟢 Low Cancer Risk")
        result = "Benign"

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.markdown("---")

    # ---------------- Feature Table ----------------

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

    # ---------------- Counterfactual Suggestions ----------------

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
                    f"Adjust **{feature_names[i]}** to approximately **{new_value:.3f}** to potentially change prediction to **Benign**."
                )

                suggestions_found = True

        if not suggestions_found:
            st.info("No small feature adjustment could change the prediction.")

    else:

        st.success("Tumor already predicted as **Benign**. No adjustment required.")

    st.markdown("---")

    # ---------------- Patient Report ----------------

    st.subheader("🧾 Patient Diagnosis Report")

    now = datetime.now().strftime("%d-%m-%Y %H:%M")

    receipt = f"""
========================================
             NAP HOSPITAL
========================================
        AI DIAGNOSTIC REPORT
========================================

Report Generated : {now}

Patient Name : {name}
Age          : {age}
Gender       : {gender}
Patient ID   : {patient_id}
Doctor       : {doctor}

----------------------------------------
Tumor Measurements
----------------------------------------
Mean Radius          : {radius}
Mean Texture         : {texture}
Mean Perimeter       : {perimeter}
Mean Area            : {area}
Mean Concave Points  : {concave_points}

----------------------------------------
AI Diagnosis Result
----------------------------------------
Prediction : {result}
Confidence : {confidence:.2f} %

Main Influencing Feature :
{top_feature}

----------------------------------------
Generated by AI Breast Cancer
Diagnostic System
NAP Hospital
----------------------------------------
"""

    st.download_button(
        "⬇ Download Patient Report",
        receipt,
        "NAP_patient_diagnosis_report.txt"
    )

    st.success("Report Generated Successfully")

# ---------------- Footer ----------------

st.markdown("---")
st.markdown("Developed for **NAP Hospital AI Research Lab**")