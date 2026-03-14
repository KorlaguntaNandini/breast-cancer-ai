import streamlit as st
import pickle
import numpy as np
import shap

# Load model and scaler
model = pickle.load(open("breast_cancer_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("🏥 Breast Cancer Diagnosis System")

st.subheader("👩 Patient Information")

name = st.text_input("Patient Name")
age = st.number_input("Age", 1, 120)
gender = st.selectbox("Gender", ["Female","Male"])
patient_id = st.text_input("Patient ID")
doctor = st.text_input("Doctor Name")

st.subheader("🔬 Tumor Measurements")

radius = st.number_input("Mean Radius")
texture = st.number_input("Mean Texture")
perimeter = st.number_input("Mean Perimeter")
area = st.number_input("Mean Area")
concave_points = st.number_input("Mean Concave Points")

if st.button("Analyze Tumor"):

    input_data = np.array([[radius, texture, perimeter, area, concave_points]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("📊 Diagnosis Result")

    if prediction[0] == 1:
        st.error("⚠️ Malignant Tumor Detected")
    else:
        st.success("✅ Benign (Non‑Cancerous)")

    confidence = np.max(probability)*100
    st.write(f"Confidence Level: {confidence:.2f}%")

    # -------- SHAP Explainability --------

    st.subheader("🔍 Main Influencing Feature")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    feature_names = [
        "Mean Radius",
        "Mean Texture",
        "Mean Perimeter",
        "Mean Area",
        "Mean Concave Points"
    ]

    if isinstance(shap_values, list):
        values = shap_values[0]
    else:
        values = shap_values

    importance = np.abs(values).mean(axis=0)
    top_feature = feature_names[np.argmax(importance)]

    st.info(f"Most Influential Feature: {top_feature}")

    # -------- Counterfactual Suggestion --------

    st.subheader("🔄 Counterfactual Suggestion")

    for i in range(len(input_data[0])):
        temp = input_data.copy()
        temp[0][i] = temp[0][i] * 0.9
        temp_scaled = scaler.transform(temp)

        new_pred = model.predict(temp_scaled)

        if new_pred[0] == 0:
            st.write(f"If **{feature_names[i]}** decreases slightly, the tumor may become **Benign**.")