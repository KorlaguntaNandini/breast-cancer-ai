import streamlit as st
import pickle
import numpy as np
import shap
from datetime import date

# Load model
model = pickle.load(open("breast_cancer_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.set_page_config(page_title="Hospital Breast Cancer AI System", layout="wide")

st.title("🏥 AI Breast Cancer Diagnostic System")
st.write("Clinical Decision Support Tool for Breast Cancer Prediction")

st.divider()

# ---------------- PATIENT DETAILS ----------------

st.header("👩 Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Female","Male"])

with col2:
    patient_id = st.text_input("Patient ID")
    doctor_name = st.text_input("Doctor Name")
    exam_date = st.date_input("Date", value=date.today())

st.divider()

# ---------------- TUMOR FEATURES ----------------

st.header("🔬 Tumor Measurement Data")

radius = st.number_input("Mean Radius")
texture = st.number_input("Mean Texture")
perimeter = st.number_input("Mean Perimeter")
area = st.number_input("Mean Area")
concave_points = st.number_input("Mean Concave Points")

st.divider()

# ---------------- PREDICTION ----------------

if st.button("🧠 Run AI Diagnosis"):

    input_data = np.array([[radius,texture,perimeter,area,concave_points]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    confidence = max(prob)*100

    st.header("📊 Diagnosis Result")

    if prediction == 1:
        st.error(f"⚠️ Malignant Tumor Detected")
    else:
        st.success(f"✅ Benign Tumor")

    st.write(f"Confidence Level: {confidence:.2f}%")

    st.divider()

    # ---------------- SHAP EXPLAINABILITY ----------------

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

    importance = np.abs(shap_values[1]).mean(axis=0)
    top_feature = feature_names[np.argmax(importance)]

    st.write("Most Influential Feature:", top_feature)

    st.divider()

    # ---------------- COUNTERFACTUAL ----------------

    st.subheader("🔄 Counterfactual Suggestion")

    for i in range(len(input_data[0])):
        temp = input_scaled.copy()
        temp[0][i] = temp[0][i] * 0.9

        new_pred = model.predict(temp)

        if new_pred == 0:
            st.write(f"If **{feature_names[i]}** decreases slightly, prediction may become benign.")