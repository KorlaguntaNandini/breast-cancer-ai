import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pytz
from fpdf import FPDF

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NAP Hospital AI Diagnostic System",
    page_icon="🏥",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Concave Points"
]

# ---------------- HEADER ----------------

st.image("logo.png", width=140)
st.title("NAP Hospital AI Diagnostic System")
st.markdown("AI Powered Breast Cancer Diagnosis Platform")

st.markdown("---")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Patient Information")

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 1, 120, 35)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
patient_id = st.sidebar.text_input("Patient ID")
doctor = st.sidebar.text_input("Doctor Name")

# ---------------- INPUT FEATURES ----------------

st.header("Tumor Measurement Inputs")

col1, col2 = st.columns(2)

with col1:
    radius = st.number_input("Mean Radius", 0.0, 50.0, 12.0)
    texture = st.number_input("Mean Texture", 0.0, 50.0, 15.0)
    perimeter = st.number_input("Mean Perimeter", 0.0, 200.0, 80.0)

with col2:
    area = st.number_input("Mean Area", 0.0, 2000.0, 500.0)
    concave_points = st.number_input("Mean Concave Points", 0.0, 1.0, 0.05)

input_data = np.array([[radius, texture, perimeter, area, concave_points]])

# ---------------- BUTTONS ----------------

colA, colB = st.columns(2)

with colA:
    analyze = st.button("🔍 Analyze Tumor")

with colB:
    if st.button("🔄 New Patient"):
        st.rerun()

# ---------------- PREDICTION ----------------

if analyze:

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    if prediction == 0:
        result = "Benign (Non‑Cancerous)"
        confidence = probability[0] * 100
    else:
        result = "Malignant Tumor Detected"
        confidence = probability[1] * 100

    st.markdown("---")
    st.header("Diagnosis Result")

    if prediction == 0:
        st.success(f"✅ {result}")
    else:
        st.error(f"⚠ {result}")

    st.info(f"Confidence Level: {confidence:.2f}%")

    # ---------------- MAIN FEATURE ----------------

    importance = np.abs(input_data[0])
    top_feature = feature_names[np.argmax(importance)]

    st.subheader("🔬 Main Influencing Feature")
    st.write(top_feature)

    # ---------------- COUNTERFACTUAL ----------------

    st.subheader("Suggested Values For Benign Prediction")

    suggestions = {
        "Mean Radius": radius * 0.9,
        "Mean Texture": texture * 0.9,
        "Mean Perimeter": perimeter * 0.9,
        "Mean Area": area * 0.9,
        "Mean Concave Points": concave_points * 0.9
    }

    for k, v in suggestions.items():
        st.write(f"{k} ≈ {round(v,2)}")

    # ---------------- PDF REPORT ----------------

    st.subheader("🧾 Download Hospital Report")

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).strftime("%d-%m-%Y %I:%M %p")

    pdf = FPDF()
    pdf.add_page()

    try:
        pdf.image("logo.png", 10, 8, 30)
    except:
        pass

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "NAP HOSPITAL", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, "AI Diagnostic Center", ln=True, align="C")
    pdf.cell(0, 8, "Hyderabad, India", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Report Generated: {now}", ln=True)

    pdf.ln(5)

    pdf.cell(0, 8, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 8, f"Age: {age}", ln=True)
    pdf.cell(0, 8, f"Gender: {gender}", ln=True)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 8, f"Doctor: {doctor}", ln=True)

    pdf.ln(8)

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
    pdf.cell(0, 8, f"Key Feature: {top_feature}", ln=True)

    pdf.ln(15)

    pdf.cell(90, 8, "________________________", align="C")
    pdf.cell(90, 8, "________________________", align="C")
    pdf.ln()

    pdf.cell(90, 8, "AI Diagnostic System", align="C")
    pdf.cell(90, 8, "Doctor Signature", align="C")

    pdf_output = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "⬇ Download Official Hospital Report",
        data=pdf_output,
        file_name="NAP_Hospital_Report.pdf",
        mime="application/pdf"
    )