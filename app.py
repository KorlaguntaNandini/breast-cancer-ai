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

col1, col2 = st.columns([1,4])

with col1:
    st.image("logo.png", width=180)

with col2:
    st.title("NAP Hospital AI Diagnostic System")
    st.markdown("### AI Powered Breast Cancer Diagnosis Platform")

st.markdown("""
📍 **NAP Hospital Address**

Beside Amrita Vishwa Vidyapeetham  
Vengal, Chennai  
Tamil Nadu  
India
""")

st.markdown("---")

# ---------------- SIDEBAR PATIENT INFO ----------------

st.sidebar.header("Patient Information")

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 1, 120, 35)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
patient_id = st.sidebar.text_input("Patient ID")
doctor = st.sidebar.text_input("Doctor Name")

# ---------------- INPUT SECTION ----------------

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

    # ---------------- FEATURE IMPORTANCE ----------------

    importance = np.abs(input_data[0])
    top_feature = feature_names[np.argmax(importance)]

    st.subheader("🔬 Main Influencing Feature")
    st.write(top_feature)

    # ---------------- COUNTERFACTUAL ----------------

    st.subheader("Suggested Benign Values")

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

# HEADER BAR
pdf.set_fill_color(0, 102, 153)
pdf.rect(0, 0, 210, 30, 'F')

# LOGO
try:
    pdf.image("logo.png", 10, 5, 20)
except:
    pass

pdf.set_text_color(255,255,255)
pdf.set_font("Arial","B",18)
pdf.cell(0,15,"NAP HOSPITAL",0,1,"C")

pdf.set_font("Arial","",12)
pdf.cell(0,5,"AI Diagnostic Center",0,1,"C")

pdf.ln(10)

pdf.set_text_color(0,0,0)

# ADDRESS
pdf.set_font("Arial","",11)
pdf.cell(0,6,"Beside Amrita Vishwa Vidyapeetham",0,1,"C")
pdf.cell(0,6,"Vengal, Chennai",0,1,"C")
pdf.cell(0,6,"Tamil Nadu, India",0,1,"C")

pdf.ln(8)

# REPORT TIME
pdf.cell(0,6,f"Report Generated: {now}",0,1)

pdf.ln(5)

# PATIENT DETAILS
pdf.set_font("Arial","B",12)
pdf.cell(0,8,"Patient Details",0,1)

pdf.set_font("Arial","",11)
pdf.cell(0,6,f"Patient Name: {name}",0,1)
pdf.cell(0,6,f"Age: {age}",0,1)
pdf.cell(0,6,f"Gender: {gender}",0,1)
pdf.cell(0,6,f"Patient ID: {patient_id}",0,1)
pdf.cell(0,6,f"Doctor: {doctor}",0,1)

pdf.ln(5)

# MEASUREMENT TABLE
pdf.set_font("Arial","B",12)
pdf.cell(0,8,"Tumor Measurements",0,1)

pdf.set_font("Arial","B",11)
pdf.cell(90,8,"Feature",1)
pdf.cell(90,8,"Value",1)
pdf.ln()

pdf.set_font("Arial","",11)

pdf.cell(90,8,"Mean Radius",1)
pdf.cell(90,8,str(radius),1)
pdf.ln()

pdf.cell(90,8,"Mean Texture",1)
pdf.cell(90,8,str(texture),1)
pdf.ln()

pdf.cell(90,8,"Mean Perimeter",1)
pdf.cell(90,8,str(perimeter),1)
pdf.ln()

pdf.cell(90,8,"Mean Area",1)
pdf.cell(90,8,str(area),1)
pdf.ln()

pdf.cell(90,8,"Mean Concave Points",1)
pdf.cell(90,8,str(concave_points),1)
pdf.ln()

pdf.ln(10)

# DIAGNOSIS RESULT
pdf.set_font("Arial","B",12)
pdf.cell(0,8,"Diagnosis Result",0,1)

pdf.set_font("Arial","",11)
pdf.cell(0,6,f"Prediction: {result}",0,1)
pdf.cell(0,6,f"Confidence: {confidence:.2f}%",0,1)
pdf.cell(0,6,f"Key Feature: {top_feature}",0,1)

pdf.ln(15)

# SIGNATURE
pdf.cell(90,8,"________________________",0,0,"C")
pdf.cell(90,8,"________________________",0,1,"C")

pdf.cell(90,8,"AI Diagnostic System",0,0,"C")
pdf.cell(90,8,"Doctor Signature",0,1,"C")

pdf_output = pdf.output(dest="S").encode("latin-1")

st.download_button(
    "⬇ Download Official Hospital Report",
    data=pdf_output,
    file_name="NAP_Hospital_Report.pdf",
    mime="application/pdf"
)