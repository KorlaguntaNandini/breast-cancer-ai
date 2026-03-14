import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pytz
from fpdf import FPDF
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NAP Hospital AI Diagnostic System",
    page_icon="🏥",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

features = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Concave Points"
]

# ---------------- HEADER ----------------

col1, col2 = st.columns([1,4])

with col1:
    st.image("logo.png", width=210)

with col2:
    st.title("NAP Hospital AI Diagnostic System")
    st.write("AI Powered Breast Cancer Detection")

st.markdown("""
📍 **NAP Hospital**

Beside Amrita Vishwa Vidyapeetham  
Vengal, Chennai  
Tamil Nadu, India
""")

st.markdown("---")

# ---------------- PATIENT INFO ----------------

st.sidebar.header("Patient Information")

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age",1,120,35)
gender = st.sidebar.selectbox("Gender",["Female","Male"])
patient_id = st.sidebar.text_input("Patient ID")
doctor = st.sidebar.text_input("Doctor Name")

# ---------------- INPUTS ----------------

st.header("Tumor Measurement Inputs")

col1,col2 = st.columns(2)

with col1:
    radius = st.number_input("Mean Radius",0.0,50.0,12.0)
    texture = st.number_input("Mean Texture",0.0,50.0,15.0)
    perimeter = st.number_input("Mean Perimeter",0.0,200.0,80.0)

with col2:
    area = st.number_input("Mean Area",0.0,2000.0,500.0)
    concave = st.number_input("Mean Concave Points",0.0,1.0,0.05)

input_data = np.array([[radius,texture,perimeter,area,concave]])

# ---------------- BUTTONS ----------------

colA,colB = st.columns(2)

with colA:
    analyze = st.button("🔍 Analyze Tumor")

with colB:
    if st.button("🔄 Reset"):
        st.rerun()

# ---------------- ANALYSIS ----------------

if analyze:

    scaled = scaler.transform(input_data)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    if pred == 0:
        result = "Benign (Non‑Cancerous)"
        confidence = prob[0]*100
    else:
        result = "Malignant Tumor Detected"
        confidence = prob[1]*100

    st.markdown("---")
    st.header("Diagnosis Result")

    if pred == 0:
        st.success(result)
    else:
        st.error(result)

    st.write(f"Confidence Level: {confidence:.2f}%")

    # ---------------- CONFIDENCE BAR ----------------

    st.subheader("AI Confidence Meter")

    st.progress(int(confidence))

    # ---------------- FEATURE VISUALIZATION ----------------

    st.subheader("Tumor Feature Visualization")

    values=[radius,texture,perimeter,area,concave]

    fig, ax = plt.subplots()

    ax.bar(features,values,color="#1f77b4")

    ax.set_ylabel("Feature Value")
    ax.set_title("Tumor Feature Analysis")

    plt.xticks(rotation=35)

    st.pyplot(fig)

    # ---------------- PDF REPORT ----------------

    st.subheader("🧾 Download Hospital Report")

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).strftime("%d-%m-%Y %I:%M %p")

    pdf = FPDF()
    pdf.add_page()

    # PAGE BORDER
    pdf.set_draw_color(0,102,153)
    pdf.set_line_width(0.6)
    pdf.rect(8,8,194,277)

    # HEADER BAR
    pdf.set_fill_color(0,102,153)
    pdf.rect(8,8,194,22,"F")

    # LOGO
    try:
        pdf.image("logo.png",12,10,18)
    except:
        pass

    # TITLE
    pdf.set_xy(0,12)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Arial","B",16)
    pdf.cell(210,10,"NAP HOSPITAL DIAGNOSTIC REPORT",0,1,"C")

    pdf.ln(6)

    pdf.set_text_color(0,0,0)

    # ADDRESS
    pdf.set_font("Arial","",10)
    pdf.cell(0,6,"Beside Amrita Vishwa Vidyapeetham, Vengal",0,1,"C")
    pdf.cell(0,6,"Chennai, Tamil Nadu, India",0,1,"C")

    pdf.ln(4)

    pdf.cell(0,6,f"Report Generated: {now}",0,1)

    pdf.ln(5)

    # PATIENT DETAILS
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Patient Details",0,1)

    pdf.set_font("Arial","",11)
    pdf.cell(0,6,f"Name: {name}",0,1)
    pdf.cell(0,6,f"Age: {age}",0,1)
    pdf.cell(0,6,f"Gender: {gender}",0,1)
    pdf.cell(0,6,f"Patient ID: {patient_id}",0,1)
    pdf.cell(0,6,f"Doctor: {doctor}",0,1)

    pdf.ln(6)

    # FEATURE TABLE
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Tumor Measurements",0,1)

    pdf.set_font("Arial","B",11)
    pdf.cell(95,8,"Feature",1)
    pdf.cell(95,8,"Value",1)
    pdf.ln()

    pdf.set_font("Arial","",11)

    for f,v in zip(features,values):
        pdf.cell(95,8,f,1)
        pdf.cell(95,8,str(v),1)
        pdf.ln()

    pdf.ln(8)

    # RESULT
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Diagnosis Result",0,1)

    if pred == 0:
        pdf.set_text_color(0,150,0)
    else:
        pdf.set_text_color(200,0,0)

    pdf.set_font("Arial","B",13)
    pdf.cell(0,8,result,0,1)

    pdf.set_text_color(0,0,0)

    pdf.set_font("Arial","",11)
    pdf.cell(0,6,f"Confidence Level: {confidence:.2f}%",0,1)

    pdf.ln(18)

    # SIGNATURES
    pdf.cell(95,8,"__________________",0,0,"C")
    pdf.cell(95,8,"__________________",0,1,"C")

    pdf.cell(95,8,"AI Diagnostic System",0,0,"C")
    pdf.cell(95,8,"Doctor Signature",0,1,"C")

    pdf_output = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "⬇ Download Hospital Report",
        data=pdf_output,
        file_name="NAP_Hospital_Report.pdf",
        mime="application/pdf"
    )