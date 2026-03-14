# ---------------- PDF REPORT ----------------

st.subheader("🧾 Download Hospital Report")

ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist).strftime("%d-%m-%Y %I:%M %p")

pdf = FPDF()
pdf.add_page()

pdf.set_left_margin(15)
pdf.set_right_margin(15)

# LOGO (same alignment)
try:
    pdf.image("logo.png",15,10,28)
except:
    pass

# TITLE
pdf.set_font("Arial","B",18)
pdf.cell(0,10,"NAP HOSPITAL",0,1,"C")

pdf.set_font("Arial","",11)
pdf.cell(0,6,"Beside Amrita Vishwa Vidyapeetham",0,1,"C")
pdf.cell(0,6,"Vengal, Chennai, Tamil Nadu, India",0,1,"C")

pdf.ln(4)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Breast Cancer Diagnostic Report",0,1,"C")

pdf.set_font("Arial","",10)
pdf.cell(0,6,f"Generated: {now}",0,1,"R")

pdf.ln(6)

# -------- PATIENT DETAILS --------

pdf.set_font("Arial","B",12)
pdf.cell(0,8,"Patient Details",0,1)

pdf.set_font("Arial","",11)

pdf.cell(95,8,f"Name: {name}",1)
pdf.cell(95,8,f"Age: {age}",1)
pdf.ln()

pdf.cell(95,8,f"Gender: {gender}",1)
pdf.cell(95,8,f"Patient ID: {patient_id}",1)
pdf.ln()

pdf.cell(190,8,f"Doctor: {doctor}",1)

pdf.ln(10)

# -------- FEATURE TABLE --------

pdf.set_font("Arial","B",12)
pdf.cell(0,8,"Tumor Feature Values",0,1)

pdf.set_font("Arial","B",11)

pdf.cell(120,8,"Feature",1)
pdf.cell(70,8,"Value",1)

pdf.ln()

pdf.set_font("Arial","",11)

for f,v in zip(features,values):
    pdf.cell(120,8,f,1)
    pdf.cell(70,8,str(v),1)
    pdf.ln()

pdf.ln(10)

# -------- RESULT BOX --------

pdf.set_font("Arial","B",13)

if pred == 0:
    pdf.set_fill_color(210,255,210)
else:
    pdf.set_fill_color(255,210,210)

pdf.cell(0,12,f"Diagnosis Result: {result}",0,1,"C",True)

pdf.set_font("Arial","",12)
pdf.cell(0,8,f"Confidence Level: {confidence:.2f}%",0,1,"C")

pdf.ln(20)

# -------- SIGNATURE --------

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