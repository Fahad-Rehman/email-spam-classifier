import streamlit as st
import pickle
from PyPDF2 import PdfReader

model = pickle.load(open("models/model.pkl", "rb"))
Vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.set_page_config(page_title="Email Spam Classifier", page_icon=":email:", layout="centered")
st.title("Email Spam Classifier")

st.write('Paste the content of the email below and click "Predict" to determine if it is spam or not.')

email_text = st.text_area("Paste Email Content Here")
uploaded_pdf = st.file_uploader("Or upload a PDF file", type=["pdf"])

if uploaded_pdf:
    reader = PdfReader(uploaded_pdf)
    pdf_text = "".join(page.extract_text() or "" for page in reader.pages)
    email_text = pdf_text.strip()

if st.button("Classify"):
    if not email_text.strip():
        st.warning("Please provide email content either by pasting text or uploading a PDF file.")
    else:
        vec = Vectorizer.transform([email_text])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("The email is classified as: SPAM")
        else:
            st.success("The email is classified as: NOT SPAM")
            