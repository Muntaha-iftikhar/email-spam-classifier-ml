# ==========================
# 📧 Email Spam Classifier App
# Author: Shahzaib Asif
# ==========================

import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ========== LOAD MODEL & VECTORIZER ==========
model = pickle.load(open("svm_spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ========== TEXT PREPROCESSING FUNCTION ==========
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_email_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned)

# ========== STREAMLIT PAGE CONFIG ==========
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== HEADER SECTION ==========
st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-top: -30px;
    }
    .subtitle {
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 25px;
        font-size: 17px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== BACKGROUND (PROFESSIONAL DARK GRADIENT) ==========
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
}
[data-testid="stHeader"], [data-testid="stSidebar"] {
    background: none;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ========== MAIN INTERFACE ==========
st.markdown('<h1 class="main-title">📧 Email Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A Machine Learning App by Shahzaib Asif</p>', unsafe_allow_html=True)

# Text input area
user_email = st.text_area("✉️ Paste your email content below:", height=200, placeholder="Type or paste email content here...")

if st.button("🔍 Analyze Email"):
    if user_email.strip() == "":
        st.warning("⚠️ Please enter some email text to analyze.")
    else:
        # Preprocess and predict
        cleaned = clean_email_text(user_email)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.markdown('<div class="result-box" style="background-color:#d9534f;">🚫 SPAM EMAIL DETECTED!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="background-color:#5cb85c;">✅ This email is safe (Not Spam).</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("<br><center>Developed by <b>Shahzaib Asif</b></center>", unsafe_allow_html=True)
